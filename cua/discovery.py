"""One discovery run end to end: the model discovers, the artifact is compiled, then replay proves it.

    validate inputs -> sign on (deterministic) -> agent loop -> compile -> save -> verify by replay

The run's evidence directory holds the structured log, the masked model transcript, the recorded
steps (with rejected locator candidates), the compiled artifact, and the verification replay. A
discovery only counts as successful if the artifact it produced replays successfully without the
model, on the same inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cua.agent.goal import GoalSpec
from cua.agent.llm import Model
from cua.agent.loop import AgentOutcome, DiscoveryAgent
from cua.agent.recorder import Recorder, RecordingError, compile_capability
from cua.agent.tools import DiscoveryTools, tool_definitions
from cua.artifact.params import validate_values
from cua.artifact.schema import Capability
from cua.artifact.store import CapabilityLibrary
from cua.evidence.log import attach_handler, get_logger, run_context
from cua.evidence.sink import RunEvidence
from cua.policy import ApprovalLedger, Policy, PolicyGate
from cua.profile import AppProfile, StateKind
from cua.replay.engine import ReplayEngine
from cua.replay.redaction import capability_vocabulary, profile_vocabulary, redact_observation
from cua.replay.result import Result
from cua.runner import new_run_id, replay
from cua.security.masking import SecretRegistry, use_registry
from cua.surface.web import WebSurface

log = get_logger(__name__)


@dataclass(frozen=True)
class DiscoveryOutcome:
    run_id: str
    status: Literal["recorded", "agent_stopped", "compile_failed", "sign_on_failed"]
    reason: str
    evidence_dir: Path
    capability: Capability | None
    verification: Result | None  # replay of the compiled artifact, no model involved

    @property
    def verified(self) -> bool:
        return self.verification is not None and self.verification.type == "success"


def discover(
    goal: GoalSpec,
    params: Mapping[str, object],
    *,
    model: Model,
    base_url: str,
    library: CapabilityLibrary,
    profile: AppProfile,
    policy: Policy,
    evidence_root: Path,
    headless: bool = True,
    max_actions: int = 30,
    verify: bool = True,
) -> DiscoveryOutcome:
    values = validate_values(goal.inputs, params)  # same contract checks as replay
    sign_on = library.get(profile.sign_on.capability) if profile.sign_on else None
    credentials = profile.sign_on_params()

    run_id = new_run_id()
    registry = SecretRegistry()
    for spec in goal.inputs:
        registry.register(spec.name, values[spec.name], spec.sensitivity)
    evidence = RunEvidence(evidence_root, run_id, registry)
    gate = PolicyGate(policy, base_url=base_url, approvals=ApprovalLedger(evidence.path("no-approvals.json")))

    with (
        attach_handler(evidence.log_handler()),
        use_registry(registry),
        run_context(run_id, mode="discovery", capability_id=goal.id),
    ):
        log.info("discovery.started", goal=goal.id, model=model.name, base_url=base_url)
        with WebSurface.launch(base_url, headless=headless, request_filter=gate.allows_request) as surface:
            if sign_on is not None:
                states = [s for s in profile.states if s.kind is not StateKind.SESSION_EXPIRED]
                signed_on = ReplayEngine(surface, registry=registry, gate=gate, states=states).run(
                    sign_on, credentials
                )
                if signed_on.type != "success":
                    return _finish(evidence, run_id, "sign_on_failed", "could not sign on", None, None)
            surface.goto(goal.entry_route)

            recorder = Recorder(checker=surface, sensitive_values=set(values.values()))
            tools = DiscoveryTools(
                surface=surface,
                recorder=recorder,
                goal=goal,
                values=values,
                policy=policy,
                profile=profile,
                registry=registry,
            )
            agent = DiscoveryAgent(model, tools, tool_definitions(goal), max_actions=max_actions)
            outcome = agent.run(goal)

        capability: Capability | None = None
        stopped: tuple[Literal["agent_stopped", "compile_failed"], str] | None = None
        if not outcome.succeeded or tools.finish_text is None:
            stopped = ("agent_stopped", f"{outcome.status}: {outcome.reason}")
        else:
            try:
                capability = compile_capability(
                    goal,
                    recorder.steps,
                    success_text=tools.finish_text,
                    success_frame=tools.finish_frame,
                    sensitive_values=recorder.sensitive_values,
                    policy=policy,
                    run_id=run_id,
                    model=model.name,
                )
            except RecordingError as exc:
                stopped = ("compile_failed", str(exc))

        # Saved whatever happened (a stopped run's transcript is the most useful evidence of all).
        vocabulary = profile_vocabulary(profile)
        if capability is not None:
            vocabulary |= capability_vocabulary(capability)
        _save_session(evidence, outcome, recorder, vocabulary)
        if stopped is not None or capability is None:
            status, reason = stopped or ("compile_failed", "no artifact")
            return _finish(evidence, run_id, status, reason, None, None)

        # The artifact lives with its evidence; verification replays it from there with no model.
        run_library = CapabilityLibrary(evidence.path("capabilities"))
        run_library.save(capability)
        if sign_on is not None:
            run_library.save(sign_on)
        log.info("discovery.compiled", steps=len(capability.steps), content_hash=capability.content_hash())

    verification = None
    if verify:
        verification = replay(
            capability.id,
            params,
            base_url=base_url,
            library=run_library,
            profile=profile,
            policy=policy,
            approvals=ApprovalLedger(evidence.path("no-approvals.json")),
            evidence_root=evidence.path("verification"),
            headless=headless,
        ).result
    return _finish(evidence, run_id, "recorded", "artifact compiled", capability, verification)


def _save_session(
    evidence: RunEvidence, outcome: AgentOutcome, recorder: Recorder, vocabulary: frozenset[str]
) -> None:
    # The transcript records what the model saw. On top of the sink's masking, observations are
    # allowlist-redacted (like failure screenshots): screen values no pattern recognizes, such as a
    # phone number or another account's balance, must not persist.
    evidence.save_json("transcript.json", [_redact_message(m, vocabulary) for m in outcome.transcript])
    evidence.save_json(
        "recorded_steps.json",
        [
            {
                "kind": s.kind,
                "intent": s.intent,
                "ref": s.element.ref,
                "role": s.element.role,
                "kept": s.target.model_dump(mode="json", exclude_none=True),
                "rejected": s.rejected,
                "expect_text": s.expect_text,
            }
            for s in recorder.steps
        ],
    )
    evidence.save_json(
        "agent_outcome.json",
        {
            "status": outcome.status,
            "reason": outcome.reason,
            "actions": outcome.actions,
            "turns": outcome.turns,
            "usage": vars(outcome.usage),
        },
    )


def _redact_message(message: dict[str, Any], vocabulary: frozenset[str]) -> dict[str, Any]:
    content = message["content"]
    if isinstance(content, str):
        return {**message, "content": redact_observation(content, vocabulary)}
    blocks = [
        {**b, "content": redact_observation(b["content"], vocabulary)}
        if b.get("type") == "tool_result" and isinstance(b.get("content"), str)
        else b
        for b in content
    ]
    return {**message, "content": blocks}


def _finish(
    evidence: RunEvidence,
    run_id: str,
    status: Literal["recorded", "agent_stopped", "compile_failed", "sign_on_failed"],
    reason: str,
    capability: Capability | None,
    verification: Result | None,
) -> DiscoveryOutcome:
    result = DiscoveryOutcome(run_id, status, reason, evidence.dir, capability, verification)
    evidence.save_json(
        "discovery_result.json",
        {
            "run_id": run_id,
            "status": status,
            "reason": reason,
            "verified": result.verified,
            "capability": capability.id if capability else None,
            "content_hash": capability.content_hash() if capability else None,
            "verification": verification.model_dump(mode="json") if verification else None,
        },
    )
    log.info("discovery.finished", status=status, reason=reason, verified=result.verified)
    return result
