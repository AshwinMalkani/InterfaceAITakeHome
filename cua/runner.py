"""One replay run end to end: sign on, run the capability, write evidence.

A run owns: a run id, a SecretRegistry (so every value it touches is masked on the way out),
an evidence directory (events.jsonl, result.json, failure screenshots), and one browser session.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cua.artifact.binding import load_binding, specialize
from cua.artifact.store import CapabilityLibrary
from cua.evidence.log import attach_handler, get_logger, run_context
from cua.evidence.sink import RunEvidence
from cua.hitl.handoff import GuardedSurface, LiveHandoff
from cua.hitl.store import HitlStore
from cua.policy import ApprovalLedger, Policy, PolicyGate, PolicyViolation
from cua.profile import AppProfile, StateKind
from cua.replay.engine import ReplayEngine
from cua.replay.redaction import profile_vocabulary
from cua.replay.result import Result
from cua.security.masking import SecretRegistry, use_registry
from cua.surface.base import Surface
from cua.surface.web import WebSurface

log = get_logger(__name__)


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    result: Result  # in-memory result holds real output values; mask before displaying/persisting
    evidence_dir: Path
    registry: SecretRegistry


def new_run_id() -> str:
    return f"run_{datetime.now(UTC):%Y%m%dT%H%M%S}_{secrets.token_hex(3)}"


def replay(
    capability_id: str,
    params: Mapping[str, object],
    *,
    base_url: str,
    library: CapabilityLibrary,
    profile: AppProfile,
    policy: Policy,
    approvals: ApprovalLedger,
    evidence_root: Path,
    allow_irreversible: bool = False,
    unredacted_screenshots: bool = False,
    request_id: str | None = None,
    headless: bool = True,
    slow_mo_ms: int = 0,
    hitl: HitlStore | None = None,
    cdp_port: int | None = None,
    unclaimed_timeout_s: float = 900,
    tenant: str | None = None,
) -> RunOutcome:
    """Replay one capability. With `hitl`, failures a person can resolve pause the run on the same live
    session and wait for an operator (see cua/hitl); without it, they are returned as failures."""
    capability = library.get(capability_id)
    if capability.app.product != profile.product:
        raise ValueError(f"{capability_id} targets {capability.app.product!r}, not {profile.product!r}")
    sign_on = library.get(profile.sign_on.capability) if profile.sign_on else None
    if tenant is not None:  # the vendor-level artifact, as it must run on this tenant
        binding = load_binding(tenant, profile.product)
        capability = specialize(capability, binding)
        sign_on = specialize(sign_on, binding) if sign_on else None
    credentials = profile.sign_on_params()  # fail fast on missing config, before opening a browser
    gate = PolicyGate(
        policy, base_url=base_url, approvals=approvals, allow_irreversible=allow_irreversible, tenant=tenant
    )
    if unredacted_screenshots and not policy.allow_unredacted_screenshots:
        raise PolicyViolation(f"policy for {policy.product!r} does not allow unredacted screenshots")
    ui_vocabulary, redact = profile_vocabulary(profile), not unredacted_screenshots

    run_id = new_run_id()
    registry = SecretRegistry()
    evidence = RunEvidence(evidence_root, run_id, registry)
    with (
        attach_handler(evidence.log_handler()),
        use_registry(registry),
        run_context(run_id, request_id=request_id, mode="replay", tenant=tenant),
    ):
        log.info(
            "run.started", capability=capability_id, base_url=base_url, allow_irreversible=allow_irreversible
        )
        result: Result | None = ReplayEngine.preflight(capability, params, gate)
        if result is None:
            with WebSurface.launch(
                base_url,
                headless=headless,
                slow_mo_ms=slow_mo_ms,
                request_filter=gate.allows_request,
                cdp_port=cdp_port,
            ) as surface:
                acting: Surface = surface
                handoff: LiveHandoff | None = None
                if hitl is not None:
                    hitl.registry = registry  # anything written to the store is masked with this run's values
                    hitl.open_session(run_id)
                    acting = GuardedSurface(surface, hitl, run_id)
                    live = "the open browser window"
                    if cdp_port:
                        live = f"CDP endpoint http://127.0.0.1:{cdp_port}"
                    handoff = LiveHandoff(
                        hitl,
                        session_id=run_id,
                        run_id=run_id,
                        surface=surface,
                        live_session=live,
                        unclaimed_timeout_s=unclaimed_timeout_s,
                    )
                # Signing on can't itself recover from "session expired", so that state is excluded.
                sign_on_engine = ReplayEngine(
                    surface,
                    registry=registry,
                    evidence=evidence,
                    gate=gate,
                    ui_vocabulary=ui_vocabulary,
                    redact_screenshots=redact,
                    states=[s for s in profile.states if s.kind is not StateKind.SESSION_EXPIRED],
                )

                def reauthenticate() -> bool:
                    return sign_on is not None and sign_on_engine.run(sign_on, credentials).type == "success"

                if sign_on is not None:
                    signed_on = sign_on_engine.run(sign_on, credentials)
                    if signed_on.type != "success":
                        result = signed_on
                if result is None:
                    engine = ReplayEngine(
                        acting,
                        registry=registry,
                        evidence=evidence,
                        gate=gate,
                        states=profile.states,
                        reauthenticate=reauthenticate,
                        escalation=handoff,
                        ui_vocabulary=ui_vocabulary,
                        redact_screenshots=redact,
                    )
                    result = engine.run(capability, params)
        result.capability.tenant = tenant
        evidence.save_json("result.json", result.model_dump(mode="json"))
        log.info("run.finished", result=result.type)
    return RunOutcome(run_id, result, evidence.dir, registry)
