"""Deterministic replay: run a capability's steps with no LLM in the loop.

Per step: scan for known/unknown states, resolve the target (first unique strategy wins; a
fallback is a drift warning), act, reject undeclared dialogs, then wait for the post-condition.

Every poll of every wait is a race, checked in this order:
  1. dialogs          an undeclared native dialog fails the step (it was dismissed, never accepted)
  2. known states     from the app profile, handled by kind:
                        interstitial     -> click its safe dismiss control, keep waiting (bounded)
                        session_expired  -> sign on again and restart, only if nothing irreversible ran
                        app_error        -> hard failure
                        business         -> business outcome shared by all capabilities
  3. unknown overlay  something undeclared covers the screen -> failure that needs a human
  4. outcomes         the capability's declared business outcomes (active from `after_step` on)
  5. post-condition   only now can the step succeed
States are checked *before* the post-condition on purpose: the expected text can be present
underneath a blocking modal, and "success" must never be reported through one.

Policy (when a PolicyGate is given): rendered routes are checked before navigating, requests the
surface blocked fail the step, and a step above the policy's unattended risk stops *before acting*
unless the capability's current content is approved and the caller allowed irreversible actions.

Human handoff (when an escalation handler is given): a failure a person can resolve (see
ESCALATABLE) pauses the run on the same live session instead of ending it. The operator's decision
comes back as resume-at-step, approve (irreversible steps only, one-shot) or abort; on resume the
state scan runs again before anything acts, because the human may have left the screen anywhere.

Retries: a step is retried once after a post-condition timeout only if its risk is `safe`.
Irreversible steps are never retried or restarted: repeating "Confirm" on a legacy app with no
idempotency token opens a second account.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from cua.artifact.params import InputError, OutputParseError, parse_output, render, validate_inputs
from cua.artifact.schema import (
    TEMPLATE,
    Capability,
    Checkpoint,
    Click,
    DialogExpectation,
    Extract,
    Fill,
    Navigate,
    Press,
    Risk,
    Select,
    Step,
    Target,
)
from cua.evidence.log import get_logger, log_context, span
from cua.evidence.sink import RunEvidence
from cua.hitl.models import Action, Resolution
from cua.policy import PolicyGate, PolicyViolation
from cua.profile import KnownState, StateKind
from cua.replay.describe import describe_checkpoint, describe_locator, describe_target
from cua.replay.redaction import capability_vocabulary
from cua.replay.result import (
    ESCALATABLE,
    NEEDS_HUMAN,
    TRANSIENT,
    BusinessOutcome,
    CapabilityRef,
    Escalated,
    Failure,
    FailureCategory,
    InterventionSummary,
    Recovery,
    ReplayWarning,
    Result,
    Success,
)
from cua.security.masking import SecretRegistry, Sensitivity
from cua.surface.base import ActionFailed, Resolved, Surface, TargetNotFound

log = get_logger(__name__)

POLL_MS = 100
SUCCESS_TIMEOUT_MS = 10_000
DISMISS_TIMEOUT_MS = 5_000
MAX_INTERSTITIALS = 3  # per invocation; more means we're looping, not recovering
MAX_RESTARTS = 1  # session restarts per invocation


class StepFailure(Exception):
    def __init__(
        self,
        category: FailureCategory,
        step_id: str,
        message: str,
        expected: str | None = None,
        needs_human: bool = False,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.step_id = step_id
        self.message = message
        self.expected = expected
        self.needs_human = needs_human or category in NEEDS_HUMAN


class _OutcomeReached(Exception):
    def __init__(self, name: str, description: str, step_id: str) -> None:
        super().__init__(name)
        self.name = name
        self.description = description
        self.step_id = step_id


class _Aborted(Exception):
    def __init__(self, failure: StepFailure, resolution: Resolution) -> None:
        super().__init__("aborted by operator")
        self.failure = failure
        self.resolution = resolution


class _Unclaimed(Exception):
    def __init__(self, failure: StepFailure, intervention_id: str) -> None:
        super().__init__("no operator claimed the intervention in time")
        self.failure = failure
        self.intervention_id = intervention_id


class EscalationHandler(Protocol):
    def escalate(
        self,
        *,
        kind: str,
        capability_id: str,
        step_id: str,
        intent: str,
        reason: str,
        steps: list[tuple[str, str]],
        screenshot: str | None,
        locations: dict[str, str],
    ) -> Resolution: ...


class _SessionLost(Exception):
    def __init__(self, step_id: str) -> None:
        super().__init__("session lost")
        self.step_id = step_id


@dataclass
class _RunState:
    capability: Capability
    values: dict[str, str]
    started: float = field(default_factory=time.monotonic)
    outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[ReplayWarning] = field(default_factory=list)
    recoveries: list[Recovery] = field(default_factory=list)
    step_index: int = -1  # -1 = entry; len(steps) = final success check
    irreversible_started: bool = False  # set *before* acting: if it may have happened, assume it did
    interstitials: int = 0
    approved_steps: set[str] = field(default_factory=set)  # one-shot operator approvals
    interventions: list[InterventionSummary] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)

    def restart(self) -> None:
        self.outputs.clear()
        self.step_index = -1


def sensitive_targets(capability: Capability) -> list[Target]:
    """Controls that hold sensitive data, per the artifact's own declarations. Masked in screenshots."""
    sensitive_inputs = {p.name for p in capability.inputs if p.sensitivity is not Sensitivity.PUBLIC}
    sensitive_outputs = {o.name for o in capability.outputs if o.sensitivity is not Sensitivity.PUBLIC}
    targets: list[Target] = []
    for step in capability.steps:
        match step.action:
            case Fill(target=target, value=value) if set(TEMPLATE.findall(value)) & sensitive_inputs:
                targets.append(target)
            case Extract(target=target, output=output) if output in sensitive_outputs:
                targets.append(target)
    return targets


def _ref(capability: Capability) -> CapabilityRef:
    return CapabilityRef(id=capability.id, version=capability.version, content_hash=capability.content_hash())


class ReplayEngine:
    def __init__(
        self,
        surface: Surface,
        *,
        registry: SecretRegistry,
        evidence: RunEvidence | None = None,
        states: Sequence[KnownState] = (),
        reauthenticate: Callable[[], bool] | None = None,
        gate: PolicyGate | None = None,
        escalation: EscalationHandler | None = None,
        ui_vocabulary: frozenset[str] = frozenset(),
        redact_screenshots: bool = True,
        success_timeout_ms: int = SUCCESS_TIMEOUT_MS,
        dismiss_timeout_ms: int = DISMISS_TIMEOUT_MS,
    ) -> None:
        self.surface = surface
        self.registry = registry
        self.evidence = evidence
        self.states = list(states)
        self.reauthenticate = reauthenticate
        self.gate = gate
        self.escalation = escalation
        self.ui_vocabulary = ui_vocabulary
        self.redact_screenshots = redact_screenshots
        self.success_timeout_ms = success_timeout_ms
        self.dismiss_timeout_ms = dismiss_timeout_ms

    @staticmethod
    def check_inputs(capability: Capability, params: Mapping[str, object]) -> Failure | None:
        """Validate params without touching the surface (callers can skip launching a browser)."""
        try:
            validate_inputs(capability, params)
        except InputError as exc:
            return Failure(
                capability=_ref(capability),
                category=FailureCategory.INVALID_INPUT,
                step_id=None,
                message=str(exc),
                retryable=False,
                duration_ms=0,
            )
        return None

    @staticmethod
    def preflight(
        capability: Capability, params: Mapping[str, object], gate: PolicyGate | None
    ) -> Failure | None:
        """Everything checkable before a browser opens: the input contract, then the policy."""
        if (invalid := ReplayEngine.check_inputs(capability, params)) is not None:
            return invalid
        if gate is not None and (problems := gate.preflight(capability)):
            return Failure(
                capability=_ref(capability),
                category=FailureCategory.POLICY_DENIED,
                step_id=None,
                message="; ".join(problems),
                retryable=False,
                duration_ms=0,
            )
        return None

    def run(self, capability: Capability, params: Mapping[str, object]) -> Result:
        with log_context(capability_id=capability.id, capability_version=capability.version):
            result = self._run(capability, params)
            details: dict[str, Any] = {}
            if isinstance(result, Failure):
                details = {
                    "category": result.category,
                    "step_id": result.step_id,
                    "needs_human": result.needs_human,
                }
            elif isinstance(result, BusinessOutcome):
                details = {"outcome": result.outcome, "step_id": result.step_id}
            log.info("capability.finished", result=result.type, duration_ms=result.duration_ms, **details)
            return result

    def _run(self, capability: Capability, params: Mapping[str, object]) -> Result:
        started = time.monotonic()
        log.info("capability.started", steps=len(capability.steps), max_risk=capability.max_risk)
        if (rejected := self.preflight(capability, params, self.gate)) is not None:
            log.warning("capability.rejected", category=rejected.category, message=rejected.message)
            return rejected
        values = validate_inputs(capability, params)
        for spec in capability.inputs:
            self.registry.register(spec.name, values[spec.name], spec.sensitivity)

        state = _RunState(capability, values, started=started)
        restarts = 0
        start = 0  # step index to (re)start from; 0 = from the entry route
        while True:
            try:
                try:
                    self._execute(state, start)
                except _SessionLost as lost:
                    failure = self._restore_session(state, lost, restarts)
                    if failure is None:
                        restarts, start = restarts + 1, 0
                        continue
                    start = self._with_human(failure, state)  # e.g. expired after an irreversible step
                    continue
            except _OutcomeReached as outcome:
                log.info("outcome.detected", outcome=outcome.name, step_id=outcome.step_id)
                return BusinessOutcome(
                    capability=_ref(capability),
                    outcome=outcome.name,
                    description=outcome.description,
                    step_id=outcome.step_id,
                    warnings=state.warnings,
                    recoveries=state.recoveries,
                    interventions=state.interventions,
                    duration_ms=state.duration_ms,
                )
            except _Aborted as aborted:
                note = aborted.resolution.note or "no note"
                failure = aborted.failure
                operator = aborted.resolution.operator
                failure.message = f"{failure.message}; aborted by operator {operator}: {note}"
                failure.needs_human = True
                return self._fail(state, failure)
            except _Unclaimed as unclaimed:
                log.warning("capability.escalated", intervention_id=unclaimed.intervention_id)
                return Escalated(
                    capability=_ref(capability),
                    intervention_id=unclaimed.intervention_id,
                    step_id=unclaimed.failure.step_id,
                    reason=unclaimed.failure.message,
                    warnings=state.warnings,
                    recoveries=state.recoveries,
                    interventions=state.interventions,
                    duration_ms=state.duration_ms,
                )
            except StepFailure as failure:
                return self._fail(state, failure)
            return Success(
                capability=_ref(capability),
                outputs=state.outputs,
                warnings=state.warnings,
                recoveries=state.recoveries,
                interventions=state.interventions,
                duration_ms=state.duration_ms,
            )

    def _execute(self, state: _RunState, start: int = 0) -> None:
        """Run steps from `start` (0 also navigates to the entry route), then verify success."""
        capability = state.capability
        if start == 0:
            with span(step_id="entry"):
                self._goto(render(capability.entry_route, state.values), "entry")
        index = start
        while True:
            state.step_index = index
            try:
                if index >= len(capability.steps):
                    with span(step_id="success"):
                        self._await(capability.success, self.success_timeout_ms, "success", state)
                    return
                step = capability.steps[index]
                with span(step_id=step.id):
                    self._run_step(step, state)
                index += 1
            except StepFailure as failure:
                index = self._with_human(failure, state)

    def _with_human(self, failure: StepFailure, state: _RunState) -> int:
        """Escalate a failure a person can resolve; return the step index to continue from.

        Re-raises the failure if there's no handler or a human can't help (policy, bad input).
        """
        if self.escalation is None or failure.category not in ESCALATABLE:
            raise failure
        steps = state.capability.steps
        ids = [s.id for s in steps]
        step = steps[ids.index(failure.step_id)] if failure.step_id in ids else None
        screenshot = self._screenshot(state, f"{state.capability.id}.{failure.step_id}.intervention.png")
        kind = "approval" if failure.category is FailureCategory.APPROVAL_REQUIRED else failure.category.value
        with span(step_id=failure.step_id, phase="intervention"):  # human actions belong to this step
            resolution = self._escalate(kind, failure, state, step, screenshot)
        return self._apply_resolution(resolution, kind, failure, state)

    def _escalate(
        self, kind: str, failure: StepFailure, state: _RunState, step: Step | None, screenshot: str | None
    ) -> Resolution:
        assert self.escalation is not None
        steps = state.capability.steps
        return self.escalation.escalate(
            kind=kind,
            capability_id=state.capability.id,
            step_id=failure.step_id,
            intent=step.intent if step else failure.step_id,
            reason=failure.message,
            steps=[(s.id, s.intent) for s in steps],
            screenshot=screenshot,
            locations=self.surface.observe().locations,
        )

    def _apply_resolution(
        self, resolution: Resolution, kind: str, failure: StepFailure, state: _RunState
    ) -> int:
        """Turn the operator's decision into the step index to continue from (or end the run)."""
        steps = state.capability.steps
        ids = [s.id for s in steps]
        state.interventions.append(
            InterventionSummary(
                id=resolution.intervention_id,
                kind=kind,
                step_id=failure.step_id,
                action=resolution.action.value if resolution.action else None,
                operator=resolution.operator,
                resume_step=resolution.resume_step,
                human_actions=len(resolution.human_actions),
            )
        )
        if resolution.timed_out:
            raise _Unclaimed(failure, resolution.intervention_id)
        if resolution.action is Action.ABORT:
            raise _Aborted(failure, resolution)
        if resolution.action is Action.APPROVE and failure.category is FailureCategory.APPROVAL_REQUIRED:
            state.approved_steps.add(failure.step_id)
            return ids.index(failure.step_id)
        # Resume: at the operator's chosen step, else retry where it failed ("success" = final check).
        target = resolution.resume_step or failure.step_id
        if target in ids:
            return ids.index(target)
        return len(steps)

    def _restore_session(self, state: _RunState, lost: _SessionLost, restarts: int) -> StepFailure | None:
        """Sign on again and restart from the entry route, if (and only if) that is safe."""
        log.warning("state.detected", state="session_expired", step_id=lost.step_id)
        if state.irreversible_started:
            return StepFailure(
                FailureCategory.SESSION_EXPIRED,
                lost.step_id,
                "session expired after an irreversible step started; its effect is unknown",
                needs_human=True,
            )
        if self.reauthenticate is None or restarts >= MAX_RESTARTS:
            return StepFailure(
                FailureCategory.SESSION_EXPIRED, lost.step_id, "session expired and could not be restored"
            )
        if not self.reauthenticate():
            return StepFailure(FailureCategory.SESSION_EXPIRED, lost.step_id, "re-authentication failed")
        state.recoveries.append(
            Recovery(
                kind="reauthenticated",
                step_id=lost.step_id,
                detail="signed on again and restarted from the entry route",
            )
        )
        log.info("recovery.applied", kind="reauthenticated", step_id=lost.step_id)
        state.restart()
        return None

    # --- steps ----------------------------------------------------------------------------

    def _run_step(self, step: Step, state: _RunState) -> None:
        started = time.monotonic()
        log.info("step.started", intent=step.intent, action=step.action.kind, risk=step.risk)
        self._scan(step.id, state)  # never act on a screen that is in a known or blocking state
        resolved = self._act(step, state)
        if step.expect is not None:
            try:
                self._await(step.expect, step.timeout_ms, step.id, state)
            except StepFailure as failure:
                timed_out = failure.category is FailureCategory.CHECKPOINT_FAILED
                if not (timed_out and self._risk(step) is Risk.SAFE):
                    raise
                state.recoveries.append(
                    Recovery(
                        kind="retried_step",
                        step_id=step.id,
                        detail="post-condition timed out; safe step retried once",
                    )
                )
                log.warning("recovery.applied", kind="retried_step")
                resolved = self._act(step, state)
                self._await(step.expect, step.timeout_ms, step.id, state)
        log.info(
            "step.succeeded",
            duration_ms=int((time.monotonic() - started) * 1000),
            strategy=resolved.strategy_kind if resolved else None,
        )

    def _act(self, step: Step, state: _RunState) -> Resolved | None:
        action = step.action
        target = getattr(action, "target", None)
        resolved = self._resolve(step, target, state) if target is not None else None
        risk = self._risk(step)
        if self.gate is not None and self.gate.policy.needs_approval(step):
            if step.id in state.approved_steps:
                state.approved_steps.discard(step.id)  # one-shot: a retry needs a new approval
                log.info("policy.irreversible_approved_by_operator", intent=step.intent)
            elif (blocker := self.gate.irreversible_blocker(state.capability)) is not None:
                raise StepFailure(
                    FailureCategory.APPROVAL_REQUIRED,
                    step.id,
                    f"stopped before {risk} step {step.intent!r}: {blocker}",
                )
            else:
                log.info("policy.irreversible_authorized", intent=step.intent)
        if risk is Risk.IRREVERSIBLE:
            state.irreversible_started = True
        try:
            match action:
                case Navigate(route=route):
                    self._goto(render(route, state.values), step.id)
                case Click(dialog=dialog):
                    assert resolved is not None
                    self.surface.click(resolved, dialog, step.timeout_ms)
                case Fill(value=value):
                    assert resolved is not None
                    self.surface.fill(resolved, render(value, state.values), step.timeout_ms)
                case Select(option=option):
                    assert resolved is not None
                    self.surface.select(resolved, render(option, state.values), step.timeout_ms)
                case Press(key=key):
                    self.surface.press(key, resolved, step.timeout_ms)
                case Extract():
                    assert resolved is not None
                    self._extract(step, action, resolved, state)
        except ActionFailed as exc:
            assert target is not None
            raise StepFailure(
                FailureCategory.ACTION_FAILED,
                step.id,
                f"{action.kind} failed: {exc}",
                expected=f"{action.kind} on {describe_target(target)}",
            ) from None
        self._check_dialogs(step.id, expected=action.dialog if isinstance(action, Click) else None)
        return resolved

    def _risk(self, step: Step) -> Risk:
        """Effective risk: the policy's heuristic can raise a step's declared risk, never lower it."""
        return self.gate.policy.effective_risk(step) if self.gate else step.risk

    def _resolve(self, step: Step, target: Target, state: _RunState) -> Resolved:
        try:
            resolved = self.surface.resolve(target, step.timeout_ms)
        except TargetNotFound as exc:
            self._scan(step.id, state)  # a known state (e.g. expired session) explains it better
            tried = "; ".join(
                f"{describe_locator(spec)}: {'ambiguous, ' if n > 1 else ''}{n} match{'es' if n != 1 else ''}"
                for spec, n in zip(target.strategies, exc.match_counts, strict=True)
            )
            raise StepFailure(
                FailureCategory.TARGET_NOT_FOUND,
                step.id,
                f"no unique match ({tried})",
                expected=describe_target(target),
            ) from None
        if resolved.strategy_index > 0:
            detail = (
                f"preferred {describe_locator(target.strategies[0])!r} did not match; "
                f"used fallback #{resolved.strategy_index} ({resolved.strategy_kind})"
            )
            state.warnings.append(ReplayWarning(kind="locator_fallback", step_id=step.id, detail=detail))
            log.warning("locator.fallback", detail=detail)
        return resolved

    def _extract(self, step: Step, action: Extract, resolved: Resolved, state: _RunState) -> None:
        text = self.surface.read_text(resolved)
        spec = next(o for o in state.capability.outputs if o.name == action.output)
        self.registry.register(spec.name, text.strip(), spec.sensitivity)  # mask the raw text too
        try:
            value = parse_output(text, action.parse)
        except OutputParseError as exc:
            raise StepFailure(
                FailureCategory.OUTPUT_PARSE_FAILED,
                step.id,
                f"{spec.name}: {exc}",
                expected=f"{spec.type} from {describe_target(action.target)}",
            ) from None
        self.registry.register(spec.name, value, spec.sensitivity)
        state.outputs[spec.name] = value
        log.info("output.extracted", output=spec.name, value=value)

    def _goto(self, route: str, step_id: str) -> None:
        if self.gate is not None:
            try:
                self.gate.check_route(route)
            except PolicyViolation as exc:
                raise StepFailure(FailureCategory.POLICY_DENIED, step_id, str(exc)) from None
        try:
            self.surface.goto(route)
        except ActionFailed as exc:
            raise StepFailure(
                FailureCategory.NAVIGATION_FAILED,
                step_id,
                f"navigation failed: {exc}",
                expected=f"page {route!r} loads",
            ) from None

    # --- waiting and state detection ------------------------------------------------------

    def _await(self, checkpoint: Checkpoint, timeout_ms: int, step_id: str, state: _RunState) -> None:
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            self._scan(step_id, state)
            if self.surface.check(checkpoint):
                # Re-scan before accepting: the page may have changed between the scan above and the
                # check, and states must be judged on a page at least as new as the one that passed.
                self._scan(step_id, state)
                return
            if time.monotonic() >= deadline:
                raise StepFailure(
                    FailureCategory.CHECKPOINT_FAILED,
                    step_id,
                    f"not reached within {timeout_ms} ms",
                    expected=describe_checkpoint(checkpoint),
                )
            self.surface.idle(POLL_MS)

    def _scan(self, step_id: str, state: _RunState) -> None:
        """Steps 1-4 of the race (see module docstring). Returns normally if nothing is in the way."""
        if blocked := self.surface.take_blocked_requests():
            raise StepFailure(
                FailureCategory.POLICY_DENIED,
                step_id,
                f"blocked {len(blocked)} request(s) outside the allowlist, first: {blocked[0]}",
            )
        self._check_dialogs(step_id)
        for known in self.states:
            if not self.surface.check(known.when):
                continue
            log.info("state.detected", state=known.name, kind=known.kind)
            match known.kind:
                case StateKind.INTERSTITIAL:
                    self._dismiss(known, step_id, state)
                    return  # the page changed; the caller's next poll re-scans
                case StateKind.SESSION_EXPIRED:
                    raise _SessionLost(step_id)
                case StateKind.APP_ERROR:
                    raise StepFailure(
                        FailureCategory.APP_ERROR,
                        step_id,
                        f"application reported an error ({known.name}): {known.description}",
                    )
                case StateKind.BUSINESS:
                    raise _OutcomeReached(known.name, known.description, step_id)
        if (frame := self.surface.blocking_overlay()) is not None:
            raise StepFailure(
                FailureCategory.UNKNOWN_STATE,
                step_id,
                f"an undeclared overlay is blocking frame {frame!r}; not proceeding",
            )
        for outcome in state.capability.outcomes:
            after = outcome.after_step
            active = after is None or state.step_index >= self._step_index(state, after)
            if active and self.surface.check(outcome.when):
                raise _OutcomeReached(outcome.name, outcome.description, step_id)

    @staticmethod
    def _step_index(state: _RunState, step_id: str) -> int:
        return next(i for i, s in enumerate(state.capability.steps) if s.id == step_id)

    def _dismiss(self, known: KnownState, step_id: str, state: _RunState) -> None:
        state.interstitials += 1
        if state.interstitials > MAX_INTERSTITIALS:
            raise StepFailure(
                FailureCategory.UNKNOWN_STATE,
                step_id,
                f"interstitial {known.name!r} keeps reappearing; not looping",
                needs_human=True,
            )
        assert known.dismiss is not None  # guaranteed by KnownState validation
        cannot_dismiss = StepFailure(
            FailureCategory.UNKNOWN_STATE,
            step_id,
            f"known interstitial {known.name!r} could not be dismissed",
            needs_human=True,
        )
        try:
            resolved = self.surface.resolve(known.dismiss, self.dismiss_timeout_ms)
            self.surface.click(resolved, None, self.dismiss_timeout_ms)
        except (TargetNotFound, ActionFailed):
            raise cannot_dismiss from None
        self._check_dialogs(step_id)
        deadline = time.monotonic() + self.dismiss_timeout_ms / 1000
        while self.surface.check(known.when):
            if time.monotonic() >= deadline:
                raise cannot_dismiss
            self.surface.idle(POLL_MS)
        state.recoveries.append(Recovery(kind="dismissed_interstitial", step_id=step_id, detail=known.name))
        log.info("recovery.applied", kind="dismissed_interstitial", state=known.name)

    def _check_dialogs(self, step_id: str, expected: DialogExpectation | None = None) -> None:
        events = self.surface.take_dialogs()
        for event in events:
            log.info(
                "dialog.handled", expected=event.expected, accepted=event.accepted, message=event.message
            )
        if unexpected := [e for e in events if not e.expected]:
            raise StepFailure(
                FailureCategory.UNEXPECTED_DIALOG,
                step_id,
                f"unexpected dialog was dismissed: {unexpected[0].message!r}",
            )
        if expected is not None and not any(e.expected for e in events):
            raise StepFailure(
                FailureCategory.EXPECTED_DIALOG_MISSING,
                step_id,
                "the declared confirmation dialog did not appear",
                expected=f"dialog containing {expected.message_contains!r}",
            )

    # --- failure evidence -----------------------------------------------------------------

    def _screenshot(self, state: _RunState, name: str) -> str | None:
        """Redacted screenshot into the run's evidence; None if there's nowhere to put it or it failed."""
        if self.evidence is None:
            return None
        vocabulary = None
        if self.redact_screenshots:
            vocabulary = capability_vocabulary(state.capability) | self.ui_vocabulary
        try:
            shot = self.surface.screenshot(mask=sensitive_targets(state.capability), vocabulary=vocabulary)
        except ActionFailed:  # never let evidence capture hide the real problem
            log.warning("evidence.screenshot_failed", name=name)
            return None
        return self.evidence.save_image(name, shot).name

    def _fail(self, state: _RunState, failure: StepFailure) -> Failure:
        observed = self.surface.observe()
        shot = self._screenshot(state, f"{state.capability.id}.{failure.step_id}.failure.png")
        evidence = [shot] if shot else []
        log.error(
            "step.failed",
            category=failure.category,
            message=failure.message,
            expected=failure.expected,
            observed=observed.locations,
            needs_human=failure.needs_human,
        )
        return Failure(
            capability=_ref(state.capability),
            category=failure.category,
            step_id=failure.step_id,
            message=failure.message,
            expected=failure.expected,
            observed={"title": observed.title, "locations": observed.locations},
            retryable=TRANSIENT[failure.category] and not state.irreversible_started,
            needs_human=failure.needs_human,
            evidence=evidence,
            warnings=state.warnings,
            recoveries=state.recoveries,
            interventions=state.interventions,
            duration_ms=state.duration_ms,
        )
