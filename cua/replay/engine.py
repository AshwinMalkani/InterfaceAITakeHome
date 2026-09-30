"""Deterministic replay: run a capability's steps with no LLM in the loop.

For each step: resolve the target (first unique strategy wins; a fallback is a drift warning),
act, check that no undeclared dialog appeared, then wait for the step's post-condition. The
first thing that goes wrong stops the run with a Failure naming the step, what was expected,
and what was observed, plus a screenshot with sensitive controls masked.

Waiting is always on state (post-conditions polled via the surface), never fixed sleeps.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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
    Select,
    Step,
    Target,
)
from cua.evidence.log import get_logger, log_context, span
from cua.evidence.sink import RunEvidence
from cua.replay.describe import describe_checkpoint, describe_locator, describe_target
from cua.replay.result import (
    RETRYABLE,
    CapabilityRef,
    Failure,
    FailureCategory,
    ReplayWarning,
    Result,
    Success,
)
from cua.security.masking import SecretRegistry, Sensitivity
from cua.surface.base import ActionFailed, Resolved, Surface, TargetNotFound

log = get_logger(__name__)

POLL_MS = 100
SUCCESS_TIMEOUT_MS = 10_000


class StepFailure(Exception):
    def __init__(
        self, category: FailureCategory, step_id: str, message: str, expected: str | None = None
    ) -> None:
        super().__init__(message)
        self.category = category
        self.step_id = step_id
        self.message = message
        self.expected = expected


@dataclass
class _RunState:
    capability: Capability
    values: dict[str, str]
    started: float = field(default_factory=time.monotonic)
    outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[ReplayWarning] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)


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
        success_timeout_ms: int = SUCCESS_TIMEOUT_MS,
    ) -> None:
        self.surface = surface
        self.registry = registry
        self.evidence = evidence
        self.success_timeout_ms = success_timeout_ms

    @staticmethod
    def check_inputs(capability: Capability, params: Mapping[str, object]) -> Failure | None:
        """Validate params without touching the surface (callers can skip launching a browser)."""
        try:
            validate_inputs(capability, params)
        except InputError as exc:
            return Failure(capability=_ref(capability), category=FailureCategory.INVALID_INPUT, step_id=None,
                           message=str(exc), retryable=False, duration_ms=0)
        return None

    def run(self, capability: Capability, params: Mapping[str, object]) -> Result:
        with log_context(capability_id=capability.id, capability_version=capability.version):
            result = self._run(capability, params)
            log.info("capability.finished", result=result.type, duration_ms=result.duration_ms,
                     **({"category": result.category, "step_id": result.step_id}
                        if isinstance(result, Failure) else {}))
            return result

    def _run(self, capability: Capability, params: Mapping[str, object]) -> Result:
        started = time.monotonic()
        log.info("capability.started", steps=len(capability.steps), max_risk=capability.max_risk)
        if (invalid := self.check_inputs(capability, params)) is not None:
            return invalid
        values = validate_inputs(capability, params)
        for spec in capability.inputs:
            self.registry.register(spec.name, values[spec.name], spec.sensitivity)

        state = _RunState(capability, values, started=started)
        try:
            with span(step_id="entry"):
                self._goto(render(capability.entry_route, values), "entry")
            for step in capability.steps:
                with span(step_id=step.id):
                    self._run_step(step, state)
            with span(step_id="success"):
                self._await(capability.success, self.success_timeout_ms, "success")
        except StepFailure as failure:
            return self._fail(state, failure)
        return Success(capability=_ref(capability), outputs=state.outputs, warnings=state.warnings,
                       duration_ms=state.duration_ms)

    # --- steps ----------------------------------------------------------------------------

    def _run_step(self, step: Step, state: _RunState) -> None:
        started = time.monotonic()
        action = step.action
        log.info("step.started", intent=step.intent, action=action.kind, risk=step.risk)

        target = getattr(action, "target", None)
        resolved = self._resolve(step, target, state) if target is not None else None
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
            raise StepFailure(FailureCategory.ACTION_FAILED, step.id, f"{action.kind} failed: {exc}",
                              expected=f"{action.kind} on {describe_target(target)}") from None

        self._check_dialogs(step.id, expected=action.dialog if isinstance(action, Click) else None)
        if step.expect is not None:
            self._await(step.expect, step.timeout_ms, step.id)
        log.info("step.succeeded", duration_ms=int((time.monotonic() - started) * 1000),
                 strategy=resolved.strategy_kind if resolved else None)

    def _resolve(self, step: Step, target: Target, state: _RunState) -> Resolved:
        try:
            resolved = self.surface.resolve(target, step.timeout_ms)
        except TargetNotFound as exc:
            tried = "; ".join(
                f"{describe_locator(spec)}: {'ambiguous, ' if n > 1 else ''}{n} match{'es' if n != 1 else ''}"
                for spec, n in zip(target.strategies, exc.match_counts, strict=True)
            )
            raise StepFailure(FailureCategory.TARGET_NOT_FOUND, step.id, f"no unique match ({tried})",
                              expected=describe_target(target)) from None
        if resolved.strategy_index > 0:
            detail = (f"preferred {describe_locator(target.strategies[0])!r} did not match; "
                      f"used fallback #{resolved.strategy_index} ({resolved.strategy_kind})")
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
            raise StepFailure(FailureCategory.OUTPUT_PARSE_FAILED, step.id, f"{spec.name}: {exc}",
                              expected=f"{spec.type} from {describe_target(action.target)}") from None
        self.registry.register(spec.name, value, spec.sensitivity)
        state.outputs[spec.name] = value
        log.info("output.extracted", output=spec.name, value=value)

    def _goto(self, route: str, step_id: str) -> None:
        try:
            self.surface.goto(route)
        except ActionFailed as exc:
            raise StepFailure(FailureCategory.NAVIGATION_FAILED, step_id, f"navigation failed: {exc}",
                              expected=f"page {route!r} loads") from None

    def _check_dialogs(self, step_id: str, expected: DialogExpectation | None = None) -> None:
        events = self.surface.take_dialogs()
        for event in events:
            log.info("dialog.handled", expected=event.expected, accepted=event.accepted,
                     message=event.message)
        if unexpected := [e for e in events if not e.expected]:
            raise StepFailure(FailureCategory.UNEXPECTED_DIALOG, step_id,
                              f"unexpected dialog was dismissed: {unexpected[0].message!r}")
        if expected is not None and not any(e.expected for e in events):
            raise StepFailure(FailureCategory.EXPECTED_DIALOG_MISSING, step_id,
                              "the declared confirmation dialog did not appear",
                              expected=f"dialog containing {expected.message_contains!r}")

    def _await(self, checkpoint: Checkpoint, timeout_ms: int, step_id: str) -> None:
        deadline = time.monotonic() + timeout_ms / 1000
        while not self.surface.check(checkpoint):
            self._check_dialogs(step_id)
            if time.monotonic() >= deadline:
                raise StepFailure(FailureCategory.CHECKPOINT_FAILED, step_id,
                                  f"not reached within {timeout_ms} ms",
                                  expected=describe_checkpoint(checkpoint))
            self.surface.idle(POLL_MS)

    # --- failure evidence -----------------------------------------------------------------

    def _fail(self, state: _RunState, failure: StepFailure) -> Failure:
        observed = self.surface.observe()
        evidence: list[str] = []
        if self.evidence is not None:
            name = f"{state.capability.id}.{failure.step_id}.failure.png"
            shot = self.surface.screenshot(mask=sensitive_targets(state.capability))
            evidence.append(self.evidence.save_image(name, shot).name)
        log.error("step.failed", category=failure.category, message=failure.message,
                  expected=failure.expected, observed=observed.locations)
        return Failure(
            capability=_ref(state.capability),
            category=failure.category,
            step_id=failure.step_id,
            message=failure.message,
            expected=failure.expected,
            observed={"title": observed.title, "locations": observed.locations},
            retryable=RETRYABLE[failure.category],
            evidence=evidence,
            warnings=state.warnings,
            duration_ms=state.duration_ms,
        )
