"""Pausing automation, ceding the live session to a human, and taking it back.

- `GuardedSurface` wraps the surface the engine acts through: every *acting* call (navigate, click,
  fill, select, press) first checks the lease. Automation physically cannot act while a human holds
  the session, so the two can never fight over the same screen.
- `LiveHandoff` is the engine's escalation handler: it files an intervention request with the
  context an operator needs, waits on the *same* live session while the operator works, records
  what they did, and returns their decision (resume at a step, approve, or abort).

Waiting rules: while a request is unclaimed it times out (the caller gets an `escalated` result with
the intervention id). Once an operator has claimed it, automation waits for them to hand back:
control is never taken away from a human mid-task.
"""

from __future__ import annotations

import contextvars
import time
from datetime import UTC, datetime

from cua.artifact.schema import Checkpoint, DialogExpectation, Target
from cua.evidence.log import get_logger
from cua.hitl.models import (
    HumanAction,
    InterventionRequest,
    InterventionStatus,
    Owner,
    Resolution,
)
from cua.hitl.store import HitlStore
from cua.surface.base import ActionFailed, DialogEvent, Observation, Resolved, Surface
from cua.surface.web import WebSurface

log = get_logger(__name__)

POLL_MS = 500


class GuardedSurface:
    """A Surface whose acting methods only work while automation holds the session's lease."""

    def __init__(self, inner: Surface, store: HitlStore, session_id: str) -> None:
        self.inner = inner
        self.store = store
        self.session_id = session_id

    def _require_control(self) -> None:
        lease = self.store.lease(self.session_id)
        if lease.owner is not Owner.AUTOMATION:
            raise ActionFailed(
                f"automation does not hold the session (held by {lease.holder}, epoch {lease.epoch})"
            )

    # acting: guarded
    def goto(self, route: str) -> None:
        self._require_control()
        self.inner.goto(route)

    def click(self, resolved: Resolved, dialog: DialogExpectation | None, timeout_ms: int) -> None:
        self._require_control()
        self.inner.click(resolved, dialog, timeout_ms)

    def fill(self, resolved: Resolved, value: str, timeout_ms: int) -> None:
        self._require_control()
        self.inner.fill(resolved, value, timeout_ms)

    def select(self, resolved: Resolved, option: str, timeout_ms: int) -> None:
        self._require_control()
        self.inner.select(resolved, option, timeout_ms)

    def press(self, key: str, resolved: Resolved | None, timeout_ms: int) -> None:
        self._require_control()
        self.inner.press(key, resolved, timeout_ms)

    # observing: pass through
    def resolve(self, target: Target, timeout_ms: int) -> Resolved:
        return self.inner.resolve(target, timeout_ms)

    def idle(self, ms: int) -> None:
        self.inner.idle(ms)

    def read_text(self, resolved: Resolved) -> str:
        return self.inner.read_text(resolved)

    def check(self, checkpoint: Checkpoint) -> bool:
        return self.inner.check(checkpoint)

    def blocking_overlay(self) -> str | None:
        return self.inner.blocking_overlay()

    def take_blocked_requests(self) -> list[str]:
        return self.inner.take_blocked_requests()

    def take_dialogs(self) -> list[DialogEvent]:
        return self.inner.take_dialogs()

    def observe(self) -> Observation:
        return self.inner.observe()

    def screenshot(self, mask: list[Target], vocabulary: frozenset[str] | None) -> bytes:
        return self.inner.screenshot(mask, vocabulary)


class LiveHandoff:
    """Escalation handler: file the request, wait on the live session, return the operator's decision."""

    def __init__(
        self,
        store: HitlStore,
        *,
        session_id: str,
        run_id: str,
        surface: WebSurface,
        live_session: str,
        unclaimed_timeout_s: float = 900,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.run_id = run_id
        self.surface = surface
        self.live_session = live_session  # how an operator reaches this browser
        self.unclaimed_timeout_s = unclaimed_timeout_s
        self._active: str | None = None
        self._context: contextvars.Context | None = None
        surface.capture_human_actions(self._on_human_action)

    def _on_human_action(self, kind: str, description: str, frame: str | None) -> None:
        # Playwright delivers binding callbacks on its own greenlet, which has its own contextvars:
        # without re-entering the escalating step's context, these log lines would lose run_id/trace
        # correlation AND the run's masking registry. (Caught by the live handoff test.)
        if self._context is not None:
            self._context.run(self._record_human_action, kind, description, frame)

    def _record_human_action(self, kind: str, description: str, frame: str | None) -> None:
        if self._active is None or self.store.lease(self.session_id).owner is not Owner.HUMAN:
            return  # automation's own clicks also fire DOM events; only the operator's are recorded
        action = HumanAction(at=datetime.now(UTC), kind=kind, description=description, frame=frame)
        self.store.add_human_action(self._active, action)
        log.info(
            "human.action",
            actor="human",
            intervention_id=self._active,
            kind=kind,
            description=description,
            frame=frame,
        )

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
    ) -> Resolution:
        request = InterventionRequest(
            session_id=self.session_id,
            run_id=self.run_id,
            capability_id=capability_id,
            kind=kind,
            step_id=step_id,
            intent=intent,
            reason=f"{reason} (live session: {self.live_session})",
            steps=steps,
            screenshot=screenshot,
            locations=locations,
        )
        intervention = self.store.create(request)
        self._active = intervention.id
        self._context = contextvars.copy_context()  # run, step span and masking registry of this escalation
        log.warning(
            "hitl.requested", intervention_id=intervention.id, kind=kind, step_id=step_id, reason=reason
        )

        deadline = time.monotonic() + self.unclaimed_timeout_s
        seen = InterventionStatus.OPEN
        try:
            while True:
                current = self.store.get(intervention.id)
                if current.status is not seen:
                    seen = current.status
                    lease = self.store.lease(self.session_id)
                    log.info(
                        "hitl.lease_transferred",
                        intervention_id=intervention.id,
                        status=seen,
                        owner=lease.owner,
                        holder=lease.holder,
                        epoch=lease.epoch,
                    )
                if current.status is InterventionStatus.RESOLVED:
                    break
                if current.status is InterventionStatus.OPEN and time.monotonic() > deadline:
                    log.warning("hitl.unclaimed", intervention_id=intervention.id)
                    return Resolution(intervention_id=intervention.id, action=None)
                self.surface.idle(POLL_MS)  # pumps browser events, which delivers captured human actions
        finally:
            self._active = None
            self._context = None

        resolution = Resolution(
            intervention_id=current.id,
            action=current.action,
            resume_step=current.resume_step,
            operator=current.operator,
            note=current.note,
            human_actions=self.store.human_actions(current.id),
        )
        log.info(
            "hitl.resolved",
            intervention_id=current.id,
            action=resolution.action,
            resume_step=resolution.resume_step,
            operator=resolution.operator,
            human_actions=len(resolution.human_actions),
        )
        return resolution
