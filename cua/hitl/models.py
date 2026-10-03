"""Human-in-the-loop data model: who controls a live session, and what a human was asked to do.

Control model: each live browser session has exactly one owner at a time, recorded as a *lease*
(owner + epoch). Every transfer bumps the epoch. Automation only acts while it holds the lease,
so "who is in control right now?" always has one answer, and a stale party can't act by accident.

    automation --escalate--> (waiting) --operator claims--> human --operator resolves--> automation
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class Owner(StrEnum):
    AUTOMATION = "automation"
    HUMAN = "human"


class InterventionStatus(StrEnum):
    OPEN = "open"  # waiting for an operator; automation is paused but still holds the lease
    CLAIMED = "claimed"  # an operator holds the lease and is working in the live session
    RESOLVED = "resolved"  # handed back with a decision


class Action(StrEnum):
    RESUME = "resume"  # continue from a chosen step (the human may have done some steps by hand)
    APPROVE = "approve"  # approval requests only: run the irreversible step this once
    ABORT = "abort"  # stop the run; the result reports the operator's decision


class Lease(BaseModel):
    session_id: str
    owner: Owner
    epoch: int
    holder: str  # "automation" or the operator's name
    updated_at: datetime


class InterventionRequest(BaseModel):
    """Everything an operator needs to act, without digging through logs."""

    session_id: str
    run_id: str
    capability_id: str
    kind: str  # "approval" | "unknown_state" | "unexpected_dialog" | "stuck" | ...
    step_id: str
    intent: str  # human-readable step description from the artifact
    reason: str
    steps: list[tuple[str, str]]  # (step_id, intent) for every step: the "resume at" choices
    screenshot: str | None = None  # file in the run's evidence directory (redacted)
    locations: dict[str, str] = {}  # frame -> path, at the time of escalation


class Intervention(InterventionRequest):
    id: str
    status: InterventionStatus
    created_at: datetime
    operator: str | None = None
    action: Action | None = None
    resume_step: str | None = None
    note: str = ""
    resolved_at: datetime | None = None


class HumanAction(BaseModel):
    """One thing the operator did in the live session (values are never captured)."""

    at: datetime
    kind: str  # click | input | change
    description: str  # e.g. "button 'Override'" or "textbox next to 'Nickname'"
    frame: str | None = None


class Resolution(BaseModel):
    """What the engine gets back from an escalation."""

    intervention_id: str
    action: Action | None  # None = nobody resolved it before the deadline
    resume_step: str | None = None
    operator: str | None = None
    note: str = ""
    human_actions: list[HumanAction] = []

    @property
    def timed_out(self) -> bool:
        return self.action is None
