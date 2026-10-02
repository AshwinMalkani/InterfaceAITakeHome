"""The replay result contract: what a calling agent gets back.

A discriminated union on `type`, deliberately separating three things callers must not confuse:

- `success`           the goal was reached; typed outputs attached
- `business_outcome`  a legitimate answer that isn't success ("no such member"): not a crash
- `failure`           something went wrong; says where, what was expected, what was observed,
                      whether a retry could help, and whether a human is needed

Recoverable conditions (a maintenance notice, an expired session, a slow page) don't appear as
result types at all: they were handled. They are listed in `recoveries` so nothing is silent.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class FailureCategory(StrEnum):
    INVALID_INPUT = "invalid_input"                      # caller error; the UI was never touched
    NAVIGATION_FAILED = "navigation_failed"              # entry route did not load
    TARGET_NOT_FOUND = "target_not_found"                # no strategy matched exactly one control
    ACTION_FAILED = "action_failed"                      # control found but not actionable (e.g. covered)
    CHECKPOINT_FAILED = "checkpoint_failed"              # post-condition or success state not reached
    UNEXPECTED_DIALOG = "unexpected_dialog"              # a native dialog nobody declared (dismissed)
    EXPECTED_DIALOG_MISSING = "expected_dialog_missing"  # the declared confirm never appeared
    OUTPUT_PARSE_FAILED = "output_parse_failed"          # extracted text isn't the declared type
    APP_ERROR = "app_error"                              # the application itself reported an error
    SESSION_EXPIRED = "session_expired"                  # session lost and could not safely be restored
    UNKNOWN_STATE = "unknown_state"                      # an undeclared state is blocking the screen


# Whether the *condition* is plausibly transient. A failure is only reported retryable if this is
# true AND no irreversible step ran in this invocation (retrying could repeat that step).
TRANSIENT: dict[FailureCategory, bool] = {
    FailureCategory.INVALID_INPUT: False,
    FailureCategory.NAVIGATION_FAILED: True,
    FailureCategory.TARGET_NOT_FOUND: False,
    FailureCategory.ACTION_FAILED: False,
    FailureCategory.CHECKPOINT_FAILED: True,
    FailureCategory.UNEXPECTED_DIALOG: False,
    FailureCategory.EXPECTED_DIALOG_MISSING: False,
    FailureCategory.OUTPUT_PARSE_FAILED: False,
    FailureCategory.APP_ERROR: True,
    FailureCategory.SESSION_EXPIRED: False,
    FailureCategory.UNKNOWN_STATE: False,
}

# Conditions a person has to look at: unknown screens and dialogs, or a lost session after an
# irreversible step (did it happen or not?). M5 routes these to an operator.
NEEDS_HUMAN = frozenset({FailureCategory.UNEXPECTED_DIALOG, FailureCategory.UNKNOWN_STATE})


class CapabilityRef(BaseModel):
    id: str
    version: str
    content_hash: str


class ReplayWarning(BaseModel):
    kind: Literal["locator_fallback"]
    step_id: str
    detail: str


class Recovery(BaseModel):
    """A recoverable condition that replay handled deliberately."""

    kind: Literal["dismissed_interstitial", "reauthenticated", "retried_step"]
    step_id: str
    detail: str


class Success(BaseModel):
    type: Literal["success"] = "success"
    capability: CapabilityRef
    outputs: dict[str, Any]
    warnings: list[ReplayWarning] = []
    recoveries: list[Recovery] = []
    duration_ms: int


class BusinessOutcome(BaseModel):
    type: Literal["business_outcome"] = "business_outcome"
    capability: CapabilityRef
    outcome: str  # a name declared in the capability's `outcomes` or the app profile's business states
    description: str
    step_id: str
    warnings: list[ReplayWarning] = []
    recoveries: list[Recovery] = []
    duration_ms: int


class Failure(BaseModel):
    type: Literal["failure"] = "failure"
    capability: CapabilityRef
    category: FailureCategory
    step_id: str | None  # None when the failure precedes any step (e.g. invalid input)
    message: str
    expected: str | None = None  # human-readable description of what should have been true
    observed: dict[str, Any] | None = None  # structural snapshot of the surface at failure time
    retryable: bool
    needs_human: bool = False
    evidence: list[str] = []  # files in the run's evidence directory
    warnings: list[ReplayWarning] = []
    recoveries: list[Recovery] = []
    duration_ms: int


Result = Annotated[Success | BusinessOutcome | Failure, Field(discriminator="type")]
