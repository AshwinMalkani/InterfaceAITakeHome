"""The replay result contract: what a calling agent gets back.

A discriminated union on `type`. M1 has `success` and `failure`; M2 adds `business_outcome`
(e.g. member_not_found: a legitimate answer, not a crash) and M5 adds `escalated`.
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


# Whether retrying the same invocation could plausibly succeed. M2 refines this per step risk.
RETRYABLE: dict[FailureCategory, bool] = {
    FailureCategory.INVALID_INPUT: False,
    FailureCategory.NAVIGATION_FAILED: True,
    FailureCategory.TARGET_NOT_FOUND: False,
    FailureCategory.ACTION_FAILED: False,
    FailureCategory.CHECKPOINT_FAILED: False,
    FailureCategory.UNEXPECTED_DIALOG: False,
    FailureCategory.EXPECTED_DIALOG_MISSING: False,
    FailureCategory.OUTPUT_PARSE_FAILED: False,
}


class CapabilityRef(BaseModel):
    id: str
    version: str
    content_hash: str


class ReplayWarning(BaseModel):
    kind: Literal["locator_fallback"]
    step_id: str
    detail: str


class Success(BaseModel):
    type: Literal["success"] = "success"
    capability: CapabilityRef
    outputs: dict[str, Any]
    warnings: list[ReplayWarning] = []
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
    evidence: list[str] = []  # files in the run's evidence directory
    warnings: list[ReplayWarning] = []
    duration_ms: int


Result = Annotated[Success | Failure, Field(discriminator="type")]
