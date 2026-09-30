"""Deterministic fault injection, so every runtime condition in the brief is reproducible.

Faults are armed through the admin endpoint (`POST /__faults`) by tests and demo scripts.
Each fault matches request paths by substring and fires `times` times (None = until cleared).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class FaultKind(StrEnum):
    SLOW = "slow"                        # transient slowness: delay the response
    SERVER_ERROR = "server_error"        # outright app error: HTTP 500 "Application Error" page
    INTERSTITIAL = "interstitial"        # known notice page that must be acknowledged to continue
    EXPIRE_SESSION = "expire_session"    # session times out right before this request
    COMPLIANCE_HOLD = "compliance_hold"  # an unexpected, unknown modal (should escalate, not be guessed at)


class Fault(BaseModel):
    kind: FaultKind
    path: str = ""  # substring of the request path; "" matches every /core page
    times: int | None = Field(default=1, ge=1)
    delay_ms: int = Field(default=3000, ge=0, le=60_000)


class FaultInjector:
    def __init__(self) -> None:
        self._faults: list[Fault] = []

    def arm(self, faults: list[Fault]) -> None:
        self._faults.extend(faults)

    def clear(self) -> None:
        self._faults.clear()

    @property
    def armed(self) -> list[Fault]:
        return list(self._faults)

    def take(self, path: str, kind: FaultKind) -> Fault | None:
        """Return (and consume one use of) the first armed fault of `kind` matching `path`."""
        for fault in self._faults:
            if fault.kind is kind and fault.path in path:
                if fault.times is not None:
                    fault.times -= 1
                    if fault.times == 0:
                        self._faults.remove(fault)
                return fault
        return None
