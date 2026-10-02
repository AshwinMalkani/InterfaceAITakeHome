"""The seam between "how we perceive and act on a surface" and "the recorded flow".

The replay engine only speaks this protocol. Artifacts describe *what* to target (role + name,
the field next to a label, a table cell); a Surface decides *how* to find it on its technology:
Playwright for web (web.py), and e.g. Windows UI Automation for a desktop surface later, where
role = ControlType, name = Name, and AutomationId becomes one more locator strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from cua.artifact.schema import Checkpoint, DialogExpectation, Target


@dataclass(frozen=True)
class Resolved:
    """A target resolved to exactly one control. `handle` is surface-specific and opaque to callers."""

    handle: Any
    strategy_index: int  # 0 = the preferred strategy; > 0 means a fallback won (drift signal)
    strategy_kind: str


@dataclass(frozen=True)
class DialogEvent:
    message: str
    expected: bool
    accepted: bool


class TargetNotFound(Exception):
    """No strategy resolved to exactly one visible control before the timeout."""

    def __init__(self, target: Target, match_counts: list[int]) -> None:
        super().__init__("target not found")
        self.target = target
        self.match_counts = match_counts  # per strategy: 0 = none, >1 = ambiguous (both rejected)


class ActionFailed(Exception):
    """The control was found, but acting on it failed (e.g. it was covered by an overlay)."""


@dataclass
class Observation:
    """What the surface looks like right now: the `observed` half of a failure report.

    Deliberately structural. Free page text is not captured, because on a banking screen it is
    mostly PII that pattern masking can't reliably catch (names, local phone numbers). The
    human-readable signal is the masked failure screenshot instead.
    """

    title: str = ""
    locations: dict[str, str] = field(default_factory=dict)  # frame name (or "top") -> path + query


class Surface(Protocol):
    def goto(self, route: str) -> None: ...

    def resolve(self, target: Target, timeout_ms: int) -> Resolved: ...

    def idle(self, ms: int) -> None:
        """Wait while letting the surface process events. Pollers must use this, not time.sleep."""
        ...

    def click(self, resolved: Resolved, dialog: DialogExpectation | None, timeout_ms: int) -> None: ...

    def fill(self, resolved: Resolved, value: str, timeout_ms: int) -> None: ...

    def select(self, resolved: Resolved, option: str, timeout_ms: int) -> None: ...

    def press(self, key: str, resolved: Resolved | None, timeout_ms: int) -> None: ...

    def read_text(self, resolved: Resolved) -> str: ...

    def check(self, checkpoint: Checkpoint) -> bool:
        """Evaluate a checkpoint once, without waiting. The engine owns waiting and timeouts."""
        ...

    def blocking_overlay(self) -> str | None:
        """Name of a frame (or "top") where something covers most of the view, else None.

        Surface-specific heuristic for *unknown* blocking states: modals nobody declared.
        """
        ...

    def take_blocked_requests(self) -> list[str]:
        """URLs the surface refused to load because policy forbids them, since the last call."""
        ...

    def take_dialogs(self) -> list[DialogEvent]:
        """Dialogs handled since the last call (expected ones answered, unexpected ones dismissed)."""
        ...

    def observe(self) -> Observation: ...

    def screenshot(self, mask: list[Target]) -> bytes:
        """Full screenshot with every control matching `mask` painted over."""
        ...
