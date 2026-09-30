"""Replay engine logic against a scripted fake surface: no browser, so this is the engine alone."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from cua.artifact.schema import Capability, Checkpoint, DialogExpectation, Target
from cua.evidence.sink import RunEvidence
from cua.replay.engine import ReplayEngine, sensitive_targets
from cua.replay.result import Failure, FailureCategory, Success
from cua.security.masking import SecretRegistry, mask_text
from cua.surface.base import ActionFailed, DialogEvent, Observation, Resolved, TargetNotFound
from tests.unit.test_schema import minimal


@dataclass
class FakeSurface:
    """Resolves every target via its first strategy unless told otherwise."""

    fallback_index: dict[str, int] = field(default_factory=dict)  # first-strategy kind -> index used
    missing: set[str] = field(default_factory=set)  # first-strategy kinds that never resolve
    unmet: set[str] = field(default_factory=set)  # checkpoint kinds that are never true
    texts: dict[str, str] = field(default_factory=lambda: {"table_cell": "$4,719.56"})
    dialogs_on_click: list[DialogEvent] = field(default_factory=list)
    blocked: bool = False
    calls: list[tuple[str, Any]] = field(default_factory=list)
    pending_dialogs: list[DialogEvent] = field(default_factory=list)

    def goto(self, route: str) -> None:
        self.calls.append(("goto", route))

    def resolve(self, target: Target, timeout_ms: int) -> Resolved:
        kind = target.strategies[0].kind
        if kind in self.missing:
            raise TargetNotFound(target, [0] + [2] * (len(target.strategies) - 1))
        index = self.fallback_index.get(kind, 0)
        return Resolved(handle=kind, strategy_index=index, strategy_kind=target.strategies[index].kind)

    def idle(self, ms: int) -> None:
        pass

    def click(self, resolved: Resolved, dialog: DialogExpectation | None, timeout_ms: int) -> None:
        if self.blocked:
            raise ActionFailed("element is covered by another element")
        self.calls.append(("click", resolved.handle))
        self.pending_dialogs += self.dialogs_on_click

    def fill(self, resolved: Resolved, value: str, timeout_ms: int) -> None:
        self.calls.append(("fill", value))

    def select(self, resolved: Resolved, option: str, timeout_ms: int) -> None:
        self.calls.append(("select", option))

    def press(self, key: str, resolved: Resolved | None, timeout_ms: int) -> None:
        self.calls.append(("press", key))

    def read_text(self, resolved: Resolved) -> str:
        return self.texts[resolved.handle]

    def check(self, checkpoint: Checkpoint) -> bool:
        return checkpoint.kind not in self.unmet

    def take_dialogs(self) -> list[DialogEvent]:
        events, self.pending_dialogs = self.pending_dialogs, []
        return events

    def observe(self) -> Observation:
        return Observation(title="CU-Core", locations={"main": "/core/member/detail"})

    def screenshot(self, mask: list[Target]) -> bytes:
        self.calls.append(("screenshot", len(mask)))
        return b"\x89PNG"


def capability(**changes: Any) -> Capability:
    data = copy.deepcopy(minimal())
    data["steps"][0]["expect"] = {"kind": "url_matches", "pattern": "^/next"}
    data.update(changes)
    return Capability.model_validate(data)


def run(
    surface: FakeSurface,
    cap: Capability | None = None,
    params: dict[str, object] | None = None,
    evidence: RunEvidence | None = None,
) -> Any:
    engine = ReplayEngine(
        surface, registry=SecretRegistry(include_env=False), evidence=evidence, success_timeout_ms=100
    )
    return engine.run(cap or capability(), params if params is not None else {"member_id": "10001"})


def test_happy_path_returns_typed_outputs() -> None:
    surface = FakeSurface()
    result = run(surface)
    assert isinstance(result, Success)
    assert result.outputs == {"balance": Decimal("4719.56")}
    assert result.warnings == []
    assert ("goto", "/start") in surface.calls and ("fill", "10001") in surface.calls


def test_invalid_input_fails_before_touching_the_surface() -> None:
    surface = FakeSurface()
    result = run(surface, params={})
    assert isinstance(result, Failure) and result.category is FailureCategory.INVALID_INPUT
    assert result.step_id is None and not result.retryable
    assert surface.calls == []


def test_fallback_locator_succeeds_with_drift_warning() -> None:
    data = minimal()
    data["steps"][0]["action"]["target"]["strategies"].append({"kind": "css", "selector": "input[name=f1]"})
    result = run(FakeSurface(fallback_index={"near_text": 1}), Capability.model_validate(data))
    assert isinstance(result, Success)
    [warning] = result.warnings
    assert warning.kind == "locator_fallback" and warning.step_id == "s1" and "css" in warning.detail


def test_target_not_found_reports_each_strategy(tmp_path: Path) -> None:
    data = minimal()
    data["steps"][0]["action"]["target"]["strategies"].append({"kind": "css", "selector": "input"})
    surface = FakeSurface(missing={"near_text"})
    result = run(surface, Capability.model_validate(data), evidence=RunEvidence(tmp_path, "r"))
    assert isinstance(result, Failure) and result.category is FailureCategory.TARGET_NOT_FOUND
    assert result.step_id == "s1"
    assert "textbox in the row of 'Id': 0 matches" in result.message
    assert "ambiguous, 2 matches" in result.message
    assert result.observed == {"title": "CU-Core", "locations": {"main": "/core/member/detail"}}
    assert result.evidence == ["app.domain.action.s1.failure.png"]
    assert (tmp_path / "r" / "app.domain.action.s1.failure.png").exists()


def test_unmet_post_condition_fails_at_that_step_with_expected() -> None:
    result = run(
        FakeSurface(unmet={"url_matches"}),
        capability(
            steps=[
                {
                    **minimal()["steps"][0],
                    "expect": {"kind": "url_matches", "pattern": "^/next"},
                    "timeout_ms": 100,
                },
                minimal()["steps"][1],
            ]
        ),
    )
    assert isinstance(result, Failure) and result.category is FailureCategory.CHECKPOINT_FAILED
    assert result.step_id == "s1"
    assert result.expected == "URL matching '^/next'"


def test_success_checkpoint_is_verified() -> None:
    result = run(FakeSurface(unmet={"text_present"}))
    assert isinstance(result, Failure) and result.step_id == "success"


def test_blocked_action_is_action_failed() -> None:
    data = minimal()
    data["steps"].insert(
        1,
        {
            "id": "s1b",
            "intent": "go",
            "risk": "safe",
            "action": {
                "kind": "click",
                "target": {"strategies": [{"kind": "role", "role": "button", "name": "Go"}]},
            },
        },
    )
    result = run(FakeSurface(blocked=True), Capability.model_validate(data))
    assert isinstance(result, Failure) and result.category is FailureCategory.ACTION_FAILED
    assert result.step_id == "s1b" and "covered" in result.message


def _with_click(dialog: dict[str, object] | None) -> Capability:
    data = minimal()
    action: dict[str, object] = {
        "kind": "click",
        "target": {"strategies": [{"kind": "role", "role": "button", "name": "Confirm"}]},
    }
    if dialog is not None:
        action["dialog"] = dialog
    data["steps"].insert(1, {"id": "s1b", "intent": "confirm", "risk": "irreversible", "action": action})
    return Capability.model_validate(data)


def test_unexpected_dialog_fails_the_step() -> None:
    surface = FakeSurface(dialogs_on_click=[DialogEvent("Are you sure?", expected=False, accepted=False)])
    result = run(surface, _with_click(None))
    assert isinstance(result, Failure) and result.category is FailureCategory.UNEXPECTED_DIALOG


def test_expected_dialog_must_actually_appear() -> None:
    result = run(FakeSurface(), _with_click({"accept": True, "message_contains": "cannot be undone"}))
    assert isinstance(result, Failure) and result.category is FailureCategory.EXPECTED_DIALOG_MISSING


def test_expected_dialog_passes() -> None:
    surface = FakeSurface(
        dialogs_on_click=[DialogEvent("... cannot be undone.", expected=True, accepted=True)]
    )
    assert isinstance(
        run(surface, _with_click({"accept": True, "message_contains": "cannot be undone"})), Success
    )


def test_unparseable_output() -> None:
    result = run(FakeSurface(texts={"table_cell": "N/A"}))
    assert isinstance(result, Failure) and result.category is FailureCategory.OUTPUT_PARSE_FAILED


def test_inputs_and_outputs_are_registered_for_masking() -> None:
    registry = SecretRegistry(include_env=False)
    cap = capability(
        outputs=[{"name": "balance", "type": "decimal", "description": "b", "sensitivity": "pii"}]
    )
    ReplayEngine(FakeSurface(), registry=registry).run(cap, {"member_id": "10001"})
    assert "10001" not in mask_text("member 10001", registry)
    assert "4719.56" not in mask_text("balance 4719.56", registry)


def test_sensitive_targets_come_from_the_artifacts_declarations() -> None:
    cap = capability(
        outputs=[{"name": "balance", "type": "decimal", "description": "b", "sensitivity": "pii"}]
    )
    kinds = [t.strategies[0].kind for t in sensitive_targets(cap)]
    assert kinds == ["near_text", "table_cell"]  # fills pii member_id; extracts pii balance
    assert sensitive_targets(capability(inputs=[{"name": "member_id", "description": "id"}])) == []


def test_evidence_screenshot_masks_declared_targets(tmp_path: Path) -> None:
    surface = FakeSurface(missing={"table_cell"})
    run(surface, evidence=RunEvidence(tmp_path, "r"))
    assert ("screenshot", 1) in surface.calls  # the pii member_id field
