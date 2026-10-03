"""Replay engine logic against a scripted fake surface: no browser, so this is the engine alone."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from cua.artifact.schema import (
    AllOf,
    AnyOf,
    Capability,
    Checkpoint,
    DialogExpectation,
    ElementVisible,
    Target,
    TextPresent,
    UrlMatches,
)
from cua.evidence.sink import RunEvidence
from cua.policy import ApprovalLedger, Policy, PolicyGate
from cua.profile import KnownState
from cua.replay.engine import MAX_INTERSTITIALS, ReplayEngine, sensitive_targets
from cua.replay.result import BusinessOutcome, Failure, FailureCategory, Success
from cua.security.masking import SecretRegistry, mask_text
from cua.surface.base import ActionFailed, DialogEvent, Observation, Resolved, TargetNotFound
from tests.unit.test_schema import minimal

Hook = Callable[["FakeSurface"], None]


@dataclass
class FakeSurface:
    """A page modelled as a set of visible texts. Actions can run hooks that change the page."""

    visible: set[str] = field(default_factory=lambda: {"Done"})
    url_ok: bool = True
    overlay: str | None = None
    fallback_index: dict[str, int] = field(default_factory=dict)  # first-strategy kind -> index used
    missing: set[str] = field(default_factory=set)  # first-strategy kinds that never resolve
    texts: dict[str, str] = field(default_factory=lambda: {"table_cell": "$4,719.56"})
    dialogs_on_click: list[DialogEvent] = field(default_factory=list)
    blocked: bool = False
    blocked_requests: list[str] = field(default_factory=list)
    hooks: dict[str, Hook] = field(default_factory=dict)  # action name -> called after the action
    calls: list[tuple[str, Any]] = field(default_factory=list)
    pending_dialogs: list[DialogEvent] = field(default_factory=list)

    def _did(self, action: str, detail: Any) -> None:
        self.calls.append((action, detail))
        if hook := self.hooks.get(action):
            hook(self)

    def goto(self, route: str) -> None:
        self._did("goto", route)

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
        self.pending_dialogs += self.dialogs_on_click
        self._did("click", resolved.handle)

    def fill(self, resolved: Resolved, value: str, timeout_ms: int) -> None:
        self._did("fill", value)

    def select(self, resolved: Resolved, option: str, timeout_ms: int) -> None:
        self._did("select", option)

    def press(self, key: str, resolved: Resolved | None, timeout_ms: int) -> None:
        self._did("press", key)

    def read_text(self, resolved: Resolved) -> str:
        return self.texts[resolved.handle]

    def check(self, checkpoint: Checkpoint) -> bool:
        match checkpoint:
            case TextPresent(text=text):
                return text in self.visible
            case UrlMatches():
                return self.url_ok
            case ElementVisible():
                return True
            case AllOf(conditions=conditions):
                return all(self.check(c) for c in conditions)
            case AnyOf(conditions=conditions):
                return any(self.check(c) for c in conditions)
        raise AssertionError(checkpoint)

    def blocking_overlay(self) -> str | None:
        return self.overlay

    def take_blocked_requests(self) -> list[str]:
        blocked, self.blocked_requests = self.blocked_requests, []
        return blocked

    def take_dialogs(self) -> list[DialogEvent]:
        events, self.pending_dialogs = self.pending_dialogs, []
        return events

    def observe(self) -> Observation:
        return Observation(title="CU-Core", locations={"main": "/core/member/detail"})

    def screenshot(self, mask: list[Target], vocabulary: frozenset[str] | None) -> bytes:
        self.calls.append(("screenshot", len(mask)))
        self.last_vocabulary = vocabulary
        return b"\x89PNG"


def capability(**changes: Any) -> Capability:
    data = copy.deepcopy(minimal())
    data["steps"][0]["expect"] = {"kind": "url_matches", "pattern": "^/next"}
    data.update(changes)
    return Capability.model_validate(data)


def state(name: str, kind: str, text: str, **extra: Any) -> KnownState:
    when = {"kind": "text_present", "text": text}
    return KnownState.model_validate({"name": name, "kind": kind, "description": name, "when": when, **extra})


NOTICE = state(
    "maintenance_notice",
    "interstitial",
    "SYSTEM NOTICE",
    dismiss={"strategies": [{"kind": "role", "role": "button", "name": "Acknowledge"}]},
)
EXPIRED = state("session_expired", "session_expired", "Your session has expired")
APP_ERROR = state("application_error", "app_error", "APPLICATION ERROR")
DENIED = state("access_denied", "business", "ERR 0403")

PII_BALANCE = [{"name": "balance", "type": "decimal", "description": "b", "sensitivity": "pii"}]
CONFIRM = {
    "id": "s1b",
    "intent": "confirm",
    "risk": "irreversible",
    "action": {
        "kind": "click",
        "target": {"strategies": [{"kind": "role", "role": "button", "name": "Confirm"}]},
    },
}


def run(
    surface: FakeSurface,
    cap: Capability | None = None,
    params: dict[str, object] | None = None,
    evidence: RunEvidence | None = None,
    states: list[KnownState] | None = None,
    reauthenticate: Callable[[], bool] | None = None,
    gate: PolicyGate | None = None,
) -> Any:
    engine = ReplayEngine(
        surface,
        registry=SecretRegistry(include_env=False),
        evidence=evidence,
        states=states or [],
        reauthenticate=reauthenticate,
        gate=gate,
        success_timeout_ms=100,
        dismiss_timeout_ms=100,
    )
    return engine.run(cap or capability(), params if params is not None else {"member_id": "10001"})


def with_step(step: dict[str, Any], at: int = 1, **changes: Any) -> Capability:
    data = copy.deepcopy(minimal())
    data["steps"].insert(at, step)
    data.update(changes)
    return Capability.model_validate(data)


class TestHappyPathAndContract:
    def test_happy_path_returns_typed_outputs(self) -> None:
        surface = FakeSurface()
        result = run(surface)
        assert isinstance(result, Success)
        assert result.outputs == {"balance": Decimal("4719.56")}
        assert result.warnings == [] and result.recoveries == []
        assert ("goto", "/start") in surface.calls and ("fill", "10001") in surface.calls

    def test_invalid_input_fails_before_touching_the_surface(self) -> None:
        surface = FakeSurface()
        result = run(surface, params={})
        assert isinstance(result, Failure) and result.category is FailureCategory.INVALID_INPUT
        assert result.step_id is None and not result.retryable
        assert surface.calls == []

    def test_inputs_and_outputs_are_registered_for_masking(self) -> None:
        registry = SecretRegistry(include_env=False)
        ReplayEngine(FakeSurface(), registry=registry).run(
            capability(outputs=PII_BALANCE), {"member_id": "10001"}
        )
        assert "10001" not in mask_text("member 10001", registry)
        assert "4719.56" not in mask_text("balance 4719.56", registry)


class TestLocators:
    def test_fallback_locator_succeeds_with_drift_warning(self) -> None:
        data = minimal()
        data["steps"][0]["action"]["target"]["strategies"].append(
            {"kind": "css", "selector": "input[name=f1]"}
        )
        result = run(FakeSurface(fallback_index={"near_text": 1}), Capability.model_validate(data))
        assert isinstance(result, Success)
        [warning] = result.warnings
        assert warning.kind == "locator_fallback" and warning.step_id == "s1" and "css" in warning.detail

    def test_target_not_found_reports_each_strategy(self, tmp_path: Path) -> None:
        data = minimal()
        data["steps"][0]["action"]["target"]["strategies"].append({"kind": "css", "selector": "input"})
        result = run(
            FakeSurface(missing={"near_text"}),
            Capability.model_validate(data),
            evidence=RunEvidence(tmp_path, "r"),
        )
        assert isinstance(result, Failure) and result.category is FailureCategory.TARGET_NOT_FOUND
        assert result.step_id == "s1"
        assert "textbox in the row of 'Id': 0 matches" in result.message
        assert "ambiguous, 2 matches" in result.message
        assert result.observed == {"title": "CU-Core", "locations": {"main": "/core/member/detail"}}
        assert result.evidence == ["app.domain.action.s1.failure.png"]
        assert (tmp_path / "r" / "app.domain.action.s1.failure.png").exists()

    def test_screenshot_masks_the_artifacts_declared_sensitive_targets(self, tmp_path: Path) -> None:
        surface = FakeSurface(missing={"table_cell"})
        run(surface, evidence=RunEvidence(tmp_path, "r"))
        assert ("screenshot", 1) in surface.calls  # the pii member_id field

    def test_sensitive_targets_come_from_declarations(self) -> None:
        kinds = [t.strategies[0].kind for t in sensitive_targets(capability(outputs=PII_BALANCE))]
        assert kinds == ["near_text", "table_cell"]
        assert sensitive_targets(capability(inputs=[{"name": "member_id", "description": "id"}])) == []


class TestCheckpointsAndRetries:
    def test_unmet_post_condition_fails_at_that_step_with_expected(self) -> None:
        data = minimal()
        data["steps"][0] |= {"expect": {"kind": "url_matches", "pattern": "^/next"}, "timeout_ms": 100}
        result = run(FakeSurface(url_ok=False), Capability.model_validate(data))
        assert isinstance(result, Failure) and result.category is FailureCategory.CHECKPOINT_FAILED
        assert result.step_id == "s1" and result.expected == "URL matching '^/next'"
        assert result.retryable  # a timeout on a safe step is transient
        assert [r.kind for r in result.recoveries] == ["retried_step"]  # it was retried once first

    def test_safe_step_is_retried_once_then_succeeds(self) -> None:
        fills: list[str] = []

        def second_fill_works(s: FakeSurface) -> None:
            fills.append("fill")
            s.url_ok = len(fills) >= 2

        data = minimal()
        data["steps"][0] |= {"expect": {"kind": "url_matches", "pattern": "^/next"}, "timeout_ms": 100}
        result = run(
            FakeSurface(url_ok=False, hooks={"fill": second_fill_works}), Capability.model_validate(data)
        )
        assert isinstance(result, Success)
        assert [r.kind for r in result.recoveries] == ["retried_step"]
        assert len(fills) == 2

    def test_irreversible_step_is_never_retried(self) -> None:
        surface = FakeSurface()
        result = run(
            surface,
            with_step({**CONFIRM, "expect": {"kind": "text_present", "text": "OPENED"}, "timeout_ms": 100}),
        )
        assert isinstance(result, Failure) and result.category is FailureCategory.CHECKPOINT_FAILED
        assert [c for c in surface.calls if c[0] == "click"] == [("click", "role")]
        assert result.recoveries == []
        assert not result.retryable  # the irreversible step may have happened

    def test_success_checkpoint_is_verified(self) -> None:
        result = run(FakeSurface(visible=set()))
        assert isinstance(result, Failure) and result.step_id == "success"

    def test_blocked_action_is_action_failed(self) -> None:
        step = {
            "id": "s1b",
            "intent": "go",
            "risk": "safe",
            "action": {
                "kind": "click",
                "target": {"strategies": [{"kind": "role", "role": "button", "name": "Go"}]},
            },
        }
        result = run(FakeSurface(blocked=True), with_step(step))
        assert isinstance(result, Failure) and result.category is FailureCategory.ACTION_FAILED
        assert result.step_id == "s1b" and "covered" in result.message

    def test_unparseable_output(self) -> None:
        result = run(FakeSurface(texts={"table_cell": "N/A"}))
        assert isinstance(result, Failure) and result.category is FailureCategory.OUTPUT_PARSE_FAILED


class TestDialogs:
    def _confirm(self, dialog: dict[str, object] | None) -> Capability:
        step = copy.deepcopy(CONFIRM)
        if dialog is not None:
            step["action"]["dialog"] = dialog
        return with_step(step)

    def test_unexpected_dialog_fails_and_needs_a_human(self) -> None:
        surface = FakeSurface(dialogs_on_click=[DialogEvent("Are you sure?", expected=False, accepted=False)])
        result = run(surface, self._confirm(None))
        assert isinstance(result, Failure) and result.category is FailureCategory.UNEXPECTED_DIALOG
        assert result.needs_human

    def test_expected_dialog_must_actually_appear(self) -> None:
        result = run(FakeSurface(), self._confirm({"accept": True, "message_contains": "cannot be undone"}))
        assert isinstance(result, Failure) and result.category is FailureCategory.EXPECTED_DIALOG_MISSING

    def test_expected_dialog_passes(self) -> None:
        surface = FakeSurface(
            dialogs_on_click=[DialogEvent("... cannot be undone.", expected=True, accepted=True)]
        )
        cap = self._confirm({"accept": True, "message_contains": "cannot be undone"})
        assert isinstance(run(surface, cap), Success)


class TestBusinessOutcomes:
    NOT_FOUND = {
        "name": "member_not_found",
        "description": "no such member",
        "when": {"kind": "text_present", "text": "No records found."},
        "after_step": "s1",
    }

    def test_declared_outcome_is_returned_not_failed(self) -> None:
        cap = capability(schema_version="1.1", outcomes=[self.NOT_FOUND])
        surface = FakeSurface(url_ok=False, hooks={"fill": lambda s: s.visible.add("No records found.")})
        result = run(surface, cap)
        assert isinstance(result, BusinessOutcome)
        assert (result.outcome, result.step_id, result.description) == (
            "member_not_found",
            "s1",
            "no such member",
        )

    def test_outcome_is_not_detected_before_its_step(self) -> None:
        # The text is on screen from the start, but the outcome only applies from s2 onwards.
        cap = capability(schema_version="1.1", outcomes=[{**self.NOT_FOUND, "after_step": "s2"}])
        result = run(FakeSurface(visible={"Done", "No records found."}), cap)
        assert isinstance(result, BusinessOutcome) and result.step_id == "s2"

    def test_profile_business_state_applies_to_every_capability(self) -> None:
        result = run(FakeSurface(visible={"Done", "ERR 0403"}), states=[DENIED])
        assert isinstance(result, BusinessOutcome) and result.outcome == "access_denied"


class TestKnownStates:
    def test_interstitial_is_dismissed_and_recorded(self) -> None:
        surface = FakeSurface(
            visible={"Done", "SYSTEM NOTICE"}, hooks={"click": lambda s: s.visible.discard("SYSTEM NOTICE")}
        )
        result = run(surface, states=[NOTICE])
        assert isinstance(result, Success)
        assert [(r.kind, r.detail) for r in result.recoveries] == [
            ("dismissed_interstitial", "maintenance_notice")
        ]

    def test_interstitial_that_will_not_go_away_fails_without_looping(self) -> None:
        surface = FakeSurface(visible={"Done", "SYSTEM NOTICE"})  # clicking Acknowledge changes nothing
        result = run(surface, states=[NOTICE])
        assert isinstance(result, Failure) and result.category is FailureCategory.UNKNOWN_STATE
        assert result.needs_human and "could not be dismissed" in result.message
        assert [c for c in surface.calls if c[0] == "click"] == [("click", "role")]

    def test_reappearing_interstitial_is_bounded(self) -> None:
        class Reappearing(FakeSurface):
            """The notice goes away when acknowledged, then comes straight back."""

            polls = 0

            def check(self, checkpoint: Checkpoint) -> bool:
                if isinstance(checkpoint, TextPresent) and checkpoint.text == "SYSTEM NOTICE":
                    self.polls += 1
                    return self.polls % 2 == 1
                return super().check(checkpoint)

        result = run(Reappearing(), states=[NOTICE])
        assert isinstance(result, Failure) and result.category is FailureCategory.UNKNOWN_STATE
        assert "keeps reappearing" in result.message and result.needs_human
        assert len(result.recoveries) == MAX_INTERSTITIALS

    def test_expired_session_reauthenticates_and_restarts(self) -> None:
        surface = FakeSurface()
        fills: list[str] = []

        def first_fill_expires(s: FakeSurface) -> None:
            fills.append("fill")
            if len(fills) == 1:
                s.visible.add("Your session has expired")

        def reauthenticate() -> bool:
            surface.visible.discard("Your session has expired")
            return True

        surface.hooks["fill"] = first_fill_expires
        result = run(surface, states=[EXPIRED], reauthenticate=reauthenticate)
        assert isinstance(result, Success)
        assert [r.kind for r in result.recoveries] == ["reauthenticated"]
        assert [c for c in surface.calls if c[0] == "goto"] == [("goto", "/start")] * 2  # restarted

    def test_expired_session_without_reauth_fails(self) -> None:
        result = run(FakeSurface(visible={"Done", "Your session has expired"}), states=[EXPIRED])
        assert isinstance(result, Failure) and result.category is FailureCategory.SESSION_EXPIRED
        assert not result.needs_human

    def test_expired_session_after_irreversible_step_needs_a_human(self) -> None:
        surface = FakeSurface(hooks={"click": lambda s: s.visible.add("Your session has expired")})
        result = run(surface, with_step(CONFIRM), states=[EXPIRED], reauthenticate=lambda: True)
        assert isinstance(result, Failure) and result.category is FailureCategory.SESSION_EXPIRED
        assert result.needs_human and not result.retryable
        assert "irreversible" in result.message

    def test_app_error_is_retryable_if_nothing_irreversible_ran(self) -> None:
        result = run(FakeSurface(visible={"Done", "APPLICATION ERROR"}), states=[APP_ERROR])
        assert isinstance(result, Failure) and result.category is FailureCategory.APP_ERROR
        assert result.retryable

    def test_app_error_after_irreversible_step_is_not_retryable(self) -> None:
        surface = FakeSurface(hooks={"click": lambda s: s.visible.add("APPLICATION ERROR")})
        result = run(surface, with_step(CONFIRM), states=[APP_ERROR])
        assert isinstance(result, Failure) and result.category is FailureCategory.APP_ERROR
        assert not result.retryable


class TestUnknownStates:
    def test_overlay_blocks_success_even_when_expected_text_is_visible(self) -> None:
        """The compliance-hold bug: expected text under an undeclared modal must not count as success."""
        result = run(FakeSurface(overlay="main"))
        assert isinstance(result, Failure) and result.category is FailureCategory.UNKNOWN_STATE
        assert result.needs_human and not result.retryable
        assert "main" in result.message

    def test_known_state_wins_over_generic_overlay(self) -> None:
        """A declared interstitial that is itself a modal is handled, not escalated."""
        surface = FakeSurface(visible={"Done", "SYSTEM NOTICE"}, overlay="main")

        def acknowledge(s: FakeSurface) -> None:
            s.visible.discard("SYSTEM NOTICE")
            s.overlay = None

        surface.hooks["click"] = acknowledge
        assert isinstance(run(surface, states=[NOTICE]), Success)


def test_state_appearing_as_the_post_condition_passes_is_attributed_to_that_step() -> None:
    """The page changes between the scan and the check: the new page's overlay belongs to this step."""

    class RacingPage(FakeSurface):
        def check(self, checkpoint: Checkpoint) -> bool:
            if isinstance(checkpoint, UrlMatches):  # s1's post-condition: the new page arrives now
                self.overlay = "main"
            return super().check(checkpoint)

    result = run(RacingPage())
    assert isinstance(result, Failure) and result.category is FailureCategory.UNKNOWN_STATE
    assert result.step_id == "s1"


class TestPolicyGate:
    POLICY = Policy(
        product="app",
        allowed_routes=["^/"],
        blocked_routes=["^/__"],
        allowed_actions=["navigate", "click", "fill", "select", "press", "extract"],
        risk={"unattended_max": "reversible", "irreversible_keywords": ["confirm"]},
    )

    def gate(self, tmp_path: Path, *, approve: Capability | None = None, allow: bool = False) -> PolicyGate:
        ledger = ApprovalLedger(tmp_path / "approvals.json")
        if approve is not None:
            ledger.approve(approve, "reviewer@cu")
        return PolicyGate(self.POLICY, base_url="http://app", approvals=ledger, allow_irreversible=allow)

    def test_irreversible_step_stops_before_acting_without_approval(self, tmp_path: Path) -> None:
        surface = FakeSurface()
        result = run(surface, with_step(CONFIRM), gate=self.gate(tmp_path, allow=True))
        assert isinstance(result, Failure) and result.category is FailureCategory.APPROVAL_REQUIRED
        assert result.step_id == "s1b" and result.needs_human and not result.retryable
        assert "not approved" in result.message
        assert ("click", "role") not in surface.calls  # never clicked Confirm

    def test_approved_but_not_allowed_still_stops(self, tmp_path: Path) -> None:
        cap = with_step(CONFIRM)
        result = run(FakeSurface(), cap, gate=self.gate(tmp_path, approve=cap, allow=False))
        assert isinstance(result, Failure) and result.category is FailureCategory.APPROVAL_REQUIRED
        assert "did not allow" in result.message

    def test_approved_and_allowed_runs(self, tmp_path: Path) -> None:
        cap = with_step(CONFIRM)
        surface = FakeSurface()
        assert isinstance(run(surface, cap, gate=self.gate(tmp_path, approve=cap, allow=True)), Success)
        assert ("click", "role") in surface.calls

    def test_blocked_request_is_a_policy_failure(self, tmp_path: Path) -> None:
        surface = FakeSurface(blocked_requests=["http://evil.example/x"])
        result = run(surface, gate=self.gate(tmp_path))
        assert isinstance(result, Failure) and result.category is FailureCategory.POLICY_DENIED
        assert "evil.example" in result.message and not result.retryable

    def test_rendered_route_is_checked_before_navigating(self, tmp_path: Path) -> None:
        data = copy.deepcopy(minimal())
        data["entry_route"] = "/m/{{inputs.member_id}}"
        surface = FakeSurface()
        result = run(
            surface,
            Capability.model_validate(data),
            params={"member_id": "../__reset"},
            gate=self.gate(tmp_path),
        )
        assert isinstance(result, Failure) and result.category is FailureCategory.POLICY_DENIED
        assert result.step_id == "entry"
        assert not [c for c in surface.calls if c[0] == "goto"]

    def test_static_violations_fail_before_touching_the_surface(self, tmp_path: Path) -> None:
        surface = FakeSurface()
        result = run(surface, capability(entry_route="/__faults"), gate=self.gate(tmp_path))
        assert isinstance(result, Failure) and result.category is FailureCategory.POLICY_DENIED
        assert result.step_id is None and surface.calls == []


class TestScreenshotRedaction:
    def test_failure_screenshots_are_redacted_with_artifact_and_ui_vocabulary(self, tmp_path: Path) -> None:
        surface = FakeSurface(missing={"table_cell"})
        engine = ReplayEngine(
            surface,
            registry=SecretRegistry(include_env=False),
            evidence=RunEvidence(tmp_path, "r"),
            ui_vocabulary=frozenset({"phone"}),
        )
        engine.run(capability(), {"member_id": "10001"})
        assert surface.last_vocabulary is not None
        assert {"id", "savings", "balance", "done", "phone"} <= surface.last_vocabulary

    def test_redaction_can_be_turned_off(self, tmp_path: Path) -> None:
        surface = FakeSurface(missing={"table_cell"})
        engine = ReplayEngine(
            surface,
            registry=SecretRegistry(include_env=False),
            evidence=RunEvidence(tmp_path, "r"),
            redact_screenshots=False,
        )
        engine.run(capability(), {"member_id": "10001"})
        assert surface.last_vocabulary is None
