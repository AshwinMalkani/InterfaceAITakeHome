"""Record -> compile -> replay against the live target app, with a scripted operator instead of an LLM.

Proves the deterministic half of discovery on its own: whatever drives the session, the recorder
only keeps locators it validated on the live page, and the compiled artifact replays through the
normal production path with the right outputs.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from cua.agent.goal import load_goal
from cua.agent.recorder import RecordedStep, Recorder, RecordingError, compile_capability
from cua.artifact.params import parse_output, render
from cua.artifact.schema import Parse, Risk, TextPresent
from cua.artifact.store import CapabilityLibrary
from cua.policy import ApprovalLedger, load_policy
from cua.profile import load_profile
from cua.replay.engine import ReplayEngine
from cua.replay.result import Success
from cua.runner import replay
from cua.security.masking import SecretRegistry
from cua.surface.snapshot import Element
from cua.surface.web import WebSurface
from tests.regression.conftest import TEST_PASSWORD, TEST_USERNAME, TargetApp

pytestmark = pytest.mark.regression

REPO = Path(__file__).resolve().parents[2]
GOAL = load_goal(REPO / "goals" / "cu_core" / "read_savings_balance.yaml")
LIBRARY = CapabilityLibrary(REPO / "capabilities")


def find(surface: WebSurface, predicate: str, frame: str | None = "main", timeout_s: float = 5) -> Element:
    """Wait for an element whose name or text equals `predicate` (what a model would pick)."""
    deadline = time.monotonic() + timeout_s
    while True:
        for e in surface.snapshot().elements:
            if e.frame == frame and predicate in (e.name, e.text, e.left_label):
                return e
        if time.monotonic() > deadline:
            raise AssertionError(f"no element {predicate!r} in frame {frame!r}")
        surface.idle(100)


def wait_text(surface: WebSurface, text: str, frame: str | None = "main") -> None:
    deadline = time.monotonic() + 5
    while not surface.check(TextPresent(kind="text_present", text=text, frame=frame)):
        assert time.monotonic() < deadline, text
        surface.idle(100)


def test_scripted_session_compiles_to_an_artifact_that_replays(
    target_apps: dict[str, TargetApp], tmp_path: Path
) -> None:
    app = target_apps["alpha"]
    app.reset()
    values = {"member_id": "10001"}
    with WebSurface.launch(app.base_url) as surface:
        ReplayEngine(surface, registry=SecretRegistry()).run(
            LIBRARY.get("cu_core.session.sign_on"), {"username": TEST_USERNAME, "password": TEST_PASSWORD}
        )
        surface.goto(GOAL.entry_route)
        recorder = Recorder(checker=surface, sensitive_values=set(values.values()))

        def act(
            kind: str,
            element: Element,
            intent: str,
            *,
            value: str | None = None,
            expect: str | None = None,
            output: str | None = None,
            parse: Parse | None = None,
        ) -> None:
            target, rejected = recorder.target_for(element)  # validated before acting
            resolved = surface.resolve_ref(element)
            if kind == "click":
                surface.click(resolved, None, 5000)
            elif kind == "fill":
                surface.fill(resolved, render(value or "", values), 5000)
            elif kind == "extract":
                text = surface.read_text(resolved)
                recorder.sensitive_values.add(text.strip())
                recorder.sensitive_values.add(str(parse_output(text, parse or Parse.TEXT)))
            if expect:
                wait_text(surface, expect)
            recorder.record(
                RecordedStep(
                    kind=kind,
                    element=element,
                    target=target,
                    intent=intent,  # type: ignore[arg-type]
                    declared_risk=Risk.SAFE,
                    value=value,
                    output=output,
                    parse=parse,
                    expect_text=expect,
                    expect_frame="main" if expect else None,
                    rejected=rejected,
                )
            )

        act(
            "click",
            find(surface, "Member Inquiry", frame="menu"),
            "Open member inquiry",
            expect="MEMBER INQUIRY",
        )
        act("fill", find(surface, "Member Number"), "Enter the member number", value="{{inputs.member_id}}")
        act("click", find(surface, "Search"), "Run the search", expect="MEMBER DETAIL")
        balance = next(
            e
            for e in surface.snapshot().elements
            if e.row_key == "Share Savings" and e.column_header == "Balance"
        )
        act(
            "extract",
            find(surface, "Avery Testwood"),
            "Read the member's name",
            output="member_name",
            parse=Parse.TEXT,
        )
        act(
            "extract",
            balance,
            "Read the share savings balance",
            output="savings_balance",
            parse=Parse.CURRENCY,
        )

        capability = compile_capability(
            GOAL,
            recorder.steps,
            success_text="MEMBER DETAIL",
            success_frame="main",
            sensitive_values=recorder.sensitive_values,
            policy=load_policy("cu_core"),
        )

    # The compiled locators are the robust ones, not refs or positions.
    kinds = [s.action.target.strategies[0].kind for s in capability.steps]  # type: ignore[union-attr]
    assert kinds == ["role", "near_text", "role", "field_value", "table_cell"]
    assert "data-cua-ref" not in capability.model_dump_json()

    library = CapabilityLibrary(tmp_path / "capabilities")
    library.save(capability)
    for dependency in ("cu_core.session.sign_on",):
        library.save(LIBRARY.get(dependency))

    app.reset()
    outcome = replay(
        GOAL.id,
        {"member_id": "10004"},
        base_url=app.base_url,
        library=library,
        profile=load_profile("cu_core"),
        policy=load_policy("cu_core"),
        approvals=ApprovalLedger(tmp_path / "approvals.json"),
        evidence_root=tmp_path / "runs",
    )
    assert isinstance(outcome.result, Success), outcome.result
    assert {k: str(v) for k, v in outcome.result.outputs.items()} == {
        "member_name": "Casey Fakeley",
        "savings_balance": "9295.94",
    }  # a different member than recorded


def test_values_seen_during_the_run_can_never_reach_the_artifact(target_apps: dict[str, TargetApp]) -> None:
    app = target_apps["alpha"]
    app.reset()
    with WebSurface.launch(app.base_url) as surface:
        ReplayEngine(surface, registry=SecretRegistry()).run(
            LIBRARY.get("cu_core.session.sign_on"), {"username": TEST_USERNAME, "password": TEST_PASSWORD}
        )
        surface.goto(GOAL.entry_route)
        recorder = Recorder(checker=surface, sensitive_values={"10001"})
        surface.click(surface.resolve_ref(find(surface, "Member Inquiry", frame="menu")), None, 5000)
        field = find(surface, "Member Number")
        target, _ = recorder.target_for(field)
        input_only = GOAL.model_copy(update={"outputs": []})  # isolate the input handling
        compile_one = lambda step: compile_capability(  # noqa: E731
            input_only,
            [step],
            success_text="MEMBER INQUIRY",
            success_frame="main",
            sensitive_values=recorder.sensitive_values,
        )

        # Layer 1, the contract: typing the literal value leaves the declared input unused.
        literal = RecordedStep(
            kind="fill", element=field, target=target, intent="Enter member", value="10001"
        )
        with pytest.raises(RecordingError, match="never used"):
            compile_one(literal)

        # Layer 2, the leak check: placeholder used correctly, but the value leaks into free text.
        chatty = RecordedStep(
            kind="fill",
            element=field,
            target=target,
            intent="Enter member 10001",
            value="{{inputs.member_id}}",
        )
        with pytest.raises(RecordingError, match="1 sensitive value"):
            compile_one(chatty)
