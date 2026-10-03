from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cua.agent.goal import load_goal
from cua.agent.recorder import (
    RecordedStep,
    Recorder,
    RecordingError,
    candidate_locators,
    compile_capability,
)
from cua.artifact.schema import Parse, Risk, Target
from cua.policy import load_policy
from cua.surface.snapshot import Element, Snapshot

GOAL = load_goal(Path(__file__).resolve().parents[2] / "goals" / "cu_core" / "read_savings_balance.yaml")


def kinds(element: Element) -> list[str]:
    return [c.kind for c in candidate_locators(element)]


class TestCandidates:
    def test_button_with_name_and_field_name(self) -> None:
        e = Element(ref="e1", frame="main", role="button", name="Search", name_attr="go", tag="input")
        assert kinds(e) == ["role", "css"]

    def test_unlabelled_legacy_textbox(self) -> None:
        e = Element(
            ref="e2", frame="main", role="textbox", left_label="Member Number", name_attr="f1", tag="input"
        )
        assert kinds(e) == ["near_text", "css"]
        assert candidate_locators(e)[1].model_dump()["selector"] == "input[name='f1']"

    def test_grid_cell_prefers_header_based_location(self) -> None:
        e = Element(
            ref="e3",
            frame="main",
            role="cell",
            text="$4,719.56",
            row_key="Share Savings",
            column_header="Balance",
            prev_cell="Share Savings",
        )
        assert kinds(e) == ["table_cell", "field_value"]

    def test_label_value_cell_without_header(self) -> None:
        e = Element(
            ref="e4", frame="main", role="cell", text="Avery Testwood", prev_cell="Name", row_key="Name"
        )
        assert kinds(e) == ["field_value"]

    def test_numeric_previous_cell_is_not_a_label(self) -> None:
        e = Element(ref="e5", frame="main", role="cell", text="x", prev_cell="00")
        assert kinds(e) == []

    def test_odd_field_names_are_not_turned_into_css(self) -> None:
        e = Element(ref="e6", frame=None, role="textbox", name_attr="a'] , body [x='", tag="input")
        assert kinds(e) == []


@dataclass
class FakeChecker:
    """Strategy kinds that resolve uniquely to the intended element; everything else fails."""

    unique: set[str] = field(default_factory=set)

    def matches_only(self, target: Target, element: Element) -> list[bool]:
        return [s.kind in self.unique for s in target.strategies]


TEXTBOX = Element(
    ref="e2", frame="main", role="textbox", left_label="Member Number", name_attr="f1", tag="input"
)


class TestRecorder:
    def test_keeps_only_validated_candidates_in_order(self) -> None:
        target, rejected = Recorder(FakeChecker({"css"})).target_for(TEXTBOX)
        assert [s.kind for s in target.strategies] == ["css"]
        assert target.frame == "main"
        assert len(rejected) == 1 and "near_text" in rejected[0]

    def test_no_valid_candidate_is_an_error(self) -> None:
        with pytest.raises(RecordingError, match="uniquely"):
            Recorder(FakeChecker()).target_for(TEXTBOX)

    def test_unidentifiable_element_is_an_error(self) -> None:
        with pytest.raises(RecordingError, match="no identifying"):
            Recorder(FakeChecker({"role"})).target_for(Element(ref="e9", frame=None, role="textbox"))

    def test_candidates_built_from_sensitive_text_are_never_used(self) -> None:
        # A results table keyed by member name: the row key is data, not a label.
        cell = Element(
            ref="e7",
            frame="main",
            role="cell",
            text="$5.00",
            row_key="Avery Testwood",
            column_header="Balance",
            prev_cell="Checking",
        )
        recorder = Recorder(FakeChecker({"table_cell", "field_value"}), sensitive_values={"Avery Testwood"})
        target, _ = recorder.target_for(cell)
        assert [s.kind for s in target.strategies] == ["field_value"]


def _step(kind: str, element: Element, strategies: list[dict[str, object]], **extra: object) -> RecordedStep:
    target = Target.model_validate({"frame": element.frame, "strategies": strategies})
    return RecordedStep(kind=kind, element=element, target=target, intent=f"{kind} it", **extra)  # type: ignore[arg-type]


def _flow(confirm_name: str = "Search") -> list[RecordedStep]:
    button = Element(ref="e3", frame="main", role="button", name=confirm_name)
    name_cell = Element(ref="e4", frame="main", role="cell", prev_cell="Name")
    balance = Element(ref="e5", frame="main", role="cell", row_key="Share Savings", column_header="Balance")
    return [
        _step(
            "fill",
            TEXTBOX,
            [{"kind": "near_text", "anchor": "Member Number", "control": "textbox"}],
            value="{{inputs.member_id}}",
        ),
        _step(
            "click",
            button,
            [{"kind": "role", "role": "button", "name": confirm_name}],
            expect_text="MEMBER DETAIL",
            expect_frame="main",
        ),
        _step(
            "extract",
            name_cell,
            [{"kind": "field_value", "label": "Name"}],
            output="member_name",
            parse=Parse.TEXT,
        ),
        _step(
            "extract",
            balance,
            [{"kind": "table_cell", "row": "Share Savings", "column": "Balance"}],
            output="savings_balance",
            parse=Parse.CURRENCY,
        ),
    ]


class TestCompile:
    def test_compiles_a_valid_discovery_artifact(self) -> None:
        cap = compile_capability(
            GOAL,
            _flow(),
            success_text="MEMBER DETAIL",
            success_frame="main",
            sensitive_values={"10001"},
            run_id="run_x",
            model="claude-opus-5-5",
        )
        assert cap.id == GOAL.id and [s.id for s in cap.steps] == ["s1", "s2", "s3", "s4"]
        assert cap.steps[1].expect is not None
        assert cap.provenance.source == "discovery" and cap.provenance.model == "claude-opus-5-5"

    def test_policy_raises_risk_of_commit_controls(self) -> None:
        cap = compile_capability(
            GOAL,
            _flow(confirm_name="Confirm"),
            success_text="MEMBER DETAIL",
            success_frame="main",
            sensitive_values=set(),
            policy=load_policy("cu_core"),
        )
        assert cap.steps[1].risk is Risk.IRREVERSIBLE

    def test_flow_must_satisfy_the_goal_contract(self) -> None:
        with pytest.raises(RecordingError, match="never extracted"):
            compile_capability(
                GOAL, _flow()[:2], success_text="x", success_frame=None, sensitive_values=set()
            )

    def test_nothing_recorded(self) -> None:
        with pytest.raises(RecordingError, match="nothing"):
            compile_capability(GOAL, [], success_text="x", success_frame=None, sensitive_values=set())


def test_snapshot_render_groups_by_frame() -> None:
    snap = Snapshot(
        elements=[
            Element(ref="e0", frame=None, role="text", text="Welcome"),
            Element(ref="e1", frame="main", role="textbox", left_label="Member Number"),
            Element(ref="e2", frame="main", role="select", options=("Savings", "Money Market")),
        ],
        locations={"top": "/core/main", "main": "/core/member/search"},
    )
    text = snap.render()
    assert text.index("[frame top") < text.index('text "Welcome"') < text.index("[frame main")
    assert "textbox (next to 'Member Number')" in text
    assert "options=['Savings', 'Money Market']" in text
