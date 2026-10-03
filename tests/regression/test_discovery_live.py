"""The full discovery pipeline against the live app, with a scripted model instead of the LLM.

The scripted model reads the same rendered observation the real model gets, finds refs by visible
text, and calls the same tools. Everything after the model is real: tool gates, recorder validation,
compilation, saving, and verification by replay. Runs in CI with no API key and no cost.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from apps.cu_core.data import MemberStore
from cua.agent.goal import load_goal
from cua.agent.llm import ModelTurn, Usage
from cua.artifact.store import CapabilityLibrary
from cua.discovery import discover
from cua.policy import load_policy
from cua.profile import load_profile
from tests.regression.conftest import TEST_PASSWORD, TargetApp
from tests.regression.harness import find_leaks

pytestmark = pytest.mark.regression

REPO = Path(__file__).resolve().parents[2]
GOAL = load_goal(REPO / "goals" / "cu_core" / "read_savings_balance.yaml")
LINE = re.compile(r"^\s+(e\d+)\s+(.*)$")


def latest_screen(messages: list[dict[str, Any]]) -> str:
    content = messages[-1]["content"]
    return content if isinstance(content, str) else "\n".join(r["content"] for r in content)


def find_ref(screen: str, needle: str, frame: str) -> str:
    current = None
    for line in screen.splitlines():
        if line.startswith("[frame "):
            current = line.split()[1]
        elif (m := LINE.match(line)) and current == frame and needle in m.group(2):
            return m.group(1)
    raise AssertionError(f"{needle!r} not on screen in frame {frame!r}:\n{screen}")


@dataclass
class OperatorScript:
    """Each step: (tool, needle, frame, args). needle=None means the tool takes no ref."""

    steps: list[tuple[str, str | None, str, dict[str, Any]]]
    name: str = "scripted-operator"
    turns: int = 0
    rejections: list[str] = field(default_factory=list)

    def respond(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> ModelTurn:
        screen = latest_screen(messages)
        if screen.startswith("Not done:"):
            self.rejections.append(screen.splitlines()[0])
        tool, needle, frame, args = self.steps.pop(0)
        if needle is not None:
            args = {**args, "ref": find_ref(screen, needle, frame)}
        self.turns += 1
        block = {"type": "tool_use", "id": f"tu{self.turns}", "name": tool, "input": args}
        return ModelTurn(content=[block], stop_reason="tool_use", usage=Usage(), model=self.name)


def _common(intent: str) -> dict[str, str]:
    return {"intent": intent, "rationale": "scripted"}


def test_discovery_pipeline_records_compiles_and_verifies(
    target_apps: dict[str, TargetApp], tmp_path: Path
) -> None:
    app = target_apps["alpha"]
    app.reset()
    script = OperatorScript(
        [
            (
                "click",
                "Member Inquiry",
                "menu",
                {**_common("Open member inquiry"), "expect_text": "MEMBER INQUIRY", "risk": "safe"},
            ),
            (
                "fill",
                "Member Number:",
                "main",
                {**_common("Enter member"), "value": "{{inputs.member_id}}"},
            ),  # a cell
            (
                "fill",
                "textbox (next to",
                "main",
                {**_common("Enter the member number"), "value": "10001"},
            ),  # literal
            (
                "fill",
                "textbox (next to",
                "main",
                {**_common("Enter the member number"), "value": "{{inputs.member_id}}"},
            ),
            (
                "click",
                '"Search"',
                "main",
                {**_common("Run the search"), "expect_text": "MEMBER DETAIL", "risk": "safe"},
            ),
            ("extract", "Testwood", "main", {**_common("Read the member's name"), "output": "member_name"}),
            (
                "extract",
                '"$4,719.56"',
                "main",
                {**_common("Read the savings balance"), "output": "savings_balance"},
            ),
            ("finish", None, "main", {"success_text": "MEMBER DETAIL", "rationale": "all outputs extracted"}),
        ]
    )
    library = CapabilityLibrary(REPO / "capabilities")
    outcome = discover(
        GOAL,
        {"member_id": "10001"},
        model=script,
        base_url=app.base_url,
        library=library,
        profile=load_profile("cu_core"),
        policy=load_policy("cu_core"),
        evidence_root=tmp_path,
    )

    assert outcome.status == "recorded", outcome.reason
    assert outcome.verified, outcome.verification
    wrong_element, literal_value = script.rejections
    assert re.fullmatch(r"Not done: e\d+ is a cell, not one of: password, textbox", wrong_element)
    assert literal_value == "Not done: type the input placeholder (e.g. {{inputs.name}}), not its value"

    cap = outcome.capability
    assert cap is not None and cap.provenance.source == "discovery"
    assert [s.action.kind for s in cap.steps] == ["click", "fill", "click", "extract", "extract"]
    first_kinds = [s.action.target.strategies[0].kind for s in cap.steps]  # type: ignore[union-attr]
    # "MEMBER INQUIRY" also matches the menu link that was clicked; the post-condition must be the
    # heading that newly appeared in the main frame, not text that was visible before the click.
    assert cap.steps[0].expect is not None and cap.steps[0].expect.frame == "main"  # type: ignore[union-attr]
    assert first_kinds == ["role", "near_text", "role", "field_value", "table_cell"]

    evidence = outcome.evidence_dir
    for name in ("events.jsonl", "transcript.json", "recorded_steps.json", "discovery_result.json"):
        assert (evidence / name).exists(), name
    # Nothing the run saw on screen persists: no seeded PII, inputs or credentials in any evidence file.
    sensitive = [*MemberStore.seeded().sensitive_values(), "10001", TEST_PASSWORD]
    assert find_leaks(evidence, sensitive) == []
    assert json.loads((evidence / "discovery_result.json").read_text())["verified"] is True


def test_irreversible_clicks_are_refused_during_discovery(
    target_apps: dict[str, TargetApp], tmp_path: Path
) -> None:
    """Without a human in the loop (M5), discovery may not click anything irreversible."""
    app = target_apps["alpha"]
    app.reset()
    script = OperatorScript(
        [
            (
                "click",
                "Member Inquiry",
                "menu",
                {**_common("Open"), "expect_text": "MEMBER INQUIRY", "risk": "safe"},
            ),
            ("click", '"Search"', "main", {**_common("Search"), "expect_text": "", "risk": "irreversible"}),
            ("request_human", None, "main", {"reason": "the next step needs approval"}),
        ]
    )
    outcome = discover(
        GOAL,
        {"member_id": "10001"},
        model=script,
        base_url=app.base_url,
        library=CapabilityLibrary(REPO / "capabilities"),
        profile=load_profile("cu_core"),
        policy=load_policy("cu_core"),
        evidence_root=tmp_path,
    )
    assert outcome.status == "agent_stopped" and "needs_human" in outcome.reason
    assert script.rejections and "needs a human's approval" in script.rejections[0]
