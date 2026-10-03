"""The discovery loop's control logic, against a scripted model and fake tools (no browser, no API)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cua.agent.goal import load_goal
from cua.agent.llm import ModelTurn, Usage
from cua.agent.loop import DiscoveryAgent, first_message
from cua.agent.tools import ToolResult, tool_definitions

GOAL = load_goal(Path(__file__).resolve().parents[2] / "goals" / "cu_core" / "read_savings_balance.yaml")


def use(name: str, n: int = 0, **args: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": f"tu_{name}_{n}", "name": name, "input": args}


@dataclass
class ScriptedModel:
    turns: list[list[dict[str, Any]]]
    stop_reason: str = "tool_use"
    name: str = "scripted"
    seen: list[list[dict[str, Any]]] = field(default_factory=list)

    def respond(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> ModelTurn:
        self.seen.append(list(messages))
        content = self.turns.pop(0) if self.turns else [{"type": "text", "text": "..."}]
        stop = self.stop_reason if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return ModelTurn(content=content, stop_reason=stop, usage=Usage(10, 5, 2, 0), model=self.name)


@dataclass
class FakeTools:
    results: dict[str, ToolResult] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    screens: list[str] = field(default_factory=list)

    def observe(self) -> str:
        return "screen 0"

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append(name)
        if name in self.results:
            return self.results[name]
        if name == "finish":
            return ToolResult("Recorded.", terminal="finish")
        if name == "request_human":
            return ToolResult("Stopping.", terminal="request_human")
        screen = self.screens.pop(0) if self.screens else f"screen {len(self.calls)}"
        return ToolResult(f"Done.\n\n{screen}")


def agent(model: ScriptedModel, tools: FakeTools, **kwargs: Any) -> DiscoveryAgent:
    return DiscoveryAgent(model, tools, tool_definitions(GOAL), **kwargs)


def test_finish_ends_the_run_successfully() -> None:
    model = ScriptedModel([[use("click", ref="e1")], [use("finish", success_text="MEMBER DETAIL")]])
    outcome = agent(model, FakeTools()).run(GOAL)
    assert outcome.status == "finished" and outcome.succeeded
    assert (outcome.actions, outcome.turns) == (1, 2)
    assert outcome.usage == Usage(20, 10, 4, 0)


def test_history_is_append_only_with_one_results_message_per_turn() -> None:
    model = ScriptedModel([[use("click", ref="e1")], [use("finish", success_text="x")]])
    agent(model, FakeTools()).run(GOAL)
    first, second = model.seen
    assert second[: len(first)] == first  # earlier turns never rewritten
    assistant, results = second[-2], second[-1]
    assert assistant["role"] == "assistant" and assistant["content"] == [use("click", ref="e1")]
    assert results["role"] == "user" and results["content"][0]["tool_use_id"] == "tu_click_0"


def test_request_human_carries_the_reason() -> None:
    model = ScriptedModel([[use("request_human", reason="unexpected supervisor override screen")]])
    outcome = agent(model, FakeTools()).run(GOAL)
    assert outcome.status == "needs_human" and "supervisor" in outcome.reason


def test_model_that_stops_calling_tools_is_nudged_once() -> None:
    model = ScriptedModel([[{"type": "text", "text": "I think I'm done"}], [use("finish", success_text="x")]])
    assert agent(model, FakeTools()).run(GOAL).status == "finished"
    model = ScriptedModel([[{"type": "text", "text": "hmm"}], [{"type": "text", "text": "hmm"}]])
    assert agent(model, FakeTools()).run(GOAL).status == "no_action"


def test_consecutive_rejections_stop_the_run() -> None:
    tools = FakeTools(results={"click": ToolResult("Not done: unknown ref", is_error=True)})
    model = ScriptedModel([[use("click", n, ref="e99")] for n in range(5)])
    outcome = agent(model, tools, max_errors=3).run(GOAL)
    assert outcome.status == "too_many_errors" and tools.calls == ["click"] * 3


def test_clicks_that_change_nothing_are_detected_as_no_progress() -> None:
    tools = FakeTools(screens=["same e1", "same e7", "same e3"])  # refs differ, screen doesn't
    model = ScriptedModel([[use("click", n, ref="e1")] for n in range(5)])
    outcome = agent(model, tools, max_unchanged=2).run(GOAL)
    assert outcome.status == "no_progress"


def test_action_budget() -> None:
    model = ScriptedModel([[use("fill", n, ref="e1", value="x")] for n in range(10)])
    assert agent(model, FakeTools(), max_actions=4).run(GOAL).status == "max_steps"


def test_refusal_or_truncation_stops_the_run() -> None:
    model = ScriptedModel([[use("click", ref="e1")]], stop_reason="refusal")
    assert agent(model, FakeTools()).run(GOAL).status == "model_stopped"


def test_every_tool_use_gets_a_result_even_after_the_run_ends() -> None:
    model = ScriptedModel([[use("finish", success_text="x"), use("click", ref="e1")]])
    tools = FakeTools()
    outcome = agent(model, tools).run(GOAL)
    assert tools.calls == ["finish"]  # the click after finish was not executed
    results = outcome.transcript[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_finish_0", "tu_click_0"] and results[1]["is_error"]


def test_goal_message_names_inputs_by_placeholder_only() -> None:
    message = first_message(GOAL)
    assert "{{inputs.member_id}}" in message and "savings_balance (decimal)" in message


def test_tools_are_strict_and_extract_is_limited_to_goal_outputs() -> None:
    tools = {t["name"]: t for t in tool_definitions(GOAL)}
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in tools.values())
    assert tools["extract"]["input_schema"]["properties"]["output"]["enum"] == [
        "member_name",
        "savings_balance",
    ]
