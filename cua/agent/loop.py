"""The discovery loop: observe -> decide (model) -> act (tools) until finished or stuck.

A hand-written loop rather than the SDK's tool runner, because every turn is a control point we own:
the stop conditions below, the policy gates inside the tools, and (M5) pausing for a human between
turns. History is strictly append-only: each model turn is appended unchanged (thinking blocks
included) and followed by one user message carrying all tool results.

Stop conditions ("stuck" is detected, never waited out):
  finished        the model called `finish` and the success text was verified on screen
  needs_human     the model called `request_human`
  no_progress     N clicks in a row left the screen exactly as it was
  too_many_errors N consecutive tool calls were rejected
  max_steps       action budget spent
  timeout         wall-clock budget spent
  no_action       the model stopped calling tools (after one nudge)
  model_stopped   the model refused or ran out of output tokens
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from cua.agent.goal import GoalSpec
from cua.agent.llm import Model, ModelTurn, Usage
from cua.agent.tools import ToolResult
from cua.evidence.log import get_logger, span

log = get_logger(__name__)

Status = Literal[
    "finished",
    "needs_human",
    "no_progress",
    "too_many_errors",
    "max_steps",
    "timeout",
    "no_action",
    "model_stopped",
]

SYSTEM_PROMPT = """\
You are operating a legacy back-office web application for a credit union, to accomplish one goal.
Everything you do is recorded and compiled into a reusable automation, so act like a careful
operator demonstrating the flow once, by the shortest path.

How to work:
- Each observation lists what is visible, grouped by frame, with a ref per element (e.g. e12).
  Act only by ref, and only with refs from the latest observation: refs change after every action.
- Inputs are given by name. Whenever you type or choose an input, use its placeholder exactly,
  e.g. {{inputs.member_id}}. Never type an input's value yourself; the system substitutes it.
- For every click, give `expect_text`: the exact heading or label you expect to see once it worked.
- To return an output, `extract` it from the element that holds the value (not from its label).
- When every output is extracted, call `finish` with text visible on screen that proves success.
- If you are blocked, the screen is unexpected, or an action needs approval, call `request_human`
  with the reason. Never guess, and never click anything that submits or commits a change unless
  the goal requires it.
- Take one action per turn and look at the result before the next.
"""


class Tools(Protocol):
    def observe(self) -> str: ...

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult: ...


@dataclass
class AgentOutcome:
    status: Status
    reason: str
    actions: int
    turns: int
    usage: Usage
    transcript: list[dict[str, Any]] = field(default_factory=list)  # masked when persisted

    @property
    def succeeded(self) -> bool:
        return self.status == "finished"


def first_message(goal: GoalSpec) -> str:
    inputs = "\n".join(f"- {{{{inputs.{p.name}}}}}: {p.description}" for p in goal.inputs) or "- (none)"
    outputs = "\n".join(f"- {o.name} ({o.type}): {o.description}" for o in goal.outputs) or "- (none)"
    return (
        f"Goal: {goal.goal.strip()}\n\n"
        f"Inputs (use these placeholders):\n{inputs}\n\n"
        f"Outputs to extract:\n{outputs}"
    )


def _field(block: Any, key: str) -> Any:
    """Content blocks are SDK objects from the real model and plain dicts from scripted ones."""
    return block[key] if isinstance(block, dict) else getattr(block, key)


def _tool_uses(content: list[Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (_field(b, "id"), _field(b, "name"), dict(_field(b, "input")))
        for b in content
        if _field(b, "type") == "tool_use"
    ]


def _dump(block: Any) -> Any:
    return block if isinstance(block, dict) else block.model_dump(mode="json")


class DiscoveryAgent:
    def __init__(
        self,
        model: Model,
        tools: Tools,
        tool_definitions: list[dict[str, Any]],
        *,
        max_actions: int = 30,
        max_errors: int = 3,
        max_unchanged: int = 3,
        timeout_s: float = 600,
    ) -> None:
        self.model = model
        self.tools = tools
        self.tool_definitions = tool_definitions
        self.max_actions = max_actions
        self.max_errors = max_errors
        self.max_unchanged = max_unchanged
        self.timeout_s = timeout_s

    def run(self, goal: GoalSpec) -> AgentOutcome:
        started = time.monotonic()
        observation = self.tools.observe()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"{first_message(goal)}\n\nCurrent screen:\n{observation}"}
        ]
        actions = turns = errors = unchanged = 0
        nudged = False
        usage = Usage()
        last_screen = _digest(observation)

        def outcome(status: Status, reason: str) -> AgentOutcome:
            log.info(
                "agent.finished",
                status=status,
                reason=reason,
                actions=actions,
                turns=turns,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_input_tokens,
            )
            return AgentOutcome(
                status,
                reason,
                actions,
                turns,
                usage,
                [{"role": m["role"], "content": _dump_content(m["content"])} for m in messages],
            )

        while True:
            if time.monotonic() - started > self.timeout_s:
                return outcome("timeout", f"exceeded {self.timeout_s:.0f}s")
            with span(step_id=f"turn{turns + 1}"):
                turn = self.model.respond(SYSTEM_PROMPT, self.tool_definitions, messages)
            turns += 1
            usage = _add(usage, turn)
            messages.append({"role": "assistant", "content": turn.content})
            log.info(
                "agent.turn",
                turn=turns,
                stop_reason=turn.stop_reason,
                model=turn.model,
                input_tokens=turn.usage.input_tokens,
                output_tokens=turn.usage.output_tokens,
                cache_read_tokens=turn.usage.cache_read_input_tokens,
            )

            if turn.stop_reason in ("refusal", "max_tokens"):
                return outcome("model_stopped", f"model stopped: {turn.stop_reason}")
            uses = _tool_uses(turn.content)
            if not uses:
                if nudged:
                    return outcome("no_action", "the model stopped calling tools")
                nudged = True
                messages.append({"role": "user", "content": "Continue by calling one of the tools."})
                continue

            results: list[dict[str, Any]] = []
            terminal: ToolResult | None = None
            for tool_id, name, args in uses:
                if terminal is not None:  # every tool_use still gets a result
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "is_error": True,
                            "content": "Skipped: the run already ended.",
                        }
                    )
                    continue
                result = self.tools.execute(name, args)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result.text,
                        "is_error": result.is_error,
                    }
                )
                if name in ("click", "fill", "select", "extract"):
                    actions += 1
                errors = errors + 1 if result.is_error else 0
                if name == "click" and not result.is_error:
                    screen = _digest(result.text)
                    unchanged = unchanged + 1 if screen == last_screen else 0
                    last_screen = screen
                if result.terminal:
                    terminal = result
            messages.append({"role": "user", "content": results})

            if terminal is not None and terminal.terminal == "finish":
                return outcome("finished", "goal completed and success verified")
            if terminal is not None:
                reason = next((a.get("reason", "") for _, n, a in uses if n == "request_human"), "")
                return outcome("needs_human", reason or "the model asked for a human")
            if errors >= self.max_errors:
                return outcome("too_many_errors", f"{errors} consecutive tool calls were rejected")
            if unchanged >= self.max_unchanged:
                return outcome("no_progress", f"{unchanged} clicks in a row left the screen unchanged")
            if actions >= self.max_actions:
                return outcome("max_steps", f"used all {self.max_actions} actions")


def _digest(text: str) -> str:
    # Refs are renumbered on every snapshot, so they're excluded from "did the screen change?".
    return hashlib.sha256(re.sub(r"\be\d+\b", "", text).encode()).hexdigest()


def _add(total: Usage, turn: ModelTurn) -> Usage:
    u = turn.usage
    return Usage(
        total.input_tokens + u.input_tokens,
        total.output_tokens + u.output_tokens,
        total.cache_read_input_tokens + u.cache_read_input_tokens,
        total.cache_creation_input_tokens + u.cache_creation_input_tokens,
    )


def _dump_content(content: Any) -> Any:
    return content if isinstance(content, str) else [_dump(b) for b in content]
