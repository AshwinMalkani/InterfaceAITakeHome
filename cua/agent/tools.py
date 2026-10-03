"""The tools the discovery model may call, and the code that executes them.

Every action goes through the same gates before it touches the app:
  ref lookup (latest snapshot only) -> recorder validation (a robust locator must exist) ->
  policy risk (irreversible needs a human) -> input placeholders (no literal input values) -> act
and its result is the next observation. The model never touches the browser directly.

Known interstitials from the app profile are dismissed automatically and are *not* recorded:
they're runtime states that replay already handles, not part of the flow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cua.agent.goal import GoalSpec
from cua.agent.recorder import ActionKind, RecordedStep, Recorder, RecordingError
from cua.artifact.params import OutputParseError, parse_output, render
from cua.artifact.schema import TEMPLATE, Parse, Risk, Step, TextPresent, ValueType
from cua.evidence.log import get_logger
from cua.policy import Policy
from cua.profile import AppProfile, StateKind
from cua.security.masking import SecretRegistry, mask_text
from cua.surface.base import ActionFailed, TargetNotFound
from cua.surface.snapshot import Element, Snapshot
from cua.surface.web import WebSurface

log = get_logger(__name__)

EXPECT_TIMEOUT_S = 10.0
_PARSE_FOR_TYPE = {
    ValueType.STRING: Parse.TEXT,
    ValueType.DECIMAL: Parse.CURRENCY,
    ValueType.INTEGER: Parse.INTEGER,
}
_RISKS = [r.value for r in Risk]
# Which element roles each action can target (click and extract accept anything: legacy apps put
# click handlers on table cells, and values live in cells, spans and text blocks).
_ACCEPTS: dict[str, set[str]] = {"fill": {"textbox", "password"}, "select": {"select"}}


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


_REF = {"type": "string", "description": "Element ref from the LATEST observation, e.g. e12"}
_INTENT = {
    "type": "string",
    "description": "Short imperative summary for a human reviewer, e.g. 'Run the search'",
}
_RATIONALE = {"type": "string", "description": "Why this action, in one sentence"}


def tool_definitions(goal: GoalSpec) -> list[dict[str, Any]]:
    outputs = [o.name for o in goal.outputs] or ["none"]
    return [
        _tool(
            "click",
            "Click a link or button.",
            {
                "ref": _REF,
                "intent": _INTENT,
                "expect_text": {
                    "type": "string",
                    "description": "Exact text that should be visible once the click worked "
                    "(a heading or label), or empty string if nothing new is expected",
                },
                "risk": {
                    "type": "string",
                    "enum": _RISKS,
                    "description": "safe = no change to data; reversible = changes data but can be undone; "
                    "irreversible = submits/commits something that can't be undone",
                },
                "rationale": _RATIONALE,
            },
        ),
        _tool(
            "fill",
            "Type into a text field. For inputs, type the placeholder, e.g. {{inputs.member_id}}.",
            {
                "ref": _REF,
                "value": {"type": "string"},
                "intent": _INTENT,
                "rationale": _RATIONALE,
            },
        ),
        _tool(
            "select",
            "Choose an option in a dropdown, by its visible label (or an input placeholder).",
            {
                "ref": _REF,
                "option": {"type": "string"},
                "intent": _INTENT,
                "rationale": _RATIONALE,
            },
        ),
        _tool(
            "extract",
            "Read the value of a goal output from the element that holds it (the value, not its label).",
            {
                "ref": _REF,
                "output": {"type": "string", "enum": outputs},
                "intent": _INTENT,
                "rationale": _RATIONALE,
            },
        ),
        _tool(
            "finish",
            "Declare the goal complete. Only after every output has been extracted.",
            {
                "success_text": {
                    "type": "string",
                    "description": "Exact text visible now that proves the goal succeeded",
                },
                "rationale": _RATIONALE,
            },
        ),
        _tool(
            "request_human",
            "Stop and ask a human: you're blocked, unsure, or an action needs approval.",
            {
                "reason": {"type": "string"},
            },
        ),
    ]


@dataclass
class ToolResult:
    text: str
    is_error: bool = False
    terminal: str | None = None  # "finish" | "request_human"


@dataclass
class DiscoveryTools:
    surface: WebSurface
    recorder: Recorder
    goal: GoalSpec
    values: dict[str, str]  # real input values, never shown to the model
    policy: Policy
    profile: AppProfile
    registry: SecretRegistry
    outputs: dict[str, Any] = field(default_factory=dict)
    finish_text: str | None = None
    finish_frame: str | None = None
    auto_dismissed: list[str] = field(default_factory=list)
    _snapshot: Snapshot = field(default_factory=Snapshot)

    # --- observation ----------------------------------------------------------------------

    def observe(self) -> str:
        """Handle known states, take a fresh snapshot, and render it (masked) for the model."""
        notes = self._handle_known_states()
        self._snapshot = self.surface.snapshot()
        rendered = mask_text(self._snapshot.render(), self.registry)
        done = ", ".join(sorted(self.outputs)) or "none"
        todo = ", ".join(o.name for o in self.goal.outputs if o.name not in self.outputs) or "none"
        footer = f"\n\nOutputs extracted: {done}. Still needed: {todo}."
        return "\n".join(notes) + ("\n" if notes else "") + rendered + footer

    def _handle_known_states(self) -> list[str]:
        notes: list[str] = []
        for state in self.profile.states:
            if not self.surface.check(state.when):
                continue
            if state.kind is StateKind.INTERSTITIAL and state.dismiss is not None:
                try:
                    self.surface.click(self.surface.resolve(state.dismiss, 5000), None, 5000)
                    self.surface.idle(500)
                    self.auto_dismissed.append(state.name)
                    notes.append(f"(System: known notice {state.name!r} was acknowledged automatically.)")
                    log.info("discovery.auto_dismissed", state=state.name)
                except (TargetNotFound, ActionFailed):
                    notes.append(
                        f"(System: known notice {state.name!r} is showing and could not be dismissed.)"
                    )
            else:
                notes.append(f"(System: known app state {state.name!r} ({state.kind}): {state.description}.)")
        return notes

    # --- execution ------------------------------------------------------------------------

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        result = self._execute(name, args)
        # Logged after running, so a value the action just extracted is already registered for masking.
        log.info(
            "agent.action",
            tool=name,
            ref=args.get("ref"),
            intent=args.get("intent"),
            rationale=args.get("rationale") or args.get("reason"),
            rejected=result.is_error,
        )
        return result

    def _execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            match name:
                case "click" | "fill" | "select" | "extract" as kind:
                    message = self._act(kind, args)
                    return ToolResult(f"{message}\n\n{self.observe()}")
                case "finish":
                    return self._finish(args)
                case "request_human":
                    return ToolResult("Stopping for a human.", terminal="request_human")
            return ToolResult(f"Unknown tool {name!r}.", is_error=True)
        except (RecordingError, ActionFailed, ValueError) as exc:
            log.warning("agent.action_rejected", tool=name, reason=str(exc))
            return ToolResult(f"Not done: {exc}\n\n{self.observe()}", is_error=True)

    def _element(self, ref: str) -> Element:
        element = self._snapshot.get(ref)
        if element is None:
            raise ValueError(
                f"unknown ref {ref!r}; refs change after every action, use the latest observation"
            )
        return element

    def _act(self, kind: ActionKind, args: dict[str, Any]) -> str:
        element = self._element(args["ref"])
        accepted = _ACCEPTS.get(kind)
        if accepted is not None and element.role not in accepted:
            raise ValueError(f"{element.ref} is a {element.role}, not one of: {', '.join(sorted(accepted))}")
        target, rejected = self.recorder.target_for(element)  # validated before the page changes
        step = RecordedStep(
            kind=kind,
            element=element,
            target=target,
            intent=args["intent"],
            rejected=rejected,
        )
        resolved = self.surface.resolve_ref(element)

        if kind == "click":
            step.declared_risk = Risk(args["risk"])
            probe = step_for_policy(step)
            if self.policy.needs_approval(probe):
                raise RecordingError(
                    "this action is irreversible and needs a human's approval: call request_human"
                )
            expect = args["expect_text"].strip()
            already = self._frames_showing(expect) if expect else set()
            self.surface.click(resolved, None, 10_000)
            if dialogs := self.surface.take_dialogs():
                raise RecordingError(
                    f"an unexpected dialog appeared and was dismissed: {dialogs[0].message!r}"
                )
            if expect:
                appeared, frame = self._wait_for_new_text(expect, already)
                if appeared:
                    step.expect_text, step.expect_frame = expect, frame
                else:
                    unproven = expect
        elif kind in ("fill", "select"):
            value = args["value"] if kind == "fill" else args["option"]
            self._check_placeholders(value)
            step.value = value
            rendered = render(value, self.values)
            if kind == "fill":
                self.surface.fill(resolved, rendered, 10_000)
            else:
                self.surface.select(resolved, rendered, 10_000)
        else:
            self._extract(step, args["output"], resolved)

        self.recorder.record(step)
        if kind == "click" and expect and not step.expect_text:
            return (
                f"Clicked, but {unproven!r} did not newly appear (it was already visible, or never showed). "
                "Next time give text that only appears after the click worked."
            )
        log.info(
            "discovery.step_recorded",
            kind=kind,
            strategies=[s.kind for s in target.strategies],
            rejected=len(rejected),
        )
        return "Done."

    def _check_placeholders(self, value: str) -> None:
        unknown = set(TEMPLATE.findall(value)) - set(self.values)
        if unknown:
            raise ValueError(f"unknown input(s) {sorted(unknown)}; declared: {sorted(self.values)}")
        literal = TEMPLATE.sub("", value)
        if any(v and v in literal for v in self.values.values()):
            raise ValueError("type the input placeholder (e.g. {{inputs.name}}), not its value")

    def _extract(self, step: RecordedStep, output: str, resolved: Any) -> None:
        spec = next((o for o in self.goal.outputs if o.name == output), None)
        if spec is None:
            raise ValueError(f"{output!r} is not a goal output")
        parse = _PARSE_FOR_TYPE.get(spec.type, Parse.TEXT)
        text = self.surface.read_text(resolved)
        try:
            value = parse_output(text, parse)
        except OutputParseError as exc:
            raise ValueError(f"that element doesn't hold a {spec.type}: {exc}") from None
        for seen in (text.strip(), str(value)):
            self.recorder.sensitive_values.add(seen)
            self.registry.register(spec.name, seen, spec.sensitivity)
        self.outputs[output] = value
        step.output, step.parse = output, parse

    def _frames(self) -> list[str | None]:
        return [None, *[f for f in self.surface.observe().locations if f != "top"]]

    def _frames_showing(self, text: str) -> set[str | None]:
        return {
            f
            for f in self._frames()
            if self.surface.check(TextPresent(kind="text_present", text=text, frame=f))
        }

    def _wait_for_new_text(self, text: str, already: set[str | None]) -> tuple[bool, str | None]:
        """(appeared, frame): the first frame where `text` shows up that didn't show it before the click.

        A post-condition must be evidence of change: text that was already visible (e.g. the link
        that was just clicked) would pass even if the click did nothing. frame None = top document.
        """
        deadline = time.monotonic() + EXPECT_TIMEOUT_S
        while time.monotonic() < deadline:
            for frame in self._frames():
                if frame not in already and self.surface.check(
                    TextPresent(kind="text_present", text=text, frame=frame)
                ):
                    return True, frame
            self.surface.idle(200)
        return False, None

    def _finish(self, args: dict[str, Any]) -> ToolResult:
        missing = [o.name for o in self.goal.outputs if o.name not in self.outputs]
        if missing:
            return ToolResult(f"Not done: outputs not extracted yet: {missing}.", is_error=True)
        text = args["success_text"].strip()
        for frame in self._frames():
            if self.surface.check(TextPresent(kind="text_present", text=text, frame=frame)):
                self.finish_text, self.finish_frame = text, frame
                return ToolResult("Recorded.", terminal="finish")
        return ToolResult(f"Not done: {text!r} is not visible on screen right now.", is_error=True)


def step_for_policy(step: RecordedStep) -> Step:
    """A minimal artifact Step for the policy's risk heuristic, built from a recorded click."""
    return Step.model_validate(
        {
            "id": "probe",
            "intent": step.intent,
            "risk": step.declared_risk,
            "action": {"kind": "click", "target": step.target.model_dump(mode="json")},
        }
    )
