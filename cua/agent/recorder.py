"""Turn the model's choices into a capability artifact: the model decides *what*, this decides *how*.

At the moment the model picks an element (before acting, while the page is unchanged) the recorder
derives candidate locators from the element's facts and keeps only those that resolve to exactly
one visible element, *and that element is the one the model meant*. A locator that merely happens
to work is never recorded. Candidates are ordered most-robust first:

    role + name  >  <label>  >  control next to a label  >  table cell by row & column header
    >  value after a label  >  form-field name (css)

Compilation then assembles the steps with the goal's contract and refuses the artifact if any
sensitive value seen during the run (inputs typed, outputs read) appears anywhere in it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from cua.agent.goal import GoalSpec
from cua.artifact.schema import (
    TEMPLATE,
    Capability,
    ControlKind,
    CssLocator,
    DialogExpectation,
    FieldValueLocator,
    LabelLocator,
    Locator,
    NearTextLocator,
    Parse,
    Risk,
    RoleLocator,
    TableCellLocator,
    Target,
)
from cua.policy import Policy
from cua.surface.snapshot import Element

ActionKind = Literal["click", "fill", "select", "extract"]

_CONTROL_KINDS = {
    "textbox": ControlKind.TEXTBOX,
    "password": ControlKind.PASSWORD,
    "select": ControlKind.SELECT,
    "button": ControlKind.BUTTON,
    "checkbox": ControlKind.CHECKBOX,
}


class RecordingError(Exception):
    """The chosen element can't be identified robustly, or the result would be unsafe to save."""


class CandidateChecker(Protocol):
    def matches_only(self, target: Target, element: Element) -> list[bool]: ...


def _labelish(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def candidate_locators(element: Element) -> list[Locator]:
    """Every strategy the element's facts support, most robust first (unvalidated)."""
    e = element
    candidates: list[Locator] = []
    if e.role in ("link", "button") and e.name:
        candidates.append(RoleLocator(kind="role", role=e.role, name=e.name))
    if e.is_control and e.label:
        candidates.append(LabelLocator(kind="label", text=e.label))
    if e.role in _CONTROL_KINDS and e.left_label:
        candidates.append(
            NearTextLocator(kind="near_text", anchor=e.left_label, control=_CONTROL_KINDS[e.role])
        )
    if e.role == "cell" and e.row_key and e.column_header:
        candidates.append(TableCellLocator(kind="table_cell", row=e.row_key, column=e.column_header))
    if e.role == "cell" and _labelish(e.prev_cell):
        candidates.append(FieldValueLocator(kind="field_value", label=e.prev_cell))
    if e.is_control and e.name_attr and re.fullmatch(r"[\w.\-]+", e.name_attr):
        candidates.append(CssLocator(kind="css", selector=f"{e.tag}[name='{e.name_attr}']"))
    return candidates


def _locator_texts(locator: Locator) -> list[str]:
    return [v for k, v in locator.model_dump().items() if k != "kind" and isinstance(v, str)]


@dataclass
class RecordedStep:
    kind: ActionKind
    element: Element
    target: Target
    intent: str
    declared_risk: Risk = Risk.SAFE
    value: str | None = None  # fill text / select option, possibly with {{inputs.x}} placeholders
    output: str | None = None
    parse: Parse | None = None
    dialog: DialogExpectation | None = None
    expect_text: str | None = None  # verified visible after the action, at record time
    expect_frame: str | None = None
    rejected: list[str] = field(default_factory=list)  # candidates dropped by validation (evidence)


@dataclass
class Recorder:
    checker: CandidateChecker
    sensitive_values: set[str] = field(default_factory=set)  # every input value typed / output read
    steps: list[RecordedStep] = field(default_factory=list)

    def target_for(self, element: Element) -> tuple[Target, list[str]]:
        """Validated target for `element`, plus descriptions of the candidates that were rejected."""
        candidates = [c for c in candidate_locators(element) if not self._mentions_sensitive(c)]
        if not candidates:
            raise RecordingError(
                f"{element.ref} ({element.role}) has no identifying label, name or field name"
            )
        verdicts = self.checker.matches_only(Target(frame=element.frame, strategies=candidates), element)
        kept = [c for c, ok in zip(candidates, verdicts, strict=True) if ok]
        rejected = [json.dumps(c.model_dump()) for c, ok in zip(candidates, verdicts, strict=True) if not ok]
        if not kept:
            raise RecordingError(f"no candidate locator uniquely identifies {element.ref}; tried {rejected}")
        return Target(frame=element.frame, strategies=kept), rejected

    def _mentions_sensitive(self, locator: Locator) -> bool:
        lowered = {v.lower() for v in self.sensitive_values if v}
        return any(text.lower() in lowered for text in _locator_texts(locator))

    def record(self, step: RecordedStep) -> RecordedStep:
        self.steps.append(step)
        return step


def _step_dict(index: int, step: RecordedStep) -> dict[str, Any]:
    target = step.target.model_dump(mode="json", exclude_none=True)
    action: dict[str, Any] = {"kind": step.kind, "target": target}
    if step.kind == "fill":
        action["value"] = step.value
    elif step.kind == "select":
        action["option"] = step.value
    elif step.kind == "extract":
        action |= {"output": step.output, "parse": step.parse}
    if step.kind == "click" and step.dialog is not None:
        action["dialog"] = step.dialog.model_dump()
    data: dict[str, Any] = {
        "id": f"s{index}",
        "intent": step.intent,
        "action": action,
        "risk": step.declared_risk,
    }
    if step.expect_text:
        data["expect"] = {"kind": "text_present", "text": step.expect_text, "frame": step.expect_frame}
    return data


def compile_capability(
    goal: GoalSpec,
    steps: list[RecordedStep],
    *,
    success_text: str,
    success_frame: str | None,
    sensitive_values: set[str],
    policy: Policy | None = None,
    run_id: str | None = None,
    model: str | None = None,
) -> Capability:
    """Assemble and validate the artifact. Raises RecordingError if it would leak or is inconsistent."""
    if not steps:
        raise RecordingError("nothing was recorded")
    data: dict[str, Any] = {
        "schema_version": "1.1",
        "id": goal.id,
        "version": goal.version,
        "description": goal.description,
        "app": goal.app.model_dump(),
        "entry_route": goal.entry_route,
        "inputs": [p.model_dump(mode="json", exclude_none=True) for p in goal.inputs],
        "outputs": [o.model_dump(mode="json") for o in goal.outputs],
        "steps": [_step_dict(i, s) for i, s in enumerate(steps, start=1)],
        "success": {"kind": "text_present", "text": success_text, "frame": success_frame},
        "provenance": {
            "source": "discovery",
            "recorded_at": datetime.now(UTC).isoformat(),
            "discovery_run_id": run_id,
            "model": model,
        },
    }
    try:
        capability = Capability.model_validate(data)
    except ValueError as exc:
        raise RecordingError(f"recorded flow doesn't satisfy the goal's contract: {exc}") from None

    if policy is not None:  # the policy heuristic can raise a step's risk, never lower it
        raised = [s.model_copy(update={"risk": policy.effective_risk(s)}) for s in capability.steps]
        capability = capability.model_copy(update={"steps": raised})

    _assert_no_leaks(capability, sensitive_values)
    return capability


def _assert_no_leaks(capability: Capability, sensitive_values: set[str]) -> None:
    """No value the run touched may appear in the artifact; templates are the only way to refer to
    inputs. Fails closed: a coincidental substring match is treated as a leak. The error says how
    many values leaked, never which."""
    blob = TEMPLATE.sub("", json.dumps(capability.canonical_dict()).lower())
    leaked = {v for v in sensitive_values if len(v) >= 3 and v.lower() in blob}
    if leaked:
        raise RecordingError(f"artifact would contain {len(leaked)} sensitive value(s) seen during the run")
