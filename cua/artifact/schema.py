"""Capability artifact schema, v1.0.

A capability is a typed, versioned, declarative description of a UI flow that an AI agent can
invoke with parameters and get typed outputs back. It is JSON data, not code:

- Declarative so it can be reviewed, diffed, hashed, and policy-checked *before* it runs.
- Templating is limited to `{{inputs.<name>}}` substitution. There is no expression language,
  so an artifact can never become a program.
- It never contains raw input values. Inputs are referenced by name, and saving refuses any
  artifact that looks like it contains sensitive data (see store.py).

Targets are identified by an ordered list of locator strategies, most robust first. The replay
engine uses the first strategy that resolves to exactly one element; if that is not the first
one, it reports a drift warning.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cua.security.masking import Sensitivity

SCHEMA_VERSION = "1.1"
# Every version this code can load. 1.1 added `outcomes` (additive: all 1.0 artifacts are valid 1.1).
SUPPORTED_SCHEMA_VERSIONS = ("1.0", "1.1")
# Fields added after 1.0. They are omitted from the canonical form while empty, so an older artifact
# serializes, and therefore hashes, exactly as it did before the upgrade. Otherwise a schema upgrade
# would change every artifact's content hash and silently void approvals bound to it.
ADDED_FIELDS = ("outcomes",)

IDENTIFIER = r"^[a-z][a-z0-9_]*$"
CAPABILITY_ID = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"  # dotted: product.domain.action
SEMVER = r"^\d+\.\d+\.\d+$"

TEMPLATE = re.compile(r"\{\{\s*inputs\.([a-z][a-z0-9_]*)\s*\}\}")
_ANY_BRACES = re.compile(r"\{\{.*?\}\}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- enums --------------------------------------------------------------------------------


class Risk(StrEnum):
    SAFE = "safe"                  # no server-side state change (navigate, read, fill a search box)
    REVERSIBLE = "reversible"      # changes state, but it can be undone
    IRREVERSIBLE = "irreversible"  # cannot be undone (moves money, opens/closes accounts, submits)


_RISK_ORDER = [Risk.SAFE, Risk.REVERSIBLE, Risk.IRREVERSIBLE]


class ValueType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"


class Parse(StrEnum):
    """How extracted text becomes a typed output."""

    TEXT = "text"          # -> string (whitespace-normalized)
    CURRENCY = "currency"  # "$1,204.55" / "(12.00)" -> decimal
    INTEGER = "integer"    # "1,204" -> integer


PARSE_RESULT_TYPE: dict[Parse, ValueType] = {
    Parse.TEXT: ValueType.STRING,
    Parse.CURRENCY: ValueType.DECIMAL,
    Parse.INTEGER: ValueType.INTEGER,
}


class ControlKind(StrEnum):
    """Kinds of form control, independent of ARIA roles (password inputs have no ARIA role)."""

    TEXTBOX = "textbox"
    PASSWORD = "password"
    SELECT = "select"
    BUTTON = "button"
    CHECKBOX = "checkbox"


class Status(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


# --- contract: inputs and outputs ---------------------------------------------------------


class ParamSpec(_Model):
    name: str = Field(pattern=IDENTIFIER)
    type: ValueType = ValueType.STRING
    description: str
    sensitivity: Sensitivity = Sensitivity.PUBLIC
    pattern: str | None = None  # full-match regex applied to the canonical string value
    choices: list[str] | None = None

    @field_validator("pattern")
    @classmethod
    def _pattern_compiles(cls, value: str | None) -> str | None:
        if value is not None:
            re.compile(value)
        return value


class OutputSpec(_Model):
    name: str = Field(pattern=IDENTIFIER)
    type: ValueType
    description: str
    sensitivity: Sensitivity = Sensitivity.PUBLIC


# --- locators: how a control is identified, most robust first -----------------------------


class RoleLocator(_Model):
    """Accessible role + name. Portable concept: web ARIA, Windows UIA, macOS AX all have it."""

    kind: Literal["role"]
    role: str
    name: str


class LabelLocator(_Model):
    """Control associated with a <label> or aria-label. Useless on most legacy apps, great on modern ones."""

    kind: Literal["label"]
    text: str


class NearTextLocator(_Model):
    """A control in the same table row as a cell whose text is `anchor` (trailing ':' ignored).

    How a human finds an unlabeled legacy field: "the box next to 'Member Number'".
    """

    kind: Literal["near_text"]
    anchor: str
    control: ControlKind


class FieldValueLocator(_Model):
    """The cell immediately after a label cell: `Name: | Avery Testwood`."""

    kind: Literal["field_value"]
    label: str


class TableCellLocator(_Model):
    """The cell in the row containing `row` text, under the column headed `column`.

    Survives column reordering and new rows, because nothing is positional.
    """

    kind: Literal["table_cell"]
    row: str
    column: str


class CssLocator(_Model):
    """Last resort. Only acceptable on stable attributes, e.g. a form field's `name`, which the
    server depends on and therefore rarely changes."""

    kind: Literal["css"]
    selector: str


Locator = Annotated[
    RoleLocator | LabelLocator | NearTextLocator | FieldValueLocator | TableCellLocator | CssLocator,
    Field(discriminator="kind"),
]


class Target(_Model):
    frame: str | None = None  # frame name, for frameset apps; None = top-level document
    strategies: list[Locator] = Field(min_length=1)


# --- checkpoints: assert state instead of assuming an action worked -----------------------


class TextPresent(_Model):
    kind: Literal["text_present"]
    text: str
    frame: str | None = None


class ElementVisible(_Model):
    kind: Literal["element_visible"]
    target: Target


class UrlMatches(_Model):
    kind: Literal["url_matches"]
    pattern: str  # regex searched in the (frame's) URL path + query
    frame: str | None = None


class AllOf(_Model):
    kind: Literal["all_of"]
    conditions: list[Checkpoint] = Field(min_length=1)


class AnyOf(_Model):
    kind: Literal["any_of"]
    conditions: list[Checkpoint] = Field(min_length=1)


Checkpoint = Annotated[
    TextPresent | ElementVisible | UrlMatches | AllOf | AnyOf,
    Field(discriminator="kind"),
]


# --- actions ------------------------------------------------------------------------------


class DialogExpectation(_Model):
    """A native dialog this action is expected to raise, and how to answer it.
    Any dialog that is *not* expected is never answered blindly."""

    accept: bool
    message_contains: str


class Navigate(_Model):
    kind: Literal["navigate"]
    route: str  # relative to the tenant's base URL; may use templates


class Click(_Model):
    kind: Literal["click"]
    target: Target
    dialog: DialogExpectation | None = None


class Fill(_Model):
    kind: Literal["fill"]
    target: Target
    value: str  # template, e.g. "{{inputs.member_id}}"


class Select(_Model):
    kind: Literal["select"]
    target: Target
    option: str  # visible option label; may use templates


class Press(_Model):
    kind: Literal["press"]
    key: str
    target: Target | None = None


class Extract(_Model):
    kind: Literal["extract"]
    target: Target
    output: str = Field(pattern=IDENTIFIER)
    parse: Parse = Parse.TEXT


Action = Annotated[Navigate | Click | Fill | Select | Press | Extract, Field(discriminator="kind")]


class Step(_Model):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    intent: str  # human-readable: shown to reviewers and in failure messages
    action: Action
    risk: Risk
    expect: Checkpoint | None = None  # post-condition; replay fails *at this step* if not reached
    timeout_ms: int = Field(default=10_000, ge=100, le=120_000)

    @model_validator(mode="after")
    def _read_only_actions_are_safe(self) -> Step:
        if isinstance(self.action, (Navigate, Extract)) and self.risk is not Risk.SAFE:
            raise ValueError(f"step {self.id}: {self.action.kind} actions must be risk 'safe'")
        return self


# --- business outcomes --------------------------------------------------------------------


class OutcomeSpec(_Model):
    """A legitimate, expected result other than success, e.g. "no such member".

    Part of the contract, like a typed error in a function signature: the calling agent can see
    every outcome it may get back. When `when` becomes true, replay stops and returns this outcome
    (not a failure). `after_step` limits detection to that step and later, so an early page can't
    trigger an outcome meant for a later one.
    """

    name: str = Field(pattern=IDENTIFIER)
    description: str
    when: Checkpoint
    after_step: str | None = None


# --- the capability -----------------------------------------------------------------------


class AppRef(_Model):
    product: str = Field(pattern=IDENTIFIER)  # the vendor product, not a tenant
    versions: str  # product version range this was recorded/verified against, e.g. ">=4.2 <5"
    surface: Literal["web"] = "web"


class Provenance(_Model):
    source: Literal["hand_written", "discovery"]
    recorded_at: datetime | None = None
    discovery_run_id: str | None = None
    model: str | None = None
    notes: str | None = None


class Capability(_Model):
    schema_version: Literal["1.0", "1.1"]
    id: str = Field(pattern=CAPABILITY_ID)
    version: str = Field(pattern=SEMVER)
    status: Status = Status.DRAFT
    description: str
    app: AppRef
    entry_route: str
    inputs: list[ParamSpec] = []
    outputs: list[OutputSpec] = []
    steps: list[Step] = Field(min_length=1)
    success: Checkpoint
    outcomes: list[OutcomeSpec] = []
    provenance: Provenance

    @model_validator(mode="after")
    def _consistent(self) -> Capability:
        _unique([s.id for s in self.steps], "step id")
        _unique([p.name for p in self.inputs], "input")
        _unique([o.name for o in self.outputs], "output")
        _unique([o.name for o in self.outcomes], "outcome")
        step_ids = {s.id for s in self.steps}
        for outcome in self.outcomes:
            if outcome.after_step is not None and outcome.after_step not in step_ids:
                raise ValueError(f"outcome {outcome.name!r} refers to unknown step {outcome.after_step!r}")

        declared_inputs = {p.name for p in self.inputs}
        used_inputs: set[str] = set()
        for text in self.template_strings():
            for braces in _ANY_BRACES.findall(text):
                match = TEMPLATE.fullmatch(braces)
                if match is None:
                    raise ValueError(f"unsupported template {braces!r}; only {{{{inputs.<name>}}}} allowed")
                used_inputs.add(match.group(1))
        if undeclared := used_inputs - declared_inputs:
            raise ValueError(f"templates reference undeclared inputs: {sorted(undeclared)}")
        if unused := declared_inputs - used_inputs:
            raise ValueError(f"declared inputs are never used: {sorted(unused)}")

        outputs = {o.name: o for o in self.outputs}
        extracts = [s.action for s in self.steps if isinstance(s.action, Extract)]
        _unique([e.output for e in extracts], "extracted output")
        for extract in extracts:
            spec = outputs.get(extract.output)
            if spec is None:
                raise ValueError(f"extract writes undeclared output {extract.output!r}")
            if PARSE_RESULT_TYPE[extract.parse] is not spec.type:
                raise ValueError(
                    f"output {spec.name!r} is {spec.type} but parse {extract.parse} yields "
                    f"{PARSE_RESULT_TYPE[extract.parse]}"
                )
        if missing := set(outputs) - {e.output for e in extracts}:
            raise ValueError(f"declared outputs are never extracted: {sorted(missing)}")
        return self

    def template_strings(self) -> list[str]:
        strings = [self.entry_route]
        for step in self.steps:
            action = step.action
            if isinstance(action, Navigate):
                strings.append(action.route)
            elif isinstance(action, Fill):
                strings.append(action.value)
            elif isinstance(action, Select):
                strings.append(action.option)
        return strings

    @property
    def max_risk(self) -> Risk:
        return max((s.risk for s in self.steps), key=_RISK_ORDER.index)

    def canonical_dict(self) -> dict[str, Any]:
        """The serialized form: JSON-compatible, None fields and empty later-added fields omitted."""
        data = self.model_dump(mode="json", exclude_none=True)
        for name in ADDED_FIELDS:
            if not data.get(name):
                data.pop(name, None)
        return data

    def content_hash(self) -> str:
        """Hash of what the capability *does*. Excludes status and provenance, so approving a
        capability doesn't change its hash, but any behavioural edit does (and voids approval)."""
        behaviour = {k: v for k, v in self.canonical_dict().items() if k not in ("status", "provenance")}
        encoded = json.dumps(behaviour, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _unique(values: list[str], what: str) -> None:
    duplicates = sorted({v for v in values if values.count(v) > 1})
    if duplicates:
        raise ValueError(f"duplicate {what}: {duplicates}")


OutcomeSpec.model_rebuild()
AllOf.model_rebuild()
AnyOf.model_rebuild()
Step.model_rebuild()
Capability.model_rebuild()
