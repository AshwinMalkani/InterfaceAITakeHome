"""Which on-screen text may appear in evidence screenshots.

Screenshots are redacted by allowlist, not denylist: every text node and input value is masked
unless its whole (normalized) text is known vocabulary. On a banking screen most text is data, and
data that no pattern recognizes (a name, a 7-digit phone) is exactly what leaks.

Vocabulary = the artifact's own words (its locators, checkpoints and outcomes were written for this
UI and are clean by construction) + the app profile's UI labels and known-state texts. Labels and
headings stay readable, so a reviewer can still see *where* replay was; values become blocks.
"""

from __future__ import annotations

import re

from cua.artifact.schema import (
    AllOf,
    AnyOf,
    Capability,
    Checkpoint,
    Click,
    CssLocator,
    ElementVisible,
    Extract,
    FieldValueLocator,
    Fill,
    LabelLocator,
    Locator,
    NearTextLocator,
    Press,
    RoleLocator,
    Select,
    TableCellLocator,
    Target,
    TextPresent,
)
from cua.profile import AppProfile


def normalize_label(text: str) -> str:
    """Must match `norm` in the surface's redaction script: collapse whitespace, drop a trailing
    colon, lowercase."""
    return re.sub(r"\s+", " ", text).strip().removesuffix(":").strip().lower()


def _locator_texts(locator: Locator) -> list[str]:
    match locator:
        case RoleLocator(name=name):
            return [name]
        case LabelLocator(text=text):
            return [text]
        case NearTextLocator(anchor=anchor):
            return [anchor]
        case FieldValueLocator(label=label):
            return [label]  # the label only; the value next to it is what we're protecting
        case TableCellLocator(row=row, column=column):
            return [row, column]
        case CssLocator():
            return []
    raise AssertionError(f"unhandled locator {locator!r}")


def _target_texts(target: Target) -> list[str]:
    return [text for locator in target.strategies for text in _locator_texts(locator)]


def checkpoint_texts(checkpoint: Checkpoint) -> list[str]:
    match checkpoint:
        case TextPresent(text=text):
            return [text]
        case ElementVisible(target=target):
            return _target_texts(target)
        case AllOf(conditions=conditions) | AnyOf(conditions=conditions):
            return [text for c in conditions for text in checkpoint_texts(c)]
    return []


def capability_vocabulary(capability: Capability) -> frozenset[str]:
    texts: list[str] = checkpoint_texts(capability.success)
    for step in capability.steps:
        # Targets name *labels* (a row, a column, the text next to a field), never the data in them,
        # and the values typed or chosen are never vocabulary.
        match step.action:
            case Click(target=target) | Fill(target=target) | Select(target=target) | Extract(target=target):
                texts += _target_texts(target)
            case Press(target=Target() as target):
                texts += _target_texts(target)
        if step.expect is not None:
            texts += checkpoint_texts(step.expect)
    for outcome in capability.outcomes:
        texts += checkpoint_texts(outcome.when)
    return frozenset(normalize_label(t) for t in texts if t.strip())


def profile_vocabulary(profile: AppProfile) -> frozenset[str]:
    texts = list(profile.ui_vocabulary)
    for state in profile.states:
        texts += checkpoint_texts(state.when)
        if state.dismiss is not None:
            texts += _target_texts(state.dismiss)
    return frozenset(normalize_label(t) for t in texts if t.strip())


_OBSERVATION_LINE = re.compile(r'^(\s+e\d+\s+(?:cell|text) )"(.*)"$')
_OPTIONS = re.compile(r"options=\[(.*)\]")


def redact_observation(text: str, vocabulary: frozenset[str]) -> str:
    """Allowlist-redact a rendered snapshot for persistence: cell/text values and select options are
    kept only if they are vocabulary. Used on discovery transcripts, which record what the model saw."""

    def block(value: str) -> str:
        return value if normalize_label(value) in vocabulary else re.sub(r"\S", "█", value)

    lines = []
    for line in text.splitlines():
        if match := _OBSERVATION_LINE.match(line):
            line = f'{match.group(1)}"{block(match.group(2))}"'
        elif match := _OPTIONS.search(line):
            options = [block(o.strip().strip("'\"")) for o in match.group(1).split(",") if o.strip()]
            line = line[: match.start()] + "options=[" + ", ".join(repr(o) for o in options) + "]"
        lines.append(line)
    return "\n".join(lines)
