"""Human-readable descriptions of targets and checkpoints, for failure reports and logs.

Everything described here comes from the artifact, which is clean by construction, so these
strings are safe to log.
"""

from __future__ import annotations

from cua.artifact.schema import (
    AllOf,
    AnyOf,
    Checkpoint,
    CssLocator,
    ElementVisible,
    FieldValueLocator,
    LabelLocator,
    Locator,
    NearTextLocator,
    RoleLocator,
    TableCellLocator,
    Target,
    TextPresent,
    UrlMatches,
)


def _in_frame(frame: str | None) -> str:
    return f" in frame {frame!r}" if frame else ""


def describe_locator(spec: Locator) -> str:
    match spec:
        case RoleLocator(role=role, name=name):
            return f"{role} named {name!r}"
        case LabelLocator(text=text):
            return f"control labelled {text!r}"
        case NearTextLocator(anchor=anchor, control=control):
            return f"{control} in the row of {anchor!r}"
        case FieldValueLocator(label=label):
            return f"value after {label!r}"
        case TableCellLocator(row=row, column=column):
            return f"{column!r} cell of the {row!r} row"
        case CssLocator(selector=selector):
            return f"css {selector!r}"
    raise AssertionError(f"unhandled locator {spec!r}")


def describe_target(target: Target) -> str:
    return describe_locator(target.strategies[0]) + _in_frame(target.frame)


def describe_checkpoint(checkpoint: Checkpoint) -> str:
    match checkpoint:
        case TextPresent(text=text, frame=frame):
            return f"text {text!r} visible{_in_frame(frame)}"
        case UrlMatches(pattern=pattern, frame=frame):
            return f"URL matching {pattern!r}{_in_frame(frame)}"
        case ElementVisible(target=target):
            return f"{describe_target(target)} visible"
        case AllOf(conditions=conditions):
            return " and ".join(f"({describe_checkpoint(c)})" for c in conditions)
        case AnyOf(conditions=conditions):
            return " or ".join(f"({describe_checkpoint(c)})" for c in conditions)
    raise AssertionError(f"unhandled checkpoint {checkpoint!r}")
