"""Invocation-time contract enforcement: validate inputs, render templates, parse outputs.

Error messages name the field and the rule, never the value: inputs are often PII.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation

from cua.artifact.schema import TEMPLATE, Capability, ParamSpec, Parse, ValueType


class InputError(ValueError):
    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"input {field!r}: {reason}")
        self.field = field
        self.reason = reason


class OutputParseError(ValueError):
    pass


def _canonical(spec: ParamSpec, raw: object) -> str:
    """Coerce a raw value to its type and return the canonical string used for templating."""
    text = str(raw).strip()
    match spec.type:
        case ValueType.STRING:
            return text
        case ValueType.INTEGER:
            if isinstance(raw, bool) or not re.fullmatch(r"-?\d+", text):
                raise InputError(spec.name, "must be an integer")
            return str(int(text))
        case ValueType.DECIMAL:
            try:
                return str(Decimal(text))
            except InvalidOperation:
                raise InputError(spec.name, "must be a decimal number") from None
        case ValueType.BOOLEAN:
            if text.lower() not in ("true", "false"):
                raise InputError(spec.name, "must be true or false")
            return text.lower()


def validate_inputs(capability: Capability, raw: Mapping[str, object]) -> dict[str, str]:
    """Check params against the capability's declared inputs; return canonical string values."""
    specs = {p.name: p for p in capability.inputs}
    if unknown := sorted(set(raw) - set(specs)):
        raise InputError(unknown[0], "not a declared input")
    values: dict[str, str] = {}
    for name, spec in specs.items():
        if name not in raw or raw[name] is None or str(raw[name]).strip() == "":
            raise InputError(name, "is required")
        value = _canonical(spec, raw[name])
        if spec.pattern is not None and not re.fullmatch(spec.pattern, value):
            raise InputError(name, f"does not match pattern {spec.pattern}")
        if spec.choices is not None and value not in spec.choices:
            raise InputError(name, f"must be one of {spec.choices}")
        values[name] = value
    return values


def render(template: str, values: Mapping[str, str]) -> str:
    """Substitute `{{inputs.x}}` placeholders. Inputs were validated, so every name exists."""
    return TEMPLATE.sub(lambda m: values[m.group(1)], template)


def parse_output(text: str, parse: Parse) -> str | Decimal | int:
    normalized = " ".join(text.split())
    match parse:
        case Parse.TEXT:
            return normalized
        case Parse.CURRENCY:
            negative = normalized.startswith("(") and normalized.endswith(")")
            digits = normalized.strip("()").replace("$", "").replace(",", "").strip()
            try:
                amount = Decimal(digits)
            except InvalidOperation:
                raise OutputParseError("expected a currency amount") from None
            return -amount if negative else amount
        case Parse.INTEGER:
            digits = normalized.replace(",", "")
            if not re.fullmatch(r"-?\d+", digits):
                raise OutputParseError("expected an integer")
            return int(digits)
