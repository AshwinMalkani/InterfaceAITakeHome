"""Single choke point for redacting sensitive data before it leaves the process.

Every egress path (log formatter, evidence files, saved artifacts, intervention requests,
CLI output) calls `mask_secrets`. Detectors run in this order:

  1. Known runtime values  - exact values registered in a per-run SecretRegistry
                             (param/output values marked pii/secret, env secrets).
  2. Sensitive key names   - dict keys like "password"/"token" have their value replaced wholesale.
  3. Patterns              - SSN, Luhn-valid card numbers, account-like digit runs, email,
                             phone, API keys / bearer tokens.

Known-value tokens carry a short HMAC (`[PII:member_id#3f9a]`) so the same value can be
correlated across one run's logs without being revealed. The HMAC key is random per registry,
so tokens cannot be correlated across runs or brute-forced offline from logs alone.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class Sensitivity(StrEnum):
    PUBLIC = "public"
    PII = "pii"
    SECRET = "secret"


# Env vars whose values are always treated as secrets if present.
SECRET_ENV_VARS: tuple[str, ...] = ("ANTHROPIC_API_KEY", "CU_CORE_PASSWORD")

# Values shorter than this are not registered: masking "1" or "No" everywhere would destroy logs.
MIN_KNOWN_VALUE_LEN = 3

MASKING_ERROR = "[MASKING_ERROR]"

_SENSITIVE_KEY = re.compile(
    r"pass(word|wd)?|secret|token|api[_-]?key|authorization|cookie|ssn|credential",
    re.IGNORECASE,
)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _mask_card(m: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    if 13 <= len(digits) <= 19 and _luhn_valid(digits):
        return f"[CARD:***{digits[-4:]}]"
    return m.group(0)


# (pattern, replacement) applied in order. Cards run before account numbers so a card
# is labelled as a card; the account rule then catches remaining long digit runs.
_PATTERNS: list[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]] = [
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{8,}"), "[SECRET:api_key]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-~+/]+=*", re.IGNORECASE), "[SECRET:bearer_token]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), _mask_card),
    (re.compile(r"\b\d{8,17}\b"), lambda m: f"[ACCT:***{m.group(0)[-4:]}]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"(?<!\d)\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"), "[PHONE]"),
]


class SecretRegistry:
    """Per-run set of concrete sensitive values to scrub, plus the HMAC key for tokens."""

    def __init__(self, *, include_env: bool = True, key: bytes | None = None) -> None:
        self._key = key or secrets.token_bytes(32)
        self._values: dict[str, tuple[str, Sensitivity]] = {}  # lowercased value -> (name, sensitivity)
        self._compiled: re.Pattern[str] | None = None
        if include_env:
            for var in SECRET_ENV_VARS:
                if value := os.environ.get(var):
                    self.register(var.lower(), value, Sensitivity.SECRET)

    def register(self, name: str, value: object, sensitivity: Sensitivity) -> None:
        if sensitivity is Sensitivity.PUBLIC or value is None:
            return
        text = str(value).strip()
        if len(text) < MIN_KNOWN_VALUE_LEN:
            return
        self._values[text.lower()] = (name, sensitivity)
        self._compiled = None

    def token(self, name: str, value: str, sensitivity: Sensitivity) -> str:
        if sensitivity is Sensitivity.SECRET:
            # Secrets get no correlation suffix: even equality of secrets is not worth leaking.
            return f"[SECRET:{name}]"
        digest = hmac.new(self._key, value.lower().encode(), hashlib.sha256).hexdigest()[:4]
        return f"[PII:{name}#{digest}]"

    def mask_known(self, text: str) -> str:
        if not self._values:
            return text
        if self._compiled is None:
            # Longest first so "John Smith" wins over "John" when both are registered.
            alternatives = sorted(self._values, key=len, reverse=True)
            self._compiled = re.compile("|".join(re.escape(v) for v in alternatives), re.IGNORECASE)

        def replace(m: re.Match[str]) -> str:
            name, sensitivity = self._values[m.group(0).lower()]
            return self.token(name, m.group(0), sensitivity)

        return self._compiled.sub(replace, text)


_current_registry: ContextVar[SecretRegistry | None] = ContextVar("secret_registry", default=None)
_fallback_registry = SecretRegistry()


def current_registry() -> SecretRegistry:
    return _current_registry.get() or _fallback_registry


@contextmanager
def use_registry(registry: SecretRegistry) -> Iterator[SecretRegistry]:
    """Make `registry` the implicit one for everything (incl. logging) inside this block."""
    token = _current_registry.set(registry)
    try:
        yield registry
    finally:
        _current_registry.reset(token)


def mask_text(text: str, registry: SecretRegistry | None = None) -> str:
    text = (registry or current_registry()).mask_known(text)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def mask_secrets(value: Any, registry: SecretRegistry | None = None) -> Any:
    """Return a masked, JSON-friendly copy of `value`. Never mutates the input."""
    reg = registry or current_registry()
    if isinstance(value, BaseModel):
        return mask_secrets(value.model_dump(mode="json"), reg)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if _SENSITIVE_KEY.search(key) and not isinstance(v, (dict, list, tuple)) and v is not None:
                out[key] = f"[REDACTED:{key}]"
            else:
                out[key] = mask_secrets(v, reg)
        return out
    if isinstance(value, (list, tuple)):
        return [mask_secrets(v, reg) for v in value]
    if isinstance(value, str):
        return mask_text(value, reg)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # A number can itself be sensitive (e.g. an account number stored as int).
        as_text = str(value)
        masked = mask_text(as_text, reg)
        return value if masked == as_text else masked
    # Unknown types (Decimal, Path, datetime, ...) are stringified then masked.
    return mask_text(str(value), reg)


def safe_mask(value: Any, registry: SecretRegistry | None = None) -> Any:
    """Fail closed: if masking itself breaks, emit a placeholder rather than the raw value."""
    try:
        return mask_secrets(value, registry)
    except Exception:
        return MASKING_ERROR
