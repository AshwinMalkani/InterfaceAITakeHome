"""Structured, correlated, masked logging.

Usage:
    log = get_logger(__name__)
    with log_context(run_id="run_1", step_id="s2", actor="replay"):
        log.info("step.started", intent="Enter member number")

- Messages are stable event names (`step.started`), never free text; details go in fields.
- Correlation fields (run_id, step_id, actor, ...) come from contextvars, not arguments.
- Every record is masked with `safe_mask` inside the formatter, so no handler can emit raw data.

Correlation model (built for log backends like Grafana Loki):
- `request_id` groups everything done for one caller request, across retries and HITL resumes.
- `run_id` identifies one execution attempt within that request.
- `trace_id` / `span_id` / `parent_span_id` follow W3C Trace Context, so an incoming `traceparent`
  joins the caller's distributed trace and logs can link to a tracing backend later.
- `service` / `env` / `version` are stamped on every record.
Only low-cardinality fields (service, env, level, mode) should ever become index labels;
ids stay in the JSON body and are queried with a line filter, e.g. `{service="cua"} |= "run_abc"`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

from cua.security.masking import safe_mask

ROOT_LOGGER = "cua"
SERVICE_NAME = "cua"

# W3C traceparent: version-traceid-parentid-flags
_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")
# Generated or format-validated hex, so exempt from pattern masking (a 16-digit all-numeric
# span id would otherwise be misread as an account number). request_id is caller-supplied
# and could contain anything, so it is deliberately NOT exempt.
_TRACE_KEYS = ("trace_id", "span_id", "parent_span_id")

_context: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Add correlation fields to every log record emitted inside this block (nests)."""
    token = _context.set({**(_context.get() or {}), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def _current() -> dict[str, Any]:
    return _context.get() or {}


@contextmanager
def run_context(
    run_id: str, *, request_id: str | None = None, traceparent: str | None = None, **fields: Any
) -> Iterator[None]:
    """Root context for one execution. Joins the caller's trace if a valid `traceparent` is given."""
    parsed = _TRACEPARENT.match(traceparent or "")
    trace_fields: dict[str, Any] = {
        "trace_id": parsed.group(1) if parsed else secrets.token_hex(16),
        "span_id": secrets.token_hex(8),
    }
    if parsed:
        trace_fields["parent_span_id"] = parsed.group(2)
    with log_context(run_id=run_id, request_id=request_id or run_id, **trace_fields, **fields):
        yield


@contextmanager
def span(**fields: Any) -> Iterator[str]:
    """Child span (e.g. one step): a new span_id whose parent is the current span."""
    span_id = secrets.token_hex(8)
    with log_context(parent_span_id=_current().get("span_id"), span_id=span_id, **fields):
        yield span_id


@cache
def _resource() -> dict[str, str]:
    try:
        package_version = version("cua")
    except PackageNotFoundError:
        package_version = "unknown"
    return {
        "service": SERVICE_NAME,
        "env": os.environ.get("CUA_ENV", "dev"),
        # Deploys set CUA_VERSION to the git SHA so behaviour changes can be tied to a commit.
        "version": os.environ.get("CUA_VERSION") or package_version,
    }


class _ContextFilter(logging.Filter):
    """Snapshot the context at record creation so async/queued handlers still see it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.context = dict(_current())
        return True


def _payload(record: logging.LogRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
        "level": record.levelname,
        "event": record.getMessage(),
        "logger": record.name,
        **_resource(),
        **getattr(record, "context", {}),
        **getattr(record, "fields", {}),
    }
    if record.exc_info:
        payload["exc"] = logging.Formatter().formatException(record.exc_info)
    trace = {k: payload.pop(k) for k in _TRACE_KEYS if payload.get(k) is not None}
    payload = {k: v for k, v in payload.items() if k not in _TRACE_KEYS}
    masked = safe_mask(payload)
    if not isinstance(masked, dict):
        return {"event": "log.masking_error", "level": "ERROR", **trace}
    return {**masked, **trace}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(_payload(record), sort_keys=False, default=str)


class ConsoleFormatter(logging.Formatter):
    _COLORS = {"DEBUG": "\033[2m", "INFO": "\033[36m", "WARNING": "\033[33m", "ERROR": "\033[31m"}
    # Shown on every line in JSON; hidden on the console to keep it readable.
    _HIDDEN = frozenset(
        {"ts", "level", "event", "logger", "run_id", "request_id", "mode", "capability_id", "tenant",
         "service", "env", "version", *_TRACE_KEYS}
    )

    def format(self, record: logging.LogRecord) -> str:
        p = _payload(record)
        color = self._COLORS.get(p["level"], "")
        reset = "\033[0m" if color else ""
        details = " ".join(f"{k}={v}" for k, v in p.items() if k not in self._HIDDEN)
        return f"{p['ts'][11:23]} {color}{p['level']:<7}{reset} {p['event']:<24} {details}".rstrip()


class EventLogger:
    """Thin wrapper so call sites read `log.info("event.name", key=value)`."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, fields: dict[str, Any], exc_info: bool = False) -> None:
        # stacklevel=3 attributes the record to the caller, not this wrapper.
        self._logger.log(level, event, extra={"fields": fields}, exc_info=exc_info, stacklevel=3)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, fields)

    def error(self, event: str, exc_info: bool = False, **fields: Any) -> None:
        self._log(logging.ERROR, event, fields, exc_info=exc_info)


def get_logger(name: str) -> EventLogger:
    if not name.startswith(ROOT_LOGGER):
        name = f"{ROOT_LOGGER}.{name}"
    return EventLogger(logging.getLogger(name))


def configure_logging(
    *,
    level: int = logging.INFO,
    console: bool = True,
    console_format: Literal["pretty", "json"] | None = None,
    extra_handlers: tuple[logging.Handler, ...] = (),
) -> logging.Logger:
    """(Re)configure the `cua` logger tree. Extra handlers (e.g. RunEvidence.log_handler()) get JSON.

    Console format defaults to $LOG_FORMAT (default "pretty"). "json" writes JSON lines to stdout,
    which is what a container log collector (e.g. Grafana Alloy/Promtail -> Loki) expects.
    """
    fmt = console_format or os.environ.get("LOG_FORMAT", "pretty")
    if fmt not in ("pretty", "json"):
        raise ValueError(f"LOG_FORMAT must be 'pretty' or 'json', got {fmt!r}")
    root = logging.getLogger(ROOT_LOGGER)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)
    root.propagate = False

    handlers: list[logging.Handler] = []
    if console:
        if fmt == "json":
            stream = logging.StreamHandler(sys.stdout)
            stream.setFormatter(JsonFormatter())
        else:
            stream = logging.StreamHandler(sys.stderr)
            stream.setFormatter(ConsoleFormatter())
        handlers.append(stream)
    for handler in extra_handlers:
        handler.setFormatter(JsonFormatter())
        handlers.append(handler)
    for handler in handlers:
        handler.addFilter(_ContextFilter())
        root.addHandler(handler)
    return root
