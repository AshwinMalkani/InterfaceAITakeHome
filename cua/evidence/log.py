"""Structured, correlated, masked logging.

Usage:
    log = get_logger(__name__)
    with log_context(run_id="run_1", step_id="s2", actor="replay"):
        log.info("step.started", intent="Enter member number")

- Messages are stable event names (`step.started`), never free text; details go in fields.
- Correlation fields (run_id, step_id, actor, ...) come from contextvars, not arguments.
- Every record is masked with `safe_mask` inside the formatter, so no handler can emit raw data.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from cua.security.masking import safe_mask

ROOT_LOGGER = "cua"

_context: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Add correlation fields to every log record emitted inside this block (nests)."""
    token = _context.set({**(_context.get() or {}), **fields})
    try:
        yield
    finally:
        _context.reset(token)


class _ContextFilter(logging.Filter):
    """Snapshot the context at record creation so async/queued handlers still see it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.context = dict(_context.get() or {})
        return True


def _payload(record: logging.LogRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
        "level": record.levelname,
        "event": record.getMessage(),
        "logger": record.name,
        **getattr(record, "context", {}),
        **getattr(record, "fields", {}),
    }
    if record.exc_info:
        payload["exc"] = logging.Formatter().formatException(record.exc_info)
    masked: dict[str, Any] = safe_mask(payload)
    return masked if isinstance(masked, dict) else {"event": "log.masking_error", "level": "ERROR"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(_payload(record), sort_keys=False, default=str)


class ConsoleFormatter(logging.Formatter):
    _COLORS = {"DEBUG": "\033[2m", "INFO": "\033[36m", "WARNING": "\033[33m", "ERROR": "\033[31m"}
    _SHOWN_CONTEXT = ("step_id", "actor")

    def format(self, record: logging.LogRecord) -> str:
        p = _payload(record)
        color = self._COLORS.get(p["level"], "")
        reset = "\033[0m" if color else ""
        skip = {"ts", "level", "event", "logger", "run_id", "mode", "capability_id", "tenant"}
        details = " ".join(f"{k}={v}" for k, v in p.items() if k not in skip)
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
    extra_handlers: tuple[logging.Handler, ...] = (),
) -> logging.Logger:
    """(Re)configure the `cua` logger tree. Extra handlers (e.g. RunEvidence.log_handler()) get JSON."""
    root = logging.getLogger(ROOT_LOGGER)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)
    root.propagate = False

    handlers: list[logging.Handler] = []
    if console:
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
