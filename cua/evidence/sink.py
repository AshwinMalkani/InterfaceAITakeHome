"""The only module allowed to write files or print to stdout.

Everything that leaves the process goes through here and is passed through `safe_mask`
first. A regression guard (tests/regression/test_guards.py) fails the build if any other
module under `cua/` writes files, so a new feature cannot add an unmasked output path.

Binary writes (screenshots, traces) cannot be text-masked; callers must apply visual
masking before handing bytes over, and only an allowlisted set of suffixes is accepted.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from cua.security.masking import SecretRegistry, safe_mask

BINARY_SUFFIXES = frozenset({".png", ".jpg", ".zip"})


def write_json(
    path: Path, data: Any, registry: SecretRegistry | None = None, *, sort_keys: bool = True
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_mask(data, registry), indent=2, sort_keys=sort_keys) + "\n")
    return path


def write_text(path: Path, text: str, registry: SecretRegistry | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(safe_mask(text, registry))
    return path


def write_bytes(path: Path, data: bytes) -> Path:
    if path.suffix not in BINARY_SUFFIXES:
        raise ValueError(f"binary writes limited to {sorted(BINARY_SUFFIXES)}, got {path.suffix!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def echo(value: Any, registry: SecretRegistry | None = None, *, reveal: bool = False) -> None:
    """Masked stdout output for the CLI. `reveal=True` is an explicit operator opt-in to raw values."""
    shown = value if reveal else safe_mask(value, registry)
    text = shown if isinstance(shown, str) else json.dumps(shown, indent=2, sort_keys=True, default=str)
    sys.stdout.write(text + "\n")


class RunEvidence:
    """Evidence directory for a single run: `<root>/<run_id>/`."""

    def __init__(self, root: Path, run_id: str, registry: SecretRegistry | None = None) -> None:
        self.run_id = run_id
        self.dir = (root / run_id).resolve()
        self.registry = registry
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        target = (self.dir / relative).resolve()
        if not target.is_relative_to(self.dir):
            raise ValueError(f"evidence path escapes run directory: {relative!r}")
        return target

    # Named save_* (not write_*) so the egress guard can tell masked sink calls apart from raw
    # pathlib writes like Path.write_text, which are forbidden outside this module.

    def save_json(self, relative: str, data: Any) -> Path:
        return write_json(self.path(relative), data, self.registry)

    def save_text(self, relative: str, text: str) -> Path:
        return write_text(self.path(relative), text, self.registry)

    def save_image(self, relative: str, data: bytes) -> Path:
        return write_bytes(self.path(relative), data)

    def log_handler(self) -> logging.Handler:
        """JSONL event log for this run. Masking happens in the formatter (cua.evidence.log)."""
        return logging.FileHandler(self.path("events.jsonl"), encoding="utf-8")
