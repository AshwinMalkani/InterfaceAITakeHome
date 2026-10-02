"""Regression harness: case manifest, golden-file comparison, leak scanning.

Cases live in `cases.yaml`; adding a regression is one YAML entry, not new test code.
Golden results are masked Result JSON with volatile fields normalized; regenerate them
deliberately with `make regress-update` and review the diff like code.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

HERE = Path(__file__).parent
CASES_FILE = HERE / "cases.yaml"
GOLDEN_DIR = HERE / "golden"

# Fields whose values legitimately change between runs; compared by presence only.
VOLATILE_KEYS = frozenset({"run_id", "ts", "started_at", "finished_at", "duration_ms", "evidence_dir"})
NORMALIZED = "<normalized>"
# Masking tokens carry a per-run HMAC suffix ([PII:name#3f9a]); keep the field name, drop the suffix.
_TOKEN_SUFFIX = re.compile(r"(\[PII:[a-z0-9_]+)#[0-9a-f]{4}\]")

TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".txt", ".md", ".html", ".yaml", ".log"})


class Expectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_type: Literal["success", "business_outcome", "failure", "escalated"]
    outcome: str | None = None  # business outcome name, when result_type == business_outcome
    category: str | None = None  # failure category, when result_type == failure
    failed_step: str | None = None
    outputs: dict[str, Any] | None = None
    recoveries: list[str] | None = None  # recovery kinds, in order
    needs_human: bool | None = None
    retryable: bool | None = None


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    capability: str
    tenant: str = "alpha"
    params: dict[str, Any] = {}
    faults: list[dict[str, Any]] = []  # apps.cu_core.faults.Fault, armed before the run
    expect: Expectation


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[Case]


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    manifest = Manifest.model_validate(yaml.safe_load(path.read_text()) or {"cases": []})
    names = [c.name for c in manifest.cases]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate case names: {sorted(duplicates)}")
    return manifest.cases


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: NORMALIZED if k in VOLATILE_KEYS else normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str):
        return _TOKEN_SUFFIX.sub(r"\1]", value)
    return value


def assert_matches_golden(name: str, actual: dict[str, Any], golden_dir: Path = GOLDEN_DIR) -> None:
    """Compare against golden/<name>.json; write it instead when REGRESS_UPDATE=1."""
    path = golden_dir / f"{name}.json"
    rendered = json.dumps(normalize(actual), indent=2, sort_keys=True) + "\n"
    if os.environ.get("REGRESS_UPDATE") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        return
    if not path.exists():
        raise AssertionError(f"no golden file for {name!r}; run `make regress-update` and review it")
    expected = path.read_text()
    assert rendered == expected, (
        f"result for {name!r} differs from {path.name} (update deliberately if intended)"
    )


def find_leaks(root: Path, sensitive_values: list[str]) -> list[tuple[Path, str]]:
    """Every (file, value) pair where a raw sensitive value appears in a text file under root."""
    leaks: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        content = path.read_text(errors="replace").lower()
        leaks.extend((path, v) for v in sensitive_values if v.lower() in content)
    return leaks
