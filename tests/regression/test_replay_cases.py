"""Replay regression cases from cases.yaml against the local target app. No LLM, no API key."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.cu_core.data import MemberStore
from cua.artifact.store import CapabilityLibrary
from cua.profile import load_profile
from cua.replay.result import Failure, Success
from cua.runner import replay
from cua.security.masking import safe_mask
from tests.regression.conftest import TEST_PASSWORD, TargetApp
from tests.regression.harness import Case, assert_matches_golden, find_leaks, load_cases

pytestmark = pytest.mark.regression

LIBRARY = CapabilityLibrary(Path(__file__).resolve().parents[2] / "capabilities")
SENSITIVE = [*MemberStore.seeded().sensitive_values(), TEST_PASSWORD]


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case.name)
def test_case(case: Case, target_apps: dict[str, TargetApp], tmp_path: Path) -> None:
    app = target_apps[case.tenant]
    app.reset()
    app.arm(case.faults)

    outcome = replay(
        case.capability,
        case.params,
        base_url=app.base_url,
        library=LIBRARY,
        profile=load_profile(case.capability.split(".", 1)[0]),
        evidence_root=tmp_path,
    )
    result = outcome.result

    # 1. The contract: result type, and where/why it failed.
    assert result.type == case.expect.result_type, result
    if isinstance(result, Failure):
        assert result.category == case.expect.category
        assert result.step_id == case.expect.failed_step
    if case.expect.outputs is not None:
        assert isinstance(result, Success)
        assert {k: str(v) for k, v in result.outputs.items()} == {
            k: str(v) for k, v in case.expect.outputs.items()
        }

    # 2. The full masked result is unchanged unless deliberately updated.
    assert_matches_golden(case.name, safe_mask(result.model_dump(mode="json"), outcome.registry))

    # 3. Nothing sensitive reached disk: no seeded PII and no credentials in any log or evidence file.
    assert find_leaks(outcome.evidence_dir, SENSITIVE) == []

    # 4. Every log line is correlated to the run (so a log backend can pull the whole run by run_id).
    events = [json.loads(line) for line in (outcome.evidence_dir / "events.jsonl").read_text().splitlines()]
    assert events and all(e.get("run_id") == outcome.run_id for e in events)
