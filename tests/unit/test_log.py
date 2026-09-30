import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from cua.evidence.log import configure_logging, get_logger, log_context
from cua.evidence.sink import RunEvidence
from cua.security.masking import SecretRegistry, Sensitivity, use_registry


@pytest.fixture
def run(tmp_path: Path) -> Iterator[RunEvidence]:
    evidence = RunEvidence(tmp_path, "run_test")
    configure_logging(console=False, extra_handlers=(evidence.log_handler(),))
    yield evidence
    configure_logging(console=False)  # closes the file handler


def read_events(run: RunEvidence) -> list[dict[str, object]]:
    for handler in logging.getLogger("cua").handlers:
        handler.flush()
    return [json.loads(line) for line in run.path("events.jsonl").read_text().splitlines()]


def test_events_are_json_with_fields(run: RunEvidence) -> None:
    get_logger("replay").info("step.succeeded", step_id="s1", duration_ms=42)
    [event] = read_events(run)
    assert event["event"] == "step.succeeded"
    assert event["duration_ms"] == 42
    assert event["logger"] == "cua.replay"
    assert event["level"] == "INFO"


def test_context_is_attached_and_nests(run: RunEvidence) -> None:
    log = get_logger("replay")
    with log_context(run_id="r1", actor="replay"):
        with log_context(step_id="s2"):
            log.info("step.started")
        log.info("run.finished")
    inner, outer = read_events(run)
    assert inner["run_id"] == "r1" and inner["step_id"] == "s2" and inner["actor"] == "replay"
    assert "step_id" not in outer


def test_log_records_are_masked(run: RunEvidence) -> None:
    reg = SecretRegistry(include_env=False)
    reg.register("member_id", "48213", Sensitivity.PII)
    with use_registry(reg), log_context(member="48213"):
        get_logger("agent").info("agent.action", rationale="Typing member 48213", password="pw")
    raw = run.path("events.jsonl").read_text()
    assert "48213" not in raw
    [event] = read_events(run)
    assert event["password"] == "[REDACTED:password]"


def test_exceptions_are_captured_and_masked(run: RunEvidence) -> None:
    try:
        raise ValueError("lookup failed for ssn 123-45-6789")
    except ValueError:
        get_logger("replay").error("step.failed", exc_info=True)
    [event] = read_events(run)
    assert "ValueError" in str(event["exc"])
    assert "123-45-6789" not in str(event["exc"])
