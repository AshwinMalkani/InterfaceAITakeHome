import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from cua.evidence.log import configure_logging, get_logger, log_context, run_context, span
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


class TestCorrelation:
    def test_run_context_sets_ids_and_request_defaults_to_run(self, run: RunEvidence) -> None:
        with run_context("run_1"):
            get_logger("replay").info("run.started")
        [event] = read_events(run)
        assert event["run_id"] == event["request_id"] == "run_1"
        assert len(str(event["trace_id"])) == 32 and len(str(event["span_id"])) == 16
        assert "parent_span_id" not in event

    def test_valid_traceparent_joins_callers_trace(self, run: RunEvidence) -> None:
        trace, parent = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
        with run_context("run_1", request_id="req_9", traceparent=f"00-{trace}-{parent}-01"):
            get_logger("replay").info("run.started")
        [event] = read_events(run)
        assert event["trace_id"] == trace and event["parent_span_id"] == parent
        assert event["request_id"] == "req_9"

    def test_malformed_traceparent_starts_a_new_trace(self, run: RunEvidence) -> None:
        with run_context("run_1", traceparent="garbage"):
            get_logger("replay").info("run.started")
        [event] = read_events(run)
        assert len(str(event["trace_id"])) == 32 and "parent_span_id" not in event

    def test_span_nests_under_current_span(self, run: RunEvidence) -> None:
        log = get_logger("replay")
        with run_context("run_1"):
            log.info("run.started")
            with span(step_id="s1") as step_span:
                log.info("step.started")
        root, step = read_events(run)
        assert step["span_id"] == step_span
        assert step["parent_span_id"] == root["span_id"]
        assert step["trace_id"] == root["trace_id"]

    def test_all_digit_trace_ids_are_not_masked_as_account_numbers(self, run: RunEvidence) -> None:
        with log_context(trace_id="1" * 32, span_id="1234567890123456"):
            get_logger("replay").info("run.started")
        [event] = read_events(run)
        assert event["span_id"] == "1234567890123456"

    def test_caller_supplied_request_id_is_still_masked(self, run: RunEvidence) -> None:
        with run_context("run_1", request_id="ssn-123-45-6789"):
            get_logger("replay").info("run.started")
        [event] = read_events(run)
        assert "123-45-6789" not in str(event["request_id"])


def test_resource_fields_on_every_record(run: RunEvidence) -> None:
    get_logger("replay").info("run.started")
    [event] = read_events(run)
    assert event["service"] == "cua" and event["env"] and event["version"]


def test_json_console_format_writes_json_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(console_format="json")
    try:
        get_logger("replay").info("run.started", step_id="s1")
    finally:
        configure_logging(console=False)
    line = capsys.readouterr().out.strip()
    assert json.loads(line)["event"] == "run.started"


def test_invalid_log_format_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "xml")
    with pytest.raises(ValueError, match="LOG_FORMAT"):
        configure_logging()
