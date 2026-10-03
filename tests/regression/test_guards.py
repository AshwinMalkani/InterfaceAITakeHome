"""Structural guards that protect the safety properties as the codebase grows."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cua.evidence.log import configure_logging, get_logger, log_context
from cua.evidence.sink import RunEvidence, echo
from cua.security.masking import SecretRegistry, Sensitivity, use_registry
from tests.regression.harness import find_leaks, load_cases

pytestmark = pytest.mark.regression

PACKAGE = Path(__file__).resolve().parents[2] / "cua"
# The sink masks files and stdout; the HITL store masks every field it writes to sqlite.
EGRESS_ALLOWED = {PACKAGE / "evidence" / "sink.py", PACKAGE / "hitl" / "store.py"}

_WRITE_ATTRS = {"write_text", "write_bytes", "FileHandler", "dump"}
_STD_STREAMS = {"stdout", "stderr"}


def egress_violations(source: str) -> list[str]:
    """Return a description of every direct output call in `source`."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "print":
            found.append(f"print() at line {node.lineno}")
        elif isinstance(func, ast.Name) and func.id == "open":
            mode_node = node.args[1] if len(node.args) > 1 else next(
                (k.value for k in node.keywords if k.arg == "mode"), None
            )
            mode = mode_node.value if isinstance(mode_node, ast.Constant) else "r"
            if not isinstance(mode, str) or set(mode) & set("wax+"):
                found.append(f"open(mode={mode!r}) at line {node.lineno}")
        elif isinstance(func, ast.Attribute):
            if func.attr in _WRITE_ATTRS:
                found.append(f".{func.attr}() at line {node.lineno}")
            elif func.attr == "connect" and isinstance(func.value, ast.Name) and func.value.id == "sqlite3":
                found.append(f"sqlite3.connect() at line {node.lineno}")
            elif func.attr == "screenshot" and any(k.arg == "path" for k in node.keywords):
                found.append(f".screenshot(path=...) at line {node.lineno}")
            elif (
                func.attr == "write"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr in _STD_STREAMS
            ):
                found.append(f"sys.{func.value.attr}.write() at line {node.lineno}")
    return found


@pytest.mark.parametrize(
    "snippet",
    [
        "print('x')",
        "open('f', 'w')",
        "open('f', mode='a')",
        "p.write_text('x')",
        "json.dump(d, f)",
        "logging.FileHandler('x')",
        "page.screenshot(path='x.png')",
        "sys.stdout.write('x')",
        "sqlite3.connect('x.db')",
    ],
)
def test_egress_detector_catches(snippet: str) -> None:
    assert egress_violations(snippet), snippet


def test_egress_detector_allows_reads() -> None:
    assert egress_violations("open('f').read(); json.dumps(d); page.screenshot()") == []


def test_only_sink_writes_output() -> None:
    """All file/stdout/database output must go through a module that masks it (sink.py, hitl/store.py)."""
    violations = {
        str(path.relative_to(PACKAGE.parent)): found
        for path in sorted(PACKAGE.rglob("*.py"))
        if path not in EGRESS_ALLOWED and (found := egress_violations(path.read_text()))
    }
    assert violations == {}, f"unmasked output paths outside sink.py: {violations}"


def test_no_sensitive_value_leaks_through_logs_or_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Push seeded sensitive values through every egress path, then scan everything written."""
    seeded = {"member_id": "48213", "member_name": "Jane Q Testperson", "balance": "$12,408.77"}
    secret = "correct-horse-battery"
    reg = SecretRegistry(include_env=False)
    for name, value in seeded.items():
        reg.register(name, value, Sensitivity.PII)
    reg.register("login_password", secret, Sensitivity.SECRET)

    run = RunEvidence(tmp_path, "run_leak", reg)
    configure_logging(console=True, extra_handlers=(run.log_handler(),))
    try:
        with use_registry(reg), log_context(run_id="run_leak"):
            log = get_logger("replay")
            log.info("step.started", value=seeded["member_id"], note=f"login {secret}")
            log.info("outputs.extracted", outputs=seeded)
            run.save_json("result.json", {"outputs": seeded})
            run.save_text("summary.txt", " ".join(seeded.values()))
            echo({"outputs": seeded}, reg)
    finally:
        configure_logging(console=False)

    all_values = [*seeded.values(), secret]
    assert find_leaks(tmp_path, all_values) == []
    captured = capsys.readouterr()
    assert not any(v in captured.out + captured.err for v in all_values)


def test_case_manifest_is_valid() -> None:
    load_cases()
