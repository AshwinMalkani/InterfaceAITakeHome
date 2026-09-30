import json
from pathlib import Path

import pytest

from cua.evidence.sink import RunEvidence, echo
from cua.security.masking import SecretRegistry, Sensitivity


@pytest.fixture
def reg() -> SecretRegistry:
    r = SecretRegistry(include_env=False)
    r.register("member_id", "48213", Sensitivity.PII)
    return r


def test_json_and_text_writes_are_masked(tmp_path: Path, reg: SecretRegistry) -> None:
    run = RunEvidence(tmp_path, "r1", reg)
    run.write_json("result.json", {"outputs": {"member": "48213"}, "token": "abc"})
    run.write_text("notes.txt", "member 48213, ssn 123-45-6789")
    data = json.loads(run.path("result.json").read_text())
    assert data["token"] == "[REDACTED:token]"
    assert "48213" not in json.dumps(data)
    assert run.path("notes.txt").read_text().endswith("[SSN]")


def test_paths_cannot_escape_run_dir(tmp_path: Path) -> None:
    run = RunEvidence(tmp_path, "r1")
    with pytest.raises(ValueError, match="escapes"):
        run.write_text("../../outside.txt", "x")


def test_binary_writes_limited_to_allowlisted_suffixes(tmp_path: Path) -> None:
    run = RunEvidence(tmp_path, "r1")
    run.write_bytes("shot.png", b"\x89PNG")
    with pytest.raises(ValueError, match="binary writes"):
        run.write_bytes("dump.txt", b"raw text would bypass masking")


def test_echo_masks_stdout(capsys: pytest.CaptureFixture[str], reg: SecretRegistry) -> None:
    echo({"member": "48213"}, reg)
    assert "48213" not in capsys.readouterr().out
