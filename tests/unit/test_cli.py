"""CLI wiring: each command's arguments reach the right code path (no browser, no API)."""

from pathlib import Path
from typing import Any

import pytest

from cua import __main__ as cli


def test_discover_dispatches_without_a_capability_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "_discover", lambda args, library: seen.update(goal=args.goal) or 0)
    assert cli.main(["discover", "goals/cu_core/read_savings_balance.yaml", "--base-url", "http://x"]) == 0
    assert seen["goal"] == Path("goals/cu_core/read_savings_balance.yaml")


def test_approve_writes_to_the_given_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cua.artifact.store import CapabilityLibrary

    repo_library = CapabilityLibrary(Path(__file__).resolve().parents[2] / "capabilities")
    library = CapabilityLibrary(tmp_path)
    library.save(repo_library.get("cu_core.member.open_sub_account"))
    assert (
        cli.main(["--library", str(tmp_path), "approve", "cu_core.member.open_sub_account", "--by", "me"])
        == 0
    )
    assert "cu_core.member.open_sub_account" in (tmp_path / "approvals.json").read_text()


def test_params_must_be_name_value() -> None:
    with pytest.raises(SystemExit, match="name=value"):
        cli._params(["member_id"])
