from pathlib import Path

import pytest

from tests.regression.harness import NORMALIZED, assert_matches_golden, find_leaks, load_cases, normalize


def test_normalize_replaces_volatile_fields_recursively() -> None:
    data = {"run_id": "r1", "steps": [{"id": "s1", "duration_ms": 12}], "outputs": {"x": 1}}
    assert normalize(data) == {
        "run_id": NORMALIZED,
        "steps": [{"id": "s1", "duration_ms": NORMALIZED}],
        "outputs": {"x": 1},
    }


def test_golden_roundtrip_and_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGRESS_UPDATE", "1")
    assert_matches_golden("case", {"type": "success", "run_id": "a"}, tmp_path)
    monkeypatch.delenv("REGRESS_UPDATE")
    assert_matches_golden("case", {"type": "success", "run_id": "b"}, tmp_path)  # volatile field differs
    with pytest.raises(AssertionError, match="differs"):
        assert_matches_golden("case", {"type": "failure", "run_id": "a"}, tmp_path)


def test_missing_golden_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="no golden file"):
        assert_matches_golden("absent", {}, tmp_path)


def test_manifest_rejects_duplicates_and_unknown_fields(tmp_path: Path) -> None:
    case = (
        "- {name: a, description: d, capability: c, "
        "expect: {result_type: success}}\n"
    )
    dup = tmp_path / "dup.yaml"
    dup.write_text("cases:\n" + case + case)
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(dup)

    extra = tmp_path / "extra.yaml"
    extra.write_text("cases:\n" + case.replace("capability: c,", "capability: c, typo: 1,"))
    with pytest.raises(ValueError):
        load_cases(extra)


def test_find_leaks_reports_file_and_value(tmp_path: Path) -> None:
    (tmp_path / "events.jsonl").write_text('{"member": "48213"}')
    (tmp_path / "shot.png").write_bytes(b"48213")  # binary files are out of scope for text scanning
    assert find_leaks(tmp_path, ["48213"]) == [(tmp_path / "events.jsonl", "48213")]
