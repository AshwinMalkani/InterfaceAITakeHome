import json
from pathlib import Path

import pytest

from cua.artifact.schema import Capability
from cua.artifact.store import ArtifactLeakError, CapabilityLibrary, load_capability, save_capability
from tests.unit.test_schema import build, minimal


def test_round_trip(tmp_path: Path) -> None:
    cap = build()
    path = save_capability(cap, tmp_path / "cap.json")
    assert load_capability(path) == cap


def test_saved_json_keeps_model_field_order_and_omits_nulls(tmp_path: Path) -> None:
    path = save_capability(build(), tmp_path / "cap.json")
    keys = list(json.loads(path.read_text()))
    assert keys[:3] == ["schema_version", "id", "version"]
    assert "null" not in path.read_text()


@pytest.mark.parametrize(
    "leak",
    ["Example member 123-45-6789", "Contact jane@example.com", "Account 000123456789"],
)
def test_refuses_to_save_artifacts_containing_sensitive_data(tmp_path: Path, leak: str) -> None:
    with pytest.raises(ArtifactLeakError) as exc:
        save_capability(build(description=leak), tmp_path / "cap.json")
    assert "$.description" in str(exc.value)
    assert leak not in str(exc.value)
    assert not (tmp_path / "cap.json").exists()


def test_refuses_literal_secret_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CU_CORE_PASSWORD", "hunter2hunter2")
    data = minimal()
    data["steps"][0]["action"]["value"] = "hunter2hunter2 {{inputs.member_id}}"
    with pytest.raises(ArtifactLeakError, match=r"steps\[0\].action.value"):
        save_capability(Capability.model_validate(data), tmp_path / "cap.json")


class TestLibrary:
    def test_save_and_get_by_id(self, tmp_path: Path) -> None:
        lib = CapabilityLibrary(tmp_path)
        path = lib.save(build())
        assert path == tmp_path / "app" / "app.domain.action.json"
        assert lib.get("app.domain.action") == build()
        assert [c.id for c in lib.all()] == ["app.domain.action"]

    def test_missing_capability(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError):
            CapabilityLibrary(tmp_path).get("app.domain.missing")

    def test_file_must_declare_matching_id(self, tmp_path: Path) -> None:
        lib = CapabilityLibrary(tmp_path)
        save_capability(build(), lib.path_for("app.domain.other"))
        with pytest.raises(ValueError, match="declares id"):
            lib.get("app.domain.other")
