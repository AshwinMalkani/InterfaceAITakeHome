"""Guards for the artifact contract: everything recorded keeps loading as the schema evolves."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cua.artifact.store import SCHEMA_FILE, CapabilityLibrary, json_schema, load_capability

pytestmark = pytest.mark.regression

REPO = Path(__file__).resolve().parents[2]
LIBRARY = CapabilityLibrary(REPO / "capabilities")
FROZEN = Path(__file__).parent / "fixtures" / "capabilities"


def test_committed_capabilities_are_valid_and_canonical() -> None:
    """Every artifact validates, and its file is exactly what saving it would write (no hand drift)."""
    capabilities = LIBRARY.all()
    assert capabilities, "no capabilities found"
    for cap in capabilities:
        on_disk = LIBRARY.path_for(cap.id).read_text()
        assert on_disk == json.dumps(cap.canonical_dict(), indent=2) + "\n", cap.id


@pytest.mark.parametrize("path", sorted(FROZEN.rglob("*.json")), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_frozen_artifacts_from_every_schema_version_still_load(path: Path) -> None:
    """Artifacts saved under older schema versions are frozen here and must keep loading."""
    load_capability(path)


def test_frozen_fixtures_exist() -> None:
    assert list(FROZEN.rglob("*.json")), "freeze at least one artifact per schema version"


def test_published_json_schema_is_current() -> None:
    """schema/capability.schema.json is what agents and reviewers read; regenerate with `make schema`."""
    assert json.loads(SCHEMA_FILE.read_text()) == json_schema()
