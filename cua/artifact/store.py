"""Loading, saving and cataloguing capability artifacts.

Artifacts must be free of sensitive data *by construction* (inputs are templates, never values).
Saving runs the artifact through `mask_secrets` as a tripwire: if masking would change anything,
the save is refused rather than silently writing a redacted (and therefore different) artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cua.artifact.schema import Capability
from cua.evidence import sink
from cua.security.masking import SecretRegistry, mask_secrets

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "schema" / "capability.schema.json"


class ArtifactLeakError(ValueError):
    pass


def _differences(original: Any, masked: Any, path: str = "$") -> list[str]:
    """JSON paths where masking changed something (paths only; never the values)."""
    if isinstance(original, dict) and isinstance(masked, dict):
        return [d for k in original for d in _differences(original[k], masked.get(k), f"{path}.{k}")]
    if isinstance(original, list) and isinstance(masked, list):
        return [d for i, (o, m) in enumerate(zip(original, masked, strict=True))
                for d in _differences(o, m, f"{path}[{i}]")]
    return [] if original == masked else [path]


def save_capability(capability: Capability, path: Path) -> Path:
    data = capability.canonical_dict()
    masked = mask_secrets(data, SecretRegistry())
    if changed := _differences(data, masked):
        raise ArtifactLeakError(f"artifact appears to contain sensitive data at: {changed}")
    # Model field order (id, intent, action, ...) reads better than alphabetical and is deterministic.
    return sink.write_json(path, data, sort_keys=False)


def load_capability(path: Path) -> Capability:
    return Capability.model_validate_json(path.read_text())


class CapabilityLibrary:
    """Artifacts on disk, one file per capability: `<root>/<product>/<id>.json`."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, capability_id: str) -> Path:
        return self.root / capability_id.split(".", 1)[0] / f"{capability_id}.json"

    def get(self, capability_id: str) -> Capability:
        path = self.path_for(capability_id)
        if not path.exists():
            raise KeyError(f"no capability {capability_id!r} at {path}")
        capability = load_capability(path)
        if capability.id != capability_id:
            raise ValueError(f"{path} declares id {capability.id!r}, expected {capability_id!r}")
        return capability

    def all(self) -> list[Capability]:
        return [load_capability(p) for p in sorted(self.root.glob("*/*.json"))]

    def save(self, capability: Capability) -> Path:
        return save_capability(capability, self.path_for(capability.id))


def json_schema() -> dict[str, Any]:
    return Capability.model_json_schema()


def write_json_schema(path: Path = SCHEMA_FILE) -> Path:
    return sink.write_json(path, json_schema())
