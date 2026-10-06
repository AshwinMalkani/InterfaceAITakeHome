"""Tenant bindings: reuse one vendor-level capability across institutions running the same product.

Many tenants run the same vendor product configured differently. Overwhelmingly that means
*relabelled* screens with the same structure ("Member Number" becomes "Account #"). So a capability is
recorded once per product, and a tenant gets a small, reviewable overlay instead of a re-recording:

  labels  product-wide text substitutions, applied to every capability of the product: locator
          texts (role names, anchors, row/column headers, field labels) and checkpoint texts
  steps   rare structural patches for one capability, keyed by stable step id, merged into that
          step's action (e.g. a different target, or a different option label)

Specialization happens at load time, and the result is validated like any artifact. Its content hash
differs from the base capability's, so approval of an irreversible capability binds to what actually
runs on that tenant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cua.artifact.schema import IDENTIFIER, Capability

TENANTS_DIR = Path(__file__).resolve().parents[2] / "config" / "tenants"

# Keys whose values are never UI text, even inside targets and checkpoints.
_STRUCTURAL_KEYS = frozenset({"kind", "frame", "control", "selector", "pattern", "role"})


class TenantBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant: str = Field(pattern=IDENTIFIER)
    product: str = Field(pattern=IDENTIFIER)
    description: str = ""
    labels: dict[str, str] = {}
    steps: dict[str, dict[str, dict[str, Any]]] = {}  # capability id -> step id -> action fields to merge


def load_binding(tenant: str, product: str, directory: Path = TENANTS_DIR) -> TenantBinding:
    path = directory / f"{tenant}.yaml"
    binding = TenantBinding.model_validate(yaml.safe_load(path.read_text()))
    if (binding.tenant, binding.product) != (tenant, product):
        raise ValueError(f"{path} is for {binding.tenant}/{binding.product}, not {tenant}/{product}")
    return binding


def _relabel(node: Any, labels: dict[str, str]) -> Any:
    """Replace exact UI-text values (never structural keys) anywhere inside a target or checkpoint."""
    if isinstance(node, dict):
        return {k: v if k in _STRUCTURAL_KEYS else _relabel(v, labels) for k, v in node.items()}
    if isinstance(node, list):
        return [_relabel(v, labels) for v in node]
    if isinstance(node, str):
        return labels.get(node, node)
    return node


def specialize(capability: Capability, binding: TenantBinding) -> Capability:
    """The capability as it must run on `binding.tenant`. Raises if the result isn't a valid artifact."""
    if capability.app.product != binding.product:
        raise ValueError(f"binding is for {binding.product!r}, capability targets {capability.app.product!r}")
    data = capability.model_dump(mode="json")
    patches = binding.steps.get(capability.id, {})
    unknown = set(patches) - {s["id"] for s in data["steps"]}
    if unknown:
        raise ValueError(f"binding patches unknown steps of {capability.id}: {sorted(unknown)}")

    labels = binding.labels
    for step in data["steps"]:
        action = step["action"]
        if "target" in action:
            action["target"] = _relabel(action["target"], labels)
        if step.get("expect"):
            step["expect"] = _relabel(step["expect"], labels)
        action.update(patches.get(step["id"], {}))
    data["success"] = _relabel(data["success"], labels)
    for outcome in data.get("outcomes", []):
        outcome["when"] = _relabel(outcome["when"], labels)
    # Inputs, outputs, ids and templates are the contract and are never touched by a binding.
    return Capability.model_validate(data)
