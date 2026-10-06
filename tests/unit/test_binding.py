from pathlib import Path

import pytest

from cua.artifact.binding import TenantBinding, load_binding, specialize
from cua.artifact.schema import Click, Extract, Fill, Select
from cua.artifact.store import CapabilityLibrary
from cua.policy import ApprovalLedger

LIBRARY = CapabilityLibrary(Path(__file__).resolve().parents[2] / "capabilities")
BALANCE = LIBRARY.get("cu_core.member.get_share_balance")


def binding(**changes: object) -> TenantBinding:
    return TenantBinding.model_validate({"tenant": "beta", "product": "cu_core", **changes})


def test_labels_are_applied_to_targets_and_checkpoints() -> None:
    beta = specialize(BALANCE, load_binding("beta", "cu_core"))
    fill = next(s.action for s in beta.steps if isinstance(s.action, Fill))
    assert fill.target.strategies[0].model_dump()["anchor"] == "Account #"
    search = beta.steps[2].action
    assert isinstance(search, Click) and search.target.strategies[0].model_dump()["name"] == "Find"
    balance = beta.steps[4].action
    assert (
        isinstance(balance, Extract) and balance.target.strategies[0].model_dump()["row"] == "Primary Share"
    )


def test_contract_and_structure_are_never_touched() -> None:
    # A label colliding with an input name, an output name, a role, a frame and a css selector.
    hostile = binding(
        labels={
            "member_id": "x",
            "member_name": "x",
            "button": "x",
            "main": "x",
            "input[name='f1']": "x",
            "{{inputs.member_id}}": "x",
        }
    )
    beta = specialize(BALANCE, hostile)
    assert [p.name for p in beta.inputs] == ["member_id"]
    assert [o.name for o in beta.outputs] == ["member_name", "savings_balance"]
    assert beta.steps[1].action.value == "{{inputs.member_id}}"  # type: ignore[union-attr]
    assert beta.steps[2].action.target.frame == "main"  # type: ignore[union-attr]
    assert beta.steps[2].action.target.strategies[0].model_dump()["role"] == "button"  # type: ignore[union-attr]
    assert beta.steps[1].action.target.strategies[1].model_dump()["selector"] == "input[name='f1']"  # type: ignore[union-attr]


def test_step_patches_merge_into_one_capability_only() -> None:
    beta = specialize(LIBRARY.get("cu_core.member.open_sub_account"), load_binding("beta", "cu_core"))
    funding = next(s.action for s in beta.steps if s.id == "s8")
    assert isinstance(funding, Select) and funding.option == "00 - Primary Share"


def test_patching_an_unknown_step_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown steps"):
        specialize(BALANCE, binding(steps={BALANCE.id: {"s99": {"option": "x"}}}))


def test_a_binding_for_another_product_is_refused() -> None:
    with pytest.raises(ValueError, match="binding is for"):
        specialize(BALANCE, binding(product="other_core"))


def test_specialized_content_has_its_own_hash() -> None:
    assert specialize(BALANCE, binding()).content_hash() == BALANCE.content_hash()  # empty binding: identical
    assert specialize(BALANCE, load_binding("beta", "cu_core")).content_hash() != BALANCE.content_hash()


def test_every_committed_capability_specializes_validly_for_beta() -> None:
    beta = load_binding("beta", "cu_core")
    for capability in LIBRARY.all():
        specialize(capability, beta)


def test_approval_is_per_tenant(tmp_path: Path) -> None:
    ledger = ApprovalLedger(tmp_path / "approvals.json")
    base = LIBRARY.get("cu_core.member.open_sub_account")
    beta = specialize(base, load_binding("beta", "cu_core"))
    ledger.approve(base, "reviewer")
    assert ledger.is_approved(base) and not ledger.is_approved(beta, "beta")
    ledger.approve(beta, "reviewer", tenant="beta")
    assert ledger.is_approved(beta, "beta") and ledger.is_approved(base)  # neither overwrote the other
