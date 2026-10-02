import copy
from pathlib import Path
from typing import Any

import pytest

from cua.artifact.schema import Capability, Risk, Step
from cua.policy import ApprovalLedger, Policy, PolicyGate, PolicyViolation, load_policy
from tests.unit.test_schema import minimal

BASE = "http://127.0.0.1:8001"


@pytest.fixture
def policy() -> Policy:
    return load_policy("cu_core")


def gate(policy: Policy, tmp_path: Path, allow: bool = False) -> PolicyGate:
    return PolicyGate(
        policy, base_url=BASE, approvals=ApprovalLedger(tmp_path / "approvals.json"), allow_irreversible=allow
    )


def click(name: str, risk: str = "safe") -> Step:
    return Step.model_validate(
        {
            "id": "s1",
            "intent": "click",
            "risk": risk,
            "action": {
                "kind": "click",
                "target": {"strategies": [{"kind": "role", "role": "button", "name": name}]},
            },
        }
    )


def cap(**changes: Any) -> Capability:
    data = copy.deepcopy(minimal())
    data["app"]["product"] = "cu_core"
    data.update(changes)
    return Capability.model_validate(data)


class TestRoutes:
    @pytest.mark.parametrize("path", ["/login", "/core/main", "/core/member/detail?mbr=1"])
    def test_allowed(self, policy: Policy, path: str) -> None:
        assert policy.route_allowed(path)

    @pytest.mark.parametrize("path", ["/__reset", "/__faults", "/core/signoff", "/admin", "/"])
    def test_blocked_or_unlisted(self, policy: Policy, path: str) -> None:
        assert not policy.route_allowed(path)

    @pytest.mark.parametrize(
        "path", ["/core/../__reset", "/core/./../core/signoff", "/login/../__faults?x=1"]
    )
    def test_dot_segments_cannot_escape_the_allowlist(self, policy: Policy, path: str) -> None:
        assert not policy.route_allowed(path)

    def test_only_the_tenants_own_origin(self, policy: Policy) -> None:
        assert policy.url_allowed(f"{BASE}/core/main", BASE)
        assert not policy.url_allowed("http://evil.example/core/main", BASE)
        assert not policy.url_allowed("http://127.0.0.1:9999/core/main", BASE)  # same host, other port
        assert not policy.url_allowed("https://127.0.0.1:8001/core/main", BASE)  # other scheme


class TestRisk:
    def test_keywords_mark_commit_controls_irreversible(self, policy: Policy) -> None:
        assert policy.classify(click("Confirm")) is Risk.IRREVERSIBLE
        assert policy.classify(click("Post Transaction")) is Risk.IRREVERSIBLE

    def test_whole_words_only(self, policy: Policy) -> None:
        assert policy.classify(click("Confirmation Number")) is Risk.SAFE
        assert policy.classify(click("Open Sub-Account")) is Risk.SAFE

    def test_artifact_can_raise_but_never_lower_risk(self, policy: Policy) -> None:
        assert policy.effective_risk(click("Confirm", risk="safe")) is Risk.IRREVERSIBLE
        assert policy.effective_risk(click("Search", risk="irreversible")) is Risk.IRREVERSIBLE

    def test_only_above_unattended_max_needs_approval(self, policy: Policy) -> None:
        assert not policy.needs_approval(click("Search", risk="reversible"))
        assert policy.needs_approval(click("Confirm", risk="irreversible"))


class TestPreflight:
    def test_clean_capability_passes(self, policy: Policy, tmp_path: Path) -> None:
        assert gate(policy, tmp_path).preflight(cap(entry_route="/core/main")) == []

    def test_violations_are_all_reported(self, policy: Policy, tmp_path: Path) -> None:
        data = copy.deepcopy(minimal())
        data["app"]["product"] = "cu_core"
        data["entry_route"] = "/__reset"
        data["status"] = "deprecated"
        data["steps"].insert(
            0,
            {
                "id": "s0",
                "intent": "go",
                "risk": "safe",
                "action": {"kind": "navigate", "route": "/core/signoff"},
            },
        )
        data["steps"].insert(
            1,
            {
                "id": "s0b",
                "intent": "commit",
                "risk": "safe",
                "action": {
                    "kind": "click",
                    "target": {"strategies": [{"kind": "role", "role": "button", "name": "Submit"}]},
                },
            },
        )
        problems = gate(policy, tmp_path).preflight(Capability.model_validate(data))
        assert any("deprecated" in p for p in problems)
        assert any("entry route '/__reset'" in p for p in problems)
        assert any("s0: route '/core/signoff'" in p for p in problems)
        assert any("s0b: looks irreversible but is declared 'safe'" in p for p in problems)

    def test_disallowed_action_kind(self, tmp_path: Path) -> None:
        read_only = Policy(product="cu_core", allowed_routes=["^/"], allowed_actions=["navigate", "extract"])
        problems = gate(read_only, tmp_path).preflight(cap(entry_route="/core/main"))
        assert any("action 'fill' is not allowed" in p for p in problems)

    def test_rendered_routes_are_checked_at_run_time(self, policy: Policy, tmp_path: Path) -> None:
        with pytest.raises(PolicyViolation):
            gate(policy, tmp_path).check_route("/__reset")


class TestApprovals:
    def test_approval_binds_to_exact_content(self, policy: Policy, tmp_path: Path) -> None:
        ledger = ApprovalLedger(tmp_path / "approvals.json")
        original = cap(entry_route="/core/main")
        ledger.approve(original, "reviewer@cu")
        assert ApprovalLedger(tmp_path / "approvals.json").is_approved(original)  # persisted
        edited = cap(entry_route="/core/main", description="edited after approval")
        assert not ledger.is_approved(edited)

    def test_status_field_is_not_trusted(self, policy: Policy, tmp_path: Path) -> None:
        assert not ApprovalLedger(tmp_path / "approvals.json").is_approved(cap(status="approved"))

    def test_irreversible_needs_approval_and_explicit_allow(self, policy: Policy, tmp_path: Path) -> None:
        capability = cap(entry_route="/core/main")
        assert "not approved" in str(gate(policy, tmp_path, allow=True).irreversible_blocker(capability))
        g = gate(policy, tmp_path, allow=False)
        g.approvals.approve(capability, "reviewer@cu")
        assert "did not allow" in str(g.irreversible_blocker(capability))
        g.allow_irreversible = True
        assert g.irreversible_blocker(capability) is None


def test_committed_capabilities_pass_preflight(policy: Policy, tmp_path: Path) -> None:
    from cua.artifact.store import CapabilityLibrary

    library = CapabilityLibrary(Path(__file__).resolve().parents[2] / "capabilities")
    for capability in library.all():
        assert gate(policy, tmp_path).preflight(capability) == [], capability.id
