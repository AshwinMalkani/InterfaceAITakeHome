"""Control transfer (leases), intervention lifecycle, masking on write, and the lease guard."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from cua.hitl.handoff import GuardedSurface
from cua.hitl.models import Action, HumanAction, InterventionRequest, InterventionStatus, Owner
from cua.hitl.store import HitlStore, LeaseConflict
from cua.security.masking import SecretRegistry, Sensitivity
from cua.surface.base import ActionFailed, Resolved


def request(session: str = "run_1", **changes: Any) -> InterventionRequest:
    data: dict[str, Any] = {
        "session_id": session,
        "run_id": session,
        "capability_id": "cu_core.member.x",
        "kind": "unknown_state",
        "step_id": "s3",
        "intent": "Run the search",
        "reason": "an undeclared overlay is blocking frame 'main'",
        "steps": [("s1", "Open"), ("s3", "Run the search"), ("s4", "Read name")],
        "locations": {"main": "/core/member/detail?mbr=10001"},
    }
    return InterventionRequest.model_validate(data | changes)


@pytest.fixture
def store(tmp_path: Path) -> HitlStore:
    s = HitlStore(tmp_path / "hitl.db")
    s.open_session("run_1")
    return s


class TestLease:
    def test_new_session_is_owned_by_automation(self, store: HitlStore) -> None:
        lease = store.lease("run_1")
        assert (lease.owner, lease.epoch) == (Owner.AUTOMATION, 0)

    def test_claim_and_resolve_transfer_control_and_bump_the_epoch(self, store: HitlStore) -> None:
        i = store.create(request())
        store.claim(i.id, "ops-alice")
        lease = store.lease("run_1")
        assert (lease.owner, lease.holder, lease.epoch) == (Owner.HUMAN, "ops-alice", 1)
        resolved = store.resolve(i.id, Action.RESUME, resume_step="s4", note="cleared the hold")
        assert resolved.status is InterventionStatus.RESOLVED and resolved.resume_step == "s4"
        lease = store.lease("run_1")
        assert (lease.owner, lease.epoch) == (Owner.AUTOMATION, 2)

    def test_only_one_operator_can_take_control(self, store: HitlStore) -> None:
        i = store.create(request())
        store.claim(i.id, "ops-alice")
        with pytest.raises(LeaseConflict):
            store.claim(i.id, "ops-bob")

    def test_cannot_hand_back_what_was_never_taken(self, store: HitlStore) -> None:
        i = store.create(request())
        with pytest.raises(LeaseConflict, match="claimed"):
            store.resolve(i.id, Action.RESUME)

    def test_lease_is_compare_and_swap(self, store: HitlStore) -> None:
        a, b = store.create(request()), store.create(request())
        store.claim(a.id, "ops-alice")  # the session is now the human's
        with pytest.raises(LeaseConflict):
            store.claim(b.id, "ops-bob")  # a second request for the same session can't steal it


class TestStore:
    def test_everything_written_is_masked(self, tmp_path: Path) -> None:
        registry = SecretRegistry(include_env=False)
        registry.register("member_id", "10001", Sensitivity.PII)
        s = HitlStore(tmp_path / "hitl.db", registry)
        s.open_session("run_1")
        i = s.create(request(reason="member 10001 has a hold; call 555-123-4567"))
        s.claim(i.id, "ops-alice")
        s.add_human_action(i.id, HumanAction(at=datetime.now(UTC), kind="click", description="link '10001'"))
        raw = (tmp_path / "hitl.db").read_bytes()
        assert b"10001" not in raw and b"555-123-4567" not in raw

    def test_lists_open_and_claimed_by_default(self, store: HitlStore) -> None:
        a, b = store.create(request()), store.create(request(step_id="s4"))
        store.claim(a.id, "ops")
        store.resolve(a.id, Action.ABORT)
        assert [i.id for i in store.interventions()] == [b.id]
        assert {i.id for i in store.interventions(include_resolved=True)} == {a.id, b.id}

    def test_shared_between_processes_via_the_file(self, store: HitlStore) -> None:
        i = store.create(request())
        other = HitlStore(store.path)  # e.g. the console process
        other.claim(i.id, "ops")
        assert store.lease("run_1").owner is Owner.HUMAN


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: self.calls.append(name)


def test_guarded_surface_refuses_to_act_while_a_human_holds_control(store: HitlStore) -> None:
    inner = _Recorder()
    guarded = GuardedSurface(inner, store, "run_1")  # type: ignore[arg-type]
    resolved = Resolved(handle=None, strategy_index=0, strategy_kind="role")
    guarded.click(resolved, None, 1000)
    store.claim(store.create(request()).id, "ops-alice")
    with pytest.raises(ActionFailed, match="held by ops-alice"):
        guarded.click(resolved, None, 1000)
    with pytest.raises(ActionFailed):
        guarded.goto("/core/main")
    guarded.observe()  # looking is always allowed
    assert inner.calls == ["click", "observe"]
