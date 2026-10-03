"""Human handoff on the live app: escalate, operator takes over the SAME session over CDP, hand back, resume.

The operator runs in another thread with its own Playwright connection to the browser's CDP endpoint,
which is exactly how a remote operator (or a co-browsing UI) reaches a live session. Nothing about
the handoff is mocked: the lease moves through the shared store, the operator's click happens in the
real browser, and it is captured by the run as a human action.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from apps.cu_core.data import MemberStore
from cua.artifact.store import CapabilityLibrary
from cua.hitl.models import Action
from cua.hitl.store import HitlStore
from cua.policy import ApprovalLedger, load_policy
from cua.profile import load_profile
from cua.replay.result import Success
from cua.runner import replay
from tests.regression.conftest import TEST_PASSWORD, TargetApp
from tests.regression.harness import find_leaks

pytestmark = pytest.mark.regression

LIBRARY = CapabilityLibrary(Path(__file__).resolve().parents[2] / "capabilities")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def _operator(store: HitlStore, cdp_port: int, errors: list[BaseException]) -> None:
    """A supervisor: wait for the request, take control, override the hold in the live browser, hand back."""
    try:
        deadline = time.monotonic() + 30
        while not (pending := store.interventions()):
            assert time.monotonic() < deadline, "no intervention was raised"
            time.sleep(0.2)
        intervention = pending[0]
        store.claim(intervention.id, "ops-alice")
        with sync_playwright() as pw:
            browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{cdp_port}")
            page = browser.contexts[0].pages[0]  # the run's own page: same session, same cookies
            main = page.frame(name="main")
            assert main is not None
            main.get_by_role("button", name="Override").click()
            page.wait_for_timeout(300)
        store.resolve(intervention.id, Action.RESUME, resume_step="s4", note="supervisor override applied")
    except BaseException as exc:  # surfaced in the main thread
        errors.append(exc)


def test_operator_takes_over_the_live_session_and_hands_back(
    target_apps: dict[str, TargetApp], tmp_path: Path
) -> None:
    app = target_apps["alpha"]
    app.reset()
    app.arm([{"kind": "compliance_hold", "path": "/member/detail"}])
    store = HitlStore(tmp_path / "hitl.db")
    port = _free_port()
    errors: list[BaseException] = []
    operator = threading.Thread(target=_operator, args=(store, port, errors), daemon=True)
    operator.start()

    outcome = replay(
        "cu_core.member.get_share_balance",
        {"member_id": "10001"},
        base_url=app.base_url,
        library=LIBRARY,
        profile=load_profile("cu_core"),
        policy=load_policy("cu_core"),
        approvals=ApprovalLedger(tmp_path / "approvals.json"),
        evidence_root=tmp_path / "runs",
        hitl=store,
        cdp_port=port,
        unclaimed_timeout_s=30,
    )
    operator.join(timeout=10)
    assert not errors, errors

    # The run finished on the same session, after the human cleared the hold.
    result = outcome.result
    assert isinstance(result, Success), result
    assert result.outputs["member_name"] == "Avery Testwood"
    [summary] = result.interventions
    assert (summary.kind, summary.step_id, summary.action, summary.resume_step, summary.operator) == (
        "unknown_state",
        "s3",
        "resume",
        "s4",
        "ops-alice",
    )

    # What the human did was captured, and only the human's actions (no automation clicks).
    actions = store.human_actions(summary.id)
    assert [a.description for a in actions] == ['button "Override"']
    assert summary.human_actions == 1

    # The handoff is in the run's log, correlated to the run.
    events = [json.loads(line) for line in (outcome.evidence_dir / "events.jsonl").read_text().splitlines()]
    names = [e["event"] for e in events]
    assert names.index("hitl.requested") < names.index("human.action") < names.index("hitl.resolved")
    human = next(e for e in events if e["event"] == "human.action")
    assert human["actor"] == "human" and human["step_id"] == "s3"
    # Every line is correlated, including those emitted from Playwright's callback greenlet.
    assert all(e.get("run_id") == outcome.run_id for e in events)
    transfers = [(e["status"], e["owner"]) for e in events if e["event"] == "hitl.lease_transferred"]
    assert transfers == [("claimed", "human"), ("resolved", "automation")]

    # The request carried what an operator needs: a redacted screenshot of the live screen.
    intervention = store.get(summary.id)
    assert intervention.screenshot and (outcome.evidence_dir / intervention.screenshot).exists()
    assert find_leaks(outcome.evidence_dir, [*MemberStore.seeded().sensitive_values(), TEST_PASSWORD]) == []
