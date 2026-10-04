"""Discovery with a human in the loop, on the live app (scripted model, simulated operator thread)."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from apps.cu_core.data import MemberStore
from cua.agent.goal import load_goal
from cua.artifact.schema import Click, Risk
from cua.artifact.store import CapabilityLibrary
from cua.discovery import discover
from cua.hitl.models import Action, Intervention
from cua.hitl.store import HitlStore
from cua.policy import load_policy
from cua.profile import load_profile
from cua.replay.result import Failure, FailureCategory
from tests.regression.conftest import TEST_PASSWORD, TargetApp
from tests.regression.harness import find_leaks
from tests.regression.test_discovery_live import OperatorScript, _common

pytestmark = pytest.mark.regression

REPO = Path(__file__).resolve().parents[2]
LIBRARY = CapabilityLibrary(REPO / "capabilities")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def operator_thread(
    store: HitlStore, decide: Callable[[Intervention], None]
) -> tuple[threading.Thread, list]:
    errors: list[BaseException] = []

    def run() -> None:
        try:
            deadline = time.monotonic() + 60
            while not (pending := store.interventions()):
                assert time.monotonic() < deadline, "no intervention was raised"
                time.sleep(0.2)
            store.claim(pending[0].id, "ops-alice")
            decide(pending[0])
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, errors


def _discover(
    app: TargetApp,
    goal_file: str,
    params: dict[str, str],
    script: OperatorScript,
    store: HitlStore,
    tmp_path: Path,
    cdp_port: int | None = None,
):  # type: ignore[no-untyped-def]
    goal = load_goal(REPO / "goals" / "cu_core" / goal_file)
    return discover(
        goal,
        params,
        model=script,
        base_url=app.base_url,
        library=LIBRARY,
        profile=load_profile("cu_core"),
        policy=load_policy("cu_core"),
        evidence_root=tmp_path,
        hitl=store,
        cdp_port=cdp_port,
        unclaimed_timeout_s=30,
    )


def test_operator_approves_the_irreversible_step_during_discovery(
    target_apps: dict[str, TargetApp], tmp_path: Path
) -> None:
    app = target_apps["alpha"]
    app.reset()
    store = HitlStore(tmp_path / "hitl.db")
    thread, errors = operator_thread(
        store, lambda i: store.resolve(i.id, Action.APPROVE, note="customer confirmed by phone")
    )
    script = OperatorScript(
        [
            (
                "click",
                "Member Inquiry",
                "menu",
                {**_common("Open member inquiry"), "expect_text": "MEMBER INQUIRY", "risk": "safe"},
            ),
            (
                "fill",
                "textbox (next to",
                "main",
                {**_common("Enter member"), "value": "{{inputs.member_id}}"},
            ),
            (
                "click",
                '"Search"',
                "main",
                {**_common("Search"), "expect_text": "MEMBER DETAIL", "risk": "safe"},
            ),
            (
                "click",
                '"Open Sub-Account"',
                "main",
                {**_common("Start a sub-account"), "expect_text": "OPEN SUB-ACCOUNT", "risk": "safe"},
            ),
            (
                "select",
                "(next to 'Account Type')",
                "main",
                {**_common("Choose type"), "option": "{{inputs.account_type}}"},
            ),
            ("fill", "(next to 'Nickname')", "main", {**_common("Nickname"), "value": "{{inputs.nickname}}"}),
            (
                "fill",
                "(next to 'Opening Deposit')",
                "main",
                {**_common("Deposit"), "value": "{{inputs.opening_deposit}}"},
            ),
            (
                "select",
                "(next to 'Funding Account')",
                "main",
                {**_common("Fund from savings"), "option": "00 - Share Savings"},
            ),
            (
                "click",
                '"Continue"',
                "main",
                {**_common("Review"), "expect_text": "REVIEW NEW SUB-ACCOUNT", "risk": "safe"},
            ),
            (
                "click",
                '"Confirm"',
                "main",
                {
                    **_common("Confirm and open the account"),
                    "expect_text": "SUB-ACCOUNT OPENED",
                    "risk": "irreversible",
                },
            ),
            ("extract", '"C-0', "main", {**_common("Read confirmation"), "output": "confirmation_number"}),
            ("extract", '"20"', "main", {**_common("Read new suffix"), "output": "new_suffix"}),
            ("finish", None, "main", {"success_text": "SUB-ACCOUNT OPENED", "rationale": "done"}),
        ]
    )
    params = {
        "member_id": "10002",
        "account_type": "Money Market",
        "nickname": "Rainy day",
        "opening_deposit": "100.00",
    }
    outcome = _discover(app, "open_funded_sub_account.yaml", params, script, store, tmp_path)
    thread.join(timeout=10)
    assert not errors, errors

    assert outcome.status == "recorded", outcome.reason
    [intervention] = outcome.interventions
    assert (intervention.kind, intervention.action, intervention.operator) == (
        "approval",
        "approve",
        "ops-alice",
    )

    cap = outcome.capability
    assert cap is not None
    # The "OPEN SUB-ACCOUNT" heading replaces a same-named button in the same frame: it still counts as
    # evidence of change because the frame loaded a new document.
    start = next(s for s in cap.steps if s.intent == "Start a sub-account")
    assert start.expect is not None
    confirm = next(s for s in cap.steps if s.risk is Risk.IRREVERSIBLE)
    assert isinstance(confirm.action, Click) and confirm.action.dialog is not None
    assert "cannot be undone" in confirm.action.dialog.message_contains  # the confirm it raised, recorded

    # Verified up to the irreversible step, and stopped there by policy: never re-executed.
    assert outcome.verification_scope == "until_irreversible" and outcome.verified
    assert isinstance(outcome.verification, Failure)
    assert outcome.verification.category is FailureCategory.APPROVAL_REQUIRED
    assert outcome.verification.step_id == confirm.id
    assert find_leaks(outcome.evidence_dir, [*MemberStore.seeded().sensitive_values(), TEST_PASSWORD]) == []


def test_agent_asks_for_help_and_an_operator_fixes_the_live_screen(
    target_apps: dict[str, TargetApp], tmp_path: Path
) -> None:
    app = target_apps["alpha"]
    app.reset()
    app.arm([{"kind": "compliance_hold", "path": "/member/detail"}])
    store = HitlStore(tmp_path / "hitl.db")
    port = _free_port()

    def override_in_the_live_browser(intervention: Intervention) -> None:
        with sync_playwright() as pw:
            page = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}").contexts[0].pages[0]
            frame = page.frame(name="main")
            assert frame is not None
            frame.get_by_role("button", name="Override").click()
            page.wait_for_timeout(300)
        store.resolve(intervention.id, Action.RESUME, resume_step="next", note="supervisor override")

    thread, errors = operator_thread(store, override_in_the_live_browser)
    script = OperatorScript(
        [
            (
                "click",
                "Member Inquiry",
                "menu",
                {**_common("Open member inquiry"), "expect_text": "MEMBER INQUIRY", "risk": "safe"},
            ),
            (
                "fill",
                "textbox (next to",
                "main",
                {**_common("Enter member"), "value": "{{inputs.member_id}}"},
            ),
            (
                "click",
                '"Search"',
                "main",
                {**_common("Search"), "expect_text": "MEMBER DETAIL", "risk": "safe"},
            ),
            ("request_human", None, "main", {"reason": "a compliance hold covers the member record"}),
            ("extract", "Testwood", "main", {**_common("Read name"), "output": "member_name"}),
            ("extract", '"$4,719.56"', "main", {**_common("Read balance"), "output": "savings_balance"}),
            ("finish", None, "main", {"success_text": "MEMBER DETAIL", "rationale": "done"}),
        ]
    )
    outcome = _discover(
        app, "read_savings_balance.yaml", {"member_id": "10001"}, script, store, tmp_path, cdp_port=port
    )
    thread.join(timeout=10)
    assert not errors, errors

    assert outcome.status == "recorded" and outcome.verified, (outcome.reason, outcome.verification)
    [intervention] = outcome.interventions
    assert (intervention.kind, intervention.action, intervention.human_actions) == (
        "agent_request",
        "resume",
        1,
    )
    transcript = (outcome.evidence_dir / "transcript.json").read_text()
    assert "covering the screen" in transcript  # the agent was told something blocked the screen
    assert 'They did: button \\"Override\\"' in transcript  # and what the human did about it
