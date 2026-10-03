"""Operator console: the HTML flow and the JSON API move the lease exactly like the store does."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cua.hitl.console import create_console
from cua.hitl.models import Owner
from cua.hitl.store import HitlStore
from tests.unit.test_hitl import request


@pytest.fixture
def setup(tmp_path: Path) -> tuple[TestClient, HitlStore, str]:
    store = HitlStore(tmp_path / "hitl.db")
    store.open_session("run_1")
    intervention = store.create(request(screenshot="shot.png"))
    (tmp_path / "evidence" / "run_1").mkdir(parents=True)
    (tmp_path / "evidence" / "run_1" / "shot.png").write_bytes(b"\x89PNG")
    client = TestClient(create_console(store, tmp_path / "evidence"), follow_redirects=False)
    return client, store, intervention.id


def test_html_flow_take_control_then_hand_back(setup: tuple[TestClient, HitlStore, str]) -> None:
    client, store, iid = setup
    assert iid in client.get("/").text
    page = client.get(f"/interventions/{iid}").text
    assert "Take control" in page and "undeclared overlay" in page

    assert client.post(f"/interventions/{iid}/claim", data={"operator": "ops-alice"}).status_code == 303
    assert store.lease("run_1").owner is Owner.HUMAN
    page = client.get(f"/interventions/{iid}").text
    assert "Hand control back" in page and "Approve" not in page  # not an approval request

    client.post(f"/interventions/{iid}/resolve", data={"action": "resume", "resume_step": "s4", "note": "ok"})
    assert store.lease("run_1").owner is Owner.AUTOMATION
    assert store.get(iid).resume_step == "s4"


def test_api_rejects_invalid_decisions(setup: tuple[TestClient, HitlStore, str]) -> None:
    client, _, iid = setup
    assert client.post(f"/api/interventions/{iid}/resolve", json={"action": "resume"}).status_code == 409
    client.post(f"/api/interventions/{iid}/claim", json={"operator": "ops"})
    assert client.post(f"/api/interventions/{iid}/claim", json={"operator": "other"}).status_code == 409
    bad_step = client.post(
        f"/api/interventions/{iid}/resolve", json={"action": "resume", "resume_step": "s99"}
    )
    assert bad_step.status_code == 400
    approve = client.post(f"/api/interventions/{iid}/resolve", json={"action": "approve"})
    assert approve.status_code == 400  # only approval requests can be approved
    assert client.get("/api/interventions/int_missing").status_code == 404


def test_only_redacted_screenshots_inside_the_evidence_dir_are_served(
    setup: tuple[TestClient, HitlStore, str],
) -> None:
    client, _, _ = setup
    assert client.get("/evidence/run_1/shot.png").status_code == 200
    assert client.get("/evidence/run_1/events.jsonl").status_code == 404
    assert client.get("/evidence/..%2F..%2Fetc/passwd.png").status_code == 404
