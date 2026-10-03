"""Shared state for human handoff: leases, intervention requests, captured human actions.

SQLite, because the run process and the operator console are separate processes that must agree on
who holds a session; one file with transactions is the simplest correct thing. (In production this
would be a database plus a queue that pages an operator; the interface stays the same.)

This is an egress point like any other: every text field is passed through `safe_mask` before it is
written, and the egress guard only allows sqlite in this module.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cua.hitl.models import (
    Action,
    HumanAction,
    Intervention,
    InterventionRequest,
    InterventionStatus,
    Lease,
    Owner,
)
from cua.security.masking import SecretRegistry, safe_mask

_SCHEMA = """
CREATE TABLE IF NOT EXISTS leases (
    session_id TEXT PRIMARY KEY, owner TEXT NOT NULL, epoch INTEGER NOT NULL,
    holder TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS interventions (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
    request TEXT NOT NULL, operator TEXT, action TEXT, resume_step TEXT, note TEXT NOT NULL DEFAULT '',
    resolved_at TEXT);
CREATE TABLE IF NOT EXISTS human_actions (
    intervention_id TEXT NOT NULL, at TEXT NOT NULL, kind TEXT NOT NULL, description TEXT NOT NULL,
    frame TEXT);
"""


class LeaseConflict(Exception):
    """The lease changed under us: someone else took or returned control first."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class HitlStore:
    def __init__(self, path: Path, registry: SecretRegistry | None = None) -> None:
        self.path = path
        self.registry = registry  # masks run-specific values (inputs, outputs) on the way in
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10, isolation_level="IMMEDIATE")
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def _mask(self, value: Any) -> Any:
        return safe_mask(value, self.registry)

    # --- leases -----------------------------------------------------------------------------

    def open_session(self, session_id: str) -> Lease:
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO leases VALUES (?, ?, 0, ?, ?)",
                (session_id, Owner.AUTOMATION, Owner.AUTOMATION, _now()),
            )
        return self.lease(session_id)

    def lease(self, session_id: str) -> Lease:
        with self._db() as db:
            row = db.execute("SELECT * FROM leases WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"no session {session_id!r}")
        return Lease.model_validate(dict(row))

    def _transfer(
        self, db: sqlite3.Connection, session_id: str, to: Owner, holder: str, expected: Owner
    ) -> None:
        """Compare-and-swap: only succeeds if the session is currently owned by `expected`."""
        changed = db.execute(
            "UPDATE leases SET owner = ?, holder = ?, epoch = epoch + 1, updated_at = ? "
            "WHERE session_id = ? AND owner = ?",
            (to, self._mask(holder), _now(), session_id, expected),
        ).rowcount
        if changed != 1:
            raise LeaseConflict(f"session {session_id!r} is not owned by {expected}")

    # --- interventions ----------------------------------------------------------------------

    def create(self, request: InterventionRequest) -> Intervention:
        intervention_id = "int_" + secrets.token_hex(4)
        payload = json.dumps(self._mask(request.model_dump(mode="json")))
        with self._db() as db:
            db.execute(
                "INSERT INTO interventions (id, session_id, status, created_at, request) "
                "VALUES (?, ?, ?, ?, ?)",
                (intervention_id, request.session_id, InterventionStatus.OPEN, _now(), payload),
            )
        return self.get(intervention_id)

    def get(self, intervention_id: str) -> Intervention:
        with self._db() as db:
            row = db.execute("SELECT * FROM interventions WHERE id = ?", (intervention_id,)).fetchone()
        if row is None:
            raise KeyError(f"no intervention {intervention_id!r}")
        return self._intervention(row)

    def interventions(self, *, include_resolved: bool = False) -> list[Intervention]:
        query = "SELECT * FROM interventions" + ("" if include_resolved else " WHERE status != 'resolved'")
        with self._db() as db:
            rows = db.execute(query + " ORDER BY created_at DESC").fetchall()
        return [self._intervention(r) for r in rows]

    @staticmethod
    def _intervention(row: sqlite3.Row) -> Intervention:
        data = dict(row)  # (sqlite3.Row's `in` checks values, so convert rather than test keys)
        request = json.loads(data.pop("request"))
        return Intervention.model_validate({**request, **data})

    def claim(self, intervention_id: str, operator: str) -> Intervention:
        """Operator takes control of the live session."""
        with self._db() as db:
            row = db.execute(
                "SELECT session_id, status FROM interventions WHERE id = ?", (intervention_id,)
            ).fetchone()
            if row is None or row["status"] != InterventionStatus.OPEN:
                raise LeaseConflict(f"intervention {intervention_id!r} is not open")
            self._transfer(db, row["session_id"], Owner.HUMAN, operator, expected=Owner.AUTOMATION)
            db.execute(
                "UPDATE interventions SET status = ?, operator = ? WHERE id = ?",
                (InterventionStatus.CLAIMED, self._mask(operator), intervention_id),
            )
        return self.get(intervention_id)

    def resolve(
        self, intervention_id: str, action: Action, *, resume_step: str | None = None, note: str = ""
    ) -> Intervention:
        """Operator hands control back with a decision."""
        with self._db() as db:
            row = db.execute(
                "SELECT session_id, status FROM interventions WHERE id = ?", (intervention_id,)
            ).fetchone()
            if row is None or row["status"] != InterventionStatus.CLAIMED:
                raise LeaseConflict(f"intervention {intervention_id!r} must be claimed before it is resolved")
            self._transfer(db, row["session_id"], Owner.AUTOMATION, Owner.AUTOMATION, expected=Owner.HUMAN)
            db.execute(
                "UPDATE interventions SET status = ?, action = ?, resume_step = ?, note = ?, resolved_at = ? "
                "WHERE id = ?",
                (InterventionStatus.RESOLVED, action, resume_step, self._mask(note), _now(), intervention_id),
            )
        return self.get(intervention_id)

    # --- captured human actions --------------------------------------------------------------

    def add_human_action(self, intervention_id: str, action: HumanAction) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO human_actions VALUES (?, ?, ?, ?, ?)",
                (
                    intervention_id,
                    action.at.isoformat(),
                    action.kind,
                    self._mask(action.description),
                    action.frame,
                ),
            )

    def human_actions(self, intervention_id: str) -> list[HumanAction]:
        with self._db() as db:
            rows = db.execute(
                "SELECT at, kind, description, frame FROM human_actions WHERE intervention_id = ? "
                "ORDER BY at",
                (intervention_id,),
            ).fetchall()
        return [HumanAction.model_validate(dict(r)) for r in rows]
