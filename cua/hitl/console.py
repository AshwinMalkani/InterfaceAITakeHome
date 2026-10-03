"""A minimal operator console: see what needs a human, take control, hand it back.

Deliberately bare (the brief scopes a full co-browsing console out). What it does is real: claiming
moves the live session's lease to the operator, resolving moves it back with a decision. The live
browser itself is reached through the endpoint shown in the request (the headed window locally, or
the CDP endpoint, e.g. via chrome://inspect). In production a co-browsing UI would sit on that same
endpoint; the control-transfer model would not change.

Run with `python -m cua console`. JSON equivalents of every action live under /api.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel

from cua.hitl.models import Action, HumanAction, Intervention, InterventionStatus
from cua.hitl.store import HitlStore, LeaseConflict

_STYLE = """<style>
body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px;color:#1d1d1f}
h1{font-size:20px} table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid #ddd;padding:6px;
text-align:left;vertical-align:top} .open{color:#b00020} .claimed{color:#a15c00} .resolved{color:#2e7d32}
img{max-width:100%;border:1px solid #ccc} form{margin:12px 0}
.box{background:#f6f6f7;padding:12px;border-radius:6px}
</style>"""


class ClaimBody(BaseModel):
    operator: str


class ResolveBody(BaseModel):
    action: Action
    resume_step: str | None = None
    note: str = ""


def create_console(store: HitlStore, evidence_root: Path) -> FastAPI:
    app = FastAPI(title="cua operator console", docs_url=None, redoc_url=None)

    def find(intervention_id: str) -> Intervention:
        try:
            return store.get(intervention_id)
        except KeyError:
            raise HTTPException(404, "no such intervention") from None

    def claim(intervention_id: str, operator: str) -> Intervention:
        find(intervention_id)
        try:
            return store.claim(intervention_id, operator.strip() or "operator")
        except LeaseConflict as exc:
            raise HTTPException(409, str(exc)) from None

    def resolve(intervention_id: str, body: ResolveBody) -> Intervention:
        current = find(intervention_id)
        if body.action is Action.APPROVE and current.kind != "approval":
            raise HTTPException(400, "only approval requests can be approved")
        if body.resume_step and body.resume_step not in {s for s, _ in current.steps} | {"success"}:
            raise HTTPException(400, "unknown step")
        try:
            return store.resolve(intervention_id, body.action, resume_step=body.resume_step, note=body.note)
        except LeaseConflict as exc:
            raise HTTPException(409, str(exc)) from None

    # --- JSON API -------------------------------------------------------------------------

    @app.get("/api/interventions")
    def api_list(include_resolved: bool = False) -> list[Intervention]:
        return store.interventions(include_resolved=include_resolved)

    @app.get("/api/interventions/{intervention_id}")
    def api_get(intervention_id: str) -> dict[str, object]:
        return {
            "intervention": find(intervention_id),
            "human_actions": store.human_actions(intervention_id),
            "lease": store.lease(find(intervention_id).session_id),
        }

    @app.post("/api/interventions/{intervention_id}/claim")
    def api_claim(intervention_id: str, body: ClaimBody) -> Intervention:
        return claim(intervention_id, body.operator)

    @app.post("/api/interventions/{intervention_id}/resolve")
    def api_resolve(intervention_id: str, body: ResolveBody) -> Intervention:
        return resolve(intervention_id, body)

    # --- evidence (redacted screenshots only) ---------------------------------------------

    @app.get("/evidence/{run_id}/{name}")
    def screenshot(run_id: str, name: str) -> FileResponse:
        root = evidence_root.resolve()
        path = (root / run_id / name).resolve()
        if not path.is_relative_to(root) or path.suffix != ".png" or not path.exists():
            raise HTTPException(404, "not found")
        return FileResponse(path)

    # --- HTML -----------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        rows = (
            "".join(
                f"<tr><td><a href='/interventions/{i.id}'>{i.id}</a></td>"
                f"<td class='{i.status}'>{i.status}</td>"
                f"<td>{escape(i.kind)}</td><td>{escape(i.capability_id)}</td><td>{escape(i.step_id)}: "
                f"{escape(i.intent)}</td><td>{i.created_at:%H:%M:%S}</td></tr>"
                for i in store.interventions(include_resolved=True)
            )
            or "<tr><td colspan=6>Nothing needs a human.</td></tr>"
        )
        return (
            f"<html><head><title>Operator console</title>{_STYLE}"
            "<meta http-equiv='refresh' content='3'></head><body><h1>Interventions</h1>"
            "<table><tr><th>id</th><th>status</th><th>kind</th><th>capability</th><th>step</th>"
            f"<th>created</th></tr>{rows}</table></body></html>"
        )

    @app.get("/interventions/{intervention_id}", response_class=HTMLResponse)
    def detail(intervention_id: str) -> str:
        i = find(intervention_id)
        lease = store.lease(i.session_id)
        shot = f"<img src='/evidence/{i.run_id}/{escape(i.screenshot)}'>" if i.screenshot else ""
        steps = (
            "".join(
                f"<option value='{escape(s)}' {'selected' if s == i.step_id else ''}>"
                f"{escape(s)}: {escape(t)}</option>"
                for s, t in i.steps
            )
            + "<option value='success'>final success check</option>"
        )
        if i.status is InterventionStatus.OPEN:
            controls = (
                f"<form method='post' action='/interventions/{i.id}/claim'>Operator name "
                "<input name='operator' required> <button>Take control of the live session</button></form>"
            )
        elif i.status is InterventionStatus.CLAIMED:
            approve = (
                (
                    "<label><input type='radio' name='action' value='approve'> "
                    "Approve this irreversible step "
                    "(this run only)</label><br>"
                )
                if i.kind == "approval"
                else ""
            )
            controls = (
                f"<form method='post' action='/interventions/{i.id}/resolve' class='box'>"
                "<b>Hand control back</b><br>"
                "<label><input type='radio' name='action' value='resume' checked> Resume at step </label>"
                f"<select name='resume_step'>{steps}</select><br>{approve}"
                "<label><input type='radio' name='action' value='abort'> Abort the run</label><br>"
                "Note <input name='note' size=60> <button>Hand back</button></form>"
            )
        else:
            controls = (
                f"<p class='box'>Resolved by {escape(i.operator or '?')}: <b>{i.action}</b>"
                f"{' at ' + escape(i.resume_step) if i.resume_step else ''}. {escape(i.note)}</p>"
            )
        actions = "".join(
            f"<li>{a.at:%H:%M:%S} {escape(a.kind)} {escape(a.description)}"
            f"{' in ' + escape(a.frame) if a.frame else ''}</li>"
            for a in _actions(store, i.id)
        )
        return (
            f"<html><head><title>{i.id}</title>{_STYLE}</head><body><a href='/'>&larr; all</a>"
            f"<h1>{escape(i.kind)} at {escape(i.step_id)}: {escape(i.intent)}</h1>"
            f"<p><b>Status</b> <span class='{i.status}'>{i.status}</span> &middot; <b>Session held by</b> "
            f"{escape(lease.holder)} (epoch {lease.epoch}) &middot; <b>Run</b> {escape(i.run_id)}</p>"
            f"<p><b>Why</b> {escape(i.reason)}</p>{controls}"
            f"<h3>Operator actions captured</h3><ul>{actions or '<li>none yet</li>'}</ul>"
            f"<h3>Screen at escalation (redacted)</h3>{shot}</body></html>"
        )

    @app.post("/interventions/{intervention_id}/claim")
    def form_claim(intervention_id: str, operator: Annotated[str, Form()]) -> RedirectResponse:
        claim(intervention_id, operator)
        return RedirectResponse(f"/interventions/{intervention_id}", status_code=303)

    @app.post("/interventions/{intervention_id}/resolve")
    def form_resolve(
        intervention_id: str,
        action: Annotated[Action, Form()],
        resume_step: Annotated[str, Form()] = "",
        note: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        resolve(intervention_id, ResolveBody(action=action, resume_step=resume_step or None, note=note))
        return RedirectResponse(f"/interventions/{intervention_id}", status_code=303)

    return app


def _actions(store: HitlStore, intervention_id: str) -> list[HumanAction]:
    return store.human_actions(intervention_id)
