"""CU-Core: an intentionally legacy-style credit-union back-office app used as the automation target.

Deliberately hostile to DOM-selector automation, like the real long tail:
framesets, table layouts, `<td>` text instead of `<label for>`, inputs named f1..f5, no ids or
test ids, no semantic headings, a native JS confirm() before the irreversible step, and HTTP 200
for most error screens.

Runtime conditions from the brief are all reproducible (see faults.py) and admin endpoints
(`/__*`) exist only for tests and demos; the automation's allowlist must never include them.
"""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from apps.cu_core.data import Member, MemberStore
from apps.cu_core.faults import Fault, FaultInjector, FaultKind
from apps.cu_core.tenants import Tenant

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
SESSION_COOKIE = "JSESSIONID"
DEMO_USER = "teller1"
MEMBER_ID_FORMAT = re.compile(r"^\d{5}$")
NEW_ACCOUNT_KINDS = ("certificate", "money_market", "holiday_club")
MIN_DEPOSIT = Decimal("5.00")


@dataclass
class Session:
    user: str
    last_seen: float


class SessionStore:
    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self._sessions: dict[str, Session] = {}

    def create(self, user: str) -> str:
        token = secrets.token_urlsafe(24)
        self._sessions[token] = Session(user, time.monotonic())
        return token

    def get(self, token: str | None) -> Session | None:
        session = self._sessions.get(token or "")
        if session is None:
            return None
        if time.monotonic() - session.last_seen > self.timeout_s:
            del self._sessions[token or ""]
            return None
        session.last_seen = time.monotonic()
        return session

    def expire(self, token: str | None) -> None:
        self._sessions.pop(token or "", None)

    def clear(self) -> None:
        self._sessions.clear()


class SessionExpired(Exception):
    pass


def _parse_amount(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace("$", "").replace(",", "").strip()).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def create_app(
    tenant: Tenant,
    *,
    password: str,
    session_timeout_s: float = 300,
    enable_admin: bool = True,
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    store = MemberStore.seeded()
    sessions = SessionStore(session_timeout_s)
    faults = FaultInjector()

    def render(request: Request, template: str, status_code: int = 200, **ctx: Any) -> HTMLResponse:
        ctx |= {"t": tenant, "overlay": getattr(request.state, "overlay", False)}
        return TEMPLATES.TemplateResponse(request, template, ctx, status_code=status_code)

    def require_session(request: Request) -> Session:
        session = sessions.get(request.cookies.get(SESSION_COOKIE))
        if session is None:
            raise SessionExpired
        return session

    def load_member(request: Request, member_id: str) -> Member | HTMLResponse:
        member = store.get(member_id)
        if member is None:
            return render(request, "message.html", title="Member Inquiry", message="No records found.")
        if member.restricted:
            return render(
                request, "message.html", title="Member Inquiry",
                message="ERR 0403: Insufficient privileges to access this member record.",
            )
        return member

    @app.exception_handler(SessionExpired)
    async def _session_expired(request: Request, exc: SessionExpired) -> Response:
        return RedirectResponse("/login?reason=expired", status_code=303)

    @app.middleware("http")
    async def _inject_faults(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if not path.startswith("/core"):
            return await call_next(request)
        if fault := faults.take(path, FaultKind.SLOW):
            await asyncio.sleep(fault.delay_ms / 1000)
        if faults.take(path, FaultKind.EXPIRE_SESSION):
            sessions.expire(request.cookies.get(SESSION_COOKIE))
        if faults.take(path, FaultKind.SERVER_ERROR):
            ref = secrets.token_hex(4).upper()
            return render(request, "app_error.html", status_code=500, reference=ref)
        if request.method == "GET" and faults.take(path, FaultKind.INTERSTITIAL):
            return render(request, "notice.html", continue_url=str(request.url.path) + (
                f"?{request.url.query}" if request.url.query else ""))
        if faults.take(path, FaultKind.COMPLIANCE_HOLD):
            request.state.overlay = True
        return await call_next(request)

    # --- sign on / shell ---------------------------------------------------------------

    @app.get("/")
    async def root() -> Response:
        return RedirectResponse("/login", status_code=303)

    @app.get("/login")
    async def login_page(request: Request, reason: str = "") -> Response:
        return render(request, "login.html", expired=reason == "expired", error=None)

    @app.post("/login")
    async def login(
        request: Request, userid: str = Form(""), password_: str = Form("", alias="password")
    ) -> Response:
        if userid != DEMO_USER or not secrets.compare_digest(password_, password):
            return render(request, "login.html", expired=False, error="Invalid user ID or password.")
        response = RedirectResponse("/core/main", status_code=303)
        response.set_cookie(SESSION_COOKIE, sessions.create(userid), httponly=True, samesite="lax")
        return response

    @app.get("/core/main")
    async def shell(request: Request) -> Response:
        require_session(request)
        return render(request, "frameset.html")

    @app.get("/core/banner")
    async def banner(request: Request) -> Response:
        session = require_session(request)
        return render(request, "banner.html", user=session.user)

    @app.get("/core/menu")
    async def menu(request: Request) -> Response:
        require_session(request)
        return render(request, "menu.html")

    @app.get("/core/welcome")
    async def welcome(request: Request) -> Response:
        require_session(request)
        return render(request, "welcome.html")

    @app.get("/core/signoff")
    async def signoff(request: Request) -> Response:
        sessions.expire(request.cookies.get(SESSION_COOKIE))
        return RedirectResponse("/login", status_code=303)

    # --- member inquiry (read-only) -----------------------------------------------------

    @app.get("/core/member/search")
    async def search_page(request: Request) -> Response:
        require_session(request)
        return render(request, "search.html", error=None, message=None, query="")

    @app.post("/core/member/search")
    async def search(request: Request, f1: str = Form("")) -> Response:
        require_session(request)
        query = f1.strip()
        if not MEMBER_ID_FORMAT.match(query):
            error = f"ERR 1042: Invalid {tenant.label('member_number')} format. Enter 5 digits."
            return render(request, "search.html", error=error, message=None, query=query)
        if store.get(query) is None:
            return render(request, "search.html", error=None, message="No records found.", query=query)
        return RedirectResponse(f"/core/member/detail?mbr={query}", status_code=303)

    @app.get("/core/member/detail")
    async def detail(request: Request, mbr: str = "") -> Response:
        require_session(request)
        member = load_member(request, mbr)
        if isinstance(member, Response):
            return member
        return render(request, "detail.html", m=member)

    # --- open sub-account (irreversible) ------------------------------------------------

    def validate_new_account(
        member: Member, kind: str, deposit_raw: str, funding: str
    ) -> tuple[Decimal | None, str | None]:
        if kind not in NEW_ACCOUNT_KINDS:
            return None, "ERR 2204: Select an account type."
        deposit = _parse_amount(deposit_raw)
        if deposit is None:
            return None, "ERR 2201: Invalid amount."
        if deposit < MIN_DEPOSIT:
            return None, f"ERR 2202: Minimum opening deposit is ${MIN_DEPOSIT}."
        source = member.account(funding)
        if source is None:
            return None, "ERR 2205: Select a funding account."
        if deposit > source.available:
            return None, "ERR 2210: Insufficient funds in funding account."
        return deposit, None

    @app.get("/core/member/subacct/new")
    async def new_subacct(request: Request, mbr: str = "") -> Response:
        require_session(request)
        member = load_member(request, mbr)
        if isinstance(member, Response):
            return member
        return render(request, "subacct_new.html", m=member, kinds=NEW_ACCOUNT_KINDS, error=None, form={})

    @app.post("/core/member/subacct/review")
    async def review_subacct(
        request: Request, mbr: str = Form(""), f2: str = Form(""), f3: str = Form(""),
        f4: str = Form(""), f5: str = Form(""),
    ) -> Response:
        require_session(request)
        member = load_member(request, mbr)
        if isinstance(member, Response):
            return member
        form = {"f2": f2, "f3": f3, "f4": f4, "f5": f5}
        deposit, error = validate_new_account(member, f2, f4, f5)
        if error:
            return render(
                request, "subacct_new.html", m=member, kinds=NEW_ACCOUNT_KINDS, error=error, form=form
            )
        return render(request, "subacct_review.html", m=member, form=form, deposit=deposit)

    @app.post("/core/member/subacct/commit")
    async def commit_subacct(
        request: Request, mbr: str = Form(""), f2: str = Form(""), f3: str = Form(""),
        f4: str = Form(""), f5: str = Form(""),
    ) -> Response:
        require_session(request)
        member = load_member(request, mbr)
        if isinstance(member, Response):
            return member
        deposit, error = validate_new_account(member, f2, f4, f5)
        if error or deposit is None:
            return render(request, "message.html", title="Open Sub-Account", message=error)
        # No idempotency token, like many legacy apps: a replayed POST opens a second account.
        # That is exactly why the replay engine must never auto-retry an irreversible step.
        suffix, confirmation = store.open_sub_account(member, f2, f5, deposit)
        return render(request, "subacct_done.html", m=member, suffix=suffix, confirmation=confirmation,
                      kind=f2, deposit=deposit, nickname=f3)

    # --- admin (tests and demos only) ---------------------------------------------------

    if enable_admin:

        @app.get("/__health")
        async def health() -> dict[str, str]:
            return {"tenant": tenant.key, "version": tenant.product_version}

        @app.post("/__faults")
        async def arm_faults(new: list[Fault]) -> list[Fault]:
            faults.arm(new)
            return faults.armed

        @app.get("/__faults")
        async def list_faults() -> list[Fault]:
            return faults.armed

        @app.delete("/__faults")
        async def clear_faults() -> JSONResponse:
            faults.clear()
            return JSONResponse({"cleared": True})

        @app.post("/__reset")
        async def reset() -> JSONResponse:
            nonlocal store
            store = MemberStore.seeded()
            faults.clear()
            sessions.clear()
            return JSONResponse({"reset": True})

    return app
