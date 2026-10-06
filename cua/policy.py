"""Safety policy: what automation may do on a product, and how risky actions are authorized.

Kept separate from the app profile on purpose. The profile is automation knowledge (how the app
behaves); the policy is the security boundary (what we're allowed to do to it). They have
different owners and different review paths.

Enforcement happens in code, never in prompts, at three points:
  1. pre-flight   a capability is checked statically before any browser opens
  2. per action   every step is checked again right before it acts (rendered routes included)
  3. network      every request the browser makes is filtered, so a click that navigates
                  somewhere unexpected is stopped too, not just our own `navigate` steps

Irreversible steps run only if the capability's *exact current content* was approved (the
approval ledger stores its content hash) AND the caller explicitly allows irreversible actions
for this invocation. Otherwise the run stops right before that step and asks for a human.
"""

from __future__ import annotations

import json
import posixpath
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cua.artifact.schema import (
    IDENTIFIER,
    Capability,
    Click,
    Locator,
    NearTextLocator,
    Press,
    Risk,
    RoleLocator,
    Status,
    Step,
)
from cua.evidence import sink

POLICIES_DIR = Path(__file__).resolve().parents[1] / "config" / "policies"
APPROVALS_FILE = Path(__file__).resolve().parents[1] / "capabilities" / "approvals.json"

_RISK_ORDER = [Risk.SAFE, Risk.REVERSIBLE, Risk.IRREVERSIBLE]


class ActionKind(StrEnum):
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    PRESS = "press"
    EXTRACT = "extract"


class RiskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Highest risk that may run without approval. Above this, approval + explicit allow is required.
    unattended_max: Risk = Risk.REVERSIBLE
    # Words that mark a control as committing something. Used to catch under-declared risk: a click
    # on "Confirm" can never be treated as safe, whatever the artifact says.
    irreversible_keywords: list[str] = Field(default_factory=list)


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str = Field(pattern=IDENTIFIER)
    allowed_routes: list[str]  # regexes on path(+query), relative to the tenant base URL
    blocked_routes: list[str] = []  # deny wins over allow
    allowed_actions: list[ActionKind]
    risk: RiskPolicy = RiskPolicy()
    # Evidence screenshots are redacted by allowlist. Only a sandbox policy may permit turning that off.
    allow_unredacted_screenshots: bool = False

    def route_allowed(self, path: str) -> bool:
        path = _normalize(path)
        if any(re.search(p, path) for p in self.blocked_routes):
            return False
        return any(re.search(p, path) for p in self.allowed_routes)

    def url_allowed(self, url: str, base_url: str) -> bool:
        """Only the tenant's own origin, and only allowed routes on it."""
        target, base = urlsplit(url), urlsplit(base_url)
        if (target.scheme, target.netloc) != (base.scheme, base.netloc):
            return False
        return self.route_allowed(target.path + (f"?{target.query}" if target.query else ""))

    def classify(self, step: Step) -> Risk:
        """Heuristic risk from what the step *touches*: clicking or pressing on a control whose
        name/anchor contains an irreversible keyword is irreversible."""
        action = step.action
        if not isinstance(action, (Click, Press)) or action.target is None:
            return Risk.SAFE
        words = " ".join(_control_text(s) for s in action.target.strategies).lower()
        if any(re.search(rf"\b{re.escape(k.lower())}\b", words) for k in self.risk.irreversible_keywords):
            return Risk.IRREVERSIBLE
        return Risk.SAFE

    def effective_risk(self, step: Step) -> Risk:
        """The artifact can raise risk but never lower it below the heuristic."""
        return max(step.risk, self.classify(step), key=_RISK_ORDER.index)

    def needs_approval(self, step: Step) -> bool:
        return _RISK_ORDER.index(self.effective_risk(step)) > _RISK_ORDER.index(self.risk.unattended_max)


def _normalize(route: str) -> str:
    """Resolve '.' and '..' the way a browser will, so '/core/../__reset' is judged as '/__reset'."""
    path, sep, query = route.partition("?")
    normalized = posixpath.normpath(path or "/")
    if path.endswith("/") and normalized != "/":
        normalized += "/"
    return normalized + sep + query


def _control_text(locator: Locator) -> str:
    match locator:
        case RoleLocator(name=name):
            return name
        case NearTextLocator(anchor=anchor):
            return anchor
    return ""


def load_policy(product: str, directory: Path = POLICIES_DIR) -> Policy:
    path = directory / f"{product}.yaml"
    policy = Policy.model_validate(yaml.safe_load(path.read_text()))
    if policy.product != product:
        raise ValueError(f"{path} declares product {policy.product!r}")
    return policy


# --- approvals ----------------------------------------------------------------------------


class Approval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    content_hash: str
    approved_by: str
    approved_at: datetime
    note: str = ""


class ApprovalLedger:
    """Approvals bound to exact capability content. Editing a capability changes its hash and
    therefore voids its approval automatically; the `status` field in the artifact is never trusted
    for authorization, since anyone who can edit the file can edit that field."""

    def __init__(self, path: Path = APPROVALS_FILE) -> None:
        self.path = path
        raw = json.loads(path.read_text()) if path.exists() else {}
        self.entries = {k: Approval.model_validate(v) for k, v in raw.items()}

    @staticmethod
    def key(capability: Capability, tenant: str | None = None) -> str:
        """Approvals are per tenant: a tenant-specialized capability is different behaviour."""
        return f"{capability.id}@{tenant}" if tenant else capability.id

    def is_approved(self, capability: Capability, tenant: str | None = None) -> bool:
        entry = self.entries.get(self.key(capability, tenant))
        return entry is not None and entry.content_hash == capability.content_hash()

    def approve(
        self, capability: Capability, approved_by: str, note: str = "", tenant: str | None = None
    ) -> Approval:
        entry = Approval(
            version=capability.version,
            content_hash=capability.content_hash(),
            approved_by=approved_by,
            approved_at=datetime.now(UTC),
            note=note,
        )
        self.entries[self.key(capability, tenant)] = entry
        sink.write_json(self.path, {k: v.model_dump(mode="json") for k, v in sorted(self.entries.items())})
        return entry


# --- the gate -----------------------------------------------------------------------------


class PolicyViolation(Exception):
    pass


class PolicyGate:
    """One run's view of the policy: the tenant it's bound to and what the caller authorized."""

    def __init__(
        self,
        policy: Policy,
        *,
        base_url: str,
        approvals: ApprovalLedger,
        allow_irreversible: bool = False,
        tenant: str | None = None,
    ) -> None:
        self.policy = policy
        self.base_url = base_url
        self.approvals = approvals
        self.allow_irreversible = allow_irreversible
        self.tenant = tenant

    def preflight(self, capability: Capability) -> list[str]:
        """Static violations, found before a browser opens. Empty list = may run."""
        problems: list[str] = []
        if capability.app.product != self.policy.product:
            problems.append(f"capability targets {capability.app.product!r}, not {self.policy.product!r}")
        if capability.status is Status.DEPRECATED:
            problems.append("capability is deprecated")
        if "{{" not in capability.entry_route and not self.policy.route_allowed(capability.entry_route):
            problems.append(f"entry route {capability.entry_route!r} is not allowed")
        for step in capability.steps:
            if step.action.kind not in self.policy.allowed_actions:
                problems.append(f"step {step.id}: action {step.action.kind!r} is not allowed")
            if (
                (route := getattr(step.action, "route", None))
                and "{{" not in route
                and not self.policy.route_allowed(route)
            ):
                problems.append(f"step {step.id}: route {route!r} is not allowed")
            if self.policy.classify(step) is Risk.IRREVERSIBLE and step.risk is not Risk.IRREVERSIBLE:
                problems.append(f"step {step.id}: looks irreversible but is declared {step.risk.value!r}")
        return problems

    def check_route(self, route: str) -> None:
        """Per-action check of a rendered route (templates resolved)."""
        if not self.policy.route_allowed(route):
            raise PolicyViolation(f"route {route!r} is not allowed")

    def irreversible_blocker(self, capability: Capability) -> str | None:
        """Why an irreversible step may not run now, or None if it may."""
        if not self.approvals.is_approved(capability, self.tenant):
            return "capability's current content is not approved"
        if not self.allow_irreversible:
            return "caller did not allow irreversible actions for this run"
        return None

    def allows_request(self, url: str) -> bool:
        return self.policy.url_allowed(url, self.base_url)
