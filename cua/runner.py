"""One replay run end to end: sign on, run the capability, write evidence.

A run owns: a run id, a SecretRegistry (so every value it touches is masked on the way out),
an evidence directory (events.jsonl, result.json, failure screenshots), and one browser session.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cua.artifact.store import CapabilityLibrary
from cua.evidence.log import attach_handler, get_logger, run_context
from cua.evidence.sink import RunEvidence
from cua.profile import AppProfile
from cua.replay.engine import ReplayEngine
from cua.replay.result import Result
from cua.security.masking import SecretRegistry, use_registry
from cua.surface.web import WebSurface

log = get_logger(__name__)


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    result: Result  # in-memory result holds real output values; mask before displaying/persisting
    evidence_dir: Path
    registry: SecretRegistry


def new_run_id() -> str:
    return f"run_{datetime.now(UTC):%Y%m%dT%H%M%S}_{secrets.token_hex(3)}"


def replay(
    capability_id: str,
    params: Mapping[str, object],
    *,
    base_url: str,
    library: CapabilityLibrary,
    profile: AppProfile,
    evidence_root: Path,
    request_id: str | None = None,
    headless: bool = True,
) -> RunOutcome:
    capability = library.get(capability_id)
    if capability.app.product != profile.product:
        raise ValueError(f"{capability_id} targets {capability.app.product!r}, not {profile.product!r}")
    sign_on = library.get(profile.sign_on.capability) if profile.sign_on else None
    credentials = profile.sign_on_params()  # fail fast on missing config, before opening a browser

    run_id = new_run_id()
    registry = SecretRegistry()
    evidence = RunEvidence(evidence_root, run_id, registry)
    with (
        attach_handler(evidence.log_handler()),
        use_registry(registry),
        run_context(run_id, request_id=request_id, mode="replay"),
    ):
        log.info("run.started", capability=capability_id, base_url=base_url)
        result: Result | None = ReplayEngine.check_inputs(capability, params)
        if result is None:
            with WebSurface.launch(base_url, headless=headless) as surface:
                engine = ReplayEngine(surface, registry=registry, evidence=evidence)
                if sign_on is not None:
                    signed_on = engine.run(sign_on, credentials)
                    if signed_on.type != "success":
                        result = signed_on
                if result is None:
                    result = engine.run(capability, params)
        evidence.save_json("result.json", result.model_dump(mode="json"))
        log.info("run.finished", result=result.type)
    return RunOutcome(run_id, result, evidence.dir, registry)
