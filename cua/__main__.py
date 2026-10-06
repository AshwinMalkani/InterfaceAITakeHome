"""Command line.

python -m cua replay <capability_id> --base-url URL --param name=value ... [--allow-irreversible]
python -m cua approve <capability_id> --by NAME [--note TEXT]
python -m cua discover <goal.yaml> --base-url URL --param name=value ... [--save]
python -m cua console [--port 8090]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cua.artifact.binding import load_binding, specialize
from cua.artifact.store import CapabilityLibrary
from cua.evidence.log import configure_logging
from cua.evidence.sink import echo
from cua.hitl.store import HitlStore
from cua.policy import ApprovalLedger, load_policy
from cua.profile import load_profile
from cua.runner import replay

REPO = Path(__file__).resolve().parents[1]
EXIT_CODES = {"success": 0, "failure": 1, "business_outcome": 2, "escalated": 3}
HITL_STORE = REPO / "runs" / "hitl.db"


def _params(pairs: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--param expects name=value, got {name!r}")
        params[name] = value
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cua", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--library", type=Path, default=REPO / "capabilities")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("replay", help="replay a capability deterministically (no LLM)")
    run.add_argument("capability_id")
    run.add_argument("--base-url", required=True, help="tenant base URL, e.g. http://127.0.0.1:8001")
    run.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    run.add_argument(
        "--allow-irreversible",
        action="store_true",
        help="permit irreversible steps in this run (the capability must also be approved)",
    )
    run.add_argument(
        "--unredacted-screenshots",
        action="store_true",
        help="don't redact failure screenshots (only if the product's policy allows it)",
    )
    run.add_argument("--tenant", help="apply this tenant's binding (config/tenants/<tenant>.yaml)")
    run.add_argument(
        "--hitl",
        action="store_true",
        help="escalate to a human operator instead of failing (see `cua console`)",
    )
    run.add_argument("--cdp-port", type=int, default=None, help="expose the live session for remote takeover")
    run.add_argument("--handoff-timeout", type=float, default=900, help="seconds to wait for an operator")
    run.add_argument("--evidence-dir", type=Path, default=REPO / "runs")
    run.add_argument("--request-id", help="caller's request id, for correlating logs")
    run.add_argument("--headed", action="store_true", help="show the browser")
    run.add_argument("--slow-mo", type=int, default=0, metavar="MS", help="delay each browser action (demos)")
    run.add_argument("--reveal", action="store_true", help="print real output values (default: masked)")

    disc = commands.add_parser(
        "discover", help="LLM-driven discovery of a goal -> verified capability artifact"
    )
    disc.add_argument("goal", type=Path, help="goal spec, e.g. goals/cu_core/read_savings_balance.yaml")
    disc.add_argument("--base-url", required=True)
    disc.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    disc.add_argument("--model", default=None, help="model id (default: $CUA_MODEL or claude-opus-5-5)")
    disc.add_argument("--max-actions", type=int, default=30)
    disc.add_argument("--evidence-dir", type=Path, default=REPO / "runs")
    disc.add_argument(
        "--save", action="store_true", help="add the verified artifact to the capability library"
    )
    disc.add_argument("--overwrite", action="store_true", help="with --save, replace an existing artifact")
    disc.add_argument("--headed", action="store_true")
    disc.add_argument(
        "--hitl",
        action="store_true",
        help="let the agent ask a human, and require approval for irreversible actions (see `cua console`)",
    )
    disc.add_argument("--cdp-port", type=int, default=None)
    disc.add_argument("--handoff-timeout", type=float, default=900)

    console = commands.add_parser("console", help="operator console for human handoff")
    console.add_argument("--port", type=int, default=8090)
    console.add_argument("--store", type=Path, default=HITL_STORE)
    console.add_argument("--evidence-dir", type=Path, default=REPO / "runs")

    approve = commands.add_parser("approve", help="approve a capability's current content (irreversible use)")
    approve.add_argument("capability_id")
    approve.add_argument("--by", required=True, help="who is approving")
    approve.add_argument("--note", default="")
    approve.add_argument("--tenant", help="approve the capability as specialized for this tenant")
    args = parser.parse_args(argv)

    configure_logging()
    library = CapabilityLibrary(args.library)
    if args.command == "discover":  # the goal file names its product
        return _discover(args, library)
    if args.command == "console":
        return _console(args)
    product = args.capability_id.split(".", 1)[0]

    if args.command == "approve":
        capability = library.get(args.capability_id)
        if args.tenant:
            capability = specialize(capability, load_binding(args.tenant, product))
        ledger = ApprovalLedger(args.library / "approvals.json")
        entry = ledger.approve(capability, args.by, args.note, tenant=args.tenant)
        echo({"approved": capability.id, **entry.model_dump(mode="json")})
        return 0

    outcome = replay(
        args.capability_id,
        _params(args.param),
        base_url=args.base_url,
        library=library,
        profile=load_profile(product),
        policy=load_policy(product),
        approvals=ApprovalLedger(args.library / "approvals.json"),
        evidence_root=args.evidence_dir,
        allow_irreversible=args.allow_irreversible,
        unredacted_screenshots=args.unredacted_screenshots,
        request_id=args.request_id,
        headless=not args.headed,
        slow_mo_ms=args.slow_mo,
        hitl=HitlStore(HITL_STORE) if args.hitl else None,
        cdp_port=args.cdp_port,
        unclaimed_timeout_s=args.handoff_timeout,
        tenant=args.tenant,
    )
    echo(
        {
            "run_id": outcome.run_id,
            "evidence_dir": str(outcome.evidence_dir),
            "result": outcome.result.model_dump(mode="json"),
        },
        outcome.registry,
        reveal=args.reveal,
    )
    return EXIT_CODES[outcome.result.type]


def _console(args: argparse.Namespace) -> int:
    import uvicorn

    from cua.hitl.console import create_console

    app = create_console(HitlStore(args.store), args.evidence_dir)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def _discover(args: argparse.Namespace, library: CapabilityLibrary) -> int:
    from cua.agent.goal import load_goal
    from cua.agent.llm import AnthropicModel
    from cua.discovery import discover

    goal = load_goal(args.goal)
    outcome = discover(
        goal,
        _params(args.param),
        model=AnthropicModel(args.model),
        base_url=args.base_url,
        library=library,
        profile=load_profile(goal.app.product),
        policy=load_policy(goal.app.product),
        evidence_root=args.evidence_dir,
        headless=not args.headed,
        max_actions=args.max_actions,
        hitl=HitlStore(HITL_STORE) if args.hitl else None,
        cdp_port=args.cdp_port,
        unclaimed_timeout_s=args.handoff_timeout,
    )
    summary: dict[str, object] = {
        "run_id": outcome.run_id,
        "status": outcome.status,
        "verification_scope": outcome.verification_scope,
        "interventions": [i.model_dump(mode="json") for i in outcome.interventions],
        "reason": outcome.reason,
        "verified": outcome.verified,
        "evidence_dir": str(outcome.evidence_dir),
    }
    if outcome.capability is not None and args.save:
        if not outcome.verified:
            raise SystemExit("refusing to save: the artifact did not replay successfully")
        if library.path_for(goal.id).exists() and not args.overwrite:
            raise SystemExit(f"{library.path_for(goal.id)} exists; pass --overwrite to replace it")
        summary["saved_to"] = str(library.save(outcome.capability))
    echo(summary)
    return 0 if outcome.verified else 1


if __name__ == "__main__":
    sys.exit(main())
