"""Command line.

    python -m cua replay <capability_id> --base-url URL --param name=value ... [--allow-irreversible]
    python -m cua approve <capability_id> --by NAME [--note TEXT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cua.artifact.store import CapabilityLibrary
from cua.evidence.log import configure_logging
from cua.evidence.sink import echo
from cua.policy import ApprovalLedger, load_policy
from cua.profile import load_profile
from cua.runner import replay

REPO = Path(__file__).resolve().parents[1]
EXIT_CODES = {"success": 0, "failure": 1, "business_outcome": 2}


def _params(pairs: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--param expects name=value, got {name!r}")
        params[name] = value
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cua", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, default=REPO / "capabilities")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("replay", help="replay a capability deterministically (no LLM)")
    run.add_argument("capability_id")
    run.add_argument("--base-url", required=True, help="tenant base URL, e.g. http://127.0.0.1:8001")
    run.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    run.add_argument("--allow-irreversible", action="store_true",
                     help="permit irreversible steps in this run (the capability must also be approved)")
    run.add_argument("--unredacted-screenshots", action="store_true",
                     help="don't redact failure screenshots (only if the product's policy allows it)")
    run.add_argument("--evidence-dir", type=Path, default=REPO / "runs")
    run.add_argument("--request-id", help="caller's request id, for correlating logs")
    run.add_argument("--headed", action="store_true", help="show the browser")
    run.add_argument("--slow-mo", type=int, default=0, metavar="MS", help="delay each browser action (demos)")
    run.add_argument("--reveal", action="store_true", help="print real output values (default: masked)")

    approve = commands.add_parser("approve", help="approve a capability's current content (irreversible use)")
    approve.add_argument("capability_id")
    approve.add_argument("--by", required=True, help="who is approving")
    approve.add_argument("--note", default="")
    args = parser.parse_args(argv)

    configure_logging()
    library = CapabilityLibrary(args.library)
    product = args.capability_id.split(".", 1)[0]

    if args.command == "approve":
        capability = library.get(args.capability_id)
        entry = ApprovalLedger(args.library / "approvals.json").approve(capability, args.by, args.note)
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
    )
    echo({"run_id": outcome.run_id, "evidence_dir": str(outcome.evidence_dir),
          "result": outcome.result.model_dump(mode="json")}, outcome.registry, reveal=args.reveal)
    return EXIT_CODES[outcome.result.type]


if __name__ == "__main__":
    sys.exit(main())
