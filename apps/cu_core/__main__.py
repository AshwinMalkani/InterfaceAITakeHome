"""Run one tenant of the target app: `python -m apps.cu_core --tenant alpha --port 8001`."""

from __future__ import annotations

import argparse
import os

import uvicorn

from apps.cu_core.server import create_app
from apps.cu_core.tenants import TENANTS

# Synthetic demo credential for a synthetic app. Override with CU_CORE_PASSWORD; it is also
# registered as a secret by the masking layer, so it never appears in logs or evidence.
DEFAULT_DEMO_PASSWORD = "demo-only-password"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", choices=sorted(TENANTS), default="alpha")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--session-timeout", type=float, default=300, help="idle timeout in seconds")
    args = parser.parse_args()

    app = create_app(
        TENANTS[args.tenant],
        password=os.environ.get("CU_CORE_PASSWORD", DEFAULT_DEMO_PASSWORD),
        session_timeout_s=args.session_timeout,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
