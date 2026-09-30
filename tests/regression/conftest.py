"""Target app instances for replay regression cases: both tenants, in-process, on free ports."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import uvicorn

from apps.cu_core.server import create_app
from apps.cu_core.tenants import TENANTS

TEST_USERNAME = "teller1"
TEST_PASSWORD = "regression-only-password"


@dataclass
class TargetApp:
    tenant: str
    base_url: str

    def _post(self, path: str, body: Any = None) -> None:
        data = json.dumps(body).encode() if body is not None else b""
        request = urllib.request.Request(self.base_url + path, data=data, method="POST",
                                         headers={"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=5):
            pass

    def reset(self) -> None:
        self._post("/__reset")

    def arm(self, faults: list[dict[str, Any]]) -> None:
        if faults:
            self._post("/__faults", faults)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


@pytest.fixture(scope="session")
def target_apps() -> Iterator[dict[str, TargetApp]]:
    previous = {k: os.environ.get(k) for k in ("CU_CORE_USERNAME", "CU_CORE_PASSWORD")}
    os.environ["CU_CORE_USERNAME"] = TEST_USERNAME
    os.environ["CU_CORE_PASSWORD"] = TEST_PASSWORD

    servers: list[tuple[uvicorn.Server, threading.Thread]] = []
    apps: dict[str, TargetApp] = {}
    for key, tenant in TENANTS.items():
        port = _free_port()
        config = uvicorn.Config(create_app(tenant, password=TEST_PASSWORD), host="127.0.0.1", port=port,
                                log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        servers.append((server, thread))
        apps[key] = TargetApp(key, f"http://127.0.0.1:{port}")
    deadline = time.monotonic() + 10
    while not all(server.started for server, _ in servers):
        if time.monotonic() > deadline:
            raise RuntimeError("target apps did not start")
        time.sleep(0.05)

    yield apps

    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=5)
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
