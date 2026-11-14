"""Shared fixtures: run real uvicorn servers on free ports inside the test process."""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    def __init__(self, app):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> "ServerThread":
        self.thread.start()
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                if httpx.get(f"{self.url}/health", timeout=0.5).status_code == 200:
                    return self
            except httpx.HTTPError:
                time.sleep(0.05)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def serve():
    started: list[ServerThread] = []

    def _serve(app) -> ServerThread:
        st = ServerThread(app).start()
        started.append(st)
        return st

    yield _serve
    for st in started:
        st.stop()
