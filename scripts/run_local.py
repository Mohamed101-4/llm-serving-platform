"""Start N mock workers and the router on this machine (no GPU needed).

    uv run python scripts/run_local.py --workers 4 --policy round_robin

Ctrl+C stops everything.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time

import httpx


def wait_ready(url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{url} did not become ready")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--base-port", type=int, default=8001)
    p.add_argument("--router-port", type=int, default=8000)
    p.add_argument("--policy", default="round_robin")
    args = p.parse_args()
    # treat `kill`/SIGTERM like Ctrl+C so child servers are always cleaned up
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    procs, urls = [], []
    try:
        for i in range(args.workers):
            port = args.base_port + i
            procs.append(subprocess.Popen([sys.executable, "-m", "llmserve.mock_worker", "--port", str(port)]))
            urls.append(f"http://127.0.0.1:{port}")
        for u in urls:
            wait_ready(u)
        procs.append(subprocess.Popen([sys.executable, "-m", "llmserve.router", "--port", str(args.router_port),
                                       "--workers", ",".join(urls), "--policy", args.policy]))
        wait_ready(f"http://127.0.0.1:{args.router_port}")
        print(f"router on :{args.router_port} ({args.policy}) -> {len(urls)} mock workers. Ctrl+C to stop.", flush=True)
        while all(pr.poll() is None for pr in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            pr.wait(timeout=10)


if __name__ == "__main__":
    main()
