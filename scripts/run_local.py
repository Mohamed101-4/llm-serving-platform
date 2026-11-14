"""Start N mock workers and the router on this machine (no GPU needed).

    uv run python scripts/run_local.py --workers 4 --policy round_robin
    uv run python scripts/run_local.py --workers 3 --prefill-workers 1 --prefill-interference   # P/D split
    uv run python scripts/run_local.py --workers 4 --policy prefix_aware --slo-ttft-ms 500 --respawn-after 8
        # then open http://127.0.0.1:8000/ui (crashed workers come back after 8 s)

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
    p.add_argument("--slo-ttft-ms", type=float, default=None, help="enable admission control")
    p.add_argument("--prefill-workers", type=int, default=0,
                   help="extra workers used only for prefill (disaggregated prefill/decode)")
    p.add_argument("--prefill-interference", action="store_true",
                   help="mock workers model prefill stalling decode on a shared GPU")
    p.add_argument("--respawn-after", type=float, default=None,
                   help="restart crashed workers after this many seconds; enables the dashboard's kill button")
    args = p.parse_args()
    # treat `kill`/SIGTERM like Ctrl+C so child servers are always cleaned up
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    procs, urls, prefill_urls = [], [], []
    wflags = ["--prefill-interference"] if args.prefill_interference else []
    if args.respawn_after is not None:
        wflags.append("--chaos")
    cmds = []
    try:
        for i in range(args.workers + args.prefill_workers):
            port = args.base_port + i
            cmds.append([sys.executable, "-m", "llmserve.mock_worker", "--port", str(port)] + wflags)
            procs.append(subprocess.Popen(cmds[-1]))
            (urls if i < args.workers else prefill_urls).append(f"http://127.0.0.1:{port}")
        for u in urls + prefill_urls:
            wait_ready(u)
        router_cmd = [sys.executable, "-m", "llmserve.router", "--port", str(args.router_port),
                      "--workers", ",".join(urls), "--policy", args.policy]
        if args.slo_ttft_ms:
            router_cmd += ["--slo-ttft-ms", str(args.slo_ttft_ms)]
        if prefill_urls:
            router_cmd += ["--prefill-workers", ",".join(prefill_urls)]
        procs.append(subprocess.Popen(router_cmd))
        wait_ready(f"http://127.0.0.1:{args.router_port}")
        pd = f" + {len(prefill_urls)} prefill" if prefill_urls else ""
        print(f"router on :{args.router_port} ({args.policy}) -> {len(urls)} mock workers{pd}. Ctrl+C to stop.", flush=True)
        # Keep running while the router is up. A dead worker is the router's problem to
        # handle (health checks + failover), so it must not take the whole stack down.
        router, died = procs[-1], {}
        while router.poll() is None:
            for i, pr in enumerate(procs[:-1]):
                if pr.poll() is not None and i not in died:
                    print(f"worker {(urls + prefill_urls)[i]} exited with code {pr.returncode}", flush=True)
                    died[i] = time.time()
            if args.respawn_after is not None:
                for i, t in list(died.items()):
                    if time.time() - t >= args.respawn_after:
                        procs[i] = subprocess.Popen(cmds[i])
                        print(f"worker {(urls + prefill_urls)[i]} restarted", flush=True)
                        del died[i]
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second signal must not abort cleanup
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for pr in procs:
            pr.terminate()
        for pr in procs:
            pr.wait(timeout=10)


if __name__ == "__main__":
    main()
