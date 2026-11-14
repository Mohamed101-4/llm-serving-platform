"""Drive load into the kind deployment and record how the worker count follows queue depth.

    uv run python scripts/k8s_autoscale_demo.py --rps 25 --duration 120

Port-forwards the router, runs the open-loop load generator, and every 2 s records the
worker Deployment's desired/ready replicas plus the router's /scaling signal. Keeps sampling
for --cooldown seconds after load stops to show scale-down. Saves one JSON file to results/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from llmserve.bench.loadgen import run, summarize
from llmserve.bench.sysinfo import git_commit, hardware
from llmserve.bench.workload import WorkloadConfig

NS = "llmserve"


def replicas() -> dict:
    out = subprocess.run(["kubectl", "-n", NS, "get", "deploy", "llmserve-worker", "-o", "json"],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)
    return {"desired": st["spec"].get("replicas"), "ready": st["status"].get("readyReplicas", 0)}


async def sampler(url: str, t0: float, stop: asyncio.Event, timeline: list) -> None:
    async with httpx.AsyncClient() as c:
        while not stop.is_set():
            point = {"t": round(time.perf_counter() - t0, 1), **await asyncio.to_thread(replicas)}
            try:
                s = (await c.get(f"{url}/scaling", timeout=2)).json()
                point.update(total_load=s["total_load"], healthy_workers=s["healthy_workers"])
            except Exception:
                pass
            timeline.append(point)
            try:
                await asyncio.wait_for(stop.wait(), 2.0)
            except asyncio.TimeoutError:
                pass


async def scenario(url: str, rps: float, duration: float, cooldown: float, wcfg: WorkloadConfig):
    timeline: list = []
    stop = asyncio.Event()
    t0 = time.perf_counter()
    samp = asyncio.create_task(sampler(url, t0, stop, timeline))
    samples, wall, _, _ = await run(url, rps, duration, "mock-model", wcfg)
    await asyncio.sleep(cooldown)
    stop.set()
    await samp
    return samples, wall, timeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rps", type=float, default=25)
    p.add_argument("--duration", type=float, default=120)
    p.add_argument("--cooldown", type=float, default=120)
    p.add_argument("--port", type=int, default=18000)
    args = p.parse_args()

    pf = subprocess.Popen(["kubectl", "-n", NS, "port-forward", "svc/llmserve-router", f"{args.port}:8000"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{args.port}"
    try:
        for _ in range(50):
            try:
                if httpx.get(f"{url}/ready", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        wcfg = WorkloadConfig(num_prefixes=64)
        samples, wall, timeline = asyncio.run(scenario(url, args.rps, args.duration, args.cooldown, wcfg))
    finally:
        pf.terminate()

    summary = summarize(samples, wall)
    result = {
        "label": "phase4-autoscale-kind",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "backend": "mock (kind, KEDA metrics-api on router /scaling)",
        "policy": "prefix_aware + admission (SLO TTFT 500 ms)",
        "hardware": hardware() | {"cluster": "kind single node in Docker Desktop (4 GB VM)"},
        "load": {"target_rps": args.rps, "duration_s": args.duration, "cooldown_s": args.cooldown, "arrival": "poisson"},
        "workload": wcfg.to_dict(),
        "summary": summary,
        "replica_timeline": timeline,
        "samples": [asdict(s) for s in samples],
    }
    out = Path("results") / f"phase4-autoscale-kind-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2))
    peak = max((pt["desired"] or 0) for pt in timeline)
    print(json.dumps({"summary": {k: summary[k] for k in ("requests_sent", "requests_ok", "requests_rejected_429",
                                                         "requests_failed", "throughput_req_s", "ttft_ms", "latency_ms")},
                      "replicas_start": timeline[0], "replicas_peak_desired": peak, "replicas_end": timeline[-1]}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
