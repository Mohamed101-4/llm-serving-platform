"""Open-loop load generator.

Requests are sent at Poisson arrival times regardless of whether earlier requests
have finished (open loop), so an overloaded server shows up as growing latency
instead of the client quietly slowing down.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from .sysinfo import git_commit, gpu_memory_mib, hardware
from .workload import Workload, WorkloadConfig


@dataclass
class Sample:
    prefix_id: int
    start: float
    ttft: float | None = None
    latency: float | None = None
    output_tokens: int = 0
    status: int = 0
    worker: str | None = None
    error: str | None = None


async def one_request(client: httpx.AsyncClient, url: str, model: str, req, t0: float) -> Sample:
    s = Sample(prefix_id=req.prefix_id, start=time.perf_counter() - t0)
    body = {"model": model, "messages": req.messages, "max_tokens": req.max_tokens,
            "stream": True, "ignore_eos": True, "temperature": 0.0}
    begin = time.perf_counter()
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as resp:
            s.status = resp.status_code
            s.worker = resp.headers.get("x-llmserve-worker")
            if resp.status_code != 200:
                await resp.aread()
                s.error = resp.text[:200]
                return s
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {}) if chunk.get("choices") else {}
                if delta.get("content"):
                    if s.ttft is None:
                        s.ttft = time.perf_counter() - begin
                    s.output_tokens += 1
        s.latency = time.perf_counter() - begin
    except Exception as e:  # network errors count as failures, not crashes
        s.error = repr(e)[:200]
    return s


def pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize(samples: list[Sample], wall: float) -> dict:
    ok = [s for s in samples if s.status == 200 and s.latency is not None]
    lat = [s.latency for s in ok]
    ttft = [s.ttft for s in ok if s.ttft is not None]
    tpot = [(s.latency - s.ttft) / (s.output_tokens - 1) for s in ok if s.ttft and s.output_tokens > 1]
    out_tokens = sum(s.output_tokens for s in ok)
    per_worker: dict[str, int] = {}
    for s in ok:
        per_worker[s.worker or "?"] = per_worker.get(s.worker or "?", 0) + 1
    ms = lambda v: None if v is None else round(v * 1000, 2)  # noqa: E731
    return {
        "requests_sent": len(samples),
        "requests_ok": len(ok),
        "requests_rejected_429": sum(1 for s in samples if s.status == 429),
        "requests_failed": sum(1 for s in samples if s.status != 200 and s.status != 429),
        "wall_time_s": round(wall, 3),
        "throughput_req_s": round(len(ok) / wall, 3),
        "output_tokens_per_s": round(out_tokens / wall, 2),
        "latency_ms": {"p50": ms(pct(lat, 50)), "p95": ms(pct(lat, 95)), "p99": ms(pct(lat, 99)), "mean": ms(float(np.mean(lat)) if lat else None)},
        "ttft_ms": {"p50": ms(pct(ttft, 50)), "p95": ms(pct(ttft, 95)), "p99": ms(pct(ttft, 99))},
        "tpot_ms": {"p50": ms(pct(tpot, 50)), "p95": ms(pct(tpot, 95))},
        "requests_per_worker": per_worker,
    }


async def gpu_sampler(stop: asyncio.Event, peaks: dict) -> None:
    while not stop.is_set():
        for g in await asyncio.to_thread(gpu_memory_mib) or []:
            peaks[g["index"]] = max(peaks.get(g["index"], 0.0), g["used_mib"])
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


async def run(url: str, rps: float, duration: float, model: str, wcfg: WorkloadConfig) -> tuple[list[Sample], float, dict]:
    workload = Workload(wcfg)
    rng = np.random.default_rng(wcfg.seed + 100)
    limits = httpx.Limits(max_connections=5000, max_keepalive_connections=1000)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0), limits=limits) as client:
        stop = asyncio.Event()
        peaks: dict = {}
        sampler = asyncio.create_task(gpu_sampler(stop, peaks))
        tasks = []
        t0 = time.perf_counter()
        next_at = 0.0
        while next_at < duration:
            delay = next_at - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one_request(client, url, model, workload.sample(), t0)))
            next_at += rng.exponential(1.0 / rps)
        samples = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
        stop.set()
        await sampler
    return list(samples), wall, peaks


def main() -> None:
    p = argparse.ArgumentParser(description="Open-loop Poisson load generator")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--rps", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=60.0, help="seconds of arrivals")
    p.add_argument("--model", default="mock-model")
    p.add_argument("--label", required=True, help="name for this run, e.g. phase1-rr-mock")
    p.add_argument("--backend", default="mock", help="mock | vllm (recorded in results)")
    p.add_argument("--policy", default=None, help="router policy under test (recorded)")
    p.add_argument("--num-prefixes", type=int, default=WorkloadConfig.num_prefixes)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    wcfg = WorkloadConfig(num_prefixes=args.num_prefixes, seed=args.seed)
    samples, wall, peaks = asyncio.run(run(args.url, args.rps, args.duration, args.model, wcfg))
    summary = summarize(samples, wall)
    summary["gpu_memory_peak_mib"] = {str(k): v for k, v in peaks.items()} or None

    result = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "backend": args.backend,
        "policy": args.policy,
        "model": args.model,
        "hardware": hardware(),
        "load": {"target_rps": args.rps, "duration_s": args.duration, "arrival": "poisson"},
        "workload": wcfg.to_dict(),
        "summary": summary,
        "samples": [asdict(s) for s in samples],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "workload")}, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
