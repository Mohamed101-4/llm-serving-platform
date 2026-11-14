"""Phase 3 behaviour tests: failover when a worker dies, bounded latency under 2x overload."""

import asyncio
import os
import signal
import subprocess
import sys
import time

import httpx

from conftest import free_port
from llmserve.bench.loadgen import run, summarize
from llmserve.bench.workload import WorkloadConfig
from llmserve.mock_worker.server import create_app as worker_app
from llmserve.mock_worker.sim import SimConfig
from llmserve.router.app import create_app as router_app


def body():
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}


def test_failover_skips_dead_worker(serve):
    alive = serve(worker_app(SimConfig(decode_base_ms=1, decode_ms_per_seq=0)))
    router = serve(router_app(["http://127.0.0.1:9", alive.url]))
    codes = [httpx.post(f"{router.url}/v1/chat/completions", json=body(), timeout=10).status_code for _ in range(6)]
    assert codes == [200] * 6  # first request retries onto the live worker; later ones avoid the dead one
    dead = httpx.get(f"{router.url}/workers").json()[0]
    assert dead["healthy"] is False


def test_worker_killed_mid_load(serve):
    """SIGKILL one of three workers while load is running: only its in-flight requests fail."""
    procs, urls = [], []
    for _ in range(3):
        port = free_port()
        procs.append(subprocess.Popen([sys.executable, "-m", "llmserve.mock_worker", "--port", str(port)]))
        urls.append(f"http://127.0.0.1:{port}")
    try:
        for u in urls:
            deadline = time.time() + 15
            while True:
                try:
                    httpx.get(f"{u}/health", timeout=0.5)
                    break
                except httpx.HTTPError:
                    assert time.time() < deadline
                    time.sleep(0.1)
        router = serve(router_app(urls, "least_load", scrape_interval=0.1))
        cfg = WorkloadConfig(output_tokens_median=20, max_output_tokens=30, seed=3)

        async def scenario():
            task = asyncio.create_task(run(router.url, rps=30, duration=4.0, model="m", wcfg=cfg))
            await asyncio.sleep(1.5)
            os.kill(procs[0].pid, signal.SIGKILL)
            return await task

        samples, wall, _, _ = asyncio.run(scenario())
        late = [s for s in samples if s.start > 2.0]  # sent 0.5 s after the kill
        failed = [s for s in samples if s.status != 200 or s.error]
        assert late and all(s.status == 200 and not s.error for s in late), "requests after detection must succeed"
        assert len(failed) <= 15, f"only requests in flight on the dead worker may fail, got {len(failed)}"
        assert not any(w["healthy"] for w in httpx.get(f"{router.url}/workers").json() if w["url"] == urls[0])
    finally:
        for p in procs:
            p.kill()
            p.wait()


def overload_run(serve, slo_ttft_ms):
    # 2 workers x 4 slots, 40 tokens x 5 ms = 0.2 s per request -> capacity ~40 req/s. Offer 80 req/s.
    cfg = SimConfig(max_num_seqs=4, decode_base_ms=5, decode_ms_per_seq=0, prefill_ms_per_token=0.001)
    workers = [serve(worker_app(cfg)) for _ in range(2)]
    router = serve(router_app([w.url for w in workers], "least_load", scrape_interval=0.05,
                              slo_ttft_ms=slo_ttft_ms, default_slots=4))
    wcfg = WorkloadConfig(output_tokens_median=40, output_tokens_sigma=0.01, max_output_tokens=40, seed=5)
    samples, wall, _, _ = asyncio.run(run(router.url, rps=80, duration=4.0, model="m", wcfg=wcfg))
    return summarize(samples, wall)


def test_latency_bounded_under_2x_overload(serve):
    slo = 300
    without = overload_run(serve, slo_ttft_ms=None)
    with_ac = overload_run(serve, slo_ttft_ms=slo)
    # Without admission control the queue grows for the whole run.
    assert without["ttft_ms"]["p99"] > 4 * slo
    # With it, excess requests get a fast 429 and admitted requests stay near the SLO.
    assert with_ac["requests_rejected_429"] > 0
    assert with_ac["ttft_ms"]["p99"] < 2 * slo
    assert with_ac["latency_ms"]["p99"] < without["latency_ms"]["p99"] / 2
    assert with_ac["throughput_req_s"] > 0.7 * 40  # still serving close to capacity
