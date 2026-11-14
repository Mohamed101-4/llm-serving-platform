"""Router + mock workers over real HTTP, no GPU required."""

import asyncio

import httpx

from llmserve.bench.loadgen import run, summarize
from llmserve.bench.workload import WorkloadConfig
from llmserve.mock_worker.server import create_app as worker_app
from llmserve.mock_worker.sim import SimConfig
from llmserve.router.app import create_app as router_app

FAST = SimConfig(decode_base_ms=1, decode_ms_per_seq=0, prefill_ms_per_token=0.001)


def body(stream: bool):
    return {"model": "mock-model", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 5, "stream": stream}


def test_non_streaming_round_robin(serve):
    workers = [serve(worker_app(FAST)) for _ in range(2)]
    router = serve(router_app([w.url for w in workers]))
    seen = []
    for _ in range(4):
        r = httpx.post(f"{router.url}/v1/chat/completions", json=body(False), timeout=10)
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"].count("tok") == 5
        seen.append(r.headers["x-llmserve-worker"])
    assert seen == [workers[0].url, workers[1].url] * 2


def test_streaming_relays_sse(serve):
    worker = serve(worker_app(FAST))
    router = serve(router_app([worker.url]))
    with httpx.stream("POST", f"{router.url}/v1/chat/completions", json=body(True), timeout=10) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert len(lines) == 5 + 2  # 5 tokens, final chunk, [DONE]


def test_dead_worker_returns_502(serve):
    router = serve(router_app(["http://127.0.0.1:9"]))  # nothing listens on port 9
    r = httpx.post(f"{router.url}/v1/chat/completions", json=body(False), timeout=10)
    assert r.status_code == 502


def test_loadgen_end_to_end(serve):
    workers = [serve(worker_app(FAST)) for _ in range(2)]
    router = serve(router_app([w.url for w in workers]))
    cfg = WorkloadConfig(output_tokens_median=8, max_output_tokens=16, seed=1)
    samples, wall, _ = asyncio.run(run(router.url, rps=40, duration=1.0, model="mock-model", wcfg=cfg))
    s = summarize(samples, wall)
    assert s["requests_sent"] > 10
    assert s["requests_ok"] == s["requests_sent"]
    assert s["latency_ms"]["p99"] >= s["latency_ms"]["p50"] > 0
    assert s["ttft_ms"]["p50"] is not None
    assert set(s["requests_per_worker"]) == {w.url for w in workers}
