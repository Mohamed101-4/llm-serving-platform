"""Phase 6: disaggregated prefill/decode (vLLM kv_transfer_params protocol) on mock workers."""

import asyncio
import time

import httpx

from llmserve.mock_worker.server import create_app as worker_app
from llmserve.mock_worker.sim import SimConfig, SimWorker
from llmserve.router.app import create_app as router_app

FAST = SimConfig(decode_base_ms=1, decode_ms_per_seq=0, prefill_ms_per_token=0.001)


def chat_body(stream=True, max_tokens=5):
    return {"model": "m", "max_tokens": max_tokens, "stream": stream,
            "messages": [{"role": "system", "content": "shared system prompt " * 40}, {"role": "user", "content": "hi"}]}


def stream_tokens(url, body):
    with httpx.stream("POST", f"{url}/v1/chat/completions", json=body, timeout=10) as r:
        assert r.status_code == 200
        return r.headers["x-llmserve-worker"], [ln for ln in r.iter_lines() if ln.startswith("data: {")]


def test_worker_prefill_then_decode_protocol(serve):
    p, d = serve(worker_app(FAST)), serve(worker_app(FAST))
    body = chat_body(stream=False)
    r = httpx.post(f"{p.url}/v1/chat/completions",
                   json={**body, "max_tokens": 1, "kv_transfer_params": {"do_remote_decode": True}}).json()
    kvp = r["kv_transfer_params"]
    assert kvp["do_remote_prefill"] is True and kvp["prompt_tokens"] > 100
    out = httpx.post(f"{d.url}/v1/chat/completions", json={**body, "kv_transfer_params": kvp}).json()
    assert out["choices"][0]["message"]["content"].count("tok") == 5
    # prefill happened only on the prefill worker; the decode worker only decoded
    pw = httpx.get(f"{p.url}/metrics").text
    dw = httpx.get(f"{d.url}/metrics").text
    assert 'vllm:generation_tokens_total{model_name="mock-model"} 0' in pw
    assert 'vllm:prompt_tokens_total{model_name="mock-model"} 0' in dw
    assert 'vllm:generation_tokens_total{model_name="mock-model"} 5' in dw


def test_router_disaggregates(serve):
    prefill = [serve(worker_app(FAST)) for _ in range(2)]
    decode = [serve(worker_app(FAST)) for _ in range(2)]
    router = serve(router_app([w.url for w in decode], "least_load", scrape_interval=0.05,
                              prefill_urls=[w.url for w in prefill]))
    used = set()
    for _ in range(6):
        worker, chunks = stream_tokens(router.url, chat_body())
        assert len(chunks) == 6  # 5 tokens + finish chunk
        used.add(worker)
    assert used <= {w.url for w in decode}
    roles = {w["url"]: w["role"] for w in httpx.get(f"{router.url}/workers").json()}
    assert sorted(roles.values()) == ["decode", "decode", "prefill", "prefill"]
    gen = {w.url: httpx.get(f"{w.url}/metrics").text for w in prefill}
    assert all('vllm:generation_tokens_total{model_name="mock-model"} 0' in t for t in gen.values())
    # same system prompt -> prefix-aware prefill routing keeps it on one prefill worker
    prompts = [int(t.split('vllm:prompt_tokens_total{model_name="mock-model"} ')[1].split()[0]) for t in gen.values()]
    assert sorted(prompts)[0] == 0


def test_prefill_worker_down_fails_over(serve):
    p = serve(worker_app(FAST))
    d = serve(worker_app(FAST))
    router = serve(router_app([d.url], prefill_urls=["http://127.0.0.1:9", p.url], scrape_interval=30))
    for _ in range(3):
        stream_tokens(router.url, chat_body())


def test_prefill_interference_stalls_decode():
    """A long prefill on the same worker delays another sequence's decode steps."""
    async def go(interference):
        w = SimWorker(SimConfig(decode_base_ms=1, decode_ms_per_seq=0, prefill_ms_per_token=0.1,
                                prefill_interference=interference))
        times = []

        async def decoder():
            async for _ in w.generate("x" * 4, 30):
                times.append(time.perf_counter())

        async def big_prefill():
            await asyncio.sleep(0.01)
            await w.prefill_only("y" * 4 * 2000)  # 2000 tokens -> 200 ms prefill

        await asyncio.gather(decoder(), big_prefill())
        return max(b - a for a, b in zip(times, times[1:]))

    assert asyncio.run(go(False)) < 0.05
    assert asyncio.run(go(True)) > 0.15
