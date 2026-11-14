import asyncio

import httpx

from llmserve.mock_worker.server import create_app as worker_app
from llmserve.mock_worker.sim import SimConfig
from llmserve.router.app import create_app as router_app
from llmserve.router.discovery import DnsDiscovery, resolve, sync_workers
from llmserve.router.policies import Worker


def test_sync_workers_updates_in_place():
    workers = [Worker("http://a:1", inflight=3), Worker("http://b:1")]
    same_list = workers
    added, removed = sync_workers(workers, {"http://a:1", "http://c:1"})
    assert added == {"http://c:1"} and removed == {"http://b:1"}
    assert same_list is workers
    assert [w.url for w in workers] == ["http://a:1", "http://c:1"]
    assert workers[0].inflight == 3  # existing worker state is kept


def test_resolve_localhost():
    assert asyncio.run(resolve("localhost", 8123)) == {"http://127.0.0.1:8123"}


def test_dns_discovery_finds_worker():
    workers: list[Worker] = []
    asyncio.run(DnsDiscovery(workers, "localhost:8123").refresh())
    assert [w.url for w in workers] == ["http://127.0.0.1:8123"]


def test_router_via_dns_and_scaling_endpoint(serve):
    worker = serve(worker_app(SimConfig(decode_base_ms=1, decode_ms_per_seq=0)))
    router = serve(router_app([], worker_dns=f"localhost:{worker.port}"))
    assert httpx.get(f"{router.url}/ready").status_code == 200
    r = httpx.post(f"{router.url}/v1/chat/completions", timeout=10,
                   json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})
    assert r.status_code == 200
    s = httpx.get(f"{router.url}/scaling").json()
    assert s["healthy_workers"] == 1 and s["total_load"] >= 0


def test_not_ready_without_workers(serve):
    router = serve(router_app([]))
    assert httpx.get(f"{router.url}/health").status_code == 200
    assert httpx.get(f"{router.url}/ready").status_code == 503
