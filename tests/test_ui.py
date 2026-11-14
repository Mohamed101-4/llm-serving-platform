"""Dashboard endpoints: page, state, runtime policy/admission switches, load generator, kill."""

import subprocess
import sys
import time

import httpx

from conftest import free_port
from llmserve.mock_worker.server import create_app as worker_app
from llmserve.mock_worker.sim import SimConfig
from llmserve.router.app import create_app as router_app

FAST = SimConfig(decode_base_ms=1, decode_ms_per_seq=0, prefill_ms_per_token=0.001)


def test_page_and_state(serve):
    ws = [serve(worker_app(FAST)) for _ in range(2)]
    router = serve(router_app([w.url for w in ws], scrape_interval=0.05))
    page = httpx.get(f"{router.url}/ui")
    assert page.status_code == 200 and "llmserve" in page.text
    s = httpx.get(f"{router.url}/ui/state").json()
    assert s["policy"] == "round_robin" and "prefix_aware" in s["policies"]
    assert [w["role"] for w in s["workers"]] == ["mixed", "mixed"]
    assert s["admission"]["enabled"] is False and s["events"] == []


def test_runtime_switches_and_load(serve):
    ws = [serve(worker_app(FAST)) for _ in range(2)]
    router = serve(router_app([w.url for w in ws], scrape_interval=0.05))
    u = router.url
    assert httpx.post(f"{u}/ui/policy", json={"name": "least_load"}).json() == {"policy": "least_load"}
    assert httpx.post(f"{u}/ui/policy", json={"name": "nope"}).status_code == 400
    assert httpx.post(f"{u}/ui/admission", json={"enabled": True, "slo_ms": 800}).json()["enabled"]
    s = httpx.get(f"{u}/ui/state").json()
    assert s["policy"] == "least_load" and s["admission"]["slo_ms"] == 800
    httpx.post(f"{u}/ui/load", json={"rps": 30})
    time.sleep(1.5)
    httpx.post(f"{u}/ui/load", json={"rps": 0})
    s = httpx.get(f"{u}/ui/state").json()
    assert s["events"] and all(e["code"] == 200 for e in s["events"])
    assert s["window"]["throughput"] > 0 and s["window"]["ttft_p50"] is not None
    assert {e["worker"] for e in s["events"]} <= {w.url for w in ws}
    later = httpx.get(f"{u}/ui/state", params={"since": s["seq"]}).json()
    assert all(e["id"] > s["seq"] for e in later["events"])
    assert httpx.post(f"{u}/ui/admission", json={"enabled": False}).json() == {"enabled": False}


def test_kill_requires_chaos_and_known_worker(serve):
    w = serve(worker_app(FAST))
    router = serve(router_app([w.url]))
    assert httpx.post(f"{router.url}/ui/kill", json={"url": "http://evil:1"}).status_code == 400
    assert httpx.post(f"{router.url}/ui/kill", json={"url": w.url}).status_code == 409


def test_kill_crashes_chaos_worker(serve):
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "llmserve.mock_worker", "--port", str(port), "--chaos"])
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                httpx.get(f"{url}/health", timeout=0.5)
                break
            except httpx.HTTPError:
                time.sleep(0.1)
        router = serve(router_app([url]))
        assert httpx.post(f"{router.url}/ui/kill", json={"url": url}).status_code == 200
        assert proc.wait(timeout=5) == 1
    finally:
        proc.kill()
