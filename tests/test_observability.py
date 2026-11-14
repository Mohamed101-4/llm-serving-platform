"""Phase 5: Prometheus metrics and OpenTelemetry traces."""

import time

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client.parser import text_string_to_metric_families

from llmserve import tracing
from llmserve.mock_worker.server import create_app as worker_app
from llmserve.mock_worker.sim import SimConfig
from llmserve.router.app import create_app as router_app

FAST = SimConfig(decode_base_ms=1, decode_ms_per_seq=0, prefill_ms_per_token=0.001)
EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def in_memory_tracing():
    # Must run before any app is created in this process; other test modules leave
    # tracing unconfigured (no exporter), so reset the guard and install ours.
    from opentelemetry import trace
    from opentelemetry.util._once import Once

    tracing._configured = False
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    trace._TRACER_PROVIDER = None
    tracing.setup_tracing("test", exporter=EXPORTER)
    yield


def chat(url, stream=True, max_tokens=4):
    body = {"model": "m", "max_tokens": max_tokens, "stream": stream,
            "messages": [{"role": "system", "content": "sys " * 100}, {"role": "user", "content": "hi"}]}
    if stream:
        with httpx.stream("POST", f"{url}/v1/chat/completions", json=body, timeout=10) as r:
            r.read()
            return r
    return httpx.post(f"{url}/v1/chat/completions", json=body, timeout=10)


def samples(url):
    text = httpx.get(f"{url}/metrics").text
    out = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            out.setdefault(s.name, []).append(s)
    return out


def total(samples_list, **labels):
    return sum(s.value for s in samples_list if all(s.labels.get(k) == v for k, v in labels.items()))


def test_router_metrics(serve):
    workers = [serve(worker_app(FAST)) for _ in range(2)]
    router = serve(router_app([w.url for w in workers], "least_load", scrape_interval=0.05, slo_ttft_ms=1000))
    for _ in range(4):
        assert chat(router.url).status_code == 200
    time.sleep(0.2)
    m = samples(router.url)
    assert total(m["llmserve_requests_total"], code="200") == 4
    assert total(m["llmserve_routing_decisions_total"], policy="least_load") == 4
    assert total(m["llmserve_ttft_seconds_count"]) == 4
    assert total(m["llmserve_request_latency_seconds_count"]) == 4
    assert {s.labels["worker"] for s in m["llmserve_worker_healthy"]} == {w.url for w in workers}
    assert all(s.value == 1 for s in m["llmserve_worker_healthy"])
    assert "llmserve_worker_est_ttft_seconds" in m
    assert "llmserve_worker_kv_cache_usage" in m


def test_rejection_and_failure_metrics(serve):
    # slow scrapes, so the request (not the health check) is what discovers the dead worker
    router = serve(router_app(["http://127.0.0.1:9"], scrape_interval=30))
    assert chat(router.url, stream=False).status_code == 503
    m = samples(router.url)
    assert total(m["llmserve_worker_failures_total"], stage="connect") == 1
    assert total(m["llmserve_requests_total"], code="503") == 1


def test_trace_covers_router_and_worker_stages(serve):
    worker = serve(worker_app(FAST))
    router = serve(router_app([worker.url]))
    EXPORTER.clear()
    assert chat(router.url).status_code == 200
    time.sleep(0.2)
    spans = EXPORTER.get_finished_spans()
    by_name = {s.name: s for s in spans}
    expected = {"llmserve.request", "route", "upstream", "wait_first_token", "stream",
                "worker.request", "worker.queue", "worker.prefill", "worker.decode"}
    assert expected <= set(by_name), set(by_name)
    # one trace, correctly nested across the HTTP hop
    assert len({s.context.trace_id for s in spans if s.name in expected}) == 1
    parent = lambda name: by_name[name].parent.span_id  # noqa: E731
    sid = lambda name: by_name[name].context.span_id  # noqa: E731
    assert parent("route") == sid("llmserve.request")
    assert parent("upstream") == sid("llmserve.request")
    assert parent("worker.request") == sid("upstream")
    for stage in ("worker.queue", "worker.prefill", "worker.decode"):
        assert parent(stage) == sid("worker.request")
    assert by_name["worker.prefill"].attributes["llm.prompt_tokens"] > 0
    assert by_name["llmserve.request"].attributes["llm.ttft_ms"] > 0
