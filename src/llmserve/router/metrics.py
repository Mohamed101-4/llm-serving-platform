"""Prometheus metrics for the router.

Event counters and histograms are updated on the request path. Per-worker state
(inflight, vLLM queue, KV cache, health, admission estimate) is read from the live worker
list at scrape time by a custom collector, so it's never stale and costs nothing per request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily

if TYPE_CHECKING:
    from .admission import AdmissionController
    from .policies import Worker

# TTFT spans ms to tens of seconds under overload; latency spans up to a minute.
TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)


class WorkerCollector:
    def __init__(self, get_workers: Callable[[], list["Worker"]], admission: "AdmissionController | None"):
        self.get_workers = get_workers
        self.admission = admission

    def collect(self):
        fams = {
            "inflight": GaugeMetricFamily("llmserve_worker_inflight", "Requests this router has open on the worker", labels=["worker"]),
            "waiting": GaugeMetricFamily("llmserve_worker_queue_waiting", "vLLM num_requests_waiting (scraped)", labels=["worker"]),
            "running": GaugeMetricFamily("llmserve_worker_running", "vLLM num_requests_running (scraped)", labels=["worker"]),
            "kv": GaugeMetricFamily("llmserve_worker_kv_cache_usage", "vLLM KV-cache usage 0..1 (scraped)", labels=["worker"]),
            "healthy": GaugeMetricFamily("llmserve_worker_healthy", "1 if the worker passes health checks", labels=["worker"]),
            "est": GaugeMetricFamily("llmserve_worker_est_ttft_seconds", "Admission controller's TTFT estimate", labels=["worker"]),
        }
        for w in self.get_workers():
            m = w.metrics
            fams["inflight"].add_metric([w.url], w.inflight)
            fams["waiting"].add_metric([w.url], m.get("waiting", 0.0))
            fams["running"].add_metric([w.url], m.get("running", 0.0))
            fams["kv"].add_metric([w.url], m.get("kv_cache", 0.0))
            fams["healthy"].add_metric([w.url], 1.0 if w.healthy else 0.0)
            if self.admission:
                fams["est"].add_metric([w.url], self.admission.estimate_ttft_ms(w) / 1000)
        yield from fams.values()


class RouterMetrics:
    def __init__(self, get_workers, admission=None):
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("llmserve_requests_total", "Requests by final outcome",
                                ["worker", "code"], registry=r)
        self.routing = Counter("llmserve_routing_decisions_total",
                               "Worker picked per request; reason is policy, admission_redirect or failover",
                               ["policy", "worker", "reason"], registry=r)
        self.rejections = Counter("llmserve_admission_rejections_total", "Requests shed with 429", registry=r)
        self.worker_failures = Counter("llmserve_worker_failures_total",
                                       "Worker connection/stream failures; stage is connect or stream",
                                       ["worker", "stage"], registry=r)
        self.ttft = Histogram("llmserve_ttft_seconds", "Time to first token through the router",
                              ["worker"], buckets=TTFT_BUCKETS, registry=r)
        self.latency = Histogram("llmserve_request_latency_seconds", "End-to-end request latency through the router",
                                 ["worker"], buckets=LATENCY_BUCKETS, registry=r)
        r.register(WorkerCollector(get_workers, admission))
