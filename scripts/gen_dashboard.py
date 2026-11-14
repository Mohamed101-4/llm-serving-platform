"""Generate deploy/grafana/llmserve-dashboard.json (run after changing panels).

    uv run python scripts/gen_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
W = 'worker=~"$worker"'


def panel(pid, title, exprs, x, y, w=12, h=8, unit="short", kind="timeseries", legend="{{worker}}", desc=""):
    targets = [{"refId": chr(65 + i), "datasource": DS, "expr": e, "legendFormat": lf}
               for i, (e, lf) in enumerate(exprs if isinstance(exprs[0], tuple) else [(e, legend) for e in exprs])]
    p = {"id": pid, "type": kind, "title": title, "description": desc, "datasource": DS,
         "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
         "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
         "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}}
    if kind == "stat":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area"}
    return p


def q(metric, quant):
    return f"histogram_quantile({quant}, sum by (le) (rate({metric}_bucket{{{W}}}[$__rate_interval])))"


panels = [
    panel(1, "Request rate", [f'sum(rate(llmserve_requests_total{{{W}}}[$__rate_interval]))'], 0, 0, 6, 4,
          "reqps", "stat", "req/s"),
    panel(2, "Rejection rate (429)", [
        "sum(rate(llmserve_admission_rejections_total[$__rate_interval])) / "
        "(sum(rate(llmserve_requests_total[$__rate_interval])) > 0)"], 6, 0, 6, 4, "percentunit", "stat", "shed"),
    panel(3, "TTFT p99", [q("llmserve_ttft_seconds", 0.99)], 12, 0, 6, 4, "s", "stat", "p99"),
    panel(4, "Healthy workers", ["sum(llmserve_worker_healthy)"], 18, 0, 6, 4, "short", "stat", "healthy"),
    panel(5, "Time to first token (router)", [(q("llmserve_ttft_seconds", p), f"p{int(p * 100)}") for p in (0.5, 0.95, 0.99)],
          0, 4, 12, 8, "s", desc="Queueing + prefill as seen by clients"),
    panel(6, "End-to-end latency (router)", [(q("llmserve_request_latency_seconds", p), f"p{int(p * 100)}") for p in (0.5, 0.95, 0.99)],
          12, 4, 12, 8, "s"),
    panel(7, "Routing decisions by worker", [
        (f"sum by (worker, reason) (rate(llmserve_routing_decisions_total{{{W}}}[$__rate_interval]))", "{{worker}} {{reason}}")],
          0, 12, 12, 8, "reqps", desc="reason: policy, admission_redirect, failover"),
    panel(8, "Outcomes", [("sum by (code) (rate(llmserve_requests_total[$__rate_interval]))", "{{code}}"),
                          ("sum(rate(llmserve_admission_rejections_total[$__rate_interval]))", "429 shed")],
          12, 12, 12, 8, "reqps"),
    panel(9, "In flight per worker (router view)", [f"llmserve_worker_inflight{{{W}}}"], 0, 20, 8, 8),
    panel(10, "vLLM waiting queue per worker", [f"llmserve_worker_queue_waiting{{{W}}}"], 8, 20, 8, 8),
    panel(11, "KV-cache usage per worker", [f"llmserve_worker_kv_cache_usage{{{W}}}"], 16, 20, 8, 8, "percentunit"),
    panel(12, "Admission TTFT estimate per worker", [f"llmserve_worker_est_ttft_seconds{{{W}}}"], 0, 28, 12, 8, "s",
          desc="Requests are shed when every worker's estimate exceeds the SLO"),
    panel(13, "Worker failures", [(f"sum by (worker, stage) (increase(llmserve_worker_failures_total{{{W}}}[$__rate_interval]))",
                                   "{{worker}} {{stage}}")], 12, 28, 12, 8),
    panel(14, "Prefix-cache hit rate per worker (scraped from workers)", [
        ('sum by (instance) (rate(vllm:prefix_cache_hits_total[$__rate_interval])) / '
         'sum by (instance) (rate(vllm:prefix_cache_queries_total[$__rate_interval]))', "{{instance}}")],
          0, 36, 24, 8, "percentunit"),
]

dashboard = {
    "uid": "llmserve",
    "title": "llmserve router",
    "tags": ["llmserve", "llm", "vllm"],
    "timezone": "browser",
    "schemaVersion": 39,
    "refresh": "5s",
    "time": {"from": "now-15m", "to": "now"},
    "templating": {"list": [{
        "name": "worker", "type": "query", "datasource": DS, "label": "Worker",
        "query": {"query": "label_values(llmserve_worker_healthy, worker)", "refId": "W"},
        "includeAll": True, "multi": True, "allValue": ".*", "current": {"text": "All", "value": "$__all"}, "refresh": 2,
    }]},
    "panels": panels,
}

out = Path(__file__).resolve().parent.parent / "deploy" / "grafana" / "llmserve-dashboard.json"
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {out} ({len(panels)} panels)")
