"""OpenAI-compatible router that forwards /v1/chat/completions to a pool of workers."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from opentelemetry.trace import SpanKind, Status, StatusCode
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..tracing import setup_tracing, tracer

from .admission import AdmissionController
from .discovery import DnsDiscovery
from .metrics import RouterMetrics
from .policies import NoHealthyWorker, Policy, RequestInfo, Worker, make_policy
from .scraper import MetricsScraper
from .ui import install_ui

# Errors that mean "this worker is gone", as opposed to a slow or bad request.
WORKER_DOWN = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)


def prompt_text(body: dict) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in body.get("messages", []))


def error(status: int, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status, headers=headers)


def create_app(
    worker_urls: list[str],
    policy: Policy | str = "round_robin",
    scrape_interval: float = 0.25,
    slo_ttft_ms: float | None = None,
    default_slots: int = 16,
    max_retries: int = 2,
    worker_dns: str | None = None,
    prefill_urls: list[str] | None = None,
) -> FastAPI:
    """With ``prefill_urls``, the router disaggregates prefill and decode: each request is first
    prefilled on a prefill worker (chosen prefix-aware, so shared prompts hit its prefix cache), then
    streamed from a decode worker (chosen by ``policy``) that receives the KV cache. This uses vLLM's
    ``kv_transfer_params`` protocol, so the same code drives mock workers and vLLM NixlConnector.
    """
    workers = [Worker(url.rstrip("/")) for url in worker_urls]
    prefill_workers = [Worker(url.rstrip("/")) for url in prefill_urls or []]
    prefill_pol = make_policy("prefix_aware")
    discovery = DnsDiscovery(workers, worker_dns) if worker_dns else None
    pol = make_policy(policy) if isinstance(policy, str) else policy
    admission = AdmissionController(slo_ttft_ms, default_slots) if slo_ttft_ms else None
    metrics = RouterMetrics(lambda: workers + prefill_workers, admission)
    # Mutable control state: the dashboard can switch policy and toggle admission at runtime.
    ctl = SimpleNamespace(pol=pol, admission=admission, default_slots=default_slots,
                          events=deque(maxlen=5000), seq=0)

    def record_event(worker_url: str, code: int | str, ttft: float | None = None, latency: float | None = None):
        ctl.seq += 1
        ctl.events.append({"id": ctl.seq, "ts": time.time(), "worker": worker_url, "code": code,
                           "ttft_ms": None if ttft is None else round(ttft * 1000, 1),
                           "latency_ms": None if latency is None else round(latency * 1000, 1)})
    setup_tracing("llmserve-router")
    tr = tracer("llmserve.router")
    client_holder: dict[str, httpx.AsyncClient] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limits = httpx.Limits(max_connections=2000, max_keepalive_connections=500)
        client_holder["c"] = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=2.0), limits=limits)
        if discovery:
            await discovery.start()
        scrapers = [MetricsScraper(pool, client_holder["c"], scrape_interval) for pool in (workers, prefill_workers) if pool]
        for sc in scrapers:
            sc.start()
        yield
        for sc in scrapers:
            await sc.stop()
        if discovery:
            await discovery.stop()
        await client_holder["c"].aclose()

    app = FastAPI(title="llmserve router", lifespan=lifespan, telemetry={"tracing": False})  # our own stage spans replace per-endpoint ones
    app.state.workers = workers
    app.state.prefill_workers = prefill_workers
    app.state.policy = pol
    app.state.admission = admission
    app.state.ctl = ctl
    app.state.metrics = metrics

    @app.get("/health")
    async def health():
        """Liveness: the router process is up."""
        return {"status": "ok", "policy": ctl.pol.name, "healthy_workers": sum(w.healthy for w in workers)}

    @app.get("/ready")
    async def ready():
        """Readiness: only take traffic when at least one worker can serve it."""
        n = sum(w.healthy for w in workers)
        return JSONResponse({"ready": n > 0, "healthy_workers": n}, status_code=200 if n else 503)

    @app.get("/scaling")
    async def scaling():
        """Autoscaling signal: total queue depth (router inflight + vLLM waiting) across workers.

        KEDA's metrics-api scaler divides ``total_load`` by its target per replica.
        """
        alive = [w for w in workers if w.healthy]
        inflight = sum(w.inflight for w in alive)
        waiting = sum(w.metrics.get("waiting", 0.0) for w in alive)
        return {"total_load": inflight + waiting, "inflight": inflight, "waiting": waiting,
                "healthy_workers": len(alive), "load_per_worker": (inflight + waiting) / len(alive) if alive else 0.0}

    @app.get("/workers")
    async def list_workers():
        return [
            {"url": w.url, "role": role, "healthy": w.healthy, "inflight": w.inflight, "metrics": w.metrics,
             "est_ttft_ms": ctl.admission.estimate_ttft_ms(w) if ctl.admission else None}
            for role, pool in (("prefill", prefill_workers), ("decode" if prefill_workers else "mixed", workers))
            for w in pool
        ]

    @app.get("/metrics")
    async def metrics_endpoint():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    async def prefill_stage(body: dict, info: RequestInfo, root_ctx):
        """Prefill on a prefill worker; returns (kv_transfer_params, None) or (None, (status, msg))."""
        client = client_holder["c"]
        excluded: set[str] = set()
        pbody = {**body, "stream": False, "max_tokens": 1,
                 "kv_transfer_params": {"do_remote_decode": True, "do_remote_prefill": False,
                                        "remote_engine_id": None, "remote_block_ids": None,
                                        "remote_host": None, "remote_port": None}}
        pbody.pop("stream_options", None)
        for attempt in range(max_retries + 1):
            try:
                pw = prefill_pol.choose(info, [w for w in prefill_workers if w.url not in excluded])
            except NoHealthyWorker:
                return None, (503, "no healthy prefill workers")
            metrics.routing.labels("prefill:" + prefill_pol.name, pw.url, "failover" if attempt else "policy").inc()
            with tr.start_as_current_span("prefill_stage", context=root_ctx, kind=SpanKind.CLIENT,
                                          attributes={"server.address": pw.url, "attempt": attempt}) as sp:
                hdrs: dict[str, str] = {}
                inject(hdrs)
                pw.inflight += 1
                try:
                    r = await client.post(f"{pw.url}/v1/chat/completions", json=pbody, headers=hdrs)
                except WORKER_DOWN as e:
                    pw.healthy = False
                    excluded.add(pw.url)
                    metrics.worker_failures.labels(pw.url, "connect").inc()
                    sp.record_exception(e)
                    sp.set_status(Status(StatusCode.ERROR, "prefill worker down"))
                    continue
                finally:
                    pw.inflight -= 1
            if r.status_code != 200:
                return None, (r.status_code, f"prefill failed: {r.text[:200]}")
            return r.json().get("kv_transfer_params") or {}, None
        return None, (502, "prefill workers failed")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        pol, admission = ctl.pol, ctl.admission  # snapshot: a mid-request switch can't mix policies
        body = await request.json()
        info = RequestInfo(prompt=prompt_text(body), max_tokens=int(body.get("max_tokens") or 64))
        stream = bool(body.get("stream"))
        client = client_holder["c"]
        excluded: set[str] = set()
        root = tr.start_span("llmserve.request", kind=SpanKind.SERVER,
                             context=extract(dict(request.headers)),
                             attributes={"llm.max_tokens": info.max_tokens, "llm.stream": stream, "llm.policy": pol.name})
        root_ctx = trace.set_span_in_context(root)
        t_req = time.perf_counter()

        def reject(status: int, message: str, headers: dict | None = None, worker_label: str = "none"):
            metrics.requests.labels(worker_label, str(status)).inc()
            record_event(worker_label, status)
            root.set_attribute("http.status_code", status)
            root.set_status(Status(StatusCode.ERROR, message))
            root.end()
            return error(status, message, headers)

        if prefill_workers:
            kvp, failure = await prefill_stage(body, info, root_ctx)
            if failure:
                return reject(*failure)
            body = {**body, "kv_transfer_params": kvp}

        # Pick a worker, admit or reject, and connect. Retry on another worker if the
        # chosen one is down; nothing has been sent to the client yet, so this is safe.
        for attempt in range(max_retries + 1):
            candidates = [w for w in workers if w.url not in excluded]
            with tr.start_as_current_span("route", context=root_ctx) as route_span:
                try:
                    worker = pol.choose(info, candidates)
                except NoHealthyWorker:
                    return reject(503, "no healthy workers")
                reason = "failover" if attempt else "policy"
                queued = False
                if admission:
                    d = admission.decide(worker, candidates)
                    route_span.set_attribute("admission.est_ttft_ms", d.est_ttft_ms)
                    route_span.set_attribute("admission.admitted", d.admitted)
                    if not d.admitted:
                        metrics.rejections.inc()
                        return reject(429, f"overloaded: estimated TTFT {d.est_ttft_ms:.0f} ms exceeds SLO "
                                           f"{admission.slo_ttft_ms:.0f} ms", {"Retry-After": str(d.retry_after_s)})
                    if d.worker is not worker:
                        reason = "admission_redirect"
                    worker = d.worker
                    queued = admission.queue_ahead(worker) >= 0
                route_span.set_attribute("route.worker", worker.url)
                route_span.set_attribute("route.reason", reason)
            metrics.routing.labels(pol.name, worker.url, reason).inc()
            worker.inflight += 1
            up_span = tr.start_span("upstream", kind=SpanKind.CLIENT, context=root_ctx,
                                    attributes={"server.address": worker.url, "attempt": attempt})
            up_ctx = trace.set_span_in_context(up_span)
            wait_span = tr.start_span("wait_first_token", context=up_ctx)
            fwd_headers: dict[str, str] = {}
            inject(fwd_headers, context=up_ctx)
            try:
                upstream = await client.send(
                    client.build_request("POST", f"{worker.url}/v1/chat/completions", json=body, headers=fwd_headers),
                    stream=True)
                if not stream:
                    await upstream.aread()  # read fully here so a mid-response failure can still be retried
                break
            except WORKER_DOWN as e:
                worker.inflight -= 1
                worker.healthy = False  # passive health check; the scraper revives it when it's back
                excluded.add(worker.url)
                metrics.worker_failures.labels(worker.url, "connect").inc()
                for sp in (wait_span, up_span):
                    sp.record_exception(e)
                    sp.set_status(Status(StatusCode.ERROR, "worker down"))
                    sp.end()
                if attempt == max_retries:
                    return reject(502, "workers failed")

        headers = {"x-llmserve-worker": worker.url}
        root.set_attribute("llm.worker", worker.url)

        def finish(ttft: float | None, failed: bool = False) -> None:
            worker.inflight -= 1
            latency = time.perf_counter() - t_req
            code = "stream_error" if failed else str(upstream.status_code)
            metrics.requests.labels(worker.url, code).inc()
            record_event(worker.url, code if failed else upstream.status_code, ttft, latency)
            if upstream.status_code == 200 and not failed:
                metrics.latency.labels(worker.url).observe(latency)
                if ttft is not None:
                    metrics.ttft.labels(worker.url).observe(ttft)
                if admission:
                    admission.record(worker, latency, ttft, queued)
            root.set_attribute("http.status_code", upstream.status_code)
            if ttft is not None:
                root.set_attribute("llm.ttft_ms", ttft * 1000)
            if failed:
                root.set_status(Status(StatusCode.ERROR, "worker failed mid-stream"))
            up_span.end()
            root.end()

        if not stream:
            wait_span.end()
            finish(None)
            return Response(upstream.content, status_code=upstream.status_code,
                            media_type=upstream.headers.get("content-type"), headers=headers)

        async def relay():
            ttft = None
            failed = False
            stream_span = None
            try:
                async for chunk in upstream.aiter_raw():
                    if ttft is None:
                        ttft = time.perf_counter() - t_req
                        wait_span.end()
                        stream_span = tr.start_span("stream", context=up_ctx)
                    yield chunk
            except WORKER_DOWN:
                # Tokens already reached the client, so we can't transparently retry.
                failed = True
                worker.healthy = False
                metrics.worker_failures.labels(worker.url, "stream").inc()
                yield f"data: {json.dumps({'error': {'message': 'worker failed mid-stream'}})}\n\n".encode()
            finally:
                await upstream.aclose()
                if stream_span is None:
                    wait_span.end()
                else:
                    stream_span.end()
                finish(ttft, failed)

        return StreamingResponse(relay(), status_code=upstream.status_code,
                                 media_type=upstream.headers.get("content-type", "text/event-stream"), headers=headers)

    install_ui(app, ctl, workers, prefill_workers, metrics)
    return app


def main() -> None:
    p = argparse.ArgumentParser(description="llmserve router")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", default="", help="comma-separated worker base URLs")
    p.add_argument("--worker-dns", default=None, help="host:port to resolve for workers (k8s headless Service)")
    p.add_argument("--policy", default="round_robin")
    p.add_argument("--scrape-interval", type=float, default=0.25, help="seconds between /metrics scrapes")
    p.add_argument("--slo-ttft-ms", type=float, default=None, help="enable admission control with this TTFT target")
    p.add_argument("--prefill-workers", default="",
                   help="comma-separated prefill worker URLs; enables disaggregated prefill/decode")
    p.add_argument("--default-slots", type=int, default=16, help="assumed concurrent sequences per worker until learned")
    args = p.parse_args()
    if not args.workers and not args.worker_dns:
        p.error("give --workers or --worker-dns")
    urls = [u for u in args.workers.split(",") if u]
    app = create_app(urls, args.policy, args.scrape_interval, args.slo_ttft_ms, args.default_slots,
                     worker_dns=args.worker_dns, prefill_urls=[u for u in args.prefill_workers.split(",") if u])
    # long keep-alive avoids races where a client reuses a connection the server just closed
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_keep_alive=75)


if __name__ == "__main__":
    main()
