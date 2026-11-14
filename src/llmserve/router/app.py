"""OpenAI-compatible router that forwards /v1/chat/completions to a pool of workers."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .admission import AdmissionController
from .discovery import DnsDiscovery
from .policies import NoHealthyWorker, Policy, RequestInfo, Worker, make_policy
from .scraper import MetricsScraper

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
) -> FastAPI:
    workers = [Worker(url.rstrip("/")) for url in worker_urls]
    discovery = DnsDiscovery(workers, worker_dns) if worker_dns else None
    pol = make_policy(policy) if isinstance(policy, str) else policy
    admission = AdmissionController(slo_ttft_ms, default_slots) if slo_ttft_ms else None
    client_holder: dict[str, httpx.AsyncClient] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limits = httpx.Limits(max_connections=2000, max_keepalive_connections=500)
        client_holder["c"] = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=2.0), limits=limits)
        if discovery:
            await discovery.start()
        scraper = MetricsScraper(workers, client_holder["c"], scrape_interval)
        scraper.start()
        yield
        await scraper.stop()
        if discovery:
            await discovery.stop()
        await client_holder["c"].aclose()

    app = FastAPI(title="llmserve router", lifespan=lifespan)
    app.state.workers = workers
    app.state.policy = pol
    app.state.admission = admission

    @app.get("/health")
    async def health():
        """Liveness: the router process is up."""
        return {"status": "ok", "policy": pol.name, "healthy_workers": sum(w.healthy for w in workers)}

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
            {"url": w.url, "healthy": w.healthy, "inflight": w.inflight, "metrics": w.metrics,
             "est_ttft_ms": admission.estimate_ttft_ms(w) if admission else None}
            for w in workers
        ]

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        info = RequestInfo(prompt=prompt_text(body), max_tokens=int(body.get("max_tokens") or 64))
        stream = bool(body.get("stream"))
        client = client_holder["c"]
        excluded: set[str] = set()

        # Pick a worker, admit or reject, and connect. Retry on another worker if the
        # chosen one is down; nothing has been sent to the client yet, so this is safe.
        for attempt in range(max_retries + 1):
            candidates = [w for w in workers if w.url not in excluded]
            try:
                worker = pol.choose(info, candidates)
            except NoHealthyWorker:
                return error(503, "no healthy workers")
            queued = False
            if admission:
                d = admission.decide(worker, candidates)
                if not d.admitted:
                    return error(429, f"overloaded: estimated TTFT {d.est_ttft_ms:.0f} ms exceeds SLO "
                                      f"{admission.slo_ttft_ms:.0f} ms", {"Retry-After": str(d.retry_after_s)})
                worker = d.worker
                queued = admission.queue_ahead(worker) >= 0
            worker.inflight += 1
            start = time.perf_counter()
            try:
                upstream = await client.send(
                    client.build_request("POST", f"{worker.url}/v1/chat/completions", json=body), stream=True)
                if not stream:
                    await upstream.aread()  # read fully here so a mid-response failure can still be retried
                break
            except WORKER_DOWN:
                worker.inflight -= 1
                worker.healthy = False  # passive health check; the scraper revives it when it's back
                excluded.add(worker.url)
                if attempt == max_retries:
                    return error(502, "workers failed")

        headers = {"x-llmserve-worker": worker.url}

        def finish(ttft: float | None) -> None:
            worker.inflight -= 1
            if admission and upstream.status_code == 200:
                admission.record(worker, time.perf_counter() - start, ttft, queued)

        if not stream:
            finish(None)
            return Response(upstream.content, status_code=upstream.status_code,
                            media_type=upstream.headers.get("content-type"), headers=headers)

        async def relay():
            ttft = None
            try:
                async for chunk in upstream.aiter_raw():
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    yield chunk
            except WORKER_DOWN:
                # Tokens already reached the client, so we can't transparently retry.
                worker.healthy = False
                yield f"data: {json.dumps({'error': {'message': 'worker failed mid-stream'}})}\n\n".encode()
            finally:
                await upstream.aclose()
                finish(ttft)

        return StreamingResponse(relay(), status_code=upstream.status_code,
                                 media_type=upstream.headers.get("content-type", "text/event-stream"), headers=headers)

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
    p.add_argument("--default-slots", type=int, default=16, help="assumed concurrent sequences per worker until learned")
    args = p.parse_args()
    if not args.workers and not args.worker_dns:
        p.error("give --workers or --worker-dns")
    urls = [u for u in args.workers.split(",") if u]
    app = create_app(urls, args.policy, args.scrape_interval, args.slo_ttft_ms, args.default_slots,
                     worker_dns=args.worker_dns)
    # long keep-alive avoids races where a client reuses a connection the server just closed
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_keep_alive=75)


if __name__ == "__main__":
    main()
