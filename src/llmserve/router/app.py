"""OpenAI-compatible router that forwards /v1/chat/completions to a pool of workers."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .policies import NoHealthyWorker, Policy, RequestInfo, Worker, make_policy


def prompt_text(body: dict) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in body.get("messages", []))


def create_app(worker_urls: list[str], policy: Policy | str = "round_robin") -> FastAPI:
    workers = [Worker(url.rstrip("/")) for url in worker_urls]
    pol = make_policy(policy) if isinstance(policy, str) else policy
    client_holder: dict[str, httpx.AsyncClient] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limits = httpx.Limits(max_connections=2000, max_keepalive_connections=500)
        client_holder["c"] = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0), limits=limits)
        yield
        await client_holder["c"].aclose()

    app = FastAPI(title="llmserve router", lifespan=lifespan)
    app.state.workers = workers
    app.state.policy = pol

    @app.get("/health")
    async def health():
        return {"status": "ok", "policy": pol.name}

    @app.get("/workers")
    async def list_workers():
        return [{"url": w.url, "healthy": w.healthy, "inflight": w.inflight, "metrics": w.metrics} for w in workers]

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        info = RequestInfo(prompt=prompt_text(body), max_tokens=int(body.get("max_tokens") or 64))
        try:
            worker = pol.choose(info, workers)
        except NoHealthyWorker:
            return JSONResponse({"error": {"message": "no healthy workers"}}, status_code=503)

        client = client_holder["c"]
        url = f"{worker.url}/v1/chat/completions"
        worker.inflight += 1
        headers = {"x-llmserve-worker": worker.url}
        try:
            if body.get("stream"):
                upstream = await client.send(client.build_request("POST", url, json=body), stream=True)

                async def relay():
                    try:
                        async for chunk in upstream.aiter_raw():
                            yield chunk
                    finally:
                        await upstream.aclose()
                        worker.inflight -= 1

                return StreamingResponse(
                    relay(), status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "text/event-stream"), headers=headers,
                )
            resp = await client.post(url, json=body)
            worker.inflight -= 1
            return Response(resp.content, status_code=resp.status_code,
                            media_type=resp.headers.get("content-type"), headers=headers)
        except httpx.HTTPError as e:
            worker.inflight -= 1
            return JSONResponse({"error": {"message": f"worker error: {e!r}"}}, status_code=502)

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="llmserve router")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", required=True, help="comma-separated worker base URLs")
    p.add_argument("--policy", default="round_robin")
    args = p.parse_args()
    app = create_app(args.workers.split(","), args.policy)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
