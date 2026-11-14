"""OpenAI-compatible HTTP server around the simulated worker (stand-in for `vllm serve`)."""

from __future__ import annotations

import argparse
import json
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from opentelemetry import trace
from opentelemetry.propagate import extract
from opentelemetry.trace import SpanKind

from ..tracing import setup_tracing
from .sim import SimConfig, SimWorker


def prompt_from_messages(messages: list[dict]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def create_app(config: SimConfig | None = None, model: str = "mock-model") -> FastAPI:
    app = FastAPI(title="llmserve mock worker", telemetry={"tracing": False})  # our own stage spans replace per-endpoint ones
    worker = SimWorker(config or SimConfig())
    setup_tracing("llmserve-mock-worker")
    tr = trace.get_tracer("llmserve.mock_worker")
    app.state.worker = worker

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model, "object": "model"}]}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(worker.prometheus_text(model))

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        prompt = prompt_from_messages(body.get("messages", []))
        max_tokens = int(body.get("max_tokens") or 64)
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        root = tr.start_span("worker.request", kind=SpanKind.SERVER, context=extract(dict(request.headers)))
        ctx = trace.set_span_in_context(root)

        if body.get("stream"):

            async def sse():
                try:
                    async for tok in worker.generate(prompt, max_tokens, ctx):
                        chunk = {
                            "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    done = {
                        "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                    }
                    yield f"data: {json.dumps(done)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    root.end()

            return StreamingResponse(sse(), media_type="text/event-stream")

        try:
            text = "".join([t async for t in worker.generate(prompt, max_tokens, ctx)])
        finally:
            root.end()
        return JSONResponse({
            "id": rid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": max_tokens},
        })

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Mock vLLM worker")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--model", default="mock-model")
    p.add_argument("--max-num-seqs", type=int, default=SimConfig.max_num_seqs)
    p.add_argument("--kv-capacity-tokens", type=int, default=SimConfig.kv_capacity_tokens)
    args = p.parse_args()
    cfg = SimConfig(max_num_seqs=args.max_num_seqs, kv_capacity_tokens=args.kv_capacity_tokens)
    uvicorn.run(create_app(cfg, args.model), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
