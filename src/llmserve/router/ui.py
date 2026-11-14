"""Live control panel for the router: GET /ui, plus small JSON endpoints it polls and drives.

    /ui/state        workers, rolling 5 s stats, recent request events (since=<event id>)
    /ui/policy       switch routing policy at runtime
    /ui/admission    turn SLO admission control on/off
    /ui/load         built-in Poisson load generator (demo traffic through this router)
    /ui/kill         ask a mock worker to crash (only works on workers started with --chaos)

The page is a single static HTML file with no build step (src/llmserve/router/static/).
"""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..bench.workload import Workload, WorkloadConfig
from .admission import AdmissionController
from .policies import POLICIES, make_policy

STATIC = Path(__file__).parent / "static"
WINDOW_S = 5.0


def _pct(vals: list[float], q: float) -> float | None:
    return round(float(np.percentile(vals, q)), 1) if vals else None


class LoadGenerator:
    """Open-loop Poisson traffic from the shared-prefix workload, sent through the router itself."""

    def __init__(self, max_inflight: int = 1500):
        self.rps = 0.0
        self.max_inflight = max_inflight
        self.inflight = 0
        self.task: asyncio.Task | None = None
        self.client: httpx.AsyncClient | None = None
        self.model = "mock-model"

    async def set(self, rps: float, base_url: str, workers) -> None:
        self.rps = max(0.0, min(rps, 200.0))
        if self.rps and (self.task is None or self.task.done()):
            self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=2.0),
                                            limits=httpx.Limits(max_connections=2000))
            await self._detect_model(workers)
            self.task = asyncio.create_task(self._run(base_url.rstrip("/")))

    async def _detect_model(self, workers) -> None:
        for w in workers:
            try:
                r = await self.client.get(f"{w.url}/v1/models", timeout=2)
                self.model = r.json()["data"][0]["id"]
                return
            except Exception:  # noqa: BLE001 - any worker that answers will do
                continue

    async def _one(self, url: str, req) -> None:
        self.inflight += 1
        body = {"model": self.model, "messages": req.messages, "max_tokens": req.max_tokens,
                "stream": True, "ignore_eos": True, "temperature": 0.0}
        try:
            async with self.client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
                async for _ in r.aiter_raw():
                    pass
        except Exception:  # noqa: BLE001 - failures are already recorded by the router
            pass
        finally:
            self.inflight -= 1

    async def _run(self, url: str) -> None:
        wl = Workload(WorkloadConfig(seed=random.randrange(1_000_000)))
        rng = np.random.default_rng()
        try:
            while self.rps > 0:
                await asyncio.sleep(float(rng.exponential(1.0 / self.rps)))
                if self.rps > 0 and self.inflight < self.max_inflight:
                    asyncio.create_task(self._one(url, wl.sample()))
        finally:
            await asyncio.sleep(0)


def install_ui(app: FastAPI, ctl, workers, prefill_workers, metrics) -> None:
    gen = LoadGenerator()
    cache = {"t": 0.0, "hits": 0.0, "queries": 0.0, "rate": None}

    def hit_rate() -> float | None:
        pool = prefill_workers or workers  # prefix caching happens where prefill runs
        hits = sum(w.metrics.get("prefix_hits", 0.0) for w in pool)
        queries = sum(w.metrics.get("prefix_queries", 0.0) for w in pool)
        now = time.time()
        if now - cache["t"] >= 2.0:
            dq = queries - cache["queries"]
            if cache["t"] and dq > 0:
                cache["rate"] = round((hits - cache["hits"]) / dq, 4)
            cache.update(t=now, hits=hits, queries=queries)
        return cache["rate"]

    @app.get("/ui", response_class=HTMLResponse)
    async def ui_page():
        return HTMLResponse((STATIC / "dashboard.html").read_text())

    @app.get("/ui/state")
    async def ui_state(since: int = 0):
        now = time.time()
        recent = [e for e in ctl.events if e["ts"] >= now - WINDOW_S]
        ok = [e for e in recent if e["code"] == 200 and e["latency_ms"] is not None]
        ttft = [e["ttft_ms"] for e in ok if e["ttft_ms"] is not None]
        shed = sum(1 for e in recent if e["code"] == 429)
        new = [e for e in ctl.events if e["id"] > since][-300:]
        return {
            "policy": ctl.pol.name, "policies": sorted(POLICIES),
            "admission": {"enabled": ctl.admission is not None,
                          "slo_ms": ctl.admission.slo_ttft_ms if ctl.admission else None},
            "load_rps": gen.rps,
            "workers": [{"url": w.url, "role": role, "healthy": w.healthy, "inflight": w.inflight,
                         "running": w.metrics.get("running", 0.0), "waiting": w.metrics.get("waiting", 0.0),
                         "kv_cache": w.metrics.get("kv_cache", 0.0)}
                        for role, pool in (("prefill", prefill_workers), ("decode" if prefill_workers else "mixed", workers))
                        for w in pool],
            "window": {"seconds": WINDOW_S, "throughput": round(len(ok) / WINDOW_S, 2),
                       "ttft_p50": _pct(ttft, 50), "ttft_p99": _pct(ttft, 99),
                       "latency_p99": _pct([e["latency_ms"] for e in ok], 99),
                       "shed_pct": round(100 * shed / len(recent), 1) if recent else 0.0,
                       "failed": sum(1 for e in recent if e["code"] not in (200, 429))},
            "cache_hit_rate": hit_rate(),
            "events": new, "seq": ctl.seq,
        }

    @app.post("/ui/policy")
    async def ui_policy(request: Request):
        name = (await request.json()).get("name")
        if name not in POLICIES:
            return JSONResponse({"error": f"unknown policy {name}"}, status_code=400)
        ctl.pol = make_policy(name)
        app.state.policy = ctl.pol
        return {"policy": name}

    @app.post("/ui/admission")
    async def ui_admission(request: Request):
        body = await request.json()
        if body.get("enabled"):
            slo = float(body.get("slo_ms") or (ctl.admission.slo_ttft_ms if ctl.admission else 500))
            if ctl.admission is None or ctl.admission.slo_ttft_ms != slo:
                ctl.admission = AdmissionController(slo, ctl.default_slots)
        else:
            ctl.admission = None
        app.state.admission = ctl.admission
        metrics.collector.admission = ctl.admission
        return {"enabled": ctl.admission is not None}

    @app.post("/ui/load")
    async def ui_load(request: Request):
        rps = float((await request.json()).get("rps") or 0)
        await gen.set(rps, str(request.base_url), workers)
        return {"rps": gen.rps}

    @app.post("/ui/kill")
    async def ui_kill(request: Request):
        url = (await request.json()).get("url", "")
        if url not in {w.url for w in workers + prefill_workers}:  # only our own workers
            return JSONResponse({"error": "unknown worker"}, status_code=400)
        try:
            async with httpx.AsyncClient(timeout=2) as c:
                r = await c.post(f"{url}/admin/crash")
            if r.status_code != 200:
                return JSONResponse({"error": "worker refused (start mock workers with --chaos)"}, status_code=409)
        except httpx.HTTPError:
            pass  # already gone
        return {"killed": url}
