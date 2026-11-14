"""Polls each worker's Prometheus /metrics endpoint and keeps the latest load signals."""

from __future__ import annotations

import asyncio
import re
import time

import httpx

from .policies import Worker

# vLLM has renamed some metrics across versions; accept either spelling.
ALIASES = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache": ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"),
    "prefix_hits": ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
    "prefix_queries": ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"),
}
LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)")


def parse_metrics(text: str) -> dict[str, float]:
    """Sum each metric over its label sets, then map to our short names."""
    raw: dict[str, float] = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m and m.group(3) != "NaN":
            raw[m.group(1)] = raw.get(m.group(1), 0.0) + float(m.group(3))
    out: dict[str, float] = {}
    for short, names in ALIASES.items():
        for n in names:
            if n in raw:
                out[short] = raw[n]
                break
    return out


class MetricsScraper:
    def __init__(self, workers: list[Worker], client: httpx.AsyncClient, interval: float = 0.25):
        self.workers = workers
        self.client = client
        self.interval = interval
        self._task: asyncio.Task | None = None

    async def scrape_once(self, w: Worker) -> None:
        try:
            r = await self.client.get(f"{w.url}/metrics", timeout=2.0)
            r.raise_for_status()
            w.metrics = parse_metrics(r.text)
            w.metrics["scraped_at"] = time.time()
        except httpx.HTTPError:
            pass  # health handling comes in Phase 3; keep the last known values

    async def _loop(self) -> None:
        while True:
            await asyncio.gather(*(self.scrape_once(w) for w in self.workers))
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
