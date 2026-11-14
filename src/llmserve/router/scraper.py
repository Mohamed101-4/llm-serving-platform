"""Polls each worker's Prometheus /metrics endpoint and keeps the latest load signals."""

from __future__ import annotations

import asyncio
import re
import time

import httpx

from .policies import Worker

# vLLM has renamed some metrics across versions; accept either spelling. llama.cpp's server
# (`llama-server --metrics`, used for real models on laptops) exposes the same signals under
# llamacpp:* names; it has no KV-usage gauge in recent builds.
ALIASES = {
    "running": ("vllm:num_requests_running", "llamacpp:requests_processing"),
    "waiting": ("vllm:num_requests_waiting", "llamacpp:requests_deferred"),
    "kv_cache": ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc", "llamacpp:kv_cache_usage_ratio"),
    "prefix_hits": ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total",
                    "llamacpp:prompt_tokens_cached_total"),
    "prefix_queries": ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total",
                       "llamacpp:prompt_tokens_total"),
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
    if "vllm:prefix_cache_queries_total" not in raw and "llamacpp:prompt_tokens_cached_total" in raw:
        # llama.cpp's prompt_tokens_total counts only tokens it had to process (cache misses)
        out["prefix_queries"] = raw.get("llamacpp:prompt_tokens_total", 0.0) + raw["llamacpp:prompt_tokens_cached_total"]
    return out


class MetricsScraper:
    """Doubles as the active health check: ``fail_threshold`` failed scrapes in a row mark
    a worker unhealthy, and one successful scrape brings it back."""

    def __init__(self, workers: list[Worker], client: httpx.AsyncClient, interval: float = 0.25,
                 fail_threshold: int = 2):
        self.workers = workers
        self.client = client
        self.interval = interval
        self.fail_threshold = fail_threshold
        self._fails: dict[str, int] = {}
        self._task: asyncio.Task | None = None

    async def scrape_once(self, w: Worker) -> None:
        try:
            r = await self.client.get(f"{w.url}/metrics", timeout=max(1.0, self.interval * 4))
            r.raise_for_status()
            w.metrics = parse_metrics(r.text)
            w.metrics["scraped_at"] = time.time()
            self._fails[w.url] = 0
            w.healthy = True
        except httpx.HTTPError:
            self._fails[w.url] = self._fails.get(w.url, 0) + 1
            if self._fails[w.url] >= self.fail_threshold:
                w.healthy = False

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
