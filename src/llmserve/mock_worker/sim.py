"""A small simulation of a vLLM worker, used when no NVIDIA GPU is available.

It models the effects routing and admission care about:
  * a bounded number of concurrently running sequences (``max_num_seqs``);
    extra requests wait in a queue (``num_requests_waiting``),
  * a bounded KV cache measured in tokens (``gpu_cache_usage_perc``),
  * a block-level prefix cache: prompt blocks seen recently skip prefill,
  * decode steps that slow down as the batch grows.

The numbers it produces are NOT GPU performance numbers. They exist so the
router can be developed and tested on a laptop.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from opentelemetry import trace

CHARS_PER_TOKEN = 4
BLOCK_TOKENS = 16


def count_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def block_hashes(text: str) -> list[str]:
    """Chained hashes of full token blocks, like vLLM's automatic prefix caching."""
    block_chars = BLOCK_TOKENS * CHARS_PER_TOKEN
    hashes, prev = [], ""
    for i in range(0, len(text) - block_chars + 1, block_chars):
        prev = hashlib.blake2b((prev + text[i : i + block_chars]).encode(), digest_size=8).hexdigest()
        hashes.append(prev)
    return hashes


@dataclass
class SimConfig:
    max_num_seqs: int = 16
    kv_capacity_tokens: int = 32_000
    prefix_cache_blocks: int = 1_500
    prefill_ms_per_token: float = 0.08
    decode_base_ms: float = 12.0
    decode_ms_per_seq: float = 0.8


@dataclass
class SimStats:
    running: int = 0
    waiting: int = 0
    kv_used_tokens: int = 0
    prefix_queries: int = 0
    prefix_hits: int = 0
    requests_total: int = 0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0


class PrefixCache:
    def __init__(self, capacity_blocks: int):
        self.capacity = capacity_blocks
        self._blocks: OrderedDict[str, None] = OrderedDict()

    def lookup_and_insert(self, hashes: list[str]) -> int:
        """Return how many leading blocks were cached, then cache all of them."""
        hit = 0
        for h in hashes:
            if h in self._blocks:
                hit += 1
            else:
                break
        for h in hashes:
            self._blocks[h] = None
            self._blocks.move_to_end(h)
        while len(self._blocks) > self.capacity:
            self._blocks.popitem(last=False)
        return hit


@dataclass
class SimWorker:
    config: SimConfig = field(default_factory=SimConfig)

    def __post_init__(self) -> None:
        self.stats = SimStats()
        self.prefix_cache = PrefixCache(self.config.prefix_cache_blocks)
        self._cond = asyncio.Condition()

    def _can_admit(self, kv_tokens: int) -> bool:
        c = self.config
        return (
            self.stats.running < c.max_num_seqs
            and self.stats.kv_used_tokens + kv_tokens <= c.kv_capacity_tokens
        )

    async def generate(self, prompt: str, max_tokens: int, trace_ctx=None):
        """Async generator yielding (token_text) one per decode step.

        Emits worker.queue / worker.prefill / worker.decode spans under ``trace_ctx``,
        mirroring the per-request stages vLLM reports when its tracing is enabled.
        """
        c = self.config
        tr = trace.get_tracer("llmserve.mock_worker")
        span = tr.start_span("worker.queue", context=trace_ctx)
        prompt_tokens = count_tokens(prompt)
        kv_tokens = min(prompt_tokens + max_tokens, c.kv_capacity_tokens)
        self.stats.requests_total += 1
        self.stats.waiting += 1
        try:
            async with self._cond:
                await self._cond.wait_for(lambda: self._can_admit(kv_tokens))
                self.stats.waiting -= 1
                self.stats.running += 1
                self.stats.kv_used_tokens += kv_tokens
        except BaseException:
            self.stats.waiting -= 1
            span.end()
            raise
        span.end()
        try:
            hashes = block_hashes(prompt)
            hit_blocks = self.prefix_cache.lookup_and_insert(hashes)
            self.stats.prefix_queries += prompt_tokens
            self.stats.prefix_hits += hit_blocks * BLOCK_TOKENS
            self.stats.prompt_tokens_total += prompt_tokens
            uncached = max(1, prompt_tokens - hit_blocks * BLOCK_TOKENS)
            with tr.start_as_current_span("worker.prefill", context=trace_ctx, attributes={
                    "llm.prompt_tokens": prompt_tokens, "llm.prefix_cache_hit_tokens": hit_blocks * BLOCK_TOKENS}):
                await asyncio.sleep(uncached * c.prefill_ms_per_token / 1000)
            span = tr.start_span("worker.decode", context=trace_ctx, attributes={"llm.max_tokens": max_tokens})
            try:
                for i in range(max_tokens):
                    step_ms = c.decode_base_ms + c.decode_ms_per_seq * self.stats.running
                    await asyncio.sleep(step_ms / 1000)
                    self.stats.generation_tokens_total += 1
                    yield f"tok{i} "
            finally:
                span.end()
        finally:
            async with self._cond:
                self.stats.running -= 1
                self.stats.kv_used_tokens -= kv_tokens
                self._cond.notify_all()

    def prometheus_text(self, model: str) -> str:
        s, c = self.stats, self.config
        lbl = f'{{model_name="{model}"}}'
        lines = [
            ("vllm:num_requests_running", "gauge", s.running),
            ("vllm:num_requests_waiting", "gauge", s.waiting),
            ("vllm:gpu_cache_usage_perc", "gauge", s.kv_used_tokens / c.kv_capacity_tokens),
            ("vllm:prefix_cache_queries_total", "counter", s.prefix_queries),
            ("vllm:prefix_cache_hits_total", "counter", s.prefix_hits),
            ("vllm:request_success_total", "counter", s.requests_total),
            ("vllm:prompt_tokens_total", "counter", s.prompt_tokens_total),
            ("vllm:generation_tokens_total", "counter", s.generation_tokens_total),
        ]
        out = []
        for name, kind, value in lines:
            out.append(f"# TYPE {name} {kind}")
            out.append(f"{name}{lbl} {value}")
        out.append(f"# mock worker, generated at {time.time():.3f}")
        return "\n".join(out) + "\n"
