"""A small simulation of a vLLM worker, used when no NVIDIA GPU is available.

It models the effects routing and admission care about:
  * a bounded number of concurrently running sequences (``max_num_seqs``);
    extra requests wait in a queue (``num_requests_waiting``),
  * a bounded KV cache measured in tokens (``gpu_cache_usage_perc``),
  * a block-level prefix cache: prompt blocks seen recently skip prefill,
  * decode steps that slow down as the batch grows,
  * optionally (``prefill_interference``), prefill and decode sharing one GPU: a prefill
    occupies the GPU and stalls every running sequence's next decode step, which is the
    interference that disaggregated prefill/decode removes,
  * prefill-only and decode-only requests using vLLM's P/D ``kv_transfer_params`` protocol,
    with a simulated KV-cache transfer cost on the decode side.

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
    prefill_interference: bool = False  # off by default so Phase 1-5 results stay reproducible
    kv_transfer_ms_per_token: float = 0.005  # KV copy prefill->decode worker (~3 ms for 600 tokens)


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
        self._gpu_free_at = 0.0  # loop time when the current prefill (if any) finishes

    def _can_admit(self, kv_tokens: int) -> bool:
        c = self.config
        return (
            self.stats.running < c.max_num_seqs
            and self.stats.kv_used_tokens + kv_tokens <= c.kv_capacity_tokens
        )

    async def _acquire(self, kv_tokens: int, trace_ctx) -> None:
        span = trace.get_tracer("llmserve.mock_worker").start_span("worker.queue", context=trace_ctx)
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
            raise
        finally:
            span.end()

    async def _release(self, kv_tokens: int) -> None:
        async with self._cond:
            self.stats.running -= 1
            self.stats.kv_used_tokens -= kv_tokens
            self._cond.notify_all()

    async def _prefill(self, prompt: str, trace_ctx) -> tuple[int, int]:
        """Run prefill (with prefix caching). Returns (prompt_tokens, cached_tokens)."""
        c = self.config
        prompt_tokens = count_tokens(prompt)
        hit_blocks = self.prefix_cache.lookup_and_insert(block_hashes(prompt))
        cached = hit_blocks * BLOCK_TOKENS
        self.stats.prefix_queries += prompt_tokens
        self.stats.prefix_hits += cached
        self.stats.prompt_tokens_total += prompt_tokens
        dur = max(1, prompt_tokens - cached) * c.prefill_ms_per_token / 1000
        with trace.get_tracer("llmserve.mock_worker").start_as_current_span(
                "worker.prefill", context=trace_ctx,
                attributes={"llm.prompt_tokens": prompt_tokens, "llm.prefix_cache_hit_tokens": cached}):
            if c.prefill_interference:
                # Prefills take the whole GPU one at a time; decode steps wait for them.
                now = asyncio.get_running_loop().time()
                self._gpu_free_at = max(now, self._gpu_free_at) + dur
                await asyncio.sleep(self._gpu_free_at - now)
            else:
                await asyncio.sleep(dur)
        return prompt_tokens, cached

    async def prefill_only(self, prompt: str, trace_ctx=None) -> dict:
        """Prefill on this worker and hand the KV cache to a decode worker (P/D disaggregation)."""
        prompt_tokens = count_tokens(prompt)
        kv = min(prompt_tokens, self.config.kv_capacity_tokens)
        await self._acquire(kv, trace_ctx)
        try:
            prompt_tokens, cached = await self._prefill(prompt, trace_ctx)
        finally:
            await self._release(kv)
        return {"do_remote_prefill": True, "do_remote_decode": False, "prompt_tokens": prompt_tokens,
                "cached_tokens": cached, "remote_engine_id": id(self), "remote_block_ids": None}

    async def generate(self, prompt: str, max_tokens: int, trace_ctx=None, kv_transfer: dict | None = None):
        """Async generator yielding (token_text) one per decode step.

        With ``kv_transfer`` (from a prefill worker), prefill is replaced by a KV-cache transfer.
        Emits worker.queue / worker.prefill (or worker.kv_transfer) / worker.decode spans under
        ``trace_ctx``, mirroring the per-request stages vLLM reports when its tracing is enabled.
        """
        c = self.config
        tr = trace.get_tracer("llmserve.mock_worker")
        prompt_tokens = int(kv_transfer.get("prompt_tokens") or 0) if kv_transfer else count_tokens(prompt)
        prompt_tokens = prompt_tokens or count_tokens(prompt)
        kv_tokens = min(prompt_tokens + max_tokens, c.kv_capacity_tokens)
        await self._acquire(kv_tokens, trace_ctx)
        try:
            if kv_transfer:
                with tr.start_as_current_span("worker.kv_transfer", context=trace_ctx,
                                              attributes={"llm.prompt_tokens": prompt_tokens}):
                    await asyncio.sleep(prompt_tokens * c.kv_transfer_ms_per_token / 1000)
            else:
                await self._prefill(prompt, trace_ctx)
            span = tr.start_span("worker.decode", context=trace_ctx, attributes={"llm.max_tokens": max_tokens})
            loop = asyncio.get_running_loop()
            try:
                for i in range(max_tokens):
                    if c.prefill_interference and loop.time() < self._gpu_free_at:
                        await asyncio.sleep(self._gpu_free_at - loop.time())  # stalled behind a prefill
                    step_ms = c.decode_base_ms + c.decode_ms_per_seq * self.stats.running
                    await asyncio.sleep(step_ms / 1000)
                    self.stats.generation_tokens_total += 1
                    yield f"tok{i} "
            finally:
                span.end()
        finally:
            await self._release(kv_tokens)

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
