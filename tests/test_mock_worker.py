import asyncio

from llmserve.mock_worker.sim import PrefixCache, SimConfig, SimWorker, block_hashes


def test_block_hashes_share_prefix():
    a = block_hashes("x" * 640 + "tail one")
    b = block_hashes("x" * 640 + "a different tail that is long enough" * 4)
    assert a[:10] == b[:10]


def test_prefix_cache_hits_after_first_insert():
    cache = PrefixCache(100)
    h = block_hashes("y" * 64 * 5)
    assert cache.lookup_and_insert(h) == 0
    assert cache.lookup_and_insert(h) == 5


async def test_concurrency_limit_creates_waiting_queue():
    w = SimWorker(SimConfig(max_num_seqs=2, decode_base_ms=5, decode_ms_per_seq=0))

    async def consume():
        async for _ in w.generate("p" * 40, 10):
            pass

    tasks = [asyncio.create_task(consume()) for _ in range(5)]
    await asyncio.sleep(0.02)
    assert w.stats.running == 2
    assert w.stats.waiting == 3
    await asyncio.gather(*tasks)
    assert w.stats.running == 0 and w.stats.waiting == 0 and w.stats.kv_used_tokens == 0
    assert "vllm:num_requests_waiting" in w.prometheus_text("m")
