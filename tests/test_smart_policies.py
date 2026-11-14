from collections import Counter

from llmserve.router.policies import LeastLoad, PrefixAware, RequestInfo, Worker, load_score
from llmserve.router.scraper import parse_metrics


def req(prefix: str, suffix: str = "") -> RequestInfo:
    return RequestInfo(prompt=prefix * 100 + suffix, max_tokens=8)  # prefix part > 512 chars


def test_load_score_combines_signals():
    idle = Worker("a")
    queued = Worker("b", metrics={"waiting": 3})
    full_kv = Worker("c", metrics={"kv_cache": 0.95})
    busy = Worker("d", inflight=2)
    assert load_score(idle) == 0
    assert load_score(queued) > load_score(busy) > load_score(idle)
    assert load_score(full_kv) > load_score(idle)


def test_least_load_picks_idle_worker():
    workers = [Worker("a", inflight=5), Worker("b", inflight=1), Worker("c", metrics={"waiting": 4})]
    assert LeastLoad().choose(req("x"), workers).url == "b"


def test_least_load_spreads_ties():
    workers = [Worker(u) for u in "abc"]
    pol = LeastLoad()
    assert Counter(pol.choose(req("x"), workers).url for _ in range(30)) == {"a": 10, "b": 10, "c": 10}


def test_least_load_avoids_full_kv_cache():
    workers = [Worker("a", metrics={"kv_cache": 0.98}), Worker("b", metrics={"kv_cache": 0.1})]
    assert LeastLoad().choose(req("x"), workers).url == "b"


def test_prefix_aware_is_sticky_and_spreads_prefixes():
    workers = [Worker(f"w{i}") for i in range(4)]
    pol = PrefixAware()
    homes = {}
    for p in [f"prefix-{i} " for i in range(40)]:
        picks = {pol.choose(req(p, f"user {j}"), workers).url for j in range(5)}
        assert len(picks) == 1  # same prefix, different user text -> same worker
        homes[p] = picks.pop()
    assert len(set(homes.values())) == 4  # 40 prefixes cover all workers


def test_prefix_aware_overflows_when_home_is_overloaded():
    workers = [Worker(f"w{i}") for i in range(3)]
    pol = PrefixAware(affinity=2.0)
    r = req("hot ")
    home = pol.choose(r, workers)
    home.inflight = 2  # within affinity: stay home for the cache hit
    assert pol.choose(r, workers) is home
    home.inflight = 3  # busier than the bonus is worth: spill over
    second = pol.choose(r, workers)
    assert second is not home
    assert second is pol.ranking(r, workers)[1]  # falls back to the next stable choice


def test_prefix_aware_spills_to_least_loaded_alternative():
    workers = [Worker(f"w{i}") for i in range(3)]
    pol = PrefixAware(affinity=2.0)
    r = req("hot ")
    home, second, third = pol.ranking(r, workers)
    home.inflight, second.inflight, third.inflight = 10, 5, 1
    assert pol.choose(r, workers) is third


def test_prefix_aware_minimal_reshuffle_when_worker_removed():
    workers = [Worker(f"w{i}") for i in range(4)]
    pol = PrefixAware()
    prefixes = [f"p{i} " for i in range(200)]
    before = {p: pol.choose(req(p), workers).url for p in prefixes}
    workers[3].healthy = False
    after = {p: pol.choose(req(p), workers).url for p in prefixes}
    moved = [p for p in prefixes if before[p] != after[p]]
    assert all(before[p] == "w3" for p in moved)  # only the dead worker's prefixes move


def test_parse_metrics_handles_labels_and_aliases():
    text = """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m",engine="0"} 3
vllm:num_requests_running{model_name="m",engine="1"} 2
vllm:num_requests_waiting{model_name="m"} 7
vllm:kv_cache_usage_perc{model_name="m"} 0.42
vllm:prefix_cache_hits_total{model_name="m"} 100
vllm:prefix_cache_queries_total{model_name="m"} 400
something_else 1
"""
    m = parse_metrics(text)
    assert m == {"running": 5, "waiting": 7, "kv_cache": 0.42, "prefix_hits": 100, "prefix_queries": 400}
