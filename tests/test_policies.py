import pytest

from llmserve.router.policies import NoHealthyWorker, RequestInfo, RoundRobin, Worker, make_policy

REQ = RequestInfo(prompt="hi", max_tokens=8)


def test_round_robin_cycles_evenly():
    workers = [Worker(f"http://w{i}") for i in range(3)]
    rr = RoundRobin()
    picks = [rr.choose(REQ, workers).url for _ in range(9)]
    assert picks == ["http://w0", "http://w1", "http://w2"] * 3


def test_round_robin_skips_unhealthy():
    workers = [Worker("http://a"), Worker("http://b", healthy=False), Worker("http://c")]
    rr = RoundRobin()
    picks = {rr.choose(REQ, workers).url for _ in range(10)}
    assert picks == {"http://a", "http://c"}


def test_no_healthy_workers_raises():
    with pytest.raises(NoHealthyWorker):
        RoundRobin().choose(REQ, [Worker("http://a", healthy=False)])


def test_make_policy_unknown():
    with pytest.raises(ValueError):
        make_policy("nope")
