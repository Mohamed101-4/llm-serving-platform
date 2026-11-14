"""Routing policies. Each policy picks one healthy worker for a request."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field


@dataclass
class Worker:
    url: str
    healthy: bool = True
    inflight: int = 0  # requests this router currently has open on the worker
    metrics: dict[str, float] = field(default_factory=dict)  # last scraped worker metrics


@dataclass
class RequestInfo:
    prompt: str
    max_tokens: int


class NoHealthyWorker(Exception):
    pass


class Policy:
    name = "base"

    def choose(self, req: RequestInfo, workers: list[Worker]) -> Worker:
        raise NotImplementedError

    @staticmethod
    def healthy(workers: list[Worker]) -> list[Worker]:
        alive = [w for w in workers if w.healthy]
        if not alive:
            raise NoHealthyWorker("no healthy workers")
        return alive


class RoundRobin(Policy):
    """Baseline: cycle through healthy workers, ignoring load and content."""

    name = "round_robin"

    def __init__(self) -> None:
        self._counter = itertools.count()

    def choose(self, req: RequestInfo, workers: list[Worker]) -> Worker:
        alive = self.healthy(workers)
        return alive[next(self._counter) % len(alive)]


def load_score(w: Worker, wait_weight: float = 1.0, kv_weight: float = 4.0) -> float:
    """How busy a worker is. Lower is better.

    * ``inflight``: requests this router has open on the worker. Always fresh.
    * ``waiting``: vLLM's own queue (scraped). Catches load from other routers/clients.
    * ``kv_cache``: KV-cache utilisation 0..1 (scraped). Near 1.0 vLLM must queue or
      preempt, so a nearly full worker is penalised even with a short queue.
    """
    m = w.metrics
    return w.inflight + wait_weight * m.get("waiting", 0.0) + kv_weight * m.get("kv_cache", 0.0)


class LeastLoad(Policy):
    """GPU-aware: send to the worker with the lowest load score."""

    name = "least_load"

    def __init__(self) -> None:
        self._tiebreak = itertools.count()

    def choose(self, req: RequestInfo, workers: list[Worker]) -> Worker:
        alive = self.healthy(workers)
        offset = next(self._tiebreak)  # rotate ties so equal workers share the load
        n = len(alive)
        return min((alive[(i + offset) % n] for i in range(n)), key=load_score)


def _rendezvous(key: str, url: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{key}|{url}".encode(), digest_size=8).digest(), "big")


class PrefixAware(Policy):
    """Send requests sharing a prompt prefix to the same worker, with a load cap.

    The prefix key is the first ``prefix_chars`` characters of the prompt (the shared
    system prompt in typical chat traffic). Workers are ranked per key with rendezvous
    hashing, so each prefix has a stable "home" and only ~1/N keys move when a worker
    joins or leaves. To avoid hot-spotting a popular prefix, a worker is skipped when its
    load exceeds ``(1 + slack) * mean_load + 1`` (consistent hashing with bounded loads);
    the request then goes to the next worker in that key's ranking.
    """

    name = "prefix_aware"

    def __init__(self, prefix_chars: int = 512, slack: float = 0.25) -> None:
        self.prefix_chars = prefix_chars
        self.slack = slack

    def ranking(self, req: RequestInfo, workers: list[Worker]) -> list[Worker]:
        key = req.prompt[: self.prefix_chars]
        return sorted(workers, key=lambda w: _rendezvous(key, w.url), reverse=True)

    def choose(self, req: RequestInfo, workers: list[Worker]) -> Worker:
        alive = self.healthy(workers)
        loads = {w.url: load_score(w) for w in alive}
        cap = (1 + self.slack) * (sum(loads.values()) / len(alive)) + 1
        for w in self.ranking(req, alive):
            if loads[w.url] <= cap:
                return w
        return min(alive, key=lambda w: loads[w.url])


POLICIES: dict[str, type[Policy]] = {p.name: p for p in (RoundRobin, LeastLoad, PrefixAware)}


def make_policy(name: str) -> Policy:
    try:
        return POLICIES[name]()
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from None
