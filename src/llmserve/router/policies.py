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
    """Prefer the worker that likely has this prompt's prefix cached, unless it is busier.

    The prefix key is the first ``prefix_chars`` characters of the prompt (the shared
    system prompt in typical chat traffic). Workers are ranked per key with rendezvous
    hashing, so each prefix has a stable "home" and only ~1/N keys move when a worker
    joins or leaves.

    Each worker's cost is ``load_score(w)``, minus ``affinity`` for the prefix's home
    worker. The cheapest wins, ties broken by rendezvous rank. So the home keeps the
    request while it is at most ``affinity`` units busier than the best alternative; past
    that, load wins and the request spills over. A first version used a looser
    bounded-load cap ``(1.25 * mean + 1)`` that let popular prefixes pile up and queue
    (see RESULTS.md, Phase 2 v1).
    """

    name = "prefix_aware"

    def __init__(self, prefix_chars: int = 512, affinity: float = 2.0) -> None:
        self.prefix_chars = prefix_chars
        self.affinity = affinity

    def ranking(self, req: RequestInfo, workers: list[Worker]) -> list[Worker]:
        key = req.prompt[: self.prefix_chars]
        return sorted(workers, key=lambda w: _rendezvous(key, w.url), reverse=True)

    def choose(self, req: RequestInfo, workers: list[Worker]) -> Worker:
        ranked = self.ranking(req, self.healthy(workers))
        costs = [load_score(w) - (self.affinity if i == 0 else 0.0) for i, w in enumerate(ranked)]
        best = min(range(len(ranked)), key=lambda i: (costs[i], i))
        return ranked[best]


POLICIES: dict[str, type[Policy]] = {p.name: p for p in (RoundRobin, LeastLoad, PrefixAware)}


def make_policy(name: str) -> Policy:
    try:
        return POLICIES[name]()
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from None
