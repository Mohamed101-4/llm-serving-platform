"""Routing policies. Each policy picks one healthy worker for a request."""

from __future__ import annotations

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


POLICIES: dict[str, type[Policy]] = {RoundRobin.name: RoundRobin}


def make_policy(name: str) -> Policy:
    try:
        return POLICIES[name]()
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from None
