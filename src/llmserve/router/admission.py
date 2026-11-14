"""SLO-aware admission control.

Before forwarding a request, estimate its time-to-first-token (TTFT) on the chosen worker:

    est_ttft = queue_wait + base_ttft
    queue_wait = (requests ahead of it + 1) * mean_service_time / slots   (only if all slots are busy)

* ``slots``: how many sequences the worker runs concurrently. Learned from the scraped
  ``num_requests_running`` at moments the worker also reports a waiting queue (it's full),
  otherwise a configured default.
* ``mean_service_time``: EWMA of end-to-end request latency on that worker. ``slots`` requests
  finish every ``mean_service_time`` seconds on average, so that's the drain rate.
* ``base_ttft``: EWMA of TTFT for requests that did not have to queue (prefill cost).

If the estimate breaks the TTFT SLO on the chosen worker, try the worker with the best
estimate; if that breaks it too, reject with 429 and a Retry-After of roughly how far over
the SLO the queue is. TTFT is the SLO target because queueing shows up there directly;
end-to-end latency also includes decode time, which mostly depends on how many tokens the
client asked for, and routing can't change that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .policies import Worker


@dataclass
class WorkerStats:
    service_s: float | None = None
    base_ttft_s: float | None = None
    slots_seen: int = 0


@dataclass
class Decision:
    worker: Worker | None
    est_ttft_ms: float
    retry_after_s: int = 0

    @property
    def admitted(self) -> bool:
        return self.worker is not None


class AdmissionController:
    def __init__(self, slo_ttft_ms: float, default_slots: int = 16, alpha: float = 0.1):
        self.slo_ttft_ms = slo_ttft_ms
        self.default_slots = default_slots
        self.alpha = alpha
        self._stats: dict[str, WorkerStats] = {}
        self.admitted = 0
        self.rejected = 0

    def stats(self, w: Worker) -> WorkerStats:
        return self._stats.setdefault(w.url, WorkerStats())

    def slots(self, w: Worker) -> int:
        s = self.stats(w)
        if w.metrics.get("waiting", 0) > 0:  # saturated: running == real concurrency limit
            s.slots_seen = max(s.slots_seen, int(w.metrics.get("running", 0)))
        return s.slots_seen or self.default_slots

    def queue_ahead(self, w: Worker) -> int:
        """Requests that would be ahead of a new one in the worker's waiting queue (-1 = free slot)."""
        return max(w.inflight - self.slots(w), int(w.metrics.get("waiting", 0)) - 1, -1)

    def estimate_ttft_ms(self, w: Worker) -> float:
        s = self.stats(w)
        base = s.base_ttft_s or 0.0
        ahead = self.queue_ahead(w)
        if ahead < 0 or s.service_s is None:
            return base * 1000
        return ((ahead + 1) * s.service_s / self.slots(w) + base) * 1000

    def decide(self, chosen: Worker, workers: list[Worker]) -> Decision:
        est = self.estimate_ttft_ms(chosen)
        if est <= self.slo_ttft_ms:
            self.admitted += 1
            return Decision(chosen, est)
        alive = [w for w in workers if w.healthy]
        best = min(alive, key=self.estimate_ttft_ms)
        best_est = self.estimate_ttft_ms(best)
        if best_est <= self.slo_ttft_ms:
            self.admitted += 1
            return Decision(best, best_est)
        self.rejected += 1
        retry = max(1, math.ceil((best_est - self.slo_ttft_ms) / 1000))
        return Decision(None, best_est, retry)

    def record(self, w: Worker, latency_s: float | None, ttft_s: float | None, queued: bool) -> None:
        s, a = self.stats(w), self.alpha
        if latency_s is not None:
            s.service_s = latency_s if s.service_s is None else (1 - a) * s.service_s + a * latency_s
        if ttft_s is not None and not queued:
            s.base_ttft_s = ttft_s if s.base_ttft_s is None else (1 - a) * s.base_ttft_s + a * ttft_s
