"""Find workers through DNS, e.g. a Kubernetes headless Service.

A headless Service resolves to one A record per ready pod, so re-resolving it every few
seconds tracks scale-ups, scale-downs and restarts without restarting the router.
"""

from __future__ import annotations

import asyncio
import logging
import socket

from .policies import Worker

log = logging.getLogger("llmserve.discovery")


async def resolve(host: str, port: int) -> set[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_STREAM)
    return {f"http://{info[4][0]}:{port}" for info in infos}


def sync_workers(workers: list[Worker], urls: set[str]) -> tuple[set[str], set[str]]:
    """Update ``workers`` in place (other components hold a reference to the same list)."""
    current = {w.url for w in workers}
    added, removed = urls - current, current - urls
    workers[:] = [w for w in workers if w.url not in removed] + [Worker(u) for u in sorted(added)]
    return added, removed


class DnsDiscovery:
    def __init__(self, workers: list[Worker], target: str, interval: float = 2.0):
        host, _, port = target.rpartition(":")
        self.host, self.port = host, int(port)
        self.workers = workers
        self.interval = interval
        self._task: asyncio.Task | None = None

    async def refresh(self) -> None:
        try:
            urls = await resolve(self.host, self.port)
        except OSError as e:  # no ready pods yet, or DNS hiccup: keep the current list
            log.warning("resolving %s failed: %s", self.host, e)
            return
        added, removed = sync_workers(self.workers, urls)
        if added or removed:
            log.warning("workers changed: +%s -%s (now %d)", sorted(added), sorted(removed), len(self.workers))

    async def _loop(self) -> None:
        while True:
            await self.refresh()
            await asyncio.sleep(self.interval)

    async def start(self) -> None:
        await self.refresh()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
