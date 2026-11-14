"""Start N vLLM replicas on a Ray cluster, one per GPU (run from the Ray head pod).

    pip install vllm
    python serve_workers.py --replicas 4 --model Qwen/Qwen2.5-1.5B-Instruct

Each replica is a Ray actor that needs 1 GPU, so Ray places at most one per GPU worker pod
and the autoscaler adds pods when there aren't enough. Inside its pod the actor runs
`vllm serve` on port 8001, which the `llmserve-ray-workers` headless Service exposes to the
router. Not run in this repo's CI (no NVIDIA GPU available).
"""

from __future__ import annotations

import argparse
import subprocess
import time

import ray


@ray.remote(num_gpus=1, max_restarts=-1)
class VllmReplica:
    def __init__(self, model: str, port: int, extra: list[str]):
        cmd = ["vllm", "serve", model, "--port", str(port), "--enable-prefix-caching", *extra]
        self.proc = subprocess.Popen(cmd)
        self.node = ray.get_runtime_context().get_node_id()

    def status(self) -> dict:
        return {"node": self.node, "alive": self.proc.poll() is None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replicas", type=int, default=2)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--extra", nargs="*", default=["--gpu-memory-utilization", "0.90"])
    args = p.parse_args()

    ray.init(address="auto")
    # SPREAD keeps replicas on different pods, so every replica can bind the same port.
    replicas = [
        VllmReplica.options(name=f"vllm-{i}", lifetime="detached", scheduling_strategy="SPREAD")
        .remote(args.model, args.port, args.extra)
        for i in range(args.replicas)
    ]
    while True:
        print([ray.get(r.status.remote()) for r in replicas], flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
