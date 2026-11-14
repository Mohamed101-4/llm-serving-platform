# Build plan and hardware adjustments

## Development machine (inspected 2026-11-13)

| Item | Found | Impact |
|---|---|---|
| Chip | Apple M3, 8 CPU cores, 10-core Apple GPU (Metal) | No NVIDIA GPU, so no CUDA |
| RAM | 8 GB | Too small to run several real model servers at once |
| CUDA / `nvidia-smi` | not present | vLLM GPU workers and Triton kernels can't run locally |
| Python | system 3.9.6 | Too old; the project uses Python 3.11 via `uv` |
| Docker | Docker Desktop 4.45 (running, arm64) | Fine for kind and container builds |
| kubectl | v1.32.2 | Fine |
| kind / helm | not installed | Installed via Homebrew when Phase 4 starts |

## Main bottleneck and how we solve it

**Problem:** vLLM, CUDA and Triton need an NVIDIA GPU. This Mac has none.

**Solution: two backends, one router.**

1. **Mock worker (local, no GPU).** A small FastAPI server that speaks the
   same API as vLLM: `/v1/chat/completions` (streaming and non-streaming) and
   a Prometheus `/metrics` endpoint using **vLLM's own metric names**
   (`vllm:num_requests_running`, `vllm:num_requests_waiting`,
   `vllm:gpu_cache_usage_perc`, prefix-cache counters). Inside, it simulates
   a GPU engine: a limited number of batch slots, a KV-cache block budget,
   an LRU prefix cache, prefill time proportional to *uncached* prompt
   tokens, and decode that slows down as the batch grows. The router can't
   tell it apart from vLLM, so all routing, admission and failover code is
   developed and tested on the Mac.
2. **Real vLLM workers (rented cloud GPU).** For numbers that go in the
   README and on a resume, the exact same router and load generator run
   against real vLLM servers on a rented NVIDIA GPU (RunPod, Lambda, Vast.ai
   or similar; a single 24 GB L4/A10G can host 2 to 4 Qwen2.5-1.5B workers).
   Scripts in `scripts/cloud/` make a run take minutes, so GPU rental stays
   short.

**Honesty rule:** every results JSON records its backend (`mock` or `vllm`)
and hardware. Mock runs show that the code works and give a rough idea of
relative behaviour, but they are **not** performance claims. Only `vllm`
runs on real GPUs go into the resume summary.

## Phase-by-phase: what runs where

| Phase | On the Mac | Needs a real GPU for |
|---|---|---|
| 1 Baseline (router, round-robin, load generator) | Everything, against mock workers | Real throughput/latency/TTFT/GPU-memory numbers |
| 2 Smart routing (load + prefix aware) | Everything; mock simulates prefix cache and KV usage | Real prefix-cache hit rate and latency deltas |
| 3 Overload protection, failover | Everything; 2x overload test runs on mocks | Confirming bounded p99 on real vLLM |
| 4 Kubernetes (KubeRay, autoscaling, Helm) | kind cluster running mock-worker pods (Ray on CPU) | Optional: real multi-GPU cluster (documented, not required) |
| 5 Observability (Prometheus, OTel, Grafana) | Everything | Nothing |
| 6 Disaggregated prefill/decode (stretch) | Only the router-side logic, with mocks | **All benchmarks**: needs 2+ GPUs and vLLM KV transfer |
| 7 Triton kernel (stretch) | Nothing useful (Triton needs CUDA) | **Everything**: develop and test on a cloud GPU |

Ray: Phases 1 to 3 start workers as plain processes to keep things simple.
Ray (and KubeRay) come in with Phase 4, where they add value for placement
and autoscaling.

## Workflow

- One branch and one PR per phase, merged into `main` once tests pass.
- Small commits with real dates. No backdating.
- `RESULTS.md` rows only come from JSON files in `results/`.
