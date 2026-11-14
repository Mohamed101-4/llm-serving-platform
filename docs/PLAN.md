# Build plan and hardware adjustments

## Development machine (inspected 2026-11-13)

| Item | Value |
|---|---|
| Machine | Apple M3, 10-core GPU (Metal 3), 8 GB unified memory |
| OS | macOS 14.4 (arm64) |
| NVIDIA GPU / CUDA | **None** (`nvidia-smi`, `nvcc` not present) |
| Python | 3.12 project venv via `uv` (system 3.9, Homebrew 3.15) |
| Docker | 28.3.3, daemon running |
| kubectl | installed; kind/minikube/helm not installed yet |

## What this means

vLLM GPU workers, CUDA, and Triton kernels **cannot run on this Mac**. So:

1. **Mock worker** (`src/llmserve/mock_worker`): a small FastAPI server that speaks the
   same OpenAI-compatible API and exposes the same Prometheus metric names as vLLM
   (`vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`,
   prefix-cache hit counters). It simulates batching slowdown, a bounded KV cache and a
   prefix cache, so routing and admission logic can be developed and tested without a GPU.
2. **The router never knows which backend it talks to.** The same router points at real
   `vllm serve` processes on a GPU machine.
3. **Real numbers come from a rented cloud GPU** (e.g. one or more NVIDIA L4/A10G/A100 on
   RunPod, Lambda, Vast.ai or a cloud VM) using `scripts/gpu/`. Only runs saved in
   `results/` are quoted anywhere, together with the hardware they ran on.
4. Benchmarks on the mock worker are labelled **`mock`** in every results file and in
   RESULTS.md. They show relative routing behaviour, not real GPU performance.

## Phases and where each one runs

| Phase | Mac (mock) | Needs real GPU for real numbers |
|---|---|---|
| 1 Baseline router + load generator | yes | yes (throughput, latency, GPU memory) |
| 2 Smart routing | yes | yes (prefix-cache gains are real only on vLLM) |
| 3 Overload protection | yes (fully testable) | optional |
| 4 Kubernetes / KubeRay on kind | yes (mock workers in kind) | multi-GPU cluster docs only |
| 5 Observability | yes | no |
| 6 Disaggregated prefill/decode | yes, mock workers model prefill-decode interference (done) | **yes, 2+ GPUs required** (script written, not run) |
| 7 Triton kernel | no (Triton needs NVIDIA) | **yes** |

Phases 6 and 7 will be marked "not benchmarked" unless a GPU run actually happens.
