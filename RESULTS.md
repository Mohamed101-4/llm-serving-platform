# Results

Every number here comes from a JSON file in [`results/`](results/). Runs with
backend **mock** use the simulated worker on a laptop: they compare routing policies
under identical conditions, but they are **not GPU performance numbers**. GPU
results will be added only after real runs on NVIDIA hardware.

## Phase 1: round-robin baseline

| Metric | Baseline (round-robin) | New | Improvement | Hardware | Workload config | Commit | Results file |
|---|---|---|---|---|---|---|---|
| Throughput (req/s) | 13.67 | n/a | n/a (baseline) | Apple M3 8 GB, 4 mock workers | Poisson 15 RPS, 60 s, 8 shared prefixes, seed 0 | eebda93 | [phase1-rr-mock-20261113-220730.json](results/phase1-rr-mock-20261113-220730.json) |
| Output tokens/s | 1497.2 | n/a | n/a | same | same | eebda93 | same |
| Latency p50 / p95 / p99 (ms) | 1886 / 4121 / 6808 | n/a | n/a | same | same | eebda93 | same |
| TTFT p50 / p95 / p99 (ms) | 37.6 / 66.0 / 107.9 | n/a | n/a | same | same | eebda93 | same |
| Requests OK / sent | 870 / 870 | n/a | n/a | same | same | eebda93 | same |
| GPU memory | not measured (no NVIDIA GPU) | n/a | n/a | same | same | eebda93 | same |

**Real GPU baseline: not run yet.** Use `scripts/gpu/start_vllm_workers.sh` on an NVIDIA machine.
