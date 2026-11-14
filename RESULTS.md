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

## Phase 2: smart routing vs round-robin (mock workers)

Same workload, same seed (0), one router policy per run, 4 mock workers on Apple M3 8 GB.
Throughput in an open-loop test is set by the offered rate, so the gains show up as latency, not req/s.
Commit `900a8ba-dirty`: the only uncommitted change during these runs was the
benchmark wrapper `scripts/compare_policies.sh` (cleanup fix, committed right after), not router code.

### Near saturation: Poisson 22 RPS, 60 s, 64 shared prefixes

| Metric | Baseline (round-robin) | New (prefix_aware) | Improvement | least_load | Hardware | Workload config | Commit |
|---|---|---|---|---|---|---|---|
| TTFT p50 (ms) | 105.08 | 66.84 | -36.4% | 104.00 (-1.0%) | Apple M3 8 GB, 4 mock workers | 22 RPS, 60 s, 64 prefixes, seed 0 | 900a8ba-dirty |
| TTFT p95 (ms) | 1210.58 | 832.96 | -31.2% | 1064.17 (-12.1%) | same | same | same |
| TTFT p99 (ms) | 1785.75 | 1487.23 | -16.7% | 2065.36 (+15.7%) | same | same | same |
| Latency p50 (ms) | 2683.99 | 2588.23 | -3.6% | 2703.84 (+0.7%) | same | same | same |
| Latency p95 (ms) | 5688.44 | 5446.14 | -4.3% | 5720.21 (+0.6%) | same | same | same |
| Latency p99 (ms) | 8498.52 | 8227.42 | -3.2% | 8513.98 (+0.2%) | same | same | same |
| Prefix-cache hit rate | 0.581 | 0.639 | +10.1% | 0.588 (+1.3%) | same | same | same |
| Throughput (req/s) | 20.50 | 20.27 | -1.1% | 20.23 (-1.3%) | same | same | same |
| Requests OK / sent | 1346 / 1346 | 1346 / 1346 | | 1346 / 1346 | same | same | same |

Files: `results/phase2-64prefix-22rps-{round_robin,least_load,prefix_aware}-mock-2026111{3-223343,3-223455,3-223606}.json`

### Moderate load: Poisson 15 RPS, 60 s

| Workload | Metric | Round-robin | prefix_aware | least_load |
|---|---|---|---|---|
| 64 prefixes | TTFT p50 (ms) | 42.13 | 39.46 (-6.3%) | 41.04 (-2.6%) |
| 64 prefixes | TTFT p95 (ms) | 107.53 | 101.00 (-6.1%) | 106.24 (-1.2%) |
| 64 prefixes | Latency p95 (ms) | 4073.31 | 3969.42 (-2.6%) | 3972.73 (-2.5%) |
| 64 prefixes | Prefix-cache hit rate | 0.565 | 0.640 (+13.2%) | 0.574 (+1.6%) |
| 8 prefixes | TTFT p95 (ms) | 69.02 | 65.26 (-5.4%) | 65.07 (-5.7%) |
| 8 prefixes | Latency p99 (ms) | 6820.83 | 6459.09 (-5.3%) | 6537.02 (-4.2%) |
| 8 prefixes | Prefix-cache hit rate | 0.767 | 0.769 (+0.2%) | 0.767 (0.0%) |

With only 8 prefixes every worker's cache already holds all of them, so prefix routing has little to win.

### Notes and caveats

- **Run-to-run variance is real.** An earlier identical-config least_load run
  (`results/phase2-v1/phase2-64prefix-22rps-least_load-mock-20261113-222305.json`) measured
  TTFT p50 -41.6% vs its round-robin run; the rerun above measured -1.0%. least_load's gains are
  therefore **not consistent** on this setup. Repeated runs with confidence intervals are a TODO.
- **v1 prefix_aware was worse** (TTFT p50 +289% at 22 RPS; files in `results/phase2-v1/`). Its loose
  bounded-load cap let popular prefixes pile onto one worker (301 to 369 requests per worker vs 327 to 345 for v2).
- All Phase 2 numbers are from the **mock worker**. Real vLLM prefix caching saves more prefill time
  on long prompts, so the GPU result may differ in either direction. **GPU comparison not run yet.**
