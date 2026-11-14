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

## Phase 3: overload protection (mock workers)

### 2x overload: admission control off vs on

4 mock workers (about 21.5 req/s capacity), offered Poisson **44 RPS** for 60 s, 64 prefixes, seed 0,
`prefix_aware` routing, TTFT SLO 500 ms. Commit `36e8f2b`, Apple M3 8 GB.

| Metric | Baseline (no admission) | New (admission, SLO 500 ms) | Improvement | Hardware | Workload config | Commit |
|---|---|---|---|---|---|---|
| TTFT p50 (ms) | 27562.08 | 280.19 | -99.0% | Apple M3 8 GB, 4 mock workers | 44 RPS (2x capacity), 60 s, 64 prefixes, seed 0 | 36e8f2b |
| TTFT p95 (ms) | 53772.19 | 849.31 | -98.4% | same | same | same |
| TTFT p99 (ms) | 56900.21 | 1127.52 | -98.0% | same | same | same |
| Latency p50 (ms) | 30473.46 | 2759.68 | -90.9% | same | same | same |
| Latency p99 (ms) | 60441.80 | 8551.42 | -85.9% | same | same | same |
| Throughput (req/s) | 21.51 | 21.42 | -0.4% | same | same | same |
| Requests OK / 429 / failed | 2693 / 0 / 0 | 1412 / 1281 / 0 | 48% shed fast | same | same | same |

Files: `results/phase3-overload-44rps-no-admission-mock-20261113-235608.json`,
`results/phase3-overload-44rps-admission-mock-20261113-235719.json`

Without admission control the queue grows for the whole run, so a request waits up to ~57 s for its
first token. With it, served throughput is unchanged and the excess gets an immediate 429 + Retry-After.
**The SLO is not held exactly:** admitted TTFT p99 was 1128 ms against a 500 ms target (p50 280 ms).
The estimator uses EWMA service times and a slot count, and it lags bursts. A tighter p99 would need a
safety margin on the estimate or a lower configured SLO.

### Worker killed mid-load

4 mock workers, Poisson 15 RPS for 60 s, `prefix_aware`; worker :8002 SIGKILLed at t=20 s. Commit `bb446bf`.

| Metric | Value |
|---|---|
| Requests OK / sent | 861 / 870 |
| Failed requests | 9, all started 17.8 to 19.4 s and cut off mid-stream on the killed worker |
| Requests sent after the kill (t > 20.5 s) | 562 / 562 OK |
| Latency p99 (ms) | 6986.16 |

File: `results/phase3-failover-kill-at-20s-mock-20261114-000209.json`.
Requests already streaming tokens can't be retried transparently; everything else fails over.
Two earlier failover runs were discarded because of benchmark bugs (the launcher tore down the router
when a worker died; loadgen counted mid-stream error events as success). Both are fixed in the commits above.

## Phase 4: Kubernetes autoscaling on kind (mock workers)

kind single-node cluster in Docker Desktop (4 GB VM) on Apple M3 8 GB. KEDA metrics-api scaler on the
router's `/scaling` total_load, target 12 per worker, min 1, max 6. Router: `prefix_aware` + admission
(TTFT SLO 500 ms). Load: Poisson 25 RPS for 120 s through `kubectl port-forward`, then 120 s idle.
Commit `ef2f7d0`. File: `results/phase4-autoscale-kind-20261114-001641.json`.

| Metric | Value |
|---|---|
| Worker replicas | 2 at start, 4 desired at t=10 s, 6 ready by t=32 s, 1 by t=176 s (56 s after load stopped) |
| Requests OK / 429 / failed | 2874 / 179 / 0 of 3053 |
| Throughput (req/s) | 23.02 |
| TTFT p50 / p95 / p99 (ms) | 46.3 / 369.7 / 969.6 |
| Latency p50 / p95 / p99 (ms) | 2243 / 5059 / 7417 |

The 429s came while new pods were still starting; latency numbers include port-forward overhead.
The RayCluster manifest for GPUs was validated with `kubectl apply --dry-run=server` against the
KubeRay 1.3.2 CRD; **no Ray or GPU workload was run.**

## Phase 6: disaggregated prefill/decode (mock workers)

Same 4 workers either colocated (each does prefill and decode) or split into prefill-only and
decode-only workers. All mock workers run with `--prefill-interference` (a prefill occupies the
"GPU" and stalls running sequences' decode steps, the effect disaggregation removes) and a KV
transfer cost of 0.005 ms/token. Decode routing `least_load`, prefill routing `prefix_aware`.
Poisson 15 RPS, 60 s, Apple M3 8 GB. Script: `scripts/disagg_bench.sh`. **Mock numbers, not GPU numbers.**

### Prefill-heavy: mostly unique prompts (`PREFIXES=100000`)

Commit `303dfe7-dirty` (only the benchmark script's label option was uncommitted). Files:
`results/phase6-unique-prompts-colocated-4w-15rps-mock-20261114-083237.json`,
`results/phase6-unique-prompts-disagg-1p3d-15rps-mock-20261114-083352.json`. 870/870 OK in both.

| Metric | Colocated (4 mixed) | Disaggregated (1 prefill + 3 decode) | Change |
|---|---|---|---|
| TPOT p50 / p95 (ms) | 23.22 / 26.34 | 22.26 / 24.67 | -4.1% / -6.3% |
| Latency p50 / p95 / p99 (ms) | 2313.79 / 4761.8 / 7944.24 | 2278.06 / 4807.76 / 7613.73 | -1.5% / +1.0% / -4.2% |
| TTFT p50 / p95 / p99 (ms) | 86.88 / 166.48 / 196.22 | 113.89 / 286.64 / 384.14 | +31.1% / +72.2% / +95.8% |
| Throughput (req/s) | 13.569 | 13.575 | same |
| Prefix-cache hit rate | 0.342 | 0.363 | +0.021 |

### Cache-friendly: 8 shared system prompts (default workload)

Commit `303dfe7`. Files: `results/phase6-colocated-4w-15rps-mock-20261114-082819.json`,
`results/phase6-disagg-1p3d-15rps-mock-20261114-082928.json`,
`results/phase6-disagg-2p2d-15rps-mock-20261114-083053.json`. 870/870 OK in all three.

| Metric | Colocated (4 mixed) | 1 prefill + 3 decode | 2 prefill + 2 decode |
|---|---|---|---|
| TPOT p50 / p95 (ms) | 19.9 / 21.69 | 22.28 / 24.63 (+12.0% / +13.6%) | 25.72 / 25.85 |
| Latency p50 / p99 (ms) | 1936.09 / 6835.49 | 2192.3 / 7550.07 (+13.2% / +10.5%) | 10093.73 / 20024.82 |
| TTFT p50 / p99 (ms) | 38.0 / 117.99 | 49.28 / 184.27 (+29.7% / +56.2%) | 7571.87 / 15801.6 |
| Throughput (req/s) | 13.667 | 13.576 | 10.826 |
| Prefix-cache hit rate | 0.767 | 0.790 | 0.787 |

### What this shows

- Disaggregation **only helped when prefill was heavy**: with mostly unique prompts it made decode
  smoother (TPOT p95 -6.3%, latency p99 -4.2%) at the cost of later first tokens (an extra network
  hop, a KV transfer, and one prefill worker serving all prefills).
- With cache-friendly prompts most prefill is skipped, so there is little interference to remove,
  and taking a worker away from decode makes every metric worse (+12% TPOT).
- **The P:D ratio matters most.** 2 prefill + 2 decode starved decode capacity and queued requests
  for seconds. Real deployments tune the ratio to the prompt/output length mix.
- The vLLM version (`scripts/gpu/start_vllm_disagg.sh`, NixlConnector) uses the same router code
  path but **has not been run on GPUs**; no GPU numbers exist for this phase.

## Phase 7: Triton INT8 (W8A16) matmul kernel

Weight-only INT8 matmul (`src/llmserve/kernels/int8_matmul.py`): int8 weights with per-output-channel
scales, converted in registers, fp32 accumulation, scale applied once per output tile. Validated
against a PyTorch dequantize-then-matmul reference. Commit `f31c48a`. File:
`results/phase7-int8-correctness-interpret-20261114-134111.json`.

Hardware: **CPU via the Triton interpreter (`TRITON_INTERPRET=1`)**, Linux aarch64 in Docker Desktop
on an Apple M3 host; torch 2.15.0+cpu, triton 3.9.0. Script: `scripts/check_int8_kernel.sh`.

| Check (5 shapes, M 1 to 128, K 256 to 1024, N 256 to 1024, fp32) | Result |
|---|---|
| Matches PyTorch reference (`allclose`, rtol = atol = 1e-4) | 5 / 5 shapes |
| Max abs error vs reference | 3.8e-5 to 8.7e-5 |
| Relative (Frobenius) error vs unquantized fp32 matmul | 0.70% to 0.78% (INT8 quantization error) |
| pytest `tests/test_int8_kernel.py` | 6 passed |

**No speed numbers.** The interpreter runs the kernel in NumPy on the CPU, so timings would be
meaningless. `scripts/gpu/bench_int8_kernel.py` times the kernel against fp16 `torch.matmul` and
PyTorch dequantize+matmul on Qwen2.5-1.5B layer shapes, but **has not been run on an NVIDIA GPU.**

## Real small model on a laptop: routing policies with llama.cpp

**The first non-simulated numbers in this project.** 2 `llama-server` workers (llama.cpp build b12017,
Metal) each serving Qwen2.5-0.5B-Instruct Q4_K_M with 4 slots, on an Apple M3 MacBook Pro with 8 GB.
Poisson 2 RPS, 90 s, 8 shared system prompts (default workload), admission off. One run per policy,
fresh workers (empty caches) for each. Commit `6374c31-dirty` (only README.md uncommitted).
Files: `results/local-llm-2rps-{round_robin,least_load,prefix_aware}-llamacpp-*.json`. 172/172 OK in all three.

| Metric | round_robin | least_load | prefix_aware | Δ prefix_aware vs round_robin |
|---|---|---|---|---|
| Throughput (req/s) | 1.879 | 1.88 | 1.86 | -1.0% |
| Output tokens/s | 199.78 | 199.81 | 197.73 | -1.0% |
| Latency p50 / p95 / p99 (ms) | 1740.94 / 4651.33 / 7546.36 | 1826.28 / 5261.39 / 6840.8 | 1544.62 / 4006.45 / 6431.16 | -11.3% / -13.9% / -14.8% |
| TTFT p50 / p95 / p99 (ms) | 100.54 / 354.53 / 498.89 | 99.67 / 436.27 / 812.24 | 91.03 / 422.73 / 834.73 | -9.5% / +19.2% / +67.3% |
| TPOT p50 / p95 (ms) | 18.74 / 27.88 | 19.91 / 27.56 | 15.73 / 25.33 | -16.1% / -9.1% |
| Prefix-cache hit rate | 0.689 | 0.689 | 0.707 | +0.018 |
| Requests per worker | 86 / 86 | 88 / 84 | 113 / 59 | |

What this shows, with caveats:
- On a real engine, `prefix_aware` cut median and tail end-to-end latency by 11 to 15% and median TTFT
  by 9.5%, but its TTFT tail got worse (+67% p99): it sent 66% of requests to one worker (shared
  prompts have a home worker), so that worker queued more often. Same trade-off as on the mock.
- `least_load` brought no clear gain on 2 workers.
- Only 172 requests per run and one run each, so differences of a few percent are within noise.
  This is a small model on a laptop, **not a GPU server**; it validates the router against a real
  inference engine, and the numbers should not be compared with GPU serving.
