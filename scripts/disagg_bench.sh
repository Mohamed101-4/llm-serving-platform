#!/usr/bin/env bash
# Phase 6: colocated vs disaggregated prefill/decode with the same total number of workers.
# Mock workers model prefill stalling decode on a shared GPU (--prefill-interference) in every
# config, so the comparison is fair.
#
#   ./scripts/disagg_bench.sh
#   PREFIXES=100000 TAG=unique-prompts CONFIGS="colocated-4w:4:0 disagg-1p3d:3:1" ./scripts/disagg_bench.sh
#
# On GPUs, see scripts/gpu/start_vllm_disagg.sh; the router flags are the same.
set -euo pipefail
cd "$(dirname "$0")/.."
RPS=${RPS:-15}
DURATION=${DURATION:-60}
PREFIXES=${PREFIXES:-8}
TAG=${TAG:-}  # e.g. unique-prompts
POLICY=${POLICY:-least_load}
UV=${UV:-uv}
# label:decode_workers:prefill_workers
CONFIGS=${CONFIGS:-"colocated-4w:4:0 disagg-1p3d:3:1 disagg-2p2d:2:2"}
mkdir -p logs

STACK=""
cleanup() { [[ -n "$STACK" ]] && kill "$STACK" 2> /dev/null && wait "$STACK" 2> /dev/null; true; }
trap cleanup EXIT
wait_up() {
  for _ in $(seq 60); do curl -sf localhost:8000/health > /dev/null && return 0; sleep 0.5; done
  echo "router did not start; see logs/" >&2
  exit 1
}
if curl -sf localhost:8000/health > /dev/null; then
  echo "something is already serving on :8000; stop it first" >&2
  exit 1
fi

for CFG in $CONFIGS; do
  IFS=: read -r NAME DEC PRE <<< "$CFG"
  $UV run python scripts/run_local.py --workers "$DEC" --prefill-workers "$PRE" --prefill-interference \
    --policy "$POLICY" > "logs/run_local-$NAME.log" 2>&1 &
  STACK=$!
  wait_up
  $UV run python -m llmserve.bench.loadgen --rps "$RPS" --duration "$DURATION" --num-prefixes "$PREFIXES" \
    --policy "$POLICY" --label "phase6-${TAG:+$TAG-}$NAME-${RPS}rps-mock" | tail -1
  cleanup
  STACK=""
  while curl -sf localhost:8000/health > /dev/null; do sleep 0.5; done
  sleep 1
done
