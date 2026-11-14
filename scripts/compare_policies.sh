#!/usr/bin/env bash
# Run the same workload against each routing policy and save one results file per run.
#
#   ./scripts/compare_policies.sh                       # mock workers on this machine
#   RPS=20 PREFIXES=64 TAG=many-prefixes ./scripts/compare_policies.sh
#
# On a GPU machine, start vLLM workers first and set WORKER_URLS, BACKEND=vllm, MODEL=...
set -euo pipefail
cd "$(dirname "$0")/.."
RPS=${RPS:-15}
DURATION=${DURATION:-60}
PREFIXES=${PREFIXES:-8}
WORKERS=${WORKERS:-4}
POLICIES=${POLICIES:-"round_robin least_load prefix_aware"}
BACKEND=${BACKEND:-mock}
MODEL=${MODEL:-mock-model}
TAG=${TAG:-default}
UV=${UV:-uv}
mkdir -p logs

STACK=""
cleanup() { [[ -n "$STACK" ]] && kill "$STACK" 2> /dev/null && wait "$STACK" 2> /dev/null; true; }
trap cleanup EXIT  # never leave routers/workers running, even when a run fails

if curl -sf localhost:8000/health > /dev/null; then
  echo "something is already serving on :8000; stop it first" >&2
  exit 1
fi

for POLICY in $POLICIES; do
  if [[ -n "${WORKER_URLS:-}" ]]; then
    $UV run python -m llmserve.router --workers "$WORKER_URLS" --policy "$POLICY" > "logs/router-$POLICY.log" 2>&1 &
  else
    $UV run python scripts/run_local.py --workers "$WORKERS" --policy "$POLICY" > "logs/run_local-$POLICY.log" 2>&1 &
  fi
  STACK=$!
  until curl -sf localhost:8000/health > /dev/null; do sleep 0.5; done
  $UV run python -m llmserve.bench.loadgen --rps "$RPS" --duration "$DURATION" --num-prefixes "$PREFIXES" \
    --model "$MODEL" --backend "$BACKEND" --policy "$POLICY" --label "phase2-$TAG-$POLICY-$BACKEND" | tail -1
  cleanup
  STACK=""
  while curl -sf localhost:8000/health > /dev/null; do sleep 0.5; done
  sleep 1
done
