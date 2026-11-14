#!/usr/bin/env bash
# 2x-overload benchmark: same workload with and without admission control.
#
#   RPS=44 SLO=500 ./scripts/overload_bench.sh
#
# RPS should be about twice what the workers can sustain (4 default mock workers ~22 req/s).
set -euo pipefail
cd "$(dirname "$0")/.."
RPS=${RPS:-44}
DURATION=${DURATION:-60}
SLO=${SLO:-500}
POLICY=${POLICY:-prefix_aware}
PREFIXES=${PREFIXES:-64}
UV=${UV:-uv}
mkdir -p logs

STACK=""
cleanup() { [[ -n "$STACK" ]] && kill "$STACK" 2> /dev/null && wait "$STACK" 2> /dev/null; true; }
trap cleanup EXIT
wait_up() {  # give up after 30 s instead of hanging forever
  for _ in $(seq 60); do curl -sf localhost:8000/health > /dev/null && return 0; sleep 0.5; done
  echo "router did not start; see logs/" >&2
  exit 1
}
if curl -sf localhost:8000/health > /dev/null; then
  echo "something is already serving on :8000; stop it first" >&2
  exit 1
fi

for MODE in no-admission admission; do
  EXTRA=""  # plain string: macOS bash 3.2 treats an empty array as unbound under set -u
  [[ "$MODE" == "admission" ]] && EXTRA="--slo-ttft-ms $SLO"
  $UV run python scripts/run_local.py --workers 4 --policy "$POLICY" $EXTRA > "logs/overload-$MODE.log" 2>&1 &
  STACK=$!
  wait_up
  $UV run python -m llmserve.bench.loadgen --rps "$RPS" --duration "$DURATION" --num-prefixes "$PREFIXES" \
    --policy "$POLICY" --label "phase3-overload-${RPS}rps-$MODE-mock" | tail -1
  cleanup
  STACK=""
  while curl -sf localhost:8000/health > /dev/null; do sleep 0.5; done
  sleep 1
done
