#!/usr/bin/env bash
# Kill one of 4 mock workers (SIGKILL) partway through a load run and record what clients saw.
#
#   RPS=15 KILL_AT=20 ./scripts/failover_bench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RPS=${RPS:-15}
DURATION=${DURATION:-60}
KILL_AT=${KILL_AT:-20}
POLICY=${POLICY:-prefix_aware}
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

$UV run python scripts/run_local.py --workers 4 --policy "$POLICY" > logs/failover.log 2>&1 &
STACK=$!
wait_up
(sleep "$KILL_AT" && pkill -9 -f "llmserve.mock_worker --port 8002" && echo "killed worker :8002 at ${KILL_AT}s") &
$UV run python -m llmserve.bench.loadgen --rps "$RPS" --duration "$DURATION" --num-prefixes 64 \
  --policy "$POLICY" --label "phase3-failover-kill-at-${KILL_AT}s-mock" | tail -1
