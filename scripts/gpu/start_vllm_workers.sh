#!/usr/bin/env bash
# Start N real vLLM workers on an NVIDIA machine (e.g. a rented cloud GPU), one per GPU.
# With a single GPU, set SHARE_ONE_GPU=1 to pack N small workers onto GPU 0.
#
#   pip install vllm   # on the GPU machine
#   N=4 MODEL=Qwen/Qwen2.5-1.5B-Instruct ./scripts/gpu/start_vllm_workers.sh
#
# Then run the router and load generator exactly as on the Mac, with
#   --model $MODEL --backend vllm
set -euo pipefail
N=${N:-2}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
BASE_PORT=${BASE_PORT:-8001}
SHARE_ONE_GPU=${SHARE_ONE_GPU:-0}
mkdir -p logs

if [[ "$SHARE_ONE_GPU" == "1" ]]; then
  MEM=$(python3 -c "print(round(0.9/$N, 2))")
else
  MEM=0.9
fi

for ((i = 0; i < N; i++)); do
  GPU=$i
  [[ "$SHARE_ONE_GPU" == "1" ]] && GPU=0
  PORT=$((BASE_PORT + i))
  echo "worker $i: GPU $GPU port $PORT mem $MEM"
  CUDA_VISIBLE_DEVICES=$GPU nohup vllm serve "$MODEL" --port "$PORT" \
    --gpu-memory-utilization "$MEM" --enable-prefix-caching --disable-log-requests \
    > "logs/vllm-$PORT.log" 2>&1 &
done

for ((i = 0; i < N; i++)); do
  PORT=$((BASE_PORT + i))
  until curl -sf "localhost:$PORT/health" > /dev/null; do sleep 2; done
  echo "worker on :$PORT ready"
done
WORKERS=$(for ((i = 0; i < N; i++)); do printf "http://127.0.0.1:%d," $((BASE_PORT + i)); done)
echo "router: python -m llmserve.router --workers ${WORKERS%,} --policy round_robin"
