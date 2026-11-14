#!/usr/bin/env bash
# Start vLLM prefill and decode workers for disaggregated serving (NOT yet run on real GPUs).
# Uses vLLM's NixlConnector: KV cache is moved GPU-to-GPU from prefill to decode workers.
# Needs a vLLM release with NixlConnector (>= 0.9.2) and NIXL/UCX installed, one GPU per worker.
#
#   P=1 D=3 MODEL=Qwen/Qwen2.5-1.5B-Instruct ./scripts/gpu/start_vllm_disagg.sh
#
# The router speaks the same kv_transfer_params protocol as vLLM's toy proxy, so then run:
#   python -m llmserve.router --workers <decode urls> --prefill-workers <prefill urls> --policy least_load
set -euo pipefail
P=${P:-1}
D=${D:-1}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
BASE_PORT=${BASE_PORT:-8001}
KV='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
mkdir -p logs

start() {  # $1 = gpu index, $2 = port, $3 = role label
  echo "$3 worker: GPU $1 port $2"
  CUDA_VISIBLE_DEVICES=$1 VLLM_NIXL_SIDE_CHANNEL_PORT=$((5600 + $1)) nohup vllm serve "$MODEL" --port "$2" \
    --enable-prefix-caching --disable-log-requests --kv-transfer-config "$KV" \
    > "logs/vllm-$3-$2.log" 2>&1 &
}
PREFILL="" DECODE=""
for ((i = 0; i < P + D; i++)); do
  PORT=$((BASE_PORT + i))
  if ((i < P)); then start "$i" "$PORT" prefill; PREFILL+="http://127.0.0.1:$PORT,"
  else start "$i" "$PORT" decode; DECODE+="http://127.0.0.1:$PORT,"; fi
done
for ((i = 0; i < P + D; i++)); do
  until curl -sf "localhost:$((BASE_PORT + i))/health" > /dev/null; do sleep 2; done
done
echo "ready. router:"
echo "python -m llmserve.router --workers ${DECODE%,} --prefill-workers ${PREFILL%,} --policy least_load"
