#!/usr/bin/env bash
# One-time download of a real (small) model for laptops without an NVIDIA GPU:
#   llama.cpp's llama-server (pinned build) + Qwen2.5-0.5B-Instruct (Q4_K_M GGUF, ~490 MB).
# Files go to ~/.cache/llmserve, never into the repo. Apple Silicon Macs only (uses Metal);
# on Linux, download the matching llama.cpp build and set LLAMA_SERVER / LLAMA_MODEL.
#
#   ./scripts/local_llm/setup.sh
#   uv run python scripts/run_local.py --backend llamacpp --workers 2 --slo-ttft-ms 1500
set -euo pipefail
BUILD=${BUILD:-b12017}
D="$HOME/.cache/llmserve"
mkdir -p "$D/llama.cpp" "$D/models"
if [[ ! -x "$D/llama.cpp/llama-$BUILD/llama-server" ]]; then
  echo "downloading llama.cpp $BUILD (~17 MB)"
  curl -fL --progress-bar -o "$D/llama.tar.gz" \
    "https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/llama-$BUILD-bin-macos-arm64.tar.gz"
  tar -xzf "$D/llama.tar.gz" -C "$D/llama.cpp" && rm "$D/llama.tar.gz"
fi
M="$D/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
if [[ ! -s "$M" ]]; then
  echo "downloading Qwen2.5-0.5B-Instruct Q4_K_M (~490 MB)"
  curl -fL --progress-bar -o "$M" \
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
fi
"$D/llama.cpp/llama-$BUILD/llama-server" --version 2>&1 | head -1
echo "ready: $M"
