#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 OUTPUT_CHECKPOINT [BIND_HOST] [PORT]" >&2
  exit 2
fi

checkpoint="$(cd "$1" && pwd -P)"
bind_host="${2:-127.0.0.1}"
port="${3:-8800}"
image="vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
name="qwen36-35b-a3b-dwm-a10x3"

docker run -d \
  --name "$name" \
  --gpus all \
  --restart unless-stopped \
  --ipc host \
  -p "${bind_host}:${port}:8800" \
  -v "${checkpoint}:/model:ro" \
  "$image" \
  /model \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 8800 \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --max-num-seqs 512

echo "started ${name} on ${bind_host}:${port}"

