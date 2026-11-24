#!/bin/sh
# Validated vendor serve command for Inkling-2-Mini-Derisked.
# Edit MODEL_DIR and BIND_ADDR for your environment. No auth is built in.
set -eu

MODEL_DIR="/path/to/model/dir"   # directory containing this repo's weights
BIND_ADDR="0.0.0.0"              # set to your trusted interface address
IMAGE="vllm/vllm-openai:inkling-2-mini"

docker rm -f inkling-derisked 2>/dev/null || true

exec docker run --name inkling-derisked \
  --gpus all --ipc=host --restart unless-stopped --network host \
  -v "$MODEL_DIR:/models/inkling-derisked:ro" \
  --entrypoint vllm "$IMAGE" \
  serve /models/inkling-derisked \
  --served-model-name Inkling-2-Mini-Derisked \
  --host "$BIND_ADDR" --port 8899 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --reasoning-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
