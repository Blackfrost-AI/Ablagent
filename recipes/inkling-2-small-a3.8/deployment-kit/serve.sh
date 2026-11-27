#!/bin/bash
IMG=vllm/vllm-openai@sha256:e72944ef5ab9d1b91228d066117595847e94bf95321018a2119e70e8ba643ff6
CAND=/workspace/i2s-candidates/a3p8-L20-37
KEY=$(cat /workspace/inkling2-small/.serve_key)
docker run -d --name i2s-serve-a3p8 --gpus "\"device=4,5,6,7\"" --ipc=host --shm-size=16g \
  -p 40001:40001 \
  -v /workspace:/workspace \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_API_KEY="$KEY" \
  "$IMG" \
  --model "$CAND" --served-model-name inkling2-small-dwm-a3p8 \
  --trust-remote-code --tokenizer-mode inkling \
  --tensor-parallel-size 4 --disable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}" \
  --max-model-len 262144 --host 0.0.0.0 --port 40001
echo "SERVE_LAUNCHED=$?"
