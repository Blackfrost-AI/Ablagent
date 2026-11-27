#!/bin/bash
IMG=vllm/vllm-openai@sha256:e72944ef5ab9d1b91228d066117595847e94bf95321018a2119e70e8ba643ff6
SNAP=/workspace/hf/hub/models--thinkingmachines--Inkling-2-Small/snapshots/885d78bfca61a7e9a498e33cfa7ad35850533b8c
docker run --rm --gpus "\"device=4,5,6,7\"" --ipc=host --shm-size=16g \
  -v /workspace:/workspace \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -e PYTHONPATH=/workspace/inkling2-small/scripts \
  --entrypoint python3 "$IMG" \
  /workspace/inkling2-small/scripts/capture_hard32_vllm.py \
  --checkpoint "$SNAP" \
  --corpus /workspace/inkling2-small/input/v1_corpus_hard32.jsonl \
  --output /workspace/inkling2-small/evidence/s2-capture \
  --tp 4 --limit 64
echo "PROBE_RC=$?"
