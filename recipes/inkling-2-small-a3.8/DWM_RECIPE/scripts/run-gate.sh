#!/bin/bash
# args: CHECKPOINT TAG OUTDIR
IMG=vllm/vllm-openai@sha256:e72944ef5ab9d1b91228d066117595847e94bf95321018a2119e70e8ba643ff6
CKPT="$1"; TAG="$2"; OUTDIR="$3"
docker run --rm --gpus "\"device=4,5,6,7\"" --ipc=host --shm-size=16g \
  -v /workspace:/workspace \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -e PYTHONPATH=/workspace/inkling2-small/scripts \
  --entrypoint python3 "$IMG" \
  /workspace/inkling2-small/scripts/s3_gate_vllm.py \
  --checkpoint "$CKPT" \
  --corpus /workspace/inkling2-small/input/v1_corpus_hard32.jsonl \
  --output "$OUTDIR" --tag "$TAG" --layers 20-37 --tp 4
echo "GATE_RC=$?"
