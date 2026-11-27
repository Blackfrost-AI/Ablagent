#!/bin/bash
cd /root/inkling2-small
export CUDA_VISIBLE_DEVICES=4,5,6,7
SNAP=/workspace/hf/hub/models--thinkingmachines--Inkling-2-Small/snapshots/885d78bfca61a7e9a498e33cfa7ad35850533b8c
if [ "$1" = "probe" ]; then
  python3 scripts/capture_hard32_i2s.py --checkpoint "$SNAP" --corpus input/v1_corpus_hard32.jsonl --output evidence/s2-probe --probe
else
  python3 scripts/capture_hard32_i2s.py --checkpoint "$SNAP" --corpus input/v1_corpus_hard32.jsonl --output evidence/s2
fi
echo "CAPTURE_RC=$?"
