#!/bin/bash
cd /workspace/inkling2-small
python3 scripts/s3_bake_attn_i2s.py \
  --checkpoint /workspace/hf/hub/models--thinkingmachines--Inkling-2-Small/snapshots/885d78bfca61a7e9a498e33cfa7ad35850533b8c \
  --capture-dir evidence/s2-battery \
  --output /workspace/i2s-candidates/a3p8-L20-37 \
  --alpha 3.8 --attn-layers 20-37 \
  --directions-sha256 95a6c68f23774cba65bee0aacea9fd252965a78b06c095020d5560fc6b8e8a47 --battery-sha256 b591b177376c8b83840a2bac14653c4a47fd4c13cd92ca1c0393728d470daaaf
echo "BAKE_RC=$?"
