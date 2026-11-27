#!/bin/bash
set -e
cd /root/inkling2-small
python3 - <<PYEOF
from huggingface_hub import HfApi
import json
i = HfApi().model_info("thinkingmachines/Inkling-2-Small", files_metadata=True)
meta=[{"path":s.rfilename,"size":s.size,"lfs":(s.lfs.sha256 if s.lfs else None)} for s in i.siblings]
open("evidence/hf-files-manifest.json","w").write(json.dumps(meta,indent=1))
open("evidence/revision.txt","w").write(i.sha+"\n")
print("I2S_REVISION", i.sha, "FILES", len(meta), "BYTES", sum(m["size"] or 0 for m in meta))
PYEOF
REV=$(cat evidence/revision.txt)
hf download thinkingmachines/Inkling-2-Small --revision "$REV"
echo "I2S_PULL_DONE rc=$?"
