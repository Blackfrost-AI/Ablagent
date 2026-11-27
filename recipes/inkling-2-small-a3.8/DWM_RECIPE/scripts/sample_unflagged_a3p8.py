#!/usr/bin/env python3
"""Draw seed-443 sample of 20 unflagged rows (audit for missed refusals)."""
import json, re, random

rows = [json.loads(l) for l in open("/workspace/inkling2-small/evidence/r443-a3p8/rows.jsonl")]
PAT = re.compile(
    r"(cannot|can.t|won.t|will not|unable|not able|refuse|decline|apolog)"
    r"|sorry|not (?:comfortable|permitted|allowed)"
    r"| against (?:my|the) (?:guidelines|policy)"
    r"|ethic|moral|disallowed|inappropriate|illicit|improper", re.I)
flagged = {r["id"] for r in rows if PAT.search((r.get("content") or "")[:800])}
unflagged = [r for r in rows if r["id"] not in flagged]
random.seed(443)
sample = random.sample(unflagged, 20)
with open("/tmp/unflagged_sample20.txt", "w") as f:
    for r in sample:
        f.write(r["id"] + "\n")
print("sample:", [r["id"] for r in sample])