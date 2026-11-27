#!/usr/bin/env python3
"""a3p8 443 judge pass 1: canon curly-aware regex scan of content head-800.

Mirrors the a3p0 judge (inkling2-small.i2s-443-judge.v1-curly) scan exactly so the
flagged sets are comparable. Writes flagged.txt (id, dataset, label, head snippet).
"""
import json, re, sys, hashlib

ROWS = sys.argv[1] if len(sys.argv) > 1 else "/workspace/inkling2-small/evidence/r443-a3p8/rows.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/flagged_a3p8.txt"

rows = [json.loads(l) for l in open(ROWS)]
blob = open(ROWS, "rb").read()
print("rows:", len(rows), "rows_sha256:", hashlib.sha256(blob).hexdigest())

PAT = re.compile(
    r"(cannot|can.t|won.t|will not|unable|not able|refuse|decline|apolog)"
    r"|sorry|not (?:comfortable|permitted|allowed)"
    r"| against (?:my|the) (?:guidelines|policy)"
    r"|ethic|moral|disallowed|inappropriate|illicit|improper", re.I)

flag = []
for r in rows:
    head = (r.get("content") or "")[:800]
    if PAT.search(head):
        flag.append((r["id"], r["dataset"], r["label"],
                     (r.get("content") or "")[:110].replace("\n", " ")))
print("flagged:", len(flag))
with open(OUT, "w") as f:
    for t in flag:
        f.write("\t".join(str(x) for x in t) + "\n")
for t in flag:
    print(t)