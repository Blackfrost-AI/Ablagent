#!/usr/bin/env python3
"""S0 hash-lock: verify every downloaded file against pinned HF manifest (size + LFS sha256)."""
import hashlib, json, pathlib, sys, datetime

SNAP = pathlib.Path("/workspace/hf/hub/models--thinkingmachines--Inkling-2-Small/snapshots/885d78bfca61a7e9a498e33cfa7ad35850533b8c")
MANIFEST = pathlib.Path("/root/inkling2-small/evidence/hf-files-manifest.json")
OUT = pathlib.Path("/root/inkling2-small/evidence")

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

meta = json.load(open(MANIFEST))
results, fails = [], []
for m in sorted(meta, key=lambda x: x["path"]):
    f = SNAP / m["path"]
    if not f.exists():
        fails.append((m["path"], "MISSING")); continue
    size = f.stat().st_size
    if size != m["size"]:
        fails.append((m["path"], f"SIZE {size} != {m['size']}")); continue
    if m["lfs"]:
        d = sha256(f)
        if d != m["lfs"]:
            fails.append((m["path"], f"SHA {d} != {m['lfs']}")); continue
        results.append({"path": m["path"], "size": size, "sha256": d, "verified": "lfs-sha256"})
    else:
        results.append({"path": m["path"], "size": size, "sha256": sha256(f), "verified": "size+local-sha256"})
    print(f"OK {m['path']} {size}", flush=True)

report = {
    "schema": "inkling2-small.s0-hashlock.v1",
    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "revision": "885d78bfca61a7e9a498e33cfa7ad35850533b8c",
    "files_total": len(meta), "files_verified": len(results), "failures": fails,
    "total_bytes": sum(r["size"] for r in results),
    "results": results,
}
(OUT / "s0-hashlock.json").write_text(json.dumps(report, indent=1) + "\n")
lines = ["# Inkling-2-Small HASHES (rev 885d78bfca61a7e9a498e33cfa7ad35850533b8c)", ""]
lines += [f"{r['sha256']}  {r['size']:>14}  {r['path']}" for r in results]
(OUT / "HASHES.md").write_text("\n".join(lines) + "\n")
if fails:
    print("S0_HASHLOCK_FAIL", fails); sys.exit(1)
print(f"S0_HASHLOCK_PASS files={len(results)} bytes={report['total_bytes']}")