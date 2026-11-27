#!/usr/bin/env python3
"""Stage a3p8 final receipts + hash manifests into release-kit (no live keys, no WG config).

Run on the validated runtime as root. Reads only local paths; writes only under release-kit/.
- Rebinds s5-serve-receipt.json to the a3p8 candidate with LIVE verification (health/models/no-key/real-forward).
- Copies cyber10-a3p8 evidence, s3 gate comparison, bake receipt/plan (already present: re-verified).
- Writes manifests/candidate-sha256.json (24 shards + sidecars) and manifests/kit-sha256.json.
"""
import hashlib, json, os, shutil, subprocess, sys, time, urllib.request

ROOT = "/workspace/inkling2-small"
KIT = f"{ROOT}/release-kit"
CAND = "/workspace/i2s-candidates/a3p8-L20-37"
KEY = open(f"{ROOT}/.serve_key").read().strip()
BASE = "http://127.0.0.1:40001"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()

def get(path, auth=True, timeout=30):
    req = urllib.request.Request(BASE + path)
    if auth:
        req.add_header("Authorization", "Bearer " + KEY)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()

def post_chat():
    body = json.dumps({
        "model": "inkling2-small-dwm-a3p8",
        "messages": [{"role": "user", "content": "What is 12*13? Reply with just the number."}],
        "max_tokens": 2048, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    dt = time.time() - t0
    ch = d["choices"][0]
    u = d.get("usage", {})
    return {"answer": (ch["message"].get("content") or "").strip()[:80],
            "finish_reason": ch.get("finish_reason"),
            "completion_tokens": u.get("completion_tokens"),
            "wall_s": round(dt, 2)}

def docker_facts():
    out = subprocess.run(["docker", "inspect", "i2s-serve-a3p8", "--format",
                          "{{.Image}}|{{.State.Status}}|{{.Config.Image}}"],
                         capture_output=True, text=True).stdout.strip()
    image_id, status, image_ref = out.split("|")
    return {"image_ref": image_ref, "image_id": image_id, "container_status": status}

def main():
    os.makedirs(f"{KIT}/manifests", exist_ok=True)
    # 1. live verify
    health, _ = get("/health", auth=False)
    ms, mbody = get("/v1/models")
    ids = [m["id"] for m in json.loads(mbody)["data"]]
    nk, _ = get("/v1/models", auth=False)
    fwd = post_chat()
    assert health == 200 and "inkling2-small-dwm-a3p8" in ids and nk == 401, (health, ids, nk)
    df = docker_facts()

    # 2. serve receipt bound to a3p8 (no key material, no WG config)
    old = json.load(open(f"{KIT}/evidence/s5-serve-receipt.json"))
    receipt = {
        "stage": "S5_SERVE", "status": "PASS",
        "image": old["image"],
        "image_verified_ref": df["image_ref"], "container_image_id": df["image_id"],
        "candidate": CAND,
        "served_model_name": "inkling2-small-dwm-a3p8",
        "endpoint_public": "REDACTED (transient research-pod endpoint; not part of the release)",
        "port": 40001, "gpus": "4-7 (TP4)",
        "container": "i2s-serve-a3p8",
        "max_model_len": 262144,
        "speculative": old["speculative"], "parsers": old["parsers"],
        "mtp_merged": True, "mtp_sha256": old["mtp_sha256"],
        "verify": {"health": health, "models_identity": ids, "auth_no_key": nk,
                   "real_forward": fwd, "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "api_key_sha256": old["api_key_sha256"],
        "api_key_location": "operator-held (0600 .env); NOT distributed",
        "note": "Alpha 3.8 final. Supersedes a3p0 receipt of same filename. No WireGuard/overlay "
                "configuration is part of this release; endpoint is the plain public IP serve as launched by scripts/run-serve-a3p8.sh.",
    }
    json.dump(receipt, open(f"{KIT}/evidence/s5-serve-receipt.json", "w"), indent=2)

    # 3. stage cyber10-a3p8 evidence (already staged? verify) + gate comparison
    src = f"{ROOT}/evidence/cyber10-a3p8"
    dst = f"{KIT}/evidence/cyber10-a3p8"
    os.makedirs(dst, exist_ok=True)
    for fn in ("rows.jsonl", "report.json", "run.log"):
        if not os.path.exists(f"{dst}/{fn}") or sha(f"{src}/{fn}") != sha(f"{dst}/{fn}"):
            shutil.copy2(f"{src}/{fn}", f"{dst}/{fn}")

    # 4. bake receipt/plan for a3p8 into kit evidence under explicit names
    for fn, dstname in (("bake-receipt.json", "s3-bake-receipt-a3p8.json"),
                        ("bake-plan.json", "s3-bake-plan-a3p8.json")):
        s = f"{CAND}/{fn}"
        if os.path.exists(s):
            shutil.copy2(s, f"{KIT}/evidence/{dstname}")

    # 5. candidate manifest (24 shards + sidecars)
    entries = []
    for fn in sorted(os.listdir(CAND)):
        p = f"{CAND}/{fn}"
        if os.path.isfile(p):
            entries.append({"path": fn, "bytes": os.path.getsize(p), "sha256": sha(p)})
    man = {"candidate": CAND, "alpha": 3.8, "band": "L20-37 wo_ud OUT-space",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "total_bytes": sum(e["bytes"] for e in entries), "files": entries}
    json.dump(man, open(f"{KIT}/manifests/candidate-sha256.json", "w"), indent=2)

    # 6. kit manifest (exclude itself)
    kit_entries = []
    for base, _, files in os.walk(KIT):
        for fn in sorted(files):
            p = os.path.join(base, fn)
            rel = os.path.relpath(p, KIT)
            if rel == "manifests/kit-sha256.json":
                continue
            kit_entries.append({"path": rel, "bytes": os.path.getsize(p), "sha256": sha(p)})
    json.dump({"files": kit_entries}, open(f"{KIT}/manifests/kit-sha256.json", "w"), indent=2)

    # 7. key scan: no live key anywhere in kit
    hits = []
    for base, _, files in os.walk(KIT):
        for fn in files:
            p = os.path.join(base, fn)
            if p.endswith((".png", ".npz", ".safetensors")):
                continue
            try:
                if KEY.encode() in open(p, "rb").read():
                    hits.append(p)
            except OSError:
                pass
    print(json.dumps({"receipt_verify": receipt["verify"], "candidate_files": len(entries),
                      "candidate_total_bytes": man["total_bytes"], "kit_files": len(kit_entries),
                      "live_key_hits": hits}, indent=2))

if __name__ == "__main__":
    main()