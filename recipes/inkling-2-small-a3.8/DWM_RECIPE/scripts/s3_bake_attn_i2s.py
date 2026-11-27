#!/usr/bin/env python3
"""S3 bake: Inkling-2-Small attention-only, alpha 3.0 (operator greenlit 2026-11-27:
"run s0-s3 hard32 bake at 3.0 and serve").

Targets: attention_writer wo_ud (= self_attn.o_proj module output,
Inkling2GatedOutputProjection.linear), band layers from measured battery.
Shape [out=4608, in=6144] PyTorch-linear stored; directions live in OUT space
(dim 0, 4608). Transpose to [in,out], right-multiply rank-1 projector, back.
BF16 direct edit — NO requantization anywhere. Experts (NVFP4 L0-39 / MXFP8
L40-47), router, sconv, attn_res, vision tower, lm_head, embed, MTP/DSpark:
all untouched, byte-exact.

Math: W := W - alpha * W @ P_span, P_span = q q^T from the layer's unit
harmful-safe direction (rank-1 per layer). directions_attention_writer[48,4608],
present-masked; every band layer asserted permutation_p <= 0.0005 and
split_half >= 0.90 at runtime (band L20-37: all p=0.0005, split 0.912-0.975).

Guards (proven canon s3_bake_attn_v1.py):
- directions.npz + battery sha gates
- GPU/mem guard: port 8901 closed, MemAvailable >= 80 GiB
- source checkpoint never written; non-safetensors files copied (incl. mtp.safetensors)
- untouched shards copied byte-exact (sha-verified); untouched tensors in edited
  shards byte-exact re-read; INCOMPLETE sentinel removed only on PASS
"""
import argparse, datetime, hashlib, json, pathlib, shutil, socket

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--capture-dir", required=True)
p.add_argument("--output", required=True)
p.add_argument("--alpha", type=float, default=3.0)
p.add_argument("--attn-layers", required=True, help="inclusive band, e.g. 18-35")
p.add_argument("--directions-sha256", required=True)
p.add_argument("--battery-sha256", required=True)
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ckpt = pathlib.Path(a.checkpoint)
cap = pathlib.Path(a.capture_dir)
out = pathlib.Path(a.output)

dsha = sha(cap / "directions.npz")
bsha = sha(cap / "direction-battery.json")
if dsha != a.directions_sha256 or bsha != a.battery_sha256:
    raise SystemExit(f"SHA GATE FAIL directions={dsha} battery={bsha}")

z = np.load(cap / "directions.npz")
D = z["directions_attention_writer"].astype(np.float64)   # [48,4608] OUT space
pres = z["present_attention_writer"]
battery = json.loads((cap / "direction-battery.json").read_text())
rows = {r["layer"]: r for r in battery["families"]["attention_writer"]["rows"] if r.get("present")}

lo, hi = (int(x) for x in a.attn_layers.split("-"))
attn_layers = list(range(lo, hi + 1))
for L in attn_layers:
    r = rows[L]
    assert pres[L] and r["permutation_p"] <= 0.0005 and r["split_half_cosine"] >= 0.90, \
        f"band layer {L} fails battery gate: {r}"
    n = np.linalg.norm(D[L])
    assert 0.99 < n < 1.01, f"direction L{L} not unit: {n}"

index = json.loads((ckpt / "model.safetensors.index.json").read_text())
wm = index["weight_map"]
attn_keys = {f"model.llm.layers.{L}.attn.wo_ud.weight": L for L in attn_layers}
for k in attn_keys:
    if k not in wm:
        raise SystemExit(f"missing target {k}")
    if "mtp" in k.lower() or "visual" in k:
        raise SystemExit(f"excluded target leaked: {k}")

def projector(L):
    v = D[L] / np.linalg.norm(D[L])
    P = np.outer(v, v)
    return torch.from_numpy(P).float().cuda()

plan = {
    "schema": "inkling2-small.attn-only-s3-bake-plan.v1", "ts": utc(),
    "alpha": a.alpha, "source": str(ckpt), "output": str(out),
    "revision": "885d78bfca61a7e9a498e33cfa7ad35850533b8c",
    "attn_layers": attn_layers, "attn_targets": len(attn_keys),
    "family": "attention_writer", "key_mapping": "attn.wo_ud -> self_attn.o_proj.linear (GatedOutputProjection)",
    "directions_sha256": dsha, "battery_sha256": bsha,
    "math": "W := W - alpha * W @ P_span (rank-1 outer of unit OUT-space direction)",
    "quantization": "none — BF16 direct on BF16 attention surface; NVFP4/MXFP8 experts untouched",
    "band_evidence": {str(L): {"split_half": rows[L]["split_half_cosine"],
                               "p": rows[L]["permutation_p"],
                               "energy": rows[L]["energy_mean_projection_squared"]} for L in attn_layers},
}
out.mkdir(parents=True, exist_ok=True)
(out / "bake-plan.json").write_text(json.dumps(plan, indent=2))
print(json.dumps({"attn": len(attn_keys), "alpha": a.alpha, "layers": attn_layers}), flush=True)
if a.dry_run:
    print("DRY_RUN_PASS", flush=True)
    raise SystemExit(0)

mem = float([x for x in pathlib.Path("/proc/meminfo").read_text().splitlines()
             if x.startswith("MemAvailable:")][0].split()[1]) / 1024**2
s = socket.socket(); s.settimeout(2); port = s.connect_ex(("127.0.0.1", 8901)); s.close()
if port == 0 or mem < 80:
    raise SystemExit(f"GPU GUARD HALT port8901={port} MemAvailableGiB={mem:.1f}")

(out / "INCOMPLETE").write_text(json.dumps({"started": utc()}))
for f in ckpt.iterdir():
    if f.is_file() and f.suffix != ".safetensors":
        shutil.copy2(f, out / f.name)
# MTP/DSpark speculator file: separate from index, copy byte-exact + sha verify
mtp_src = ckpt / "mtp.safetensors"
if mtp_src.exists():
    shutil.copy2(mtp_src, out / "mtp.safetensors")
    if sha(mtp_src) != sha(out / "mtp.safetensors"):
        raise SystemExit("MTP COPY MISMATCH")

projectors = {L: projector(L) for L in attn_layers}
targets_by_shard = {}
for k, L in attn_keys.items():
    targets_by_shard.setdefault(wm[k], []).append(k)

receipt = {
    "schema": "inkling2-small.attn-only-s3-bake-receipt.v1",
    "plan_sha256": sha(out / "bake-plan.json"),
    "alpha": a.alpha, "started": utc(),
    "edited": [], "untouched_shards": [], "untouched_tensors": 0,
    "mtp_sha256": sha(out / "mtp.safetensors") if mtp_src.exists() else None,
}
n_edit = 0
for shard, keys in sorted(targets_by_shard.items()):
    src = ckpt / shard
    tensors = {}
    with safe_open(src, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            t = f.get_tensor(k)
            if k in attn_keys:
                L = attn_keys[k]
                assert tuple(t.shape) == (4608, 6144) and t.dtype == torch.bfloat16
                W = t.float().cuda()
                Wio = W.T.contiguous()                       # [6144,4608]
                P = projectors[L]                            # [4608,4608] OUT space
                edited = (Wio - a.alpha * (Wio @ P)).T.contiguous()
                cos = torch.nn.functional.cosine_similarity(
                    W.flatten(), edited.flatten(), dim=0).item()
                receipt["edited"].append({
                    "key": k, "layer": L, "kind": "attn",
                    "weight_cos_pre_post": cos,
                    "delta_abs_max": (edited - W).abs().max().item()})
                t = edited.to(torch.bfloat16).cpu()
                n_edit += 1
            tensors[k] = t
    save_file(tensors, str(out / shard), metadata=meta or {"format": "pt"})
    print(f"wrote {shard} targets={len(keys)}", flush=True)

for shard in sorted(set(wm.values())):
    if shard in targets_by_shard:
        continue
    shutil.copy2(ckpt / shard, out / shard)
    s1, s2 = sha(ckpt / shard), sha(out / shard)
    receipt["untouched_shards"].append({"shard": shard, "sha_match": s1 == s2})
    if s1 != s2:
        raise SystemExit(f"UNTOUCHED SHARD COPY MISMATCH {shard}")

edited_keys = {e["key"] for e in receipt["edited"]}
mismatch = []
total_untouched = 0
for shard in sorted(targets_by_shard):
    with safe_open(ckpt / shard, framework="pt") as f, \
         safe_open(out / shard, framework="pt") as g:
        for k in f.keys():
            if k in edited_keys:
                continue
            a_t, b_t = f.get_tensor(k), g.get_tensor(k)
            total_untouched += 1
            if a_t.dtype != b_t.dtype or a_t.shape != b_t.shape or not torch.equal(a_t, b_t):
                mismatch.append(k)
receipt["untouched_tensors"] = total_untouched
receipt["untouched_mismatches"] = mismatch[:10]
receipt["n_edited_tensors"] = n_edit
receipt["finished"] = utc()
ok = (not mismatch and n_edit == len(attn_keys)
      and all(s["sha_match"] for s in receipt["untouched_shards"]))
receipt["status"] = "PASS" if ok else "FAIL"
(out / "bake-receipt.json").write_text(json.dumps(receipt, indent=2))
if ok:
    (out / "INCOMPLETE").unlink(missing_ok=True)
print("BAKE_" + receipt["status"],
      json.dumps({"edited": n_edit, "untouched_tensors": total_untouched,
                  "untouched_shards": len(receipt["untouched_shards"])}), flush=True)