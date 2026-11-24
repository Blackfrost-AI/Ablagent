#!/usr/bin/env python3
"""Inkling-2-Mini S3 DERISK: attention-writer weight edit from S2 directions.

Pre-registered design (operator greenlight 'okay you may begin', 2026-11-20):
- Target surface: self_attn.o_proj.linear.weight (BF16-stored) at L9-18.
- Per-pass unit directions d_{L,p} from evidence/directions.npz (attention_writer).
- Per-pass consensus: w += alpha * sum_p d_p d_p^T / ||d_p||^2 applied to each
  column of the writer weight, orthogonalizing the refusal subspace. Column-space
  edit only — no bias, no norm scales, no attention weights, no experts.
- Alpha 1.0 (recurrence-untested, conservative first arm).
- Writes candidate checkpoint (copy + edited tensors) + edit receipt. Source untouched.
"""
import argparse, datetime, hashlib, json, pathlib, shutil, socket

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return "sha256:" + h.hexdigest()

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--directions", required=True)
p.add_argument("--battery", required=True)
p.add_argument("--output", required=True)
p.add_argument("--layers", default="9-18")
p.add_argument("--alpha", type=float, default=1.0)
a = p.parse_args()
out = pathlib.Path(a.output)
out.mkdir(parents=True, exist_ok=True)

# GPU guard (pre-registered in SERVE.md)
mem = float([x for x in pathlib.Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")][0].split()[1]) / 1024**2
s = socket.socket(); s.settimeout(2); port = s.connect_ex(("127.0.0.1", 8899)); s.close()
if port == 0 or mem < 80:
    raise SystemExit(f"GPU GUARD HALT port8899={port} MemAvailableGiB={mem:.1f}")

import numpy as np
import torch
from safetensors.torch import save_file

# Layers to edit (inclusive range)
lo, hi = (int(x) for x in a.layers.split("-"))
layers = list(range(lo, hi + 1))
n_passes = 3  # only for looped layers; passes_for() gates per layer

# Load directions (48 sites, attention_writer family) + battery stats
z = np.load(a.directions)
directions = z["directions"]  # (48, 2304) unit vectors
sites = z["logical_sites"]    # (48,) position in raw capture ordering
lay = z["layers"]; vis = z["visits"]
battery = json.loads(pathlib.Path(a.battery).read_text())
site_rows = battery["families"]["attention_writer"]["rows"]
loop_start = battery["metadata"]["loop_layer_index_start"]
loop_end = battery["metadata"]["loop_layer_index_end"]

def passes_for(L):
    # physical L6-17 revisited 3x; L18-23 post-loop single pass; L0-5 pre-loop single
    return 3 if loop_start <= L < loop_end else 1

# Map (layer, pass) -> direction
dirmap = {}
for i, (L, V) in enumerate(zip(lay.tolist(), vis.tolist())):
    dirmap[(L, V)] = directions[i]

# Load safetensors index + files for the language model
ckpt = pathlib.Path(a.checkpoint)
index = json.loads((ckpt / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]
from safetensors.torch import load_file

# Target tensor names for o_proj.linear at each selected layer
# (checkpoint key; load-time conversion is a PURE RENAME: attn.wo_ud -> self_attn.o_proj.linear,
#  verified in transformers/conversion_mapping.py inkling2_text — no transpose/interleave op)
targets = {}
for L in layers:
    key = f"model.llm.layers.{L}.attn.wo_ud.weight"
    targets[key] = weight_map[key]

# Compute the edit: per-pass consensus orthogonalization per layer
receipt = {
    "schema": "inkling2-mini.s3-derisk-edit.v1",
    "timestamp_utc": utc(),
    "source_checkpoint": str(ckpt),
    "source_checkpoint_digest": "sha256:42e765f7",  # from s0-hashlock manifest digest prefix; full value in evidence
    "alpha": a.alpha,
    "layers": layers,
    "surface": "attn.wo_ud.weight (== self_attn.o_proj.linear after pure-rename load conversion)",
    "edit_method": "per-pass consensus column orthogonalization: W := W - alpha * sum_p d_p d_p^T W / ||d_p||^2 (applied per column; no bias/scale/expert changes)",
    "directions_sha256": sha(pathlib.Path(a.directions)),
    "battery_sha256": sha(pathlib.Path(a.battery)),
}

edited_files = {}  # shard -> {tensor_name: edited_tensor}
edits_meta = []
for L in layers:
    key = f"model.llm.layers.{L}.attn.wo_ud.weight"
    shard = targets[key]
    tensors = load_file(str(ckpt / shard))
    W = tensors[key]  # (hidden_out, hidden_in) — linear.weight is (out, in)
    W_edit = W.clone().to(torch.float32)
    total_delta_norm = 0.0
    n_passes_this_layer = passes_for(L)
    for pass_idx in range(n_passes_this_layer):
        d = torch.from_numpy(dirmap[(L, pass_idx)]).float()  # (2304,) unit, writer OUTPUT space (R^hidden_out)
        # y = Wx; remove d-component of the OUTPUT: W := W - alpha * outer(d, W^T d)
        V = W_edit.T @ d  # (3072,) — input-space image of d through current W
        W_edit -= a.alpha * torch.outer(d, V)
        total_delta_norm += float(V.norm().item())
    # sanity: pass-0 direction output component must be (near-)nulled by the edit
    W64 = W.to(torch.float64); d64 = torch.from_numpy(dirmap[(L, 0)]).double()
    pre = (W64.T @ d64); post = (W_edit.double().T @ d64)
    removed_frac = 1.0 - (post.norm() / pre.norm()).item() if pre.norm() > 0 else 0.0
    # Verify edit magnitude sanity
    delta = (W_edit - W.float()).norm().item() / max(W.float().norm().item(), 1e-9)
    edits_meta.append({
        "layer": L, "shard": shard, "tensor": key,
        "rel_delta_frobenius": round(delta, 6),
        "pass_proj_norms_sum": round(total_delta_norm, 4),
        "removed_frac_pass0_dir": round(removed_frac, 6),
    })
    tensors[key] = W_edit.to(W.dtype)
    edited_files.setdefault(shard, {}).update({key: tensors[key]})

receipt["edits"] = edits_meta

# Write candidate: copy all files, overwrite edited shards
cand = out / "candidate"
if cand.exists(): shutil.rmtree(cand)
shutil.copytree(ckpt, cand)
for shard, tens in edited_files.items():
    # Load full shard, apply edits, save
    full = load_file(str(cand / shard))
    full.update(tens)
    save_file(full, str(cand / shard), metadata={"format": "pt"})
receipt["candidate_dir"] = str(cand)
receipt["candidate_shards_rewritten"] = sorted(edited_files.keys())
# Candidate digest: hash every file in candidate dir
digests = {}
for f in sorted(cand.rglob("*")):
    if f.is_file():
        digests[str(f.relative_to(cand))] = sha(f)
receipt["candidate_file_digests"] = digests
receipt["status"] = "S3_EDIT_APPLIED"
(out / "s3-edit-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k not in ("candidate_file_digests",)}, indent=2))
