#!/usr/bin/env python3
"""S3 gate comparison: base vs candidate (vLLM-native gate outputs).

Computes:
- writer separation: mean projection of safe-row attention_writer onto the harmful-safe
  refusal direction q per band layer. The bake removes the q-component, so the candidate's
  safe-row projection should be strongly reduced vs base (edit efficacy). Reported as
  base_proj_mean, cand_proj_mean, reduction_fraction per layer + band aggregate.
- NLL delta: candidate nll_mean - base nll_mean (capability floor; small |delta| = preserved).
- per-row NLL delta distribution.

Read-only; consumes {base,cand}-gate.npz + directions.npz. No model load.
"""
import argparse, hashlib, json, pathlib
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""): h.update(b)
    return "sha256:" + h.hexdigest()

p = argparse.ArgumentParser()
p.add_argument("--gate-dir", required=True)
p.add_argument("--directions", required=True)
p.add_argument("--layers", required=True)
p.add_argument("--alpha", type=float, required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
gd = pathlib.Path(a.gate_dir); out = pathlib.Path(a.output)
lo, hi = (int(x) for x in a.layers.split("-")); LSEL = list(range(lo, hi+1))

zb = np.load(gd / "base-gate.npz"); zc = np.load(gd / "cand-gate.npz")
Wb = zb["writer"].astype(np.float64); Wc = zc["writer"].astype(np.float64)
nllb = zb["nll"].astype(np.float64); nllc = zc["nll"].astype(np.float64)
assert Wb.shape == Wc.shape == (32, len(LSEL), 4608), f"shapes {Wb.shape} {Wc.shape}"

zdir = np.load(a.directions)
D = zdir["directions_attention_writer"].astype(np.float64)  # [48,4608] unit
Q = D[LSEL]  # [band,4608]
Q = Q / np.linalg.norm(Q, axis=-1, keepdims=True)

# projection of each safe-row writer onto q: [32, band]
projb = np.einsum("rld,ld->rl", Wb, Q)
projc = np.einsum("rld,ld->rl", Wc, Q)
base_mean = projb.mean(axis=0); cand_mean = projc.mean(axis=0)
base_absmean = np.abs(projb).mean(axis=0); cand_absmean = np.abs(projc).mean(axis=0)
reduction = 1.0 - (cand_absmean / np.maximum(base_absmean, 1e-9))

per_layer = []
for j, L in enumerate(LSEL):
    per_layer.append({"layer": L,
                      "base_proj_mean": float(base_mean[j]), "cand_proj_mean": float(cand_mean[j]),
                      "base_absmean": float(base_absmean[j]), "cand_absmean": float(cand_absmean[j]),
                      "abs_reduction_fraction": float(reduction[j])})

nll_delta = nllc - nllb
report = {
    "schema": "inkling2-small.s3-gate-comparison.v1",
    "alpha": a.alpha, "layers": LSEL,
    "base_gate_sha256": sha(gd / "base-gate.npz"), "cand_gate_sha256": sha(gd / "cand-gate.npz"),
    "directions_sha256": sha(a.directions),
    "base_nll_mean": float(nllb.mean()), "cand_nll_mean": float(nllc.mean()),
    "nll_delta_mean": float(nll_delta.mean()), "nll_delta_abs_max": float(np.abs(nll_delta).max()),
    "nll_delta_per_row": nll_delta.tolist(),
    "writer_abs_reduction_band_mean": float(reduction.mean()),
    "writer_abs_reduction_band_min": float(reduction.min()),
    "per_layer": per_layer,
    "interpretation": "abs_reduction_fraction = 1 - cand|proj|/base|proj| on safe rows; "
                      "high = refusal-direction component removed by the bake (edit efficacy). "
                      "nll_delta small = capability floor preserved.",
}
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": "S3_GATE_COMPARISON_DONE",
                  "base_nll_mean": report["base_nll_mean"], "cand_nll_mean": report["cand_nll_mean"],
                  "nll_delta_mean": report["nll_delta_mean"],
                  "writer_abs_reduction_band_mean": report["writer_abs_reduction_band_mean"]}, indent=2))