#!/usr/bin/env python3
"""S2 direction battery for Inkling-2-Small (48 physical layers, no loop passes).
Adapted from Inkling-Small direction_battery_v1.py (proven). No weight changes.
Families: layer_output, attention_writer, mlp_output, moe_experts.
No shared experts in this architecture (n_shared_experts=0) — family absent by design.
"""
import argparse, hashlib, json, pathlib, datetime
import numpy as np
p = argparse.ArgumentParser(); p.add_argument('--capture-dir', required=True); p.add_argument('--output', required=True); p.add_argument('--permutations', type=int, default=2000); a = p.parse_args()
out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True); cp = pathlib.Path(a.capture_dir)
meta = json.loads((cp / 'capture-metadata.json').read_text()); z = np.load(cp / 'raw-capture.npz')
assert meta['row_count'] == 64 and meta['pair_count'] == 32
physical = meta['physical_layers']; hidden = meta['hidden_size']
assert physical == 48 and hidden == 4608
labels = np.array([1 if x == 'harmful' else 0 for x in meta['labels']], dtype=np.int8)
assert np.array_equal(labels, np.tile([1, 0], 32))
rng = np.random.default_rng(20261127)
permutations = np.array([np.array([labels[i] for j in range(32) for i in rng.permutation([2*j, 2*j+1])], dtype=np.int8) for _ in range(a.permutations)])
assert np.all((permutations[:, ::2] + permutations[:, 1::2]) == 1)
families = ['layer_output', 'attention_writer', 'mlp_output', 'moe_experts']
presence = {f: np.array(meta['family_presence'][f], dtype=bool) for f in families}

def unit(v):
    n = np.linalg.norm(v.astype(np.float64)); return v / n if n > 0 and np.isfinite(n) else np.zeros_like(v)
def cos(x, y):
    return float(np.dot(unit(x), unit(y)))

results = {}
directions = {f: np.zeros((physical, hidden), dtype=np.float32) for f in families}
for family in families:
    x = z[family].astype(np.float64)
    assert x.shape == (64, physical, hidden) and np.isfinite(x).all()
    pres = presence[family]
    x -= x.mean(axis=0, keepdims=True)
    harmful = x[0::2]; safe = x[1::2]
    delta = (harmful - safe).mean(axis=0)
    norms = np.linalg.norm(delta, axis=1)
    unit_delta = delta / np.maximum(norms[:, None], 1e-30)
    directions[family] = unit_delta.astype(np.float32)
    halves = []
    for h in range(2):
        idx = np.arange(h, 32, 2)
        halves.append((x[2*idx] - x[2*idx+1]).mean(axis=0))
    split = np.array([cos(halves[0][j], halves[1][j]) for j in range(physical)])
    signed = (x[0::2] - x[1::2])
    null = np.empty((a.permutations, physical), dtype=np.float64)
    for n in range(a.permutations):
        signs = (permutations[n, 0::2] * 2 - 1).astype(np.float64)
        null[n] = np.linalg.norm((signed * signs[:, None, None]).mean(axis=0), axis=1)
    pvals = (1 + (null >= norms[None, :]).sum(axis=0)) / (a.permutations + 1)
    famres = {'rows': [], 'direction_norms': norms.tolist(), 'split_half_cosine': split.tolist(),
              'permutation_p': pvals.tolist(), 'present_layers': pres.tolist(),
              'null_norm_quantiles': {str(q): np.quantile(null, q, axis=0).tolist() for q in [0.05, 0.5, 0.95]},
              'null_permutations': a.permutations}
    for j in range(physical):
        if not pres[j]:
            famres['rows'].append({'layer': j, 'family': family, 'present': False}); continue
        q = x[:, j] @ unit_delta[j]
        famres['rows'].append({'layer': j, 'family': family, 'present': True,
                               'direction_norm': float(norms[j]), 'split_half_cosine': float(split[j]),
                               'permutation_p': float(pvals[j]),
                               'energy_mean_projection_squared': float(np.mean(q**2)),
                               'energy_harmful_mean': float(np.mean(q[labels == 1]**2)),
                               'energy_safe_mean': float(np.mean(q[labels == 0]**2))})
    results[family] = famres

crossfam = {}
for l in range(physical):
    crossfam[str(l)] = {f"{fa}|{fb}": cos(directions[fa][l], directions[fb][l])
                        for fa in families for fb in families if fa < fb and presence[fa][l] and presence[fb][l]}
(out / 'cross-family-cosines.json').write_text(json.dumps(crossfam, indent=1) + '\n')
(out / 'direction-battery.json').write_text(json.dumps({
    'schema': 'inkling2-small.s2-direction-battery.v1', 'metadata': meta,
    'permutation_count': a.permutations,
    'split_half': 'even pairs vs odd pairs; cosine of paired difference vectors',
    'normalization': 'paired per-site L2 centroid differences; per-site harmful/safe centring',
    'families': results}, indent=2) + '\n')
np.savez_compressed(out / 'directions.npz', labels=labels, layers=np.arange(physical, dtype=np.int16),
                    **{f'directions_{f}': directions[f] for f in families},
                    **{f'present_{f}': presence[f] for f in families})
summary = {'status': 'S2_BATTERY_MEASURED', 'measurement_complete': True,
           'reason': 'S2 capture/battery complete; band selection follows from measured rows; operator greenlit S0-S3 alpha 3.0',
           'capture_rows': 64, 'physical_layers': physical, 'families': families,
           'corpus_sha256': meta['corpus_sha256'],
           'directions_sha256': 'sha256:' + hashlib.sha256((out / 'directions.npz').read_bytes()).hexdigest(),
           'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out / 's2-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))