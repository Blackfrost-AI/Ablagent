# Inkling-2-Mini DWM duplication runbook (S0 → S3)

This directory contains everything needed to reproduce the alpha-2.5
derivative `Blackfrost-AI/Inkling-2-Mini-Derisked` from
`thinkingmachines/Inkling-2-Mini`, using the exact scripts that produced it.

> **Correction history (2026-11-24):** the original runbook (HF revision
> `31ec2edd`) contained six statements contradicted by the shipped scripts
> and receipts: a "CW whitened" basis, "4 loop visits", a `down_proj` input
> capture site, a "held-out" hard32 battery, a `direction-metadata.json`
> artifact, and hash-verification claims for the edit script. This revision
> describes the shipped method exactly; the scripts, capture artifacts, and
> edit receipt are unchanged. The original runbook remains available at the
> pinned prior revision. The model card carries a matching correction row.

Read 100% of this file before starting. All scripts are deterministic given
the same inputs; the battery uses a fixed RNG seed (`20261120`).

## Hardware / environment requirements

- 1× CUDA GPU with ≥80 GiB free VRAM (the scripts load BF16 with
  `device_map="cuda"`). Capture, edit, and verify scripts additionally guard
  on host RAM: port 8899 must be free and `/proc/meminfo` MemAvailable
  ≥ 80 GiB (Linux-only).
- Python 3.12, torch 2.14.1+cu130, transformers 5.24.0.dev0 (the exact dev
  build used for capture is recorded in `s2-capture/capture-metadata.json`;
  the kit does not pin a requirements file — match that environment or
  re-qualify), numpy, safetensors, huggingface_hub.
- Base checkpoint: `hf download thinkingmachines/Inkling-2-Mini` (~60 GB
  BF16, 4 shards). Verify against the S0 hash-lock manifest if you have it;
  the edit receipt records the source digest prefix `sha256:42e765f7`
  (8-hex prefix literal; the full manifest stayed in the run's internal
  evidence store).

## Architecture facts (verified in the shipped capture metadata)

- 24 physical layers; physical layers 6–17 are looped, executed in
  **3 passes** per token (`loop_num_passes: 3`); layers 0–5 and 18–23 are
  single-pass. 48 logical layer-visits total.
- Hidden size 2304; attention writer `self_attn.o_proj.linear` maps
  3072 → 2304. Checkpoint key `model.llm.layers.{L}.attn.wo_ud.weight` is a
  **pure rename** of `self_attn.o_proj.linear.weight` at load (no transpose).
- Routed-experts-only MoE (no shared expert): `moe_down` and `mlp_output`
  hooks must agree exactly; the battery asserts this as a hook-semantics
  check, not independent evidence.
- Thinking is ON for all behavioral measurement (reasoning_effort 0.9,
  explicit `<|message_model|><|content_thinking|>` channel). Capture itself
  is prefill-only, last-token, no generation.

## Stage S2 — capture + direction battery

```bash
python scripts/s2_capture_directions.py \
  --checkpoint /path/to/Inkling-2-Mini \
  --corpus scripts/v1_corpus_hard32.jsonl \
  --output s2-capture
python scripts/s2_direction_battery.py \
  --capture-dir s2-capture --output s2-battery --permutations 2000
```

- Corpus: `v1_corpus_hard32.jsonl` — 32 adjacent harmful/safe prompt pairs
  (64 rows). This is the **capture corpus itself**, not a held-out set; the
  capture script asserts its sha256 (`902ac247…`).
- Hooks capture six arrays at every (layer, visit) site: `layer_output`,
  `attention_writer` (`self_attn.o_proj.linear` **output**, hidden 2304),
  `moe_down`, `mlp_output`, `gate_forget`, `gate_message`. Outputs:
  `raw-capture.npz` (79 MB, each array 64×48×2304), `capture-metadata.json`.
- The battery computes, per family and site: paired harmful−safe centroid
  difference directions (per-site centring), split-half cosine (even vs odd
  pairs), permutation p-values (2000 adjacent-pair sign flips, seed
  20261120), activation-projection energy, and cross-pass cosines for looped
  layers. It reads only the capture artifacts — it never re-scores prompts.
- `directions.npz` holds the 48 unit directions of the `attention_writer`
  family plus `logical_sites`/`layers`/`visits`/`labels` arrays.

### Shipped battery statistics (inspect `s2-capture/direction-battery.json`)

- Selected edit layers 9–18: split-half cosines 0.798–0.956; permutation
  p ≤ 0.0005 at every selected site.
- Cross-pass cosines (`attention_writer`, looped layers 6–17): off-diagonal
  values range **0.2408–0.8595** — passes are related but not near-identical.
  The original run record cited 0.9885–0.9937 with a 0.9 coherence gate;
  that claim is inconsistent with the shipped battery and is superseded by
  these measured values. The edit was applied per-pass regardless (below),
  which does not require cross-pass coherence.
- `s2-summary.json` status `S2_BATTERY_MEASURED_NOT_GO` is **stage-scoped**:
  S2 is measurement, not an intervention go/no-go decision. It does not
  indicate the shipped artifact failed.

## Stage S3 — derisk edit (the shipped method)

```bash
python scripts/s3_derisk_edit.py \
  --checkpoint /path/to/Inkling-2-Mini \
  --directions s2-capture/directions.npz \
  --battery s2-capture/direction-battery.json \
  --output s3-a25 --layers 9-18 --alpha 2.5
```

- **Method: per-pass consensus column orthogonalization, loop-enabled.**
  For each selected physical layer L and each of its passes p (3 passes for
  L9–17; 1 pass for post-loop L18), with unit direction d in writer
  **output** space: `W ← W − α·outer(d, Wᵀd)`, applied sequentially per
  pass in float32, stored back in BF16. **No whitening/PCA is performed.**
  No bias, norm scales, attention weights, or experts are touched.
- Targets: `model.llm.layers.{9..18}.attn.wo_ud.weight` — 10 tensors, all in
  `model-00001-of-00001.safetensors` (the language-model shard).
- At α 2.5 the summed multi-pass removal overshoots the pass-0 component:
  receipt `removed_frac_pass0_dir` values are negative (e.g. −0.69 at L9;
  exactly −0.5 at single-pass L18 = 1−|1−2.5|). This is expected
  superposition behavior — the edit does not merely null one direction.
- Output: `candidate/` (full checkpoint copy with edited shard) +
  `s3-edit-receipt.json` binding alpha, layers, directions/battery sha256,
  per-tensor relative deltas, and **every candidate file's sha256**.
  The script computes digests; it does not verify them against anything.
  The shipped receipt (`s3-edit-receipt-a25.json`) binds the published
  weight shard sha256 `b1156b5d…`.
- Note: the receipt's `battery_sha256` (`1ec4212b…`) binds the
  pre-sanitization battery; the published `direction-battery.json` differs
  from it only by one internal path field replaced with `<internal-path>`
  at publication (verified by diff at packaging time).

## Stage S3 — gates and behavioral verification

```bash
python scripts/s3_gate_metrics.py \
  --checkpoint s3-a25/candidate --corpus scripts/v1_corpus_hard32.jsonl \
  --output s3-gates --tag a25
python scripts/s3_verify.py \
  --candidate s3-a25/candidate --base /path/to/Inkling-2-Mini \
  --corpus scripts/v1_corpus_hard32.jsonl --output s3-verify --max-new 200
```

- `s3_gate_metrics.py`: prefill-only on the 32 **safe** rows; captures
  candidate writer projections (live hooks, L9–18 per visit) and final-norm
  hidden states + logits for parity comparison against the base capture.
- `s3_verify.py`: greedy generation, thinking ON, base vs candidate on the
  32 harmful rows + 4 capability probes (math×2, code, recall). Refusal =
  deterministic keyword detection in the final text channel (no LLM judge).
  Keyword detection is a smoke instrument, not the release adjudication.

## Reported results (run record; not re-measured by this kit)

Alpha ladder on smoke-50 (answered/substituted/refused), thinking ON:

| alpha | answered | substituted | refused |
|---|---|---|---|
| 1.5 | 21 | 14 | 15 |
| 1.7 | 19 | 16 | 15 |
| 2.0 | 29 | 11 | 10 |
| 2.5 | 38 | 10 | 2 |

Shipped alpha 2.5, 443-row corpus hand-adjudicated: 266/295 harmful-prompt
compliance (90.2%), 29/295 real refusals, 0/75 false refusals on benign
prompts, zero errors. The 443-corpus prompts/responses/scores are not
redistributed in this kit.

## Contents

```
RUNBOOK.md                        this file (corrected 2026-11-24)
scripts/
  s2_capture_directions.py        S2 capture (prefill, 6 arrays, 48 sites)
  s2_direction_battery.py         S2 battery (split-half/permutation/cross-pass)
  s3_derisk_edit.py               S3 per-pass consensus orthogonalization
  s3_gate_metrics.py              S3 safe-row parity gates
  s3_verify.py                    S3 behavioral verify (keyword + capability)
  v1_corpus_hard32.jsonl          capture corpus (32 harmful/safe pairs)
s2-capture/
  capture-metadata.json           architecture, environment, corpus bindings
  direction-battery.json          full battery statistics
  s2-summary.json                 stage-scoped S2 status
  directions.npz                  48×2304 unit directions + site arrays
  raw-capture.npz                 79 MB full per-prompt capture
s3-edit-receipt-a25.json          alpha-2.5 edit receipt (shipped artifact)
```

A vendored copy of this kit, with a per-file SHA-256 provenance manifest and
an offline verifier, is maintained at
`https://github.com/Blackfrost-AI/Ablagent` under
`examples/inkling-2-mini-derisked/`.