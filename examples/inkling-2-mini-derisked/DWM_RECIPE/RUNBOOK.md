# DWM_RECIPE — Inkling-2-Mini duplication runbook

This directory contains everything needed to **reproduce the alpha-2.5 derivative from a clean upstream base**: the exact capture/edit/verify scripts, the captured refusal direction, the hard-32 direction battery, capture evidence, and this runbook.

**Read 100% of this file and `../deployment-kit/AGENT_INSTRUCTIONS.md` before running anything.**

## 1. Architecture (why this model is different)

Inkling-2-Mini is a **loop transformer**:

- 24 physical layers; physical layers **6–17** form a **loop block** executed **3 times**.
- The model presents **48 logical layer-visits** (`model.layers[i]` for `i in 0..47`).
- Per logical layer, `layer_idx` (physical, 0–23) and `loop_count` (which visit) describe the mapping.
- Each attention block exposes a fused down-projection target `attn.wo_ud` with input dimension 2304 — the refusal direction is defined in that down-projection input space.
- MTP (`model.mtp`) provides the draft head used by DSpark speculative decoding; it is a separate module and must be preserved.

Because of the loop, a direction captured at one logical visit can bleed into the other visits of the same physical layer. Direction targeting therefore **zeroes entries whose physical layer falls outside the allowed set**, and the edit target is selected by `layer_idx` (physical), not by logical index.

## 2. Stage graph

```
S0 intake → S1 characterize → S2 capture+battery → S3 edit (alpha) → S3 verify → S5 serve → S6 bench
```

### S0 — intake
Obtain the clean upstream base `thinkingmachines/Inkling-2-Mini` and hash-lock every source file. Record the base revision. Never begin an edit on a previously modified checkpoint.

### S1 — characterize
Inventory the architecture: confirm `model_type == "inkling2mini"`, the loop block range, `attn.wo_ud` shapes, and the MTP module. Record `S1-characterize.json`.

### S2 — refusal-direction capture
`scripts/s2_capture_directions.py` runs the base on CUDA and captures, at `layers{i}.attn.down_proj` for logical layers 9–18, the residual stream for 32 harmful + 32 benign prompts (max 256 tokens each). For each logical layer it computes:

```
dir = normalize( mean(harmful residual) − mean(benign residual) )
cosine_invariance = cos(dir@pass0, dir@pass1, dir@pass2)
```

- Output `directions.npz` is `(48, 2304)` float32 = **432 KiB raw (442,368 bytes)**; the NPZ file is **406,708 bytes (~397 KiB)**.
- **48 direction sites**: layers 9–18, each with 4 loop visits + metadata.
- `raw-capture.npz` is **79,118,677 bytes (~75.45 MiB)** — includes 64 prompts × hidden states.
- **Hooks:** a `register_forward_hook` on each `attn.down_proj` module reads `output[:, -1, :].float()` (last-token residual) and stores it as `t{loop}h_{layer}_resid`.
- Layers whose loop-pass cosine invariance is below the 0.9 threshold are quarantined; this capture passed with invariance 0.9885–0.9937.

### S2 battery — direction validation
`scripts/s2_direction_battery.py` scores the candidate direction on **hard32** (`scripts/v1_corpus_hard32.jsonl`), the held-out direction-only battery, per selected layer and per loop pass. Output: `s2-capture/direction-battery.json`. hard32 is a direction-selection instrument, not a release verdict.

### S3 — edit (this artifact = alpha 2.5)
`scripts/s3_derisk_edit.py`:

- target: `attn.wo_ud` only (down-projection input space)
- layers: **9–18** (physical `layer_idx`)
- alpha: **2.5** → 10 target tensors edited (one per physical layer in range)
- basis: **Whitened (CW)** — PCA whitening on benign residuals (PCA dims = 10% of benign corpus), whitened direction `w = W^T d`, renormalized
- loop: **enabled**, `visits_key = "three_passes"`; the edit is applied identically across all three loop visits
- always builds **from the clean base**; never stacks alpha on alpha

Command (paths are placeholders — set your own):

```sh
python scripts/s3_derisk_edit.py \
  --checkpoint /path/to/clean/base \
  --directions s2-capture/directions.npz \
  --battery s2-capture/direction-battery.json \
  --output /path/to/output \
  --layers 9-18 --alpha 2.5
```

The script verifies file hashes (only the target weight shard may differ), runs an artifact self-test, and emits an edit receipt.

### S3 verify
`scripts/s3_verify.py` + `scripts/s3_gate_metrics.py` re-run the benign/harmful probes on the candidate and confirm quality gates (KL divergence, PPL delta, coherence). Receipts in `s3-edit-receipt-a25.json`.

### S5 — serve
Serve exactly per `../deployment-kit/SERVE.md`: vLLM `inkling-2-mini` image, `inkling` reasoning parser, DSpark `num_speculative_tokens=7`, MTP preserved, max-model-len 262144.

### S6 — bench
443-row R1 refusal benchmark (advbench + strongreject + xstest) and a 50-prompt smoke set, bare prompts, 32k answer budget, 8 workers. This artifact: harm-arm compliance 266/295 (90.2%), false refusals 0/75, zero errors.

## 3. Reproducing a different alpha

Re-run S3 from the **same clean base** with a different `--alpha`. Do not apply an edit on top of an existing derivative. Observed smoke-50 answer/refusal counts: alpha 1.5 = 21/15, 1.7 = 19/15, 2.0 = 29/10, 2.5 = 38/2.

## 4. Files in this directory

| Path | Role |
|---|---|
| `scripts/s2_capture_directions.py` | S2 capture (hooks + per-loop direction) |
| `scripts/s2_direction_battery.py` | hard32 direction battery |
| `scripts/s3_derisk_edit.py` | S3 weight edit (CW basis, loop-enabled) |
| `scripts/s3_verify.py`, `scripts/s3_gate_metrics.py` | S3 verification |
| `scripts/v1_corpus_hard32.jsonl` | hard32 direction battery corpus |
| `s2-capture/directions.npz` | Captured refusal direction (48×2304 f32) |
| `s2-capture/raw-capture.npz` | Full per-prompt residual capture |
| `s2-capture/direction-metadata.json`, `capture-metadata.json` | Capture provenance |
| `s2-capture/direction-battery.json` | hard32 battery result |
| `s2-capture/s2-summary.json` | S2 stage summary |
| `s3-edit-receipt-a25.json` | Final edit receipt |

## 5. Hard rules (repeat)

- Always from clean base. Never stack edits.
- Always preserve MTP byte-identical.
- Target `attn.wo_ud` down-projection space; select by physical `layer_idx`.
- Whitened (CW) basis, loop-enabled, three_passes.
- Sanitize before publishing: no IPs, hostnames, paths, or credentials.
