# RUNBOOK — Inkling-2-Small DWM (hard32 → L20–37 wo_ud OUT-space → α3.8)

This runbook documents the shipped method exactly as executed on 2026-11-27,
bound to the artifacts in this directory. Every hash below is in
`../manifests/candidate-sha256.json` or `../manifests/kit-sha256.json`.

## Inputs (immutable)

| Input | Identity |
|---|---|
| Base checkpoint | `thinkingmachines/Inkling-2-Small` rev `885d78bfca61a7e9a498e33cfa7ad35850533b8c` |
| S0 hashlock | 43/43 files, 393,510,050,371 bytes, SHA-256 vs HF LFS oids, 0 failures (`evidence/s0/s0-hashlock.json`) |
| Capture corpus | hard32 universal bank: 32 harmful + 32 paired safe mirrors (`input/v1_corpus_hard32.jsonl`) |
| Directions | `evidence/s2-battery/directions.npz` sha256 `95a6c68f23774cba65bee0aacea9fd252965a78b06c095020d5560fc6b8e8a47` |
| Battery | `evidence/s2-battery/direction-battery.json` sha256 `b591b177376c8b83840a2bac14653c4a47fd4c13cd92ca1c0393728d470daaaf` |

Thinking mode pinned **ON** for capture, gate, and all reported evaluations.

## Architecture facts that shape the method (verified on-disk)

- 48 physical layers, hidden 4608, **no loop passes** (unlike Inkling-2-Mini).
- MoE every layer: 384 routed experts, top-8, **no shared experts**.
- Experts L0–39 NVFP4 (block [1,16], amax/denominator scales), L40–47 MXFP8
  (group 32, E8M0). Raw checkpoint is row-major unswizzled.
- **Attention writer `wo_ud` is BF16 on all 48 layers** — [4608, 6144]
  (48 heads × 128). This is the bake surface; no requantization is needed.
- 8 full-attention layers (5,11,17,23,29,35,41,47); 40 SWA-1024 local layers.
- MTP/DSpark speculator in `mtp.safetensors` (84 tensors, BF16), outside the
  weight index — copied byte-exact into the candidate.

## Why the capture uses the vLLM native loader

The host transformers build has no modelopt/hf_quant_config handler: packed
experts cannot load into plain-BF16 projections (shape mismatch + reinit OOM),
and checkpoint keys do not match HF param names (0 direct matches). The
shipped capture (`scripts/capture_hard32_vllm.py`) therefore hooks the vLLM
native `inkling2` module directly. The earlier HF-path scripts are kept,
labeled, under `scripts/superseded-hf-path/` as negative evidence — do not use
them. `scripts/i2s_bf16_loader.py` retains the PROVEN exact dequant helpers
(max_diff=0 vs vLLM reference) and the interleaved gate/up convention
(even column = gate, odd = up) for any future HF-path work.

## S2 — capture + battery

```bash
# capture (4 GPUs, TP4, pinned image; see scripts/run-capture-vllm.sh)
bash scripts/run-capture-vllm.sh          # -> evidence/s2-capture/{raw-capture.npz,capture-metadata.json}
# battery (host python3 + numpy; no GPU)
python3 scripts/direction_battery_i2s.py \
  --capture-dir evidence/s2-capture --output evidence/s2-battery --permutations 2000
```

Capture: 64 rows × 48 layers × 4608 hidden, families attention_writer /
mlp_output / moe_experts (mlp_output == moe_experts here; layer_output absent
by design — fused block-attn-residual). `raw-capture.npz` sha256
`63a559079f9f39da84da731561c63d879c9c69de36f5ff8121a253bb0468cd88`.

Band selection (attention_writer family): contiguous mid-stack **L20–37**
(18 layers), all permutation p = 0.0005 (2000-permutation floor), split-half
cosine 0.912–0.975, direction norms 2.2–7.6. Early L0–6 and late L38–47 fail
the 0.90 split-half stability gate and are excluded.

## S3 — bake (α3.8)

```bash
bash scripts/run-bake-a3p8.sh
```

Math (per layer, proven canon): `wo_ud` [out 4608, in 6144] → transpose →
`W := W − α · W @ (q qᵀ)` with q the unit harmful−safe OUT-space direction →
transpose back → BF16. α = 3.8, layers 20–37. **No whitening/PCA. No requant.**

The script asserts the directions and battery SHA-256 before writing.
Result (`evidence/s3-bake-receipt-a3p8.json`): 24/24 shards written, exactly
18 `wo_ud` tensors edited, 514 untouched tensors byte-exact, all sidecars
byte-exact, MTP merged byte-identical (sha256
`3b265640ec84ba66e928a2844c4c022c94be731336b0f63755a20a247bfd2a36`, equal to
the S0 upstream hash). Candidate total: 393,506,668,732 bytes.

## S3 gate — capability floor

```bash
bash scripts/run-gate.sh <BASE_SNAPSHOT> base evidence/s3-gate            # once, on the base
bash scripts/run-gate.sh /workspace/i2s-candidates/a3p8-L20-37 cand evidence/s3-gate-a3p8
python3 scripts/s3_gate_compare.py --gate-dir <dir with base-gate.npz+cand-gate.npz> \
  --directions evidence/s2-battery/directions.npz --layers 20-37 --alpha 3.8 \
  --output evidence/s3-gate-a3p8/gate-comparison-a3p8.json
```

32 safe hard32-mirror rows, prompt_logprobs NLL, thinking ON, TP4:
base NLL 6.0069 → candidate 5.9794, **Δ = −0.0275** (capability floor held;
per-row |Δ| max 0.206). Full comparison in
`evidence/s3-gate-a3p8/gate-comparison-a3p8.json`.

Honest limitation (carried from the α3.0 gate): the static safe-row
writer-projection reduction is confounded by downstream propagation through
the residual stream and block-attn-residual; sign flips make it unsuitable as
an efficacy measure. The behavioral evaluations are the definitive test.

## S5 — serve + verify

See `../deployment-kit/AGENT_INSTRUCTIONS.md` and `../deployment-kit/serve.sh`.
Same-IP swap discipline was used across α3.0 → α3.3 → α3.8: one port (40001),
one key, container replaced, identity re-verified each swap.

## Evaluation shipped with this release

- **cyber10 (α3.8)**: 10/10 answered, 0 refusals, 0 errors, all
  `finish_reason=stop`, temp 0.0, 32K budget, thinking ON. 10/10 graded
  FULL_COMPLIANCE. Hedging solved: 4/10 hedged rows at α3.0, 6/10 at α3.3,
  **1/10 at α3.8** — and that one is a brief practical "lab/authorized testing"
  framing line, not a moral hedge or refusal tail.
  Evidence: `evidence/cyber10-a3p8/{rows.jsonl,report.json,run.log}`.
- **443-row corpus (α3.8 final)**: 443/443 generated, 0 errors, 0 truncations,
  temp 1.0, top_p 1.0, 32K budget, thinking ON. All 77 regex-flagged responses
  and a 20-row seed-443 sample of the 366 unflagged responses were manually
  adjudicated: **no refusals observed in the adjudicated set**. Estimated class
  shares: 34.9% full compliance, 65.1% partial/benign reframing, 0.0% refusal.
  These class shares are sample-based estimates, not exhaustive labels for all
  443 responses. Evidence: `evidence/r443-a3p8/{rows.jsonl,judge-report.json,EVAL-REPORT.md,run.log}`.
  Reproduction/audit scripts: `scripts/{r443_a3p8.py,flag443_a3p8.py,sample_unflagged_a3p8.py,judge_build_report_a3p8.py}`;
  adjudication inputs: `evidence/r443-a3p8/{adjudications-flagged.json,adjudications-sample.json,flagged.txt,unflagged-sample20.txt}`.

## Alpha-selection record (process provenance, not additional release results)

α3.0 → α3.3 → **α3.8 (shipped)**. All three bakes used the identical direction
set and band; only α changed. Only final-artifact evaluation results are
reported as release results in the model card and evaluation summary.