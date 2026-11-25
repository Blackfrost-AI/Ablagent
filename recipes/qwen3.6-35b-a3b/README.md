# Qwen3.6-35B-A3B DWM

This is Blackfrost AI's public, Apache-2.0 agent package for reproducing the
validated **stock Qwen3.6-35B-A3B BF16 Domain Weight Modification at alpha 1.0
for three passes**. It locks the exact source revision, 32+32 contrast bank,
initial direction and raw activations, thinking-on capture procedure, layer
range, target matrices, projection operator, norm restoration, transplant save
path, comparison gate, and serving profile used by the recorded run.

The repository name keeps Blackfrost's established `recepie` spelling. The
documents use the ordinary word *recipe*.

No Qwen checkpoint is redistributed. See [`MODEL-NOTICE.md`](MODEL-NOTICE.md)
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Recorded result

The package is derived from controlled run
`20261010-qwen36-35b-stock-frosty`. One exact candidate was built, checked,
served, and approved by the operator during hands-on testing.

| Item | Recorded value |
|---|---:|
| Source | `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Source identity | `sha256:88760187ccb2560500b1d51f0e75610658c956be010b4477bc11bb8265c7413b` |
| Precision | BF16 |
| Alpha and passes | `1.0 × 3` |
| Direction capture | Fresh thinking-on hard32 capture before every pass |
| Edited language layers | `2–39` inclusive |
| Edited tensors | 38 attention output weights |
| Non-target changes | 0 |
| MTP exactness | 19/19 tensors exact |
| Peak separation trace | `7.3863759041 → 3.5371067524 → 2.1556081772` |
| Pinned L39 separation removed | 97.06894375% |
| Held-out Jensen-Shannon divergence | 0.0052088025 |
| Held-out top-1 agreement | 0.9556962025 |
| Held-out relative NLL delta | −0.0244480805 |
| Serving smoke | Passed on vLLM 0.31.0, BF16 |
| Operator HITL | Approved |
| Release qualification | Incomplete: S1, S6, S7, and SR were not run |

The evidence above establishes a structurally correct candidate and a small
distribution check. It is not a substitute for the missing capability,
refusal, and release red-line gates.

## What the method does

### 1. Capture a direction at every language layer

The bank has 32 harmful and 32 safe prompts. Each prompt is rendered
individually with the model's chat template, `add_generation_prompt=True`, and
`enable_thinking=True`. There is no system prompt and no truncation. The
capture records the last prefill token from the output of each of the 40
language decoder layers.

For layer `l`, in float32:

```text
delta[l] = mean(harmful_activations[l]) - mean(safe_activations[l])
r[l] = delta[l] / ||delta[l]||₂
```

The initial capture selected layer 39 as the peak, with separation
`7.386375904083252`, split-half cosine `0.9535326957702637`, SNR
`5.622357368469238`, and rank-90 `22`. The package includes both the exact
initial direction tensor and its raw baseline activations.

### 2. Edit the attention output at each corresponding layer

For each pass and each eligible matrix `W` at layer `l`:

```text
E = W.float() - 1.0 * r[l][:, None] * (r[l] @ W.float())[None, :]
W' = BF16(E * (||W||F / ||E||F))
```

The scalar restores the global Frobenius norm of the whole matrix after the
rank-one subtraction. The direction is captured again from the current edited
model before pass 2 and before pass 3. Reusing the first direction for all
three passes is a different method.

### 3. Limit writes to 38 matrices

The edit surface is:

- 28 `linear_attn.out_proj.weight` tensors;
- 10 `self_attn.o_proj.weight` tensors; and
- language layers 2 through 39 inclusive.

Language layers 0 and 1, MLP weights, routers, experts, vision weights, and
all MTP tensors remain untouched. Because the Transformers wrapper omits the
upstream `mtp.*` payload, the executor copies the immutable source checkpoint
and transplants only the 38 edited tensors into their original safetensor
shards. It then inventories every tensor and fails if any non-target changed.

## Why the initial capture is bundled

The executor still captures a fresh direction on pass 1. It compares that
direction with the bundled S2 direction and requires L39 cosine of at least
`0.99`. The recorded run reached `0.9999986887`. This check catches changes in
the source, bank, chat formatting, thinking state, capture point, or runtime
before any output is accepted.

The bundled baseline activations also let the comparison script calculate the
same pinned-direction separation geometry without recapturing a different
baseline.

## Repository map

| Path | Purpose |
|---|---|
| `RUNBOOK.md` | Exact ordered download, build, verify, measure, serve, and qualification procedure |
| `recipe.lock.json` | Machine-readable method, asset, candidate, and claims contract |
| `environment.lock.json` | Recorded container, CUDA, Torch, GPU, and platform facts |
| `assets/source-lock.json` | All 40 files in the pinned upstream source inventory |
| `assets/v1_corpus_hard32.jsonl` | Exact ordered 32+32 direction bank |
| `assets/directions_on.safetensors` | Exact initial 40×2048 BF16 direction tensor |
| `assets/capture-activations.safetensors` | Exact initial `[32,40,2048]` BF16 activations for both classes |
| `assets/heldout_distribution.jsonl` | Exact eight-prompt distribution comparison set |
| `scripts/run_locked.py` | Three-pass executor with no method overrides |
| `scripts/locked_source.py` | Shared 40-file source verifier used by capture, build, measurement, and output verification |
| `scripts/verify_output.py` | Independent target and non-target checkpoint verifier |
| `scripts/measure_compare.py` | Recorded NLL, KL, JS, TV, entropy, top-1, and geometry checks |
| `scripts/verify_package.py` | Fail-closed package and optional source verifier |
| `scripts/serve_vllm.sh` | Pinned vLLM serving profile |
| `evidence/` | Sanitized capture, build, measurement, serving, HITL, and claims receipts |
| `plans/ablagent-read-only-audit.json` | Eight-stage standalone Ablagent audit plan |

## Quick package audit

```bash
python3 scripts/verify_package.py
```

Expected result: `passed: true`, recipe ID
`qwen3.6-35b-a3b-stock-bf16-a1p0x3`, 32+32 prompts, 40 direction tensors,
three passes, 38 targets, 19 protected MTP tensors, eight read-only Ablagent
stages, and complete `SHA256SUMS` coverage.

The complete commands are in [`RUNBOOK.md`](RUNBOOK.md).

## Ablagent Standalone Harness 1.0.0

The included plan lets an Ablagent-compatible model study and audit the method
inside a narrow, read-only workspace:

```bash
cp config.ablagent.example.json config.local.json
/path/to/Ablagent-Standalone-Harness-1.0.0/bin/ablagent \
  --config "$PWD/config.local.json" --probe
/path/to/Ablagent-Standalone-Harness-1.0.0/bin/ablagent \
  --config "$PWD/config.local.json" \
  --workspace "$PWD" \
  --plan plans/ablagent-read-only-audit.json
```

The optional paired model is `Alblagent-9B`, reached through the harness's
ordinary OpenAI-compatible endpoint configuration. The plan exposes only
`read_file`, `grep`, and `glob`; it cannot edit weights or start services.

## License

Copyright 2026 Blackfrost AI. Blackfrost-authored code, documentation, prompt
material, manifests, and plans are licensed under Apache-2.0. Upstream model
materials and runtime dependencies retain their own licenses.

