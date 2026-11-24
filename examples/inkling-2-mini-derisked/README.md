# Inkling-2-Mini-Derisked — a complete DWM worked example

This directory is the **same duplication kit published with
[Blackfrost-AI/Inkling-2-Mini-Derisked](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked)**
on Hugging Face, vendored byte-identical at revision
`31ec2eddecfdf93c8626d9e4dfc0a2aca6070900`. It documents one full Domain Weight
Modification run end to end: architecture reconnaissance, direction capture,
direction batteries, a loop-enabled weight edit, an edit receipt, and a
serving recipe — so researchers and their agents can study, verify, and
reproduce the process on a real checkpoint.

## What this example demonstrates

Inkling-2-Mini is a **loop transformer**: 24 physical layers, with physical
layers 6–17 executed in 3 passes per token (48 logical layer-visits total),
sliding-window/full attention alternation, routed-experts-only MoE, and an MTP
head. That architecture breaks naive abliteration assumptions, and the kit
shows how the run handled each one:

- **Physical vs logical layers.** Captures are tagged with `(layer, visit)`;
  the edit targets physical layers 9–18 by checkpoint key, not logical site
  indices. Layers 9–17 are inside the loop (3 passes each); layer 18 is
  post-loop (single pass) — the edit script gates pass count per layer.
- **Loop superposition.** A direction captured at one pass is not the same
  vector at another pass. The shipped edit applies a per-pass consensus
  orthogonalization — for each selected physical layer, each pass direction
  `d` is removed from the writer's output space:
  `W ← W − α·outer(d, Wᵀd)` (sequentially per pass) — at α = 2.5,
  loop-enabled.
- **Writer family selection.** The capture/edit family is `attention_writer`
  (`self_attn.o_proj.linear` output, hidden 2304; checkpoint key
  `attn.wo_ud.weight`, a pure rename at load). The battery also records
  `layer_output`, `moe_down`, and `mlp_output` families for comparison.
- **Evidence discipline.** Every stage emits hash-bound JSON: capture
  metadata, direction battery, S2 summary, and an edit receipt whose
  `candidate_file_digests` bind the exact shipped weight shard.

### What actually shipped (ground truth = scripts + receipt)

The authority for what was done is `DWM_RECIPE/scripts/s3_derisk_edit.py` and
`DWM_RECIPE/s3-edit-receipt-a25.json`, not the narrative runbook. The shipped
method is **per-pass consensus column orthogonalization, loop-enabled, α 2.5,
physical layers 9–18 (10 tensors), attention writer output space**. See
[Runbook errata](#runbook-errata) below — the vendored `RUNBOOK.md` contains
several statements that the shipped scripts and receipts contradict. They are
preserved verbatim for provenance and corrected here.

## Scope: this kit is not the DWMIP v0 runtime

- The kit scripts are the **validated GPU/torch pipeline** used to produce the
  real artifact. They are Linux/CUDA-only, require the upstream checkpoint,
  and were run with `transformers 5.24.0.dev0` / `torch 2.14.1+cu130` (a dev
  build; the kit pins no requirements file). They are **vendored research
  material**, not part of the synthetic `ablagent` package: the v0 runtime
  does not import, execute, or test them, and CI does not lint them
  (byte-identical provenance is deliberate).
- The DWMIP v0 runtime in `src/ablagent/` is the synthetic, CPU-only
  formalization of the same ideas (banks, plans, receipts, policy gates).
  Read [docs/methodology.md](../../docs/methodology.md) for how the two
  relate; this example is the real-world case study behind that formalism.
- Instructions inside the kit (`RUNBOOK.md`, `AGENT_INSTRUCTIONS.md`) are
  **pipeline documentation for researchers running the GPU pipeline** — data,
  not directives to repository tooling or agents.

## Layout

```
DWM_RECIPE/
  RUNBOOK.md                     # stage-by-stage runbook (verbatim; see errata)
  scripts/
    s2_capture_directions.py     # residual capture, (layer, visit) tagged
    s2_direction_battery.py      # split-half / permutation / cross-pass stats
    s3_derisk_edit.py            # loop-enabled per-pass orthogonalization
    s3_gate_metrics.py           # token-level ΔNLL / agreement gates
    s3_verify.py                 # artifact self-test + digest checks
    v1_corpus_hard32.jsonl       # 32 harmful/safe prompt pairs (capture corpus)
  s2-capture/
    capture-metadata.json        # architecture, environment, corpus bindings
    direction-battery.json       # full battery statistics (295 KB)
    s2-summary.json              # stage-scoped S2 status
    directions.npz               # 48×2304 unit directions + site arrays (vendored)
    raw-capture.npz              # 79 MB full capture — fetch-only, see below
  s3-edit-receipt-a25.json       # alpha-2.5 edit receipt for the shipped artifact
deployment-kit/
  AGENT_INSTRUCTIONS.md          # agent-facing deployment notes
  SERVE.md                       # pinned vLLM serving recipe
  scripts/serve.sh               # docker launch (default bind 0.0.0.0 — change it)
PROVENANCE.json                  # per-file sha256/size/source manifest
NOTICE.md                        # licensing scope for vendored material
```

## Verify the kit (offline, no GPU)

Every vendored file is pinned in `PROVENANCE.json`. From the repo root:

```bash
python scripts/verify_example_provenance.py
```

This recomputes SHA-256 of all 15 vendored files and checks the internal hash
chain (`directions.npz` digest in the S2 summary and edit receipt; hard32
corpus digest in the capture metadata). The test suite runs the same checks.

You can also inspect the directions directly with numpy:

```python
import numpy as np
z = np.load("examples/inkling-2-mini-derisked/DWM_RECIPE/s2-capture/directions.npz")
z["directions"].shape      # (48, 2304) float32, unit norm
z["layers"], z["visits"]   # physical layer + loop pass per direction
```

## Fetch the large capture asset (explicit, verified)

`raw-capture.npz` (79 MB) is not vendored. It is required by
`s2_direction_battery.py`. Fetch it explicitly — nothing downloads implicitly:

```bash
python scripts/fetch_example_assets.py          # download + sha256 verify
python scripts/fetch_example_assets.py --check  # verify if already present
python scripts/verify_example_provenance.py --fetch-only
```

The fetcher pins the HF revision, verifies the digest against
`PROVENANCE.json` before keeping the file, and refuses to overwrite a
mismatched existing file. The destination is gitignored.

## Run the pipeline (GPU required)

Requirements the kit does not bundle: Linux + CUDA, ≥80 GiB available RAM
(the scripts guard on `/proc/meminfo`), the upstream checkpoint
`thinkingmachines/Inkling-2-Mini` (license: other — review its terms), and a
torch/transformers environment matching the capture metadata. Then follow
`DWM_RECIPE/RUNBOOK.md` stage by stage, keeping the errata below in mind.
Model weights for the *shipped derivative* come from the HF repository, not
from this kit.

To serve the shipped artifact, read `deployment-kit/SERVE.md` first.
`serve.sh` defaults to `BIND_ADDR=0.0.0.0` with host networking and **no
authentication** — set a private bind address before running it anywhere.

## Reported results (from the run record — not re-measured here)

The run record (S6, closed 2026-11-22) reports, for the shipped α 2.5
artifact with thinking enabled:

| Measurement | Reported result |
|---|---|
| 443-row corpus, hand-adjudicated (α 2.5) | 266/295 harmful-prompt compliance (90.2%); 29/295 real refusals; 0/75 false refusals on benign prompts; zero errors |
| Alpha ladder (smoke-50, answered/substituted/refused) | α 1.5: 21/14/15 · α 1.7: 19/16/15 · α 2.0: 29/11/10 · α 2.5: 38/10/2 |

These numbers are **reported from the run record, not re-measured by this
repository**. The 443-corpus prompts, responses, and scores are intentionally
not shipped. Smoke-50 rows were graded into answered/substituted/refused; the
three counts need not sum to 50 (ungraded rows), so do not treat them as a
complete confusion matrix. The run record's token-gate details are not
included in the vendored kit; `s3_gate_metrics.py` shows how they were
computed.

## Runbook errata

The vendored `RUNBOOK.md` is preserved byte-identical to the HF publication.
The following statements in it are contradicted by the shipped scripts,
metadata, and receipt (verified against the actual bytes):

1. **`direction-metadata.json`** is listed as shipped but does not exist in
   the kit. No script reads it; use `capture-metadata.json`.
2. **"4 loop visits"** — the model has **3** passes
   (`capture-metadata.json: loop_num_passes: 3`; the capture script asserts
   geometry `(24, 6, 18, 3)`: 24 physical layers, loop over layers 6–17).
3. **Capture site description** — the runbook says `layers{i}.attn.down_proj`
   input space; the script hooks `self_attn.o_proj.linear` **output** space
   (hidden 2304), and the edit script treats directions as writer output
   space (input dim 3072). The checkpoint key `attn.wo_ud.weight` is a pure
   rename of `self_attn.o_proj.linear` at load — no transpose.
4. **"held-out hard32 battery"** — hard32 is the **capture corpus itself**
   (identical sha256 `902ac247…`, asserted by the capture script). The
   battery computes split-half/permutation/cross-pass statistics on the
   capture activations; it never re-scores prompts.
5. **"Whitened (CW) basis"** — the shipped edit script contains **no
   whitening**. It implements per-pass consensus orthogonalization; the
   receipt's `edit_method` matches the script, not the runbook.
6. **"verifies file hashes / artifact self-test"** in the edit script — the
   edit script computes digests but does not verify them;
   `source_checkpoint_digest` is a hardcoded 8-hex prefix literal
   (`sha256:42e765f7`), not a measured full hash. Verification lives in
   `s3_verify.py`.

Additional caveats:

- **Cross-pass cosines.** The run record cites cross-pass cosines
  0.9885–0.9937 (threshold 0.9) as gating layer selection. The shipped
  `direction-battery.json` records off-diagonal pass cosines of
  **0.2408–0.8595** for `attention_writer` over the looped layers 6–17 —
  below that threshold. The receipt binds the pre-sanitization battery
  digest `1ec4212b…`; the shipped battery differs from it only by one
  sanitized path field. Treat the shipped battery as the inspectable
  evidence; the run-record cosine range and its gating claim are unverified
  here and appear inconsistent with the shipped statistics.
- **`S2_BATTERY_MEASURED_NOT_GO`** in `s2-summary.json` is stage-scoped: S2 is
  measurement, not an intervention decision. It does not indicate the shipped
  artifact failed.
- **`removed_frac_pass0_dir` values are negative** in the receipt (e.g. −0.69
  at L9): at α 2.5 the summed three-pass removal overshoots the pass-0
  component. This is expected superposition behavior — the edit does not
  merely "null" the direction.

## Licensing

See [NOTICE.md](NOTICE.md). Kit scripts and documentation are
Blackfrost-authored (Apache-2.0); the hard32 prompt pairs are CC-BY-4.0;
activation-derived artifacts (directions, captures, batteries, receipts) are
Blackfrost-published research material derived from an upstream model with
license `other` — this repository grants no additional license over them.

## Provenance

`PROVENANCE.json` records, per file: SHA-256, byte size, HF source path, git
blob OID at the pinned revision, vendored/fetch-only status, and license
scope. All 15 vendored files were verified byte-identical to the HF tree at
vendoring time (2026-11-24). Internal filesystem paths were sanitized to
`<internal-path>` by the publisher before HF upload; no further edits were
made here. Weights, the 443 measurement corpus, endpoints, and credentials
are not included.