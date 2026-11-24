# Quickstart

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Python 3.11+ is required. No GPU, API credential, model repository, remote code, or network service is used by the reference runtime.

## Run the synthetic vertical slice

```bash
.venv/bin/ablagent demo --run-dir .runs/reference
```

The run creates:

- `baseline.graph.json`: declared graph, sites, spaces, and writer.
- `baseline.safetensors`: exact original matrix and graph metadata.
- `candidate.direction.json` and `bound.direction.json`: contrast and explicit unit-direction binding.
- `proposed.plan.json`: bounded alpha-zero and alpha-one proposals.
- `candidate.safetensors` and `candidate.manifest.json`: one explicitly compiled update.
- `trial-reference.trial.json`: retained trial lineage.
- `candidate.evidence.json`: typed observation and metrics with exact artifact bindings.

## Verify independently

```bash
.venv/bin/ablagent verify .runs/reference/candidate.manifest.json
.venv/bin/ablagent verify-evidence .runs/reference/candidate.evidence.json
```

Verification recomputes the operation and observation from the stored graph and baseline, validates candidate metadata and non-target tensor bytes, and verifies the selected trial chain and retained-output digests. For an attested inert mock bundle, pass its selected bank explicitly:

```bash
.venv/bin/ablagent verify-evidence run/attested.evidence.json \
  --bank examples/banks/synthetic-release-mock.bank.json
```

## Evaluate eligibility

```bash
.venv/bin/ablagent gate .runs/reference/candidate.evidence.json \
  --bank examples/banks/synthetic-release-mock.bank.json
```

Expected result: **blocked, exit status 2**. The demo does not include the required private behavioral red-line evidence. A public mock bank only exercises policy schemas; it is not a real holdout or safety evaluation.

## Plan separately from compile

```bash
.venv/bin/ablagent plan bound.direction.json \
  --run-dir .runs/reference --alpha 1 --output one.plan.json
.venv/bin/ablagent compile bound.direction.json \
  --run-dir .runs/reference --plan one.plan.json --artifact-id independent
```

A plan with multiple alpha values proposes separate trials; it is not executable as a batch. `compile` accepts exactly one intervention and checks the persisted direction and baseline. There is no planner execution endpoint and no release/publish command.

Run artifacts and private data are ignored by Git. Use a fresh run directory after a failed attempt; the runtime preserves completed artifacts rather than silently overwriting them.

## Run the real pipeline on a HuggingFace checkpoint

The optional `[real]` extra adds a CPU executor for small dense safetensors
models (torch, transformers, huggingface_hub). This is the only path that
touches the network — it downloads the pinned checkpoint revision.

```bash
.venv/bin/python -m pip install -e '.[real]'

.venv/bin/ablagent capabilities --adapter hf   # preflight; fails closed if deps missing

.venv/bin/ablagent run --pack derisk --executor real \
  --model https://huggingface.co/HuggingFaceTB/SmolLM2-135M \
  --alpha 1.0
```

Options: `--writer <writer_id|tensor_name>` selects the edit site (default:
strongest captured contrast); `--max-pairs` (4–16) sizes the capture bank;
`--baseline-rows` sizes S1 generation. A local HF-format directory also works
as `--model` for offline runs.

The run directory contains:

- `hf-snapshot/`: the pinned checkpoint, every file hashed in the intake receipt.
- `artifacts/`: the same DWMIP chain as the synthetic path — baseline graph and
  tensors, capture bank, direction candidates, bound direction, compiled
  candidate, `candidate.manifest.json`, `CORPUS-REGISTRY.json`.
- `control/CONTROL.json`: every stage transition; skipped stages carry reasons.
- `evidence/`: intake, baseline generation, capture, bake, export, gate results,
  secret scans, and an `s9_card_draft` labeled `draft_experimental_not_a_release`.
- `export/`: an HF-loadable copy of the checkpoint with exactly one tensor
  changed, original dtype restored, plus `ABLAGENT-EXPORT-RECEIPT.json`.

Verify independently:

```bash
.venv/bin/ablagent verify <run_dir>/artifacts/candidate.manifest.json
.venv/bin/ablagent status <run_dir>
```

**Honest boundaries.** S4 quant, S5 serve, S6/S7 behavior/capability verify, the
operator red-line holdout, and S10 deliver are recorded as **skipped —
untested**. Every stage's release state stays `not_eligible`. The export is an
experimental research artifact; do not publish or deploy it without running the
missing verification gates. The v0 envelope is CPU, `dense_out_in` safetensors
(llama/qwen/mistral/gemma-style), ≤ ~900 MiB float32 editable surface. Oversize
shards are rejected before download; Conv1D families (gpt2/gpt_neo/gptj) fail
closed at intake. Other non-dense architectures (MoE, SSM) are unqualified in
v0 — treat any run on them as unvalidated.
