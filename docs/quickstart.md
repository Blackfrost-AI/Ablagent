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

Verification recomputes the operation and observation from the stored graph and baseline, validates candidate metadata and non-target tensor bytes, and loads the retained trial.

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
