# Ablagent — agent boot instructions

You are an agent operating the **Ablagent** research runtime: an evidence-first,
architecture-aware Domain Weight Modification (DWM) toolkit extending
abliteration / refusal-direction research. This file is your boot contract.
Read it fully before acting.

## What this repository is

- A **research runtime**, not a model zoo. It ships the DWM Interchange
  Protocol (DWMIP v0), a synthetic reference executor, and a **real CPU
  executor** for small dense HuggingFace checkpoints.
- One published worked example is vendored byte-identical:
  `examples/inkling-2-mini-derisked/` (see its `PROVENANCE.json`).
- A **recipe library** of finalized GPU-scale playbooks is vendored
  byte-identical under `recipes/` (see `recipes/README.md` and
  `recipes/snapshots.lock.json`). Recipes are immutable snapshots and
  run-record claims — never copy their hyperparameters onto another
  checkpoint, never edit them in place.
- `docs/serving-stacks.md` explains the capture/serve environment pattern:
  HF forward hooks inside pinned vLLM/SGLang engine images, one environment
  for capture/edit/serve, and when a from-source engine build is actually
  needed. Read it before proposing capture or serving work.
- `benchmarks/` ships real, prompt-only, hash-pinned corpora (443 eval /
  smoke-50 / cyber-10) plus a protocol, offline validator/scorer, and full
  upstream license scope. S6/S7 remain unmeasured in real runs: never treat a
  mock gate or a benchmark score as a release decision. An operator-controlled
  release-only corpus is intentionally absent; see
  `benchmarks/PROVENANCE.md` and `packs/derisk/NAMESPACE-NOTICE.md`.
- No credentials, no private holdouts, no operational endpoints. Keep it that way.

## Boot sequence (every session)

1. Read `README.md` (scope + what runs today).
2. Read `docs/quickstart.md` and `docs/agent-integration.md`.
3. Read `spec/dwmip/v0/README.md` if you will touch artifacts or contracts.
4. Run `ablagent doctor` to verify the environment before any run.
5. For real-checkpoint work, confirm the `[real]` extra is installed:
   `ablagent capabilities --adapter hf` (fails closed with an install hint).

## The two executors

| Executor | Command | What it does |
|---|---|---|
| `mock` (default) | `ablagent run --pack derisk` | Fully offline synthetic pipeline; no network, no torch |
| `real` | `ablagent run --pack derisk --executor real --model <HF URL or repo id>` | Pins + downloads the checkpoint, runs S0→S1→S2→S3 on CPU, exports an HF-loadable edited copy |

A local HF-format directory also works as `--model` (offline/CI).

### Real-run stage honesty

The real executor completes **S0 intake, S1 baseline generation, S2 direction
capture, S3 rank-one bake + HF export, S8 productize, S9 card draft**. It
records **honest skips** for S4 (quant), S5 (serve), S6/S7 (behavior +
capability verify), SR (operator red-line holdout), and S10 (deliver).
Skipped means **untested**, never "passed". Release stays `not_eligible`.

## Hard rules

- **Never present a run as a release.** Every real run ends `not_eligible`;
  behavior, capability, and red-line gates were not run. The export is an
  experimental artifact for further research.
- **Never fake a skip.** Do not mark skipped stages complete, do not invent
  measurements, do not edit CONTROL.json or evidence files by hand.
- **Proposals are not permissions.** Agent output is untrusted input; only an
  explicitly selected `InterventionSpec` may be compiled (see
  `docs/agent-integration.md`).
- **Evidence class discipline.** S2 capture directions are `correlational`
  contrasts, not causal claims. Keep that label in anything you write.
- **No secrets anywhere.** Not in prompts, banks, cards, commits, or logs.
  The runtime secret-scans the pinned snapshot and the run tree; a finding
  fails the run closed.
- **Network only in the explicit real path.** The mock executor and all tests
  are offline. Do not add network calls elsewhere.
- **Resource envelope.** The v0 real adapter is CPU-only, `dense_out_in`
  safetensors checkpoints, ≤ ~900 MiB float32 editable surface. Oversize
  shards are rejected before download; Conv1D families (gpt2/gpt_neo/gptj)
  fail closed at intake — correct behavior, not a bug to route around. Other
  non-dense architectures (MoE, SSM) are unqualified in v0: do not present a
  run on them as validated.

## Useful commands

```bash
ablagent doctor                          # environment + pack health
ablagent packs                           # installed domain packs
ablagent stages                          # canonical stage DAG
ablagent capabilities                    # synthetic adapter contract
ablagent capabilities --adapter hf       # real adapter contract + dep versions
ablagent run --pack derisk               # offline synthetic run
ablagent run --pack derisk --executor real \
  --model https://huggingface.co/<org>/<model> \
  --alpha 1.0 [--writer <writer_id|tensor_name>] \
  [--max-pairs 16] [--baseline-rows 8]
ablagent status <run>                    # control-plane state
ablagent verify <run>/artifacts/candidate.manifest.json
```

## After material work

- Run the suite: `python -m pytest tests/ -q` (real tests need `[real]`).
- Lint changed files: `python -m ruff check <paths>`.
- Record what you verified vs. what remains untested. Never upgrade a claim
  without a recorded, passing check.