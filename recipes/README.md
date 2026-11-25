# Recipes — finalized DWM playbooks

This directory is the **recipe library**: exact, evidence-bound procedures for
recorded model lines, shared so researchers (and their agents) can study,
verify, and reproduce published Blackfrost DWM runs. It grows as runs are
finalized — each new recipe is added as a self-contained, hash-locked pack.

## What a recipe is (and is not)

- A recipe records **one model line's** method: source identity, contrast
  material, capture conditions, edit arithmetic, target tensors, serialization
  rules, verification gates, and known limits. Per the uniqueness principle,
  **never copy a recipe's alpha, passes, or targets onto a different checkpoint**
  — training history changes the recipe even on the same base model.
- Recipes are **run-record claims with attached evidence**, not measurements
  made by this repository. The vendored receipts describe the original runs.
- These recipes target **GPU-scale checkpoints** (Qwen 9B-class, Qwen3.6-35B-A3B
  MoE, GLM-5.3 BF16/Flash) and carry their own locked scripts and runbooks.
  They are **not executed by the shipped `--executor real` runtime**, whose
  envelope is small dense CPU checkpoints. Study them as method references;
  reproduce them on suitable hardware via their own runbooks.
- The executable, synthetic-corpus domain packs live in [`packs/`](../packs/);
  the published small-model worked example lives in
  [`examples/inkling-2-mini-derisked/`](../examples/inkling-2-mini-derisked/).

## Library

| Recipe | Method | Origin (public at import) |
|---|---|---|
| [`qwen-9b-class/`](qwen-9b-class/START-HERE.md) | α 1.0, three fresh-capture rounds, norm-preserving residual-write edits | `Blackfrost-AI/blackfrost-dwm-qwen9b-runbook` @ `1352ce4` |
| [`qwen3.6-35b-a3b/`](qwen3.6-35b-a3b/AGENTS.md) | Stock BF16 α 1.0 ×3, fresh per-pass capture, 38 attention-output targets | `Blackfrost-AI/Qwen3.6-35B-A3B-DWM-recepie` @ `f23bbc8` |
| [`glm-5.3-bf16-a2.2/`](glm-5.3-bf16-a2.2/README.md) | DWM32 α 2.2, pinned direction derivation, NVFP4 serialization | `Blackfrost-AI/GLM-5.3-DWM32-A2.2-Recipe` @ `476bbc5` |
| [`glm-5.3-flash-a2.8/`](glm-5.3-flash-a2.8/README.md) | Flash-variant DWM α 2.8 with pinned directions | `Blackfrost-AI/GLM-5.3-Flash-DWM-recepie` @ `77443f9` |

## Known limits at import

- **GLM-5.3 Flash α2.8 is study/reproduction-with-recapture only out of the box:**
  its pinned 151 MB `directions.safetensors` is deliberately **not published**
  (`assets/directions.asset.json`, `archive_status: missing`). Execution requires
  recovering the original tensor from the recorded run directory or recapturing
  with the bundled runtime + capture script, accepted only at the exact pinned
  digest. The verifier reports this honestly and still passes package integrity.
- **GLM-5.3 BF16 α2.2** ships pinned prompts/pairs; its 194 MB direction tensor
  is a **fetch-only public GitHub release asset** (`directions-v1`, exact
  SHA-256 in `assets/directions.asset.json`) — not vendored here.
- **Qwen3.6-35B-A3B** ships its direction/capture tensors in-tree.
- **Qwen 9B-class** re-measures directions on the target checkpoint itself by
  design — never reuse directions from another model.
- All four package verifiers pass at import (qwen-9b-class via
  `run_dwm.py --self-test`; the two GLM verifiers exit 0 with the asset status
  reported honestly).

## Provenance and integrity

Imported **byte-for-byte** from
[`Blackfrost-AI/Ablagent-base-legacy`](https://github.com/Blackfrost-AI/Ablagent-base-legacy)
@ `288d8d6` ("Align recipe library with Blackfrost Pro Series"), which itself
snapshots four recipe repositories that were public at import time.

- [`catalog.json`](catalog.json) — origins, commits, snapshot identities, and
  per-recipe verify commands (copied unchanged from the source monorepo).
- [`snapshots.lock.json`](snapshots.lock.json) — every imported file bound by
  byte count and SHA-256. All 100 recipe files re-verified against this lock
  after import.
- Each recipe also carries its own `recipe.lock.json` / `SHA256SUMS` and a
  `scripts/verify_package.py` (or `run_dwm.py --self-test` for qwen-9b-class).

Verify the whole library against the lock:

```bash
python3 - <<'EOF'
import hashlib, json, pathlib
lock = json.load(open('recipes/snapshots.lock.json'))
bad = 0
for r in lock['recipes']:
    for f in r['files']:
        p = pathlib.Path(r['path']) / f['path']
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != f['sha256']:
            print('DRIFT', p); bad += 1
raise SystemExit(1 if bad else print('all recipe files match the lock'))
EOF
```

## Adding or updating a recipe (maintainer workflow)

Recipes are **immutable snapshots**: never edit files inside an imported
recipe directory — that breaks the lock and the provenance claim. To publish
a finalized run or a corrected recipe:

1. Finalize the run in your own workspace: complete the pipeline gates you
   claim, keep receipts, and record what was **not** tested.
2. Package it as a new self-contained directory `recipes/<model-line>-<method>/`
   with: `README.md` (method + limits), `RUNBOOK.md` (exact procedure),
   pinned inputs (`assets/`), evidence receipts (`evidence/`), a
   `recipe.lock.json` + `SHA256SUMS`, and a `scripts/verify_package.py`.
   Follow the layout of the existing packs; include `NOTICE` /
   `THIRD_PARTY_NOTICES.md` and a `MODEL-NOTICE.md` for upstream model terms.
3. Add an entry to `catalog.json` (origin repository/commit or run id,
   snapshot identity, verify command) and regenerate the `snapshots.lock.json`
   entry for the new directory. For a corrected version of an existing line,
   add a **new versioned directory** and mark the old one superseded in the
   catalog — do not mutate the old snapshot.
4. Re-run the lock verification above plus each recipe's own verifier.
5. **Review before publishing:** no credentials, private endpoints, internal
   host paths, operator-only holdout prompts, or unpublished model weights.
   Prompt banks must be publishable research material with recorded provenance.
6. Open a PR (external contributors: see [CONTRIBUTING.md](../CONTRIBUTING.md)).
   Recipe additions are operator-reviewed; a recipe is a claim about a real
   run, so evidence completeness is a merge gate.

## License

The recipe library is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), matching this repository.
Upstream **model weights are not included** and remain under their own terms —
see each recipe's `MODEL-NOTICE.md`.