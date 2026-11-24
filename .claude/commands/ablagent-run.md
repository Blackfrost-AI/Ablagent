---
description: Start a REAL DWM run on a HuggingFace checkpoint (needs [real] extra)
argument-hint: <HF URL or repo id> [alpha]
---

The operator named this checkpoint: $ARGUMENTS

1. Confirm the real extra: `ablagent capabilities --adapter hf`. If it fails,
   show the install hint (`pip install 'ablagent[real]'`) and stop.
2. Run the real pipeline (CPU; small dense safetensors models only):

```bash
ablagent run --pack derisk --executor real --model <checkpoint> --alpha 1.0
```

3. Verify the result independently:

```bash
ablagent verify <run_dir>/artifacts/candidate.manifest.json
ablagent status <run_dir>
```

4. Report: completed stages, honest skips (untested), selected writer, export
   path, and that release is `not_eligible`. Never call the export a release.
   Skipped stages (quant/serve/verify/red-line/deliver) are UNTESTED, not passed.