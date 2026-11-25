---
name: qwen3.6-35b-a3b-dwm
description: Audit or reproduce Blackfrost AI's stock Qwen3.6-35B-A3B BF16 DWM alpha-1.0 x3 method with fresh per-pass directions and exact target and MTP verification.
---

# Qwen3.6-35B-A3B DWM

1. Read `AGENTS.md`, `recipe.lock.json`, and `RUNBOOK.md` completely.
2. Treat every method field in `recipe.lock.json` as immutable.
3. Run `python3 scripts/verify_package.py` before acquiring large inputs.
4. Download and verify the pinned stock BF16 source revision.
5. Use `scripts/run_locked.py`; it has no alpha, pass, bank, direction, layer,
   or target override.
6. Verify the output independently with `scripts/verify_output.py`.
7. Reproduce the small distribution check with `scripts/measure_compare.py`.
8. Record new results separately and complete the missing qualification gates
   before making a release claim.

Use `plans/ablagent-read-only-audit.json` with Ablagent Standalone Harness
1.0.0. That plan audits the package and cannot execute the weight edit.

