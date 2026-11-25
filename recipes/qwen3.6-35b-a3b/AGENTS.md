# Agent contract: Qwen3.6-35B-A3B stock DWM

Read these files completely and in this order before proposing a command:

1. `recipe.lock.json`
2. `RUNBOOK.md`
3. `evidence/capture-receipt.json`
4. `evidence/verification-summary.json`
5. `MODEL-NOTICE.md`

This method is immutable. It applies **alpha 1.0 for exactly three passes** to
the pinned stock BF16 source. Each pass captures a fresh direction from the
current candidate with the exact 32+32 bank, the upstream chat template, a
generation prompt, thinking enabled, and the last prefill token at every
language decoder layer. It edits language layers 2 through 39 at exactly 38
attention output matrices and restores each matrix's global Frobenius norm.

Never substitute the source revision, prompt bank, initial direction, capture
condition, layer range, target surface, alpha, pass count, arithmetic, norm
rule, or save procedure and call the result this method. MLP, router, expert,
vision, and MTP weights are outside the edit surface. All 19 MTP tensors and
every other non-target tensor must remain exact.

Run `python3 scripts/verify_package.py` before acquiring model weights. Run it
again with `--source` before building. Use a new output path. Stop on every
mismatch and keep the source read-only.

The Ablagent Standalone Harness plan is a bounded, read-only audit. It may read
only the listed files with `read_file`, `grep`, and `glob`. The harness does not
download weights, use a GPU, edit a checkpoint, start a service, or approve a
release. A shell-capable operator follows `RUNBOOK.md` for those actions.

This package records a successful build, structural gate, small held-out
distribution gate, vLLM smoke test, and operator HITL approval for one exact
candidate. S1, S6, S7, and the release red-line gate were not executed. Do not
describe the candidate or a reproduction as release qualified.

