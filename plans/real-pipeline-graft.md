# Plan: Complete the vision — real pipeline in the public runtime

Operator decision (2026-11-24): option 2 (port the real pipeline into the runtime),
graft ablagentv2's control plane into the public repo (one package, DWMIP stays the
evidence spine), take sterile v2 material, then agent boot docs + shortcuts.

## Sources (surveyed, sterility verified)
- Public: /Volumes/craigssd/Ablagent (DWMIP protocol + evidence/gate engine, 98 tests)
- v2: /Volumes/craigssd/code/Observatory/ablagentv2 — TAKE: constants.py (S0–S10 DAG),
  control/ (~750 lines), executors/ (local/ssh/mock, injection-safe), research/ (fetch
  with provenance), packs.py + packs/derisk (synthetic corpora), run.py, pipeline.py,
  security/ (redline classifier, secret_scan, evidence), tests (627 lines).
  DO NOT TAKE: reference/v1/ (internal charter/doctrine/hosts). Sanitize: OPERATIONS.md
  fleet names. Secret-scan hits in pack = false positives (corpus-id shape).
- v1 proven scripts: behavior reference only (port behavior, not bytes).

## Design
1. Graft v2 modules into src/ablagent/ (no collisions except cli.py/__init__.py —
   merge CLI subcommands: keep DWMIP commands, add packs/stages/doctor/init/status/run).
   New core dep: pyyaml. Optional extra [real]: torch, transformers, huggingface_hub.
2. Real stage adapters (src/ablagent/adapters/):
   - S0 intake: resolve HF repo id/URL → snapshot download (hf_hub), pin revision +
     file hashes → RUN manifest. CPU-only, no torch needed for intake.
   - S2 capture: torch forward hooks on declared writer outputs over pack capture
     corpus → mean-difference DirectionCandidate (correlational, labeled).
   - S3 bake: rank-one W' = W − α·d(dᵀW) on safetensors, declared coordinate only,
     non-target byte preservation, emit DWMIP-compatible evidence.
   - Unmet stages (S4 quant, S5 serve, S6/S7 bench, SR) → honest NOT_RUN with reason;
     never fake PASS. SR holdout stays operator-supplied (v1 ruling).
   - capabilities: real_checkpoint_support true only when [real] installed; declare
     supported families explicitly; unknown arch → UNSUPPORTED (uniqueness doctrine).
3. Pipeline execution: DAG order from constants; per-stage evidence into run dir;
   phase-gated authority from v2 control store; mock executor path preserved for tests.
4. Agent boot docs: AGENTS.md (canonical), CLAUDE.md shim, .claude/commands/
   (ablagent-run, ablagent-status), SHADOW.md, docs/agent-integration.md rewrite,
   README "Use with your agent" section. Clone-to-run, fork-to-contribute.
5. Verify: full suite + ruff + real tiny-model CPU smoke (HF link → S0→S2→S3→verify
   on a tiny random-weights model, no GPU needed for the math); reviewer; push.
   README/site truth update: what is runtime-executed vs agent-executed vs NOT_RUN.

## Non-goals (this pass)
- No GPU serve/quant/bench adapters (declared NOT_RUN, documented).
- No internal hosts/partner data/operator-only holdout content.
- No change to DWMIP protocol semantics or gate names.
## Review decisions (2026-11-24, reviewer pass before build)
1. Skip eligibility stays GLOBAL-SAFE: keep SKIP_OPTIONAL_STAGES={SR}; add
   `allowed_stages` param to store.skip_stage (default None → constant). Real
   path passes REAL_UNMET_STAGES={S4,S5,S6,S7,S10} explicitly. test_skip_redline
   unchanged. release_check event details gain skipped list (additive).
2. Capture bank is DERIVED: first 16 corpus prompts × {hostile=refusal prefill,
   benign=helpful prefill}; BankRecord.prompt = JSON {prompt, assistant_prefill};
   records_digest binds actual inputs; lineage in notes + receipt. Sign test:
   projection onto d after bake ≈ (1−α)× before.
3. Tied weights: declare ONLY canonical stored tensor (shard name); lm_head
   alias excluded; tie recorded in provenance. Capture ranking excludes
   embeddings + aliases (module-output dim ≠ writer output dim for Embedding).
4. bf16/f16 via torch (safetensors.torch load/save); export = original shard
   bytes untouched except edited tensor recast to original dtype; export
   receipt binds f32 candidate tensor digest → exported bytes digest +
   per-shard hashes; non-edited shards byte-identical check.
5. Envelope: editable-surface checkpoint ≤ ~900MB f32 (under 1GiB cap);
   fail closed at intake with clear error. Document small-dense-model scope.
6. Algorithm id: additive Literal widening + HF_ALGORITHM_ID
   "hf.contrast.unit.v0"; default unchanged → existing digests identical.
   AdapterCapabilities.real_checkpoint_support widened to bool=False;
   capabilities --adapter {synthetic,hf} selector. Re-export spec schemas.
7. Digest traps honored: graph id == baseline stem ("baseline"); verbatim
   baseline ref; direction_path always passed; bank persisted; slug ids;
   -0.0 normalized (x+0.0) in all float tuples.
8. Capture: eval()+no_grad, tuple outputs unwrapped, last-token via
   attention_mask.sum(-1)-1, batch=1, torch/transformers versions in receipt.
   Conv1D (in,out) families: transpose at intake + reverse on export, recorded.
9. CLI: --executor real; lazy imports; preflight [real] deps BEFORE run dir;
   docstring updated (network only in explicit real path). scan_tree gains
   binary-suffix/size skip; HF snapshot pre-scanned at S0 (fail early).
10. S1 baseline: real greedy generation on S6+SR corpora (evaluation role,
    never release_redline); baseline ≠ gate. S8 secret scan + S9 card draft
    reused; card never claims release. Release stays not_eligible; summary
    says so explicitly.
