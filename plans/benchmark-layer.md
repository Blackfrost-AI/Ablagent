# Plan — real refusal-removal benchmark layer + deep-review fixes (v2)

Sanitized public plan. Only generic references to an operator-controlled
release corpus appear here; no private identifiers, paths, or digests are
recorded in this repository.

## Verified facts (local inspection, 2026-11-25)

| Fact | Evidence |
|---|---|
| 443 eval corpus | 146 advbench / 149 strongreject / 148 xstest = 443 prompts; fields `id,dataset,prompt,label`; prompt-only, no responses/scores |
| Smoke-50 | 50 rows all advbench; fields `dataset,prompt,label` (no `id`); **prompt-sequence equals the first 50 prompts of the 443**, not a byte-wise row slice |
| Cyber-10 | 10 first-party authored security reasoning probes; fields `id,topic,prompt`; prompt-only; no source-license encumbrance identified, no attribution claim beyond "Blackfrost-authored" |
| Upstream terms (primary-source fetch) | llm-attacks (AdvBench) MIT; StrongREJECT MIT; XSTest CC BY 4.0 (no ShareAlike); redistribution with attribution allowed; verbatim license texts committed |
| Exclusion | An operator-controlled, release-only red-line corpus is intentionally absent. It was not opened, copied, or referenced by identity. No such content belongs in the public repo. |
| Content sweep | no responses/score/answer columns, credentials, private endpoints, or internal paths. Public URLs inside upstream source prompts are source content. |
| Pack integrity | `packs/derisk/corpora/*` and `PACK.yaml` are hash-bound. Row bytes, corpus IDs, file names, and pins stay UNCHANGED; de-conflation is non-inline only. |
| Real stale issues | `docs/support-matrix.md` claims real checkpoint/capture unsupported; `packs/derisk/PROVENANCE.md:3` cites a non-existent generator; README/site cite 443 figures without reproducible corpora. |

## Implementation (ranked by release risk)

1. **Public/private separation (P0)** — no protected corpus id/hash/row-count literals anywhere in the repo. Only the generic statement above. `validate.py` uses a path/namespace allowlist for published corpora; it does not encode forbidden literals. No private run paths/endpoint names in any public file.
2. **Non-inline de-conflation (P0)** — add `packs/derisk/NAMESPACE-NOTICE.md`: the 9-row fixture is a public synthetic development fixture; legacy `redline_holdout`/`release_redline` vocabulary is deprecated; the operator-controlled release corpus is absent; mock SR results are wiring demonstrations only. Short pointers from GUIDE/PROVENANCE text (non-hash-bound files only). `PACK.yaml`, all corpora, digests untouched. CI asserts the 9-row digest equals its pin.
3. **Executable offline benchmark (P0)** in `benchmarks/`:
   - corpora copied **byte-for-byte, original schemas preserved** — `R1-FALSE-REFUSAL-443-v1.jsonl`, `R1-SMOKE-50-v1.jsonl`, `CYBER-REASONING-10-v1.jsonl`; registry records exact sha256 + rows + composition + allowed stages + source-license refs.
   - `validate.py` — strict offline verifier: digest/row/schema/corpus-id checks, smoke50 prompt sequence == first-50-of-443, no protected-namespace fields, no answer-like columns, no credential/private-path patterns, license files present. Includes tamper tests.
   - `score.py` — **genuinely executable** offline scorer. Consumes frozen `responses.jsonl` + `judgments.jsonl`, validates IDs/corpus digest/evaluator rubric hash, computes per-dataset metrics with explicit states (`PASS/FAIL/MISSING/NOT_RUN/ERROR/NOT_APPLICABLE/NEEDS_REVIEW`); deterministic Unicode-normalized refusal classifier as a **prefilter only**; typed judgments authoritative; unfinished rows never silently counted as success; hash-bound metrics receipt. No network, no content generation.
   - `collect.py` — optional **network-isolated** collector for OpenAI-compatible endpoints; records raw completion, finish reason, model id, decoding params, template, timestamps, response digest; refuses to write anywhere except the caller's output path; prompts-only corpus never modified. Operator supplies endpoint explicitly; no default endpoint.
   - schemas: `responses.schema.json`, `judgments.schema.json`, `metrics.schema.json`; small **synthetic** response/judgment fixture with unit tests (Unicode curly apostrophes, truncation, timeout, duplicate id, missing judgment, empty response).
   - `protocol/SCORING.md` / `protocol/RUNS.md`: bare prompts, thinking ON where supported, answer-region extraction full text, human/LLM adjudication, negative controls (xstest/safe false-refusal rate; xstest/unsafe correct refusal NOT automatically failure), no tuning on 443, smoke-50 tuning contaminates 443, cyber-10 qualitative. Historical run-record figures labeled claims, not measurements by this repo.
   - `PROVENANCE.md` + `LICENSES/` — verbatim MIT texts with exact copyright holders, CC BY 4.0 legal code or pinned copy + hash, NOTICE with citations, explicit "modified/adapted subset" statement for XSTest, dataset-level attribution (per-row upstream attribution is NOT claimed — the 443 IDs are anonymous), upstream URLs + revisions, retrieval date, transformation = byte copy, rebuild/verify receipt.
4. **Genuine stale fixes (P1)** — `docs/support-matrix.md` rows for real dense CPU executor (intake/capture/bake/export implemented; behavior/serving/quantization/eligibility unmeasured); `packs/derisk/PROVENANCE.md` dead generator reference corrected; `README.md`, `docs/index.html`, `docs/quickstart.md` benchmark section with accurate reproducibility statement; mock S6/S7 prose labeled row-presence wiring check, not behavioral pass. **Runtime evidence typing** (`evaluator: synthetic-mock-v0`, `measurement_class: wiring-only`, `semantic_untested: true`) added in `_evaluate_generic_gate` output with tests.
5. **CI (P1)** — `.github/workflows/verify.yml`: `python benchmarks/validate.py --strict`, `pytest` benchmark tests, `ruff check src tests scripts benchmarks`, `ruff format --check` same, link checker if present. No new CLI subcommand needed.
6. **Verify + publish** — full pytest suite + ruff + schemas export + example provenance + benchmark validate/score/collect dry tests + secrets scan + link check; then commit/push, Pages rebuild check.

## Hard boundaries

- No private/protected corpus material, identifiers, digests, run paths, or endpoints in public files.
- No responses, scores, model outputs, or generated answers committed.
- No mutation of hash-bound pack rows, `PACK.yaml` pins, or recipe snapshots.
- Real executor stays `not_eligible`; S6/S7 remain skipped; no claim of release readiness.
- No per-row upstream attribution claim; no ShareAlike overstatement; no unsafe dual-use content beyond the upstream research prompts, which are prompts only.
