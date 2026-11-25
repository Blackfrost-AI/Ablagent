# Running a benchmark (operator procedure)

The repository ships corpora and offline tooling. **It does not ship responses,
judgments, scores, or a served model.** Collection is an explicitly
operator-initiated action against the operator's own endpoint.

## Layout

```
benchmarks/
  REGISTRY.yaml               # corpora, digests, splits
  PROVENANCE.md / NOTICE.md  # license scope
  corpora/*.jsonl             # prompt-only, hash-pinned
  protocol/*.schema.json     # responses / judgments / metrics
  protocol/SCORING.md         # rubric (hash-bound)
  validate.py                 # offline corpus verifier
  collect.py                  # optional, explicit, network collector
  score.py                    # offline, hash-bound scorer
  fixtures/                   # synthetic response/judgment examples
```

Run artifacts (responses, judgments, metrics) belong in a **local, gitignored**
directory such as `.scratch/benchmark-runs/<run-id>/`. Never commit model
outputs or scores.

## 1. Verify the corpora before any run

```bash
python benchmarks/validate.py --strict
```

This checks all three digests, exact row schemas, the smoke-50 prompt
relation, license-file presence, and the namespace allowlist. A failure means
the corpus changed — stop, do not "fix" prompts to make a run pass.

## 2. Collect responses (optional, explicit network step)

```bash
python benchmarks/collect.py \
  --corpus R1-FALSE-REFUSAL-443-v1 \
  --endpoint "$OPERATOR_ENDPOINT" \
  --model "$MODEL_ID" \
  --out .scratch/benchmark-runs/2026-example/responses.jsonl \
  --temperature 1.0 --top-p 0.95 --max-tokens 16384 --seed 42
```

`collect.py` requires an explicit endpoint (no defaults), talks only to that
OpenAI-compatible chat-completions URL, never writes outside the given output
path, and refuses to overwrite an existing file. It records raw completion,
finish reason, decoding, model identity, timestamps, and a per-file digest.
For thinking-enabled models grade the extracted answer region per
[SCORING.md](SCORING.md) §2/§4. If your endpoint is not OpenAI-compatible,
write an adapter that emits the same `responses.jsonl` schema — never edit the
corpus to fit a harness.

## 3. Produce typed judgments

Write `judgments.jsonl` per `judgments.schema.json`. A human or an LLM judge
adjudicates each row using SCORING.md; the deterministic classifier in
`score.py` is only a prefilter. Record evaluator identity and rubric hash
(`sha256sum benchmarks/protocol/SCORING.md`). Never auto-fill `engaged` for
rows you did not read.

## 4. Score offline

```bash
python benchmarks/score.py \
  --corpus R1-FALSE-REFUSAL-443-v1 \
  --responses .scratch/benchmark-runs/2026-example/responses.jsonl \
  --judgments .scratch/benchmark-runs/2026-example/judgments.jsonl \
  --out .scratch/benchmark-runs/2026-example/metrics.json
```

The scorer validates corpus identity, duplicate ids, unknown ids, rubric hash,
and unfinished-row handling; emits a metrics receipt; and always sets
`release_eligible: false`. To record an intentionally unrun corpus:

```bash
python benchmarks/score.py --corpus R1-FALSE-REFUSAL-443-v1 --not-run
```

## 5. Report honestly

- Separate harmful-row engagement, xstest/safe false-refusal rate, xstest/unsafe
  behavior, and cyber-10 qualitative notes. Never collapse them into one
  number.
- Report the unfinished rate and all warnings alongside any rate.
- State the model digest, decoding, template, thinking mode, collector version,
  and evaluator identity.
- Label any smoke-tuned result as development-influenced.
- A score is not a release decision. The real executor in this repository
  still records S6/S7 as skipped and ends `not_eligible`.
