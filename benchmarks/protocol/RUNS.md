# Running a benchmark (v2 operator procedure)

Install the project dependencies (`pip install -e '.[dev]'`). All tooling except
explicit collection is offline. No model is served or qualified by these commands.
Keep artifacts under a local gitignored directory; never commit actual responses.

1. Run `python benchmarks/validate.py --strict` and read
   [SCORING-v2.md](SCORING-v2.md). Do not repair frozen prompts to pass validation.
2. Only with separate operator authorization, collect against the explicit endpoint:

```bash
python benchmarks/collect.py --corpus R1-FALSE-REFUSAL-443-v1 \
  --endpoint "$OPERATOR_ENDPOINT" --model "$MODEL_ID" \
  --model-digest "$CHECKPOINT_SHA256" --run-id "$RUN_ID" \
  --out .scratch/benchmark-runs/example/responses.jsonl \
  --temperature 1.0 --top-p 0.95 --max-tokens 16384 --seed 42
```

The checkpoint digest is an operator assertion, not a server attestation.
Set credentials via `ABLAGENT_BENCHMARK_TOKEN`; avoid command-line secrets.
URLs with credentials, query strings or fragments are rejected. Redirects and
environment proxies are disabled. Responses are bounded to 8 MiB per request.
Use TLS for credentials: HTTP is accepted for operator-controlled local serving
but is unencrypted. No automatic retries, resume or overwrite. Records are written
and fsynced incrementally. Completion creates `responses.jsonl.complete.json`
with the **full** SHA-256, coverage and error count. An interrupted file is incomplete;
start a new run/path. Completion with errors is not successful measurement.

Template/thinking flags record intent only: the generic adapter cannot enforce
or observe them. Decoding parameters are transmitted, not proven applied.
`observed_settings` records only server-returned model ID (null when absent),
unknown thinking and null template. Mismatched returned model ID is an error.
This does not establish checkpoint identity, comparability or serving readiness.

3. Re-adjudicate each full extracted answer using `judgments.v2.schema.json`.
   Include evaluator ID/version and, for LLM judges, judging-prompt SHA-256.
   Repeat run/model/corpus/row identity and bind the exact response record:

```python
from benchmarks.contracts import digest, jsonl, sha256
responses = jsonl(".scratch/benchmark-runs/example/responses.jsonl", "responses")
response_sha256 = digest(responses[0])  # entire canonical response object
rubric_sha256 = sha256("benchmarks/protocol/SCORING-v2.md")
```

Never auto-fill a judgment for an unread answer. Deterministic prefilters are not
adjudicators. See `tests/test_benchmarks_v2.py` for schema-valid inert examples.

4. Score offline, supplying the completion receipt to verify collection integrity:

```bash
python benchmarks/score.py --corpus R1-FALSE-REFUSAL-443-v1 \
  --responses .scratch/benchmark-runs/example/responses.jsonl \
  --judgments .scratch/benchmark-runs/example/judgments.jsonl \
  --completion .scratch/benchmark-runs/example/responses.jsonl.complete.json \
  --out .scratch/benchmark-runs/example/metrics.json
```

Omitting `--completion` produces explicitly incomplete partial analysis, never a
verified complete collection. To record no execution use `--not-run` without inputs.
Do not mix runs or models. Stale/swapped judgments, invalid schemas and unknown rows
fail closed. Corrections require new versioned evidence and re-adjudication.

5. Report all categories, denominators, unresolved counts and warnings. Harmful
engagement is not benign false refusal. Smoke/cyber have no aggregate pass rates.
Label smoke-tuned results development-influenced. A score does not complete S6/S7,
capability, serving or release. Every receipt remains `release_eligible: false`.

## Legacy v1

The unversioned schemas and `SCORING.md` remain frozen for audit. V1 inputs require
`score.py --legacy-v1`; their unbound judgments and historical mixed-population
semantics are explicitly legacy, not v2 evidence. Do not simply relabel them.
The placeholder files under `fixtures/` are legacy illustrations only.
