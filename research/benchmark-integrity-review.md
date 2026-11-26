# Benchmark integrity implementation and local verification

Status: implemented and final local verification passed. Parent review finding resolved:
`contracts.jsonl_bytes` now splits records on literal LF bytes, removes only one terminal
newline, preserves interior blank-line rejection, and accepts CRLF JSON whitespace.
No commit/push. Independent review was partial (iteration ceiling); this fix was rechecked
by the full mandatory offline suite and lint.
No commit/push. No live endpoint, downloaded checkpoint, benchmark/model qualification,
corpus edit, recipe edit or vendored-example edit. Only dependency installation used
network. Tests use inert records and temporary tiny locally constructed HF fixtures.

## Changes

- `benchmarks/contracts.py`: strict JSON duplicate/nonfinite/Unicode rejection,
  local schema enforcement, full-record canonical response digest.
- `benchmarks/protocol/{responses,judgments,metrics,completion}.v2.schema.json`:
  strict versioned contracts; human/LLM adjudication only; full binding and coverage.
- `benchmarks/protocol/SCORING-v2.md`: versioned binding/rubric, population semantics,
  provenance and migration boundaries. Frozen v1 contracts/rubric unchanged.
- `benchmarks/scoring_v2.py`, `benchmarks/score.py`: strict v2 default, explicit legacy
  v1 mode, mixed-run/stale/swapped evidence rejection, separate harmful engagement
  and benign false refusal; smoke/cyber categories only; completion verification.
- `benchmarks/collect.py`: validated explicit URL, no redirects/environment proxies,
  8 MiB reads, controlled malformed transport results, exclusive incremental/fsynced
  output, exclusive full-SHA completion receipt, requested/observed provenance.
- `benchmarks/registry.py`, `benchmarks/validate.py`: shared structured duplicate-key
  rejecting YAML loader, corpus allowlists/path confinement, rows/IDs/labels/composition.
- `tests/test_benchmarks.py`: legacy regressions preserved, negative corpus tests
  confined to temporary copies. `tests/test_benchmarks_v2.py`: 78 offline regressions.
- `tests/conftest.py`: socket denial fixture; required-real mode fails on skips.
- `.github/workflows/verify.yml`: all-corpus CLI checks and mandatory Python 3.12
  CPU real-adapter lane (torch 2.6.0 / transformers 4.48.3), offline environment,
  import preflight and skip rejection.
- `pyproject.toml`: jsonschema runtime dependency.
- `benchmarks/README.md`, `benchmarks/protocol/RUNS.md`,
  `benchmarks/fixtures/README.md`, `docs/measurement.md`, `docs/support-matrix.md`:
  current operator procedure and strict separation from pipeline/release gates.

## Observed verification (local isolated .venv)

Environment: macOS arm64, Python 3.14.3, torch 2.15.0, transformers 5.24.1,
jsonschema 4.26.0. Existing .venv reused; only jsonschema and transitive requirements
installed. CI's Linux/Python 3.12 pinned lane is configured but not executed locally.

- `ablagent doctor`: Python/PyYAML/DAG/packs OK (6 packs).
- `pytest tests/test_benchmarks.py -q`: **10 passed**.
- `pytest tests/test_benchmarks_v2.py -q`: **69 passed**.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 ABLAGENT_REQUIRE_REAL_TESTS=1 .venv/bin/python -m pytest tests/ -q`:
  **251 passed in 5.42s; no skips** (final rerun).
- `ruff check src tests scripts benchmarks`: **All checks passed**.
- `ruff format --check src tests scripts benchmarks`: **82 files already formatted**.
- `python scripts/export_schemas.py --check`: exit 0.
- `python scripts/verify_example_provenance.py`: all 15 vendored example files verified.
- `python benchmarks/validate.py --strict`: PASS, all three frozen corpus hashes.
- `git diff --check`: exit 0.
- `git diff --exit-code HEAD -- benchmarks/corpora recipes examples/inkling-2-mini-derisked benchmarks/protocol/SCORING.md benchmarks/protocol/responses.schema.json benchmarks/protocol/judgments.schema.json benchmarks/protocol/metrics.schema.json`: exit 0, frozen bytes unchanged.
- All-corpus v2 CLI NOT_RUN/schema checks execute in the parametrized v2 tests.

## Limits

Hashes establish consistency, not authenticity. Model/checkpoint settings are operator
assertions; generic collection cannot prove applied template/thinking/decoding or weight
identity. No serving readiness, efficacy, behavior, capability or release was measured.
S6/S7 remain skipped and release remains false/not_eligible. No evidence-import gate.
Unresolved rows remain explicit; rates are conditional on adjudication. V1 remains legacy
and unbound. No resume/overwrite; interruptions require fresh paths/run IDs. Transport
is stubbed in tests, not a live-service qualification. CI execution and external advisory
lookup were not performed. In-process socket denial is not an OS-level network sandbox.


## Final hardening and negative evidence

Scoring now hashes the same immutable byte snapshots it parses (responses,
judgments and completion), avoiding path re-read races. Collection compares the
persisted file against the hash of bytes it actually wrote before completion.
Registry ancillary metadata is type/key checked. Added nine cases for concurrent
output mutation, snapshot hashes, schema-valid receipt tampering, empty/truncated/
timeout responses, and pre-transport numeric/identity rejection.

An initial snapshot-race test failed twice because its injection ran during corpus
loading, before the response snapshot. Narrowing the injection to response parsing
corrected the test boundary; final full suite passed. Earlier formatting/import
lint findings were corrected before the recorded final checks. No product failure
was suppressed. Independent review statically inspected contracts, collector,
scorer, registry, response schema and CI, but did not complete all files or execute
checks; parent review is still required. The final local verification is the
executed evidence, not that partial review.
