# Synthetic fixture — no real model outputs

This directory demonstrates the response/judgment/metrics mechanics with
**invented** strings. It is not a benchmark result, not a model output, and not
a score. Fixtures are constructed dynamically by `tests/test_benchmarks.py` in a
temp directory: the rubric hash, options digests, and row ids are derived at
test time so the fixture never goes stale when a corpus or protocol changes.

`responses.sample.jsonl` and `judgments.sample.jsonl` below show the shape of
one record each, using placeholder ids — they are examples only and are not
consumed by the test suite.

These are **legacy v1** illustrations, not valid v2 evidence. Strict v2 examples
are constructed and schema-validated in `tests/test_benchmarks_v2.py`; see
`../protocol/RUNS.md` for binding and migration. Never relabel v1 judgments.
