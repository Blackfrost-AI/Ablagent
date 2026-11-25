# Namespace notice — synthetic development fixtures

**Read this before interpreting `derisk-sr-9.jsonl` or any mock SR result.**

- The nine rows in [`corpora/derisk-sr-9.jsonl`](corpora/derisk-sr-9.jsonl)
  are **public, synthetic development fixtures**: machine-written strings with
  placeholder labels, created solely to exercise the pack loader, gate graph,
  and mock executor wiring. They are not severity-rated prompts, not a
  measurement instrument, and not an evaluation result.
- The legacy inline vocabulary in those rows (`redline_holdout`,
  `release_redline`, and the corpus filename `derisk-sr-9`) is **deprecated
  naming from the pack's construction**. It does **not** refer to, derive from,
  or substitute for any operator-controlled release-only corpus. An
  operator-controlled, release-only red-line corpus is intentionally absent
  from this repository and was not opened or copied.
- Because `PACK.yaml` and the per-file manifest bind every corpus byte and its
  SHA-256, **these rows, filenames, corpus ids, and pins are intentionally
  unchanged**. Clarifying their meaning here avoids breaking hash-bound
  provenance; a semantic rename would require a new versioned pack.
- Mock-executor results for the SR/release stage are **wiring demonstrations
  only** (row presence / executor round-trip). They are not behavioral
  measurements. A skipped stage is never a passed stage, and every real run
  ends `not_eligible` until separately qualified gates are recorded.

For a real, publishable refusal-removal measurement corpus with protocol,
scorer, and license scope, see [`benchmarks/README.md`](../../benchmarks/README.md).
