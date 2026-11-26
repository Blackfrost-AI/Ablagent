# Refusal-removal benchmarks

The public, reproducible measurement layer for DWM research — the practical
answer to *"did this edit actually remove false refusals?"*

| Corpus | Rows | Role | Split |
|---|---|---|---|
| [R1-FALSE-REFUSAL-443-v1](corpora/R1-FALSE-REFUSAL-443-v1.jsonl) | 443 | full false-refusal + negative-control eval | evaluation-only |
| [R1-SMOKE-50-v1](corpora/R1-SMOKE-50-v1.jsonl) | 50 | first 50 prompts of the 443; fast serve/wiring smoke | development-only |
| [CYBER-REASONING-10-v1](corpora/CYBER-REASONING-10-v1.jsonl) | 10 | hard security-reasoning probes | qualitative-only |

These are **prompt-only** corpora. This repository ships no responses, scores,
judgments, or generated answers — you run collection and adjudication under an
explicit, hash-bound protocol, locally.

## Start here

```bash
python benchmarks/validate.py --strict     # verify digests, schemas, licenses
cat benchmarks/protocol/SCORING-v2.md         # the rubric (hash-bound into receipts)
cat benchmarks/protocol/RUNS.md           # collection → judgments → offline score
```

`score.py` defaults to strict **v2** schemas and response-bound judgments.
It reports ENGAGED / REFUSED / DEGENERATE / MISSING / ERROR / NEEDS_REVIEW /
NOT_RUN categories, separate harmful engagement and benign false-refusal metrics,
explicit denominators and unresolved counts, and `release_eligible: false`.
Smoke/cyber have no aggregate performance or pass rates. `--completion` checks
full response-file integrity; without it the receipt is explicitly incomplete.
The exclusive incremental collector rejects redirects and bounds reads. Requested
settings are not presented as observed behavior. See [RUNS.md](protocol/RUNS.md).
V1 artifacts remain frozen and require `--legacy-v1`; re-adjudicate for v2.
Tests use inert records and transport stubs, never a live endpoint.

## Honest boundaries

- **These corpora are not a release gate.** The shipped real CPU executor
  records S6/S7 as skipped and every run ends `not_eligible`. Behavior,
  capability, and authorized release evaluations are separate operator gates.
- **Mixed prompt types.** The 443 combines harmful benchmark prompts with
  XSTest safe and unsafe prompts. A refusal on an unsafe row may be correct
  behavior — the scorer reports categories only, not failure.
  The xstest/safe rows are the false-refusal negative controls.
- **Cyber-10 is qualitative.** No expected answers or rubric are shipped; it
  never produces a pass rate.
- **Smoke-50 contaminates the 443 if used for tuning.** It is the same first
  50 prompts. Record any tuning; see SCORING-v2.md.
- **Historical numbers are run-record claims.** Figures like "90.2% compliance"
  elsewhere in this repository describe specific past model builds, not
  measurements made by this code.

## Provenance & license

Upstream: AdvBench (MIT), StrongREJECT (MIT), XSTest (CC BY 4.0). Full license
texts, attribution, the adaptation notice, and the exclusion boundary are in
[PROVENANCE.md](PROVENANCE.md), [NOTICE.md](NOTICE.md), and `LICENSES/`.
Cyber-10 and all tooling are first-party under the repository's Apache-2.0.

An operator-controlled, release-only red-line corpus is intentionally absent
and was not opened or copied. The public synthetic development fixture under
[`packs/derisk/`](../packs/derisk/NAMESPACE-NOTICE.md) is unrelated to it.

See [REGISTRY.yaml](REGISTRY.yaml) for digests, composition, split policy, and
tooling pointers.