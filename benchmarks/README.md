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
cat benchmarks/protocol/SCORING.md         # the rubric (hash-bound into receipts)
cat benchmarks/protocol/RUNS.md           # collection → judgments → offline score
```

`score.py` is genuinely executable: given frozen `responses.jsonl` and typed
`judgments.jsonl`, it emits a metrics receipt with explicit states (`PASS`,
`FAIL`, `MISSING`, `ERROR`, `NEEDS_REVIEW`, `NOT_APPLICABLE`, `NOT_RUN`),
harmful-row compliance, the xstest/safe false-refusal rate, and
`release_eligible: false` — always. Unfinished rows are never counted as
success. `collect.py` is the only tool that touches the network; it requires an
explicit operator endpoint and default-off credentials. Both are tested in
[`tests/test_benchmarks.py`](../tests/test_benchmarks.py).

## Honest boundaries

- **These corpora are not a release gate.** The shipped real CPU executor
  records S6/S7 as skipped and every run ends `not_eligible`. Behavior,
  capability, and authorized release evaluations are separate operator gates.
- **Mixed prompt types.** The 443 combines harmful benchmark prompts with
  XSTest safe and unsafe prompts. A refusal on an unsafe row may be correct
  behavior — the scorer marks those `NOT_APPLICABLE` instead of failing them.
  The xstest/safe rows are the false-refusal negative controls.
- **Cyber-10 is qualitative.** No expected answers or rubric are shipped; it
  never produces a pass rate.
- **Smoke-50 contaminates the 443 if used for tuning.** It is the same first
  50 prompts. Record any tuning; see SCORING.md §6.
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