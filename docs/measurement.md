# Measurement and evidence

## Required result states

Every required metric has one of:

- `PASS`: a complete measurement meeting the declared threshold.
- `FAIL`: measured and below threshold.
- `MISSING`: required evidence was not supplied.
- `NOT_RUN`: evaluation was not executed.
- `ERROR`: execution, parsing, or validation failed.
- `NOT_APPLICABLE`: permitted only by an explicit policy predicate and recorded reason; absence alone never implies it.

Required red-line evidence in any state other than a complete `PASS` blocks eligibility. Scores cannot compensate for missing or failed mandatory gates. A distinction is preserved between “behavior failed” and “the gate could not evaluate behavior.”

## Artifact binding

An evidence bundle binds:

- Manifest path and canonical manifest digest.
- Baseline/candidate file and tensor digests.
- Graph, direction, and intervention identities.
- Typed observations and metric evidence digests.
- A selected-trial parent chain, with digest-bound retained outputs and optional verified plan references.
- Environment and decoding configuration.

`verify-evidence` reloads the graph and tensors, independently recomputes the update and synthetic observation, validates the selected parent chain, and checks metric references. For mock red-line evidence it also requires the selected bank via `--bank` and replays the inert fixture. It does not require a database, credentials, or network access.

The verifier checks the declared selected chain, not every experiment branch. Preserving rejected, failed, and superseded sibling trials and limiting repeated private-holdout queries remain the operator's evaluation-infrastructure responsibility.

## Mock-evidence boundary

The `synthetic-mock-v0` evaluator replays exactly six inert controls from the selected `mock_release_redline` bank. It binds the bank reference, baseline/candidate artifact pair, evaluator/configuration literals, canonical raw JSON, raw file hash, and all six results. Missing, extra, or altered bank records and metrics fail closed.

Private evaluation may treat `bank_publication=NOT_APPLICABLE` as informational; every substantive gate must pass. Public mock-bank evaluation is also restricted to synthetic policy-fixture testing. Neither grants release authorization. `redline_observation` records explicit non-passing diagnostics; only the complete attested mock result set may supply passing `redline` metrics.

This proves reproducible fixture wiring and tamper evidence, not evaluator authenticity, behavioral safety, red-line coverage, or release readiness.

Hashes provide tamper evidence, not author authenticity. Signed approvals and a protected release system are separate responsibilities.

## Trial discipline

A trial has `completed`, `rejected`, `failed`, or `promoted` status. Negative records retain their reason and available outputs. Promotion appends a new child record with the original parent identity and does not rewrite the parent.

Selection experiments should predeclare candidate grids, stop conditions, held-out data, and trial limits. Repeated access to a private holdout can turn it into a calibration set; private evaluation infrastructure should enforce query limits and lineage.

## Claims

The current bundle is explicitly `synthetic_reference_only`. Structural correctness, an analytic observation, and policy-conformance tests do not establish:

- A model's behavior changed as intended.
- Capability was preserved.
- A red-line gate passed on a real model.
- A serving stack loads a quantized or fused derivative.
- A third party approved or may publish the result.

Those claims require separate adapter, serving, behavioral, and human-authorized release evidence.

## Metrics must not lie by omission

A passing metric includes a score and complete denominator. Non-passing results cannot carry a passing score. Invalid responses, timeouts, and parser failures must remain in an external evaluator's denominator. Public mock rows only test the policy schema; they are not behavioral red-line measurements.
