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
- Retained trial file paths and trial digests.
- Environment and decoding configuration.

`verify-evidence` reloads the graph and tensors, independently recomputes the update and synthetic observation, validates trial lineage, and checks metric references. It does not require a database, credentials, or network access.

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
