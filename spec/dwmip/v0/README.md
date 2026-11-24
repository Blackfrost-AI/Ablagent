# DWMIP v0: Domain Weight Modification Interchange Protocol

DWMIP binds a proposed modification to exact computation semantics, artifact bytes, and retained evidence. A valid manifest is **not** a certificate of behavioral efficacy, safety, authorization, or general architecture support.

## Identity chain

```text
GraphSpec + CaptureContract + BankManifest
        → DirectionCandidate (correlational)
        → BoundDirection (one writer, one space, unit normalized)
        → InterventionSpec (one coordinate, one operation)
        → CompiledArtifact (baseline and candidate identities)
        → EvidenceBundle (trials, observations, results)
        → EligibilityDecision (policy conformance only)
```

Each transition validates the preceding identity. A candidate vector cannot be treated as a bound direction. An arbitrary graph cannot be treated as executable. A loaded artifact cannot be treated as behaviorally evaluated.

## Required identities

- Schema, algorithm, and adapter versions.
- Checkpoint, graph, tensor, dtype, shape, layout, and axis order.
- Logical site coordinate: layer plus expert, route, recurrent state/step, or loop iteration where applicable.
- Source/destination space, basis, normalization, inner product, and coordinate mapping.
- Capture/calibration/evaluation bank manifest and record-set digests.
- Candidate statistic and full counts, source vector digest, normalized direction digest.
- Intervention parameters and exact baseline/candidate tensor and file digests.
- Codec readback, transformation metadata, structural verification, and retained trial lineage.

Equal tensor dimensions, matching names, or high cosine similarity do not substitute for these identities.

## Capability declarations

An adapter declares executable topologies, operations, tensor layouts, codecs, and real-checkpoint support. Qualification separates:

1. **Well-formedness:** schema and unique/required fields.
2. **Decodability:** bounded, safe tensor deserialization.
3. **Applicability:** exact identity, geometry, space, and writer semantics.
4. **Numerical verification:** independent operation recomputation and non-target preservation.
5. **Behavioral acceptance:** separately measured and scoped; never implied by 1–4.

Unknown semantics return `UNSUPPORTED`/`ValidationError`; they do not activate a suffix or shape fallback.

## Reference v0 subset

- Descriptive graph IR can represent dense, MoE, recurrent, and looped coordinates.
- Only the analytic `dense_linear` synthetic adapter is executable.
- One unshared `dense_out_in` float32/float64 matrix per declared site.
- `W′ = W − αd(dᵀW)`, with unit `d`, one coordinate, and `0 ≤ α ≤ 1`.
- At `α=1` this is orthogonal direction removal. Arbitrary `α` is a rank-one update, not generally a projection.
- Multi-direction composition is unsupported until order and orthogonalization are explicit.
- `safetensors` supports the reference read/write/audit subset. Other formats and quantizers require independent conformance.

## Conformance

Run `pytest`. The suite supplies positive and negative examples for graph identity, missing spaces, recurrent coordinates, exact algebra, normalization, bank role/identity, unsupported topology execution, malformed tensors, transformation tampering, planner containment, trial retention, and fail-closed policy decisions.

An adapter should additionally provide:

- Valid and invalid metadata examples.
- Read/write/readback and corruption cases.
- Wrong identity, shape, dtype, space, and coordinate cases.
- Unsupported operation and unshared/shared parameter cases.
- Resource-limit and non-finite input cases.
- Independent behavioral and serving qualification before any real-model claim.

## Provenance and authenticity

SHA-256 identities detect content changes; they do not authenticate an author. Release signatures and approval identities belong to a separate release system. JSON evidence is independently verifiable without SQLite or an online service.
