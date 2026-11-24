# DWMIP v0 normative specification

## 1. Scope and conformance language

DWMIP defines artifact identities and transition requirements for Domain Weight Modification experiments. **MUST**, **MUST NOT**, and **MAY** are normative. DWMIP v0 does not authorize an experiment, certify behavioral safety, establish real-model support, or publish an artifact.

A conforming implementation MUST reject unknown required semantics. It MUST NOT infer a compatible space from tensor names, dimensions, or cosine similarity.

## 2. Identity and serialization

Manifests use the strict `dwmip.*.v0` schemas implemented by this package. Unknown fields, non-finite floats, duplicate JSON keys, invalid UTF-8, and malformed SHA-256 identities MUST be rejected.

`sha256:<lowercase hex>` identifies exact bytes. `canonical-json-v1` is the project's deterministic UTF-8, sorted-key, compact JSON representation; it is not RFC 8785. A digest detects modification but does not authenticate an author.

The graph digest binds sites, spaces, writers, checkpoint identity, codec declarations, and adapter identity. Artifact references bind role, checkpoint, graph, relative path, file bytes, and the canonical tensor set. Tensor identities include sorted tensor names, shapes, dtypes, C-contiguous bytes, and the declared graph.

## 3. Architecture model

A `GraphSpec` describes ordered sites and explicit `SpaceSpec` and `WriterSpec` objects. A logical coordinate identifies its site and, where applicable, expert, route, recurrent state/step, or loop iteration.

A writer declares exact tensor names, layout, axes, sharing, execution order, parent/child sites, and source/destination spaces. Topology descriptors alone do not confer execution support. Recurrent, looped, shared, MoE, fused, and quantized execution require separately qualified semantics.

## 4. Banks and direction binding

A capture bank identifies its source, record-set digest, normalization contract, and license. Capture, calibration, evaluation, and release roles MUST NOT be silently interchanged. Candidate statistics retain observed/eligible counts and provenance; correlations are not behavioral results.

A `DirectionCandidate` is not executable. A `BoundDirection` MUST bind one writer, source space, tensor, graph, baseline, capture bank, source algorithm, and source vector. Its vector MUST be finite and unit-normalized within the declared tolerance.

A direction from another checkpoint, graph, tensor, bank, writer, or space MUST NOT be reused without a separately defined map and qualification. No such transfer mechanism exists in v0.

## 5. Reference intervention

The only executable operation is `dense.rank_one_output.v0` on the declared `synthetic-cpu-v0` dense matrix adapter:

```text
W' = W - alpha * d * (d^T W)
```

`W` has shape `out x in`; `d` has shape `out`; the output projection is applied exactly once. The direction MUST match the writer and source space. At `alpha=1`, this is orthogonal direction removal. Other values are rank-one updates, not generally projections.

One intervention contains one direction and one coordinate. v0 does not define multi-direction ordering, implicit orthogonalization, shared-parameter replication, quantized encoding, or cross-model transfer.

## 6. Verification and evidence

Compilation MUST compare the selected intervention with the persisted direction file, verify baseline identity and graph applicability, and record the exact baseline/candidate identities. Non-target tensors MUST be unchanged.

Verification MUST independently recompute the writer output and the complete structural audit. It MUST NOT trust a stored `passed` flag or candidate-provided expected digest.

An evidence bundle binds the compiled manifest, baseline/candidate bytes, graph, direction, and intervention. It retains completed trials with matching metrics. Synthetic observations are analytic diagnostics, not language outputs.

A metric result is `PASS`, `FAIL`, `MISSING`, `NOT_RUN`, `ERROR`, or `NOT_APPLICABLE`. Passing requires a positive denominator, complete observations, and a score. A zero-eligible or missing required red-line result is not a pass.

## 7. Eligibility boundary

All six required red-line categories must have complete passing evidence for public policy eligibility. The external release bank must be explicitly licensed for public bundling. Missing or malformed gates block evaluation.

Eligibility is not authorization. v0 has no publisher, credentials, approval identity, confidentiality store, serving engine, or real behavioral evaluator. Private real release holdouts MUST remain outside the public repository.

## 8. Conformance

The bundled vectors in `conformance/` exercise valid graphs, missing spaces, unsupported execution, writer/space/normalization identity failures, exact algebra, and candidate/audit tampering. Run them with:

```bash
ablagent conformance --fixtures spec/dwmip/v0/conformance --run-dir .runs/conformance
```

Passing this suite establishes only synthetic reference conformance. New adapters, codecs, and serving/behavioral claims need independent qualification and evidence.
