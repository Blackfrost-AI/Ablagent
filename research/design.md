# Design decisions

## Scope

The repository is a clean research runtime. Prior models, weights, directions, prompts, outputs, benchmark scores, serving endpoints, credentials, run history, and model-specific hyperparameters are excluded. Only original synthetic numeric banks are bundled.

## Why the protocol comes before an executor

Architecture compatibility cannot be inferred from a tensor suffix or matching shape. DWMIP binds a checkpoint, graph, site, writer, space, capture, direction, algorithm, and intervention to explicit identities. Complex topologies are representable, but execution is disabled without an adapter with qualified semantics.

## Why v0 has one intervention

The reference update is `W' = W - alpha d(d^T W)` for a declared dense `out x in` matrix and a unit vector `d`. It is analytically checkable, has no implicit multi-direction ordering, and uses one original baseline. This is a protocol test, not proof that the edit changes language behavior.

## Evidence boundary

Candidate file identity is necessary but insufficient. Verification recomputes graph/site/space/writer bindings, transformation metadata, original and candidate tensors, and the full structural audit. Evidence replays synthetic observations and requires completed, metric-consistent retained trials. Eligibility is evaluated separately from artifacts.

Synthetic behavioral claims stay `NOT_RUN`. Mock banks exercise schema and policy wiring only. Real red-line holdouts, capability data, private artifacts, and authorization do not belong in this repository. A structural or policy pass is never a public release approval.

## Distribution boundary

Code is Apache-2.0; original synthetic fixtures are CC BY 4.0. Third-party corpora are not redistributed. The private GitHub repository is preparation for a possible open-source release, not itself an open-source publication. There is no publishing API, telemetry, downloader, or remote model-code execution in v0.

## Source documentation

- `docs/methodology.md` — research synthesis and trial semantics.
- `docs/architecture-navigation.md` — topology-specific contracts.
- `docs/measurement.md` — honest metric and missingness semantics.
- `docs/banks.md` — capture and release-bank roles.
- `docs/determinism.md` — identities, serialization, and numerical scope.
- `docs/threat-model.md` — trust boundaries and fail-closed limits.
- `spec/dwmip/v0/` — protocol and executable synthetic conformance fixtures.
