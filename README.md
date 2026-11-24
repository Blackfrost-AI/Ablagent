# Ablagent

**An evidence-first Domain Weight Modification (DWM) research protocol with a synthetic reference runtime.**

> **Status: protocol / synthetic alpha.** This repository does not support real model checkpoints, claim behavioral efficacy, certify safety, authorize releases, or ship previous model work. It provides executable contracts and tests for building qualified adapters.

## What is here

- **DWMIP v0:** explicit graph, site, vector-space, writer, tensor, bank, direction, intervention, and evidence identities.
- **Reference algebra:** one normalized direction, one coordinate, one dense `out × in` writer, and the update `W′ = W − αd(dᵀW)`.
- **Safe artifacts:** canonical JSON and `safetensors`, bounded parsing, exact transformation metadata, independent verification, and retained negative trials.
- **A bounded planner:** deterministic, proposal-only experiment plans with no execution or approval authority.
- **Fail-closed gates:** missing/error/not-run red-line evidence blocks eligibility. Visibility flags do not provide confidentiality or publication.
- **Architecture research:** adapter requirements for residual, latent-MoE, recurrent, shared-weight, fused, and quantized computation.
- **Original synthetic fixtures:** numeric prompt banks with explicit data licenses; no model generations or third-party corpus rows.

## Quick start

Requires Python 3.11 or newer. The reference runtime uses CPU NumPy and does not require a GPU, model download, remote model code, API key, or network service.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/ablagent capabilities
.venv/bin/ablagent demo --run-dir .runs/reference
.venv/bin/ablagent verify .runs/reference/candidate.manifest.json
.venv/bin/ablagent gate .runs/reference/candidate.evidence.json \
  --bank examples/banks/synthetic-release-mock.bank.json
```

The demonstration reports structural and diagnostic observations only. Its intentionally incomplete red-line evidence produces a blocked eligibility decision; this is expected. `gate` returns exit status `2` when blocked. Other contract failures return `1`. There is no publish command.

Plan and compilation are deliberately separate:

```bash
.venv/bin/ablagent plan bound.direction.json \
  --run-dir .runs/reference --alpha 1 --output one.plan.json
.venv/bin/ablagent compile bound.direction.json \
  --run-dir .runs/reference --plan one.plan.json --artifact-id independent
```

See [`docs/quickstart.md`](docs/quickstart.md) for the artifact flow and [`docs/support-matrix.md`](docs/support-matrix.md) for exact limits.

## Tests

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests
```

Tests cover hand-calculated algebra, dtype/axis/identity failures, safe tensor parsing, tampered manifests, private-bank restrictions, missing red-lines, unsupported graph execution, planner containment, negative trial retention, and an offline CLI workflow.

## Research map

- [Method overview](docs/methodology.md): candidate discovery versus validated intervention.
- [Architecture navigation](docs/architecture-navigation.md): why names and dimensions do not establish compatible spaces.
- [DWMIP specification](spec/dwmip/v0/README.md): manifests, capability qualification, and conformance.
- [Measurement contracts](docs/measurement.md): artifact identity, retained trials, and evidence states.
- [Bank policy](docs/banks.md): roles, splits, redistribution, and private release holdouts.
- [Threat model](docs/threat-model.md): untrusted planners, input files, evidence, and code execution.
- [Determinism](docs/determinism.md): exactly what reproducibility means here.

## Contributing and security

Contributions should improve reusable semantics, tests, or qualified adapters—not introduce model-specific recipes. See [CONTRIBUTING.md](CONTRIBUTING.md). Report vulnerabilities through [SECURITY.md](SECURITY.md); do not place secrets, credentials, private prompts, or operational endpoints in issues or pull requests.

## Licensing

Code is [Apache-2.0](LICENSE). Project-authored synthetic bank content is [CC BY 4.0](banks/LICENSE-CC-BY-4.0), with per-bank provenance in its manifest. A code license does not implicitly license third-party model weights, prompts, annotations, or serving stacks.
