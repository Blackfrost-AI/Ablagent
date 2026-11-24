# Contributing

## Project boundary

Contribute reusable protocol semantics, architecture adapters, codecs, tests, evidence validation, and clean licensed banks. Do not contribute prior model weights, captured directions, model-specific parameters, model generations, run results, private prompts, credentials, operational endpoints, or partner material.

## Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/ruff check src tests
.venv/bin/pip-audit
```

Run tests before proposing changes. Add positive and negative tests for each new contract. Do not replace a failing safety/identity assertion with a permissive fallback merely to make a test pass.

## Adapter requirements

An architecture adapter must define explicit graph, site, writer, space, capture, and codec contracts. It must distinguish representation from execution and declare real-checkpoint support separately. Include exact environment provenance, supported operations, unsupported cases, and independent verification.

Unknown semantics must fail closed. Never infer model support from a filename, matching tensor dimensions, suffix match, or a sibling model's recipe. New codec and serving claims require separate qualification.

## Data and licenses

Do not bundle third-party prompt/annotation corpora without a documented redistribution decision. Code licensing does not license datasets. New original synthetic fixtures need an explicit content license and provenance. Private release banks stay outside the repository.

Vendored research material (see `examples/inkling-2-mini-derisked/`) must be byte-identical to a pinned public source revision, carry a per-file `PROVENANCE.json` manifest and license scope (`NOTICE.md`), and stay excluded from lint/format so provenance is preserved. Large assets are fetch-only through explicit, digest-verified helpers — never implicit downloads.

## Security

No arbitrary remote model code, telemetry, implicit downloads, credential handling, or self-approval paths. Treat planner outputs and external artifacts as untrusted. See `SECURITY.md` and `docs/threat-model.md`.
