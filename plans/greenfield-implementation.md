# Greenfield Ablagent implementation

## Boundary

This repository contains reusable DWM/DWMIP research contracts, original synthetic numeric fixtures, and a deterministic local reference runtime. It does not contain prior model work, checkpoints, captured directions, model-specific recipes, benchmark responses, partner data, credentials, serving endpoints, or release authorization.

## Current deliverable

- Typed graph/site/space/writer, bank, direction, intervention, artifact, trial, and evidence schemas.
- Bounded canonical JSON and `safetensors` handling.
- One executable synthetic matrix adapter with exact rank-one update verification.
- Descriptive architecture navigation for residual, latent-MoE, recurrent, shared-weight, fused, and quantized systems.
- Proposal-only bounded planning and separately selected one-intervention compilation.
- Original CC BY 4.0 synthetic capture and inert mock-policy banks.
- Fail-closed eligibility policy with explicit evidence states and no publishing API.
- Apache-2.0 code, contributor/security documentation, and test/lint/audit workflows.

## Verification and publication checklist

- [x] Full tests and coverage.
- [x] Ruff lint and format check.
- [x] Dependency advisory audit.
- [x] Clean source and built-wheel CLI workflows.
- [x] Repository content and history inspected for excluded material.
- [x] Independent reviewer findings resolved or explicitly bounded.
- [x] Private `Blackfrost-AI/Ablagent` remote identity confirmed.
- [x] Initial commit and push completed; remote contents verified.

## Remote CI status

GitHub Actions run `45344338868` did not start any job because GitHub reported a failed-payment or spending-limit issue on the account. This is an external runner-provisioning block, not a failed test or lint step. Local tests, clean-wheel workflows, dependency audit, and content scans passed; rerun the workflow after the account's billing/spending issue is resolved.

## Next qualified increment

A real-model adapter is separate work. It must introduce explicit execution, capture, and codec semantics; qualify each architecture independently; add real serving and behavioral measurements; and preserve private release holdouts outside the repository. No such support is claimed by v0.
