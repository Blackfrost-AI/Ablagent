# Security policy

## Reporting

This repository is currently private. Report suspected vulnerabilities privately to the repository maintainers through an existing trusted maintainer channel. GitHub private vulnerability reporting was not enabled during initial repository setup. Do not open a public issue containing exploit details, credentials, private prompts, operational endpoints, partner data, or third-party secrets.

Include the affected commit, environment, reproduction steps, impact, and suggested remediation where known. Do not include real credentials or private evaluation material in a report.

## Scope and limitations

v0 is a local synthetic reference implementation, not a production service. It has no network client, model downloader, remote-code execution path, publishing integration, authentication system, or secure private-bank store.

The runtime is not a sandbox for hostile Python/native dependencies. Install only reviewed dependencies and adapters. Use operator-owned run directories. Treat external model artifacts and architecture adapters as untrusted until independently reviewed.

A missing, unknown, or malformed required gate must block eligibility. Do not weaken this behavior to obtain a release result. Visibility metadata does not enforce confidentiality.

## Response

Maintainers should reproduce the issue, preserve the failing evidence, patch the narrowest affected contract, add a regression test, and disclose only after remediation and affected-user coordination.
