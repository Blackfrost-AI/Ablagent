# Threat model and security boundaries

## Assets

Artifact integrity, tensor identity, research data licensing, private holdouts, approval authority, host filesystem, and truthful evidence/claims.

## Trust boundaries

| Input | Boundary | Control |
|---|---|---|
| Planner request | Untrusted structured input | Strict Pydantic schema, bounded steps, no execution/approval fields |
| Graph/metadata | Untrusted JSON | Size limits, duplicate/non-finite rejection, exact identities |
| Tensor file | Untrusted binary | `safetensors`, header/file limits, dtype/finite checks |
| Bank manifest | Untrusted data and licensing claim | Role, pair, digest, redistribution and visibility validation |
| Evidence | Untrusted claim | Independent artifact recomputation and trial binding |
| Model code | Arbitrary upstream code | No remote-code execution path in v0 |
| Network | External side effects | No runtime HTTP client, publisher, downloader, or telemetry |

## What v0 does not provide

- Confidentiality, encryption, sandboxing of untrusted native dependencies, or authenticated release signatures.
- Protection against a malicious Python caller using the library or modifying the code.
- A secure execution sandbox for arbitrary architecture adapters or third-party model repositories.
- Complete protection against symlink/race attacks in a hostile local output directory. Use a private, operator-controlled run directory.
- Secure private-bank storage. A `private` label is only a policy input.
- Legal clearance of third-party banks, codecs, model licenses, or model outputs.

## Operational expectations

Run in a dedicated environment and operator-owned output directory. Do not enable arbitrary remote code to make an unsupported checkpoint load. Review adapter code before installation. Treat model repositories and generated evidence as untrusted. Never place credentials, real private holdouts, or operational endpoints in Git.

`pip-audit` checks declared Python dependency advisories, not native-code safety or the absence of all vulnerabilities. Report suspected vulnerabilities privately as described in `SECURITY.md`.
