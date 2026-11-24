# Determinism and canonical identity

## Canonical JSON v1

Ablagent canonicalizes schema-derived JSON values as UTF-8, sorted object keys, no insignificant whitespace, and UTF-8 characters rather than ASCII escaping. It rejects non-string keys, duplicate input keys, NaN, Infinity, and negative zero.

This small format is **not RFC 8785/JCS** and does not claim arbitrary cross-language numeric canonicalization. v0 uses canonical JSON for manifest identity; numeric tensor bytes have separate dtype/shape/content identities.

Digests are domain-separated:

```text
SHA256("ablagent" || NUL || domain || NUL || "v1" || NUL || payload)
```

The result is encoded as `sha256:<lowercase hex>`. File digests are plain SHA-256 over exact file bytes. Tensor identities include names, shapes, dtypes, and contiguous bytes.

## Scope of reproducibility

The deterministic planner is a pure function of its typed request. Reference compilation is deterministic in the tested local NumPy/safetensors environment. Tests verify exact local readback and independent recomputation.

NumPy/BLAS floating-point behavior, dependency versions, hardware, filesystem metadata, serving engines, and future LLM planners do not automatically provide portable bitwise identity. This project does not claim cross-platform bitwise reproducibility.

Each real adapter must declare its numerical tolerance, precision transitions, backend versions, and any nondeterminism. A tolerant floating-point comparison is not the same as artifact-byte identity.

## Immutability and recovery

Artifact filenames are not reused. Failed operations remove incomplete candidate tensor payloads where possible; completed negative trials remain in the journal. An existing graph or manifest blocks accidental overwrite. A failed partial run is not silently resumed as if it were complete.
