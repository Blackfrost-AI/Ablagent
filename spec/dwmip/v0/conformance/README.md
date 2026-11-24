# Executable conformance vectors

Run from the repository root:

```bash
ablagent conformance --fixtures spec/dwmip/v0/conformance --run-dir .runs/conformance
```

Each case declares an `accept` or `reject` expectation. The runner constructs fresh synthetic objects, performs the named operation, and reports whether the observed result matches. A passing conformance report is limited to these synthetic protocol cases. It does not qualify a real model, architecture, codec, serving stack, behavioral evaluation, or release.

The numerical vectors contain only original matrices and directions. File-backed verification cases generate temporary run artifacts below the explicit `--run-dir`; use a fresh directory for a new run because artifacts are immutable.
