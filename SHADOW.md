# Ablagent — Shadow entry

Read [`AGENTS.md`](AGENTS.md) completely first — it is the canonical boot
contract (scope, executors, hard rules). Then:

1. `ablagent doctor` before any run.
2. Default to the offline `mock` executor; use `--executor real` only when the
   operator names a checkpoint.
3. Every real run ends `not_eligible`. Never present it as a release.

This repository is a research runtime; the worked example under
`examples/inkling-2-mini-derisked/` is vendored evidence, not a release.