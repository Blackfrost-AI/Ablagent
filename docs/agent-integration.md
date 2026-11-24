# Ablagent for agent integrators

The runtime's planner output is untrusted input. An LLM or other agent may propose an `ExperimentRequest`, but the deterministic planner and schema validator bound that request. Only an explicitly selected `InterventionSpec` may be compiled.

An agent must not:

- Write weights or manifests directly.
- Choose its own policy result or approval.
- Treat a graph schema as executable architecture support.
- Infer compatible spaces from tensor names or dimensions.
- Read private release banks or credentials.
- Enable arbitrary remote code or network publication.
- Present synthetic evidence as model behavioral proof.

Use the CLI as a narrow integration boundary. No provider integration or autonomous executor is included in v0.

## Real executor (opt-in, explicit)

With the `[real]` extra installed, an agent may boot a real session by passing
a HuggingFace URL or repo id:

```bash
ablagent capabilities --adapter hf     # preflight; fails closed without deps
ablagent run --pack derisk --executor real --model <HF URL> --alpha 1.0
```

The same trust boundary applies: the operator names the checkpoint and the
parameters; the runtime pins the revision, captures correlational directions,
compiles one bounded rank-one edit, and exports an HF-loadable copy. Stages the
v0 adapter cannot execute are recorded as skipped — untested, never passed —
and every run ends `not_eligible`. An agent must not present the export as a
release, must not mark skipped stages complete, and must not edit control or
evidence files by hand. See `AGENTS.md` for the full boot contract.
