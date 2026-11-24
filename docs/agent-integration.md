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
