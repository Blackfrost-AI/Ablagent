# Methodology: candidates, interventions, evidence

Domain Weight Modification (DWM) grew out of false-refusal removal and abliteration research. Refusal-direction interventions remain the originating application; DWM extends that intervention approach to other explicitly studied behaviors and directions. The broader name reflects a broader research scope, not a departure from refusal research.

Read [From refusal removal to DWM](refusal-research.md) for the research lineage and terminology.

> **Executable scope:** v0 runs synthetic, CPU-only reference experiments. Real-checkpoint adapters, neural activation capture, and real refusal-behavior evaluation are not implemented; the checkpoint-specific guidance below is a qualification framework for future adapters.

Domain Weight Modification needs more than a vector and a weight tensor. Ablagent separates **discovery**, **compilation**, **behavioral validation**, and **release eligibility**.

## 1. Observe the actual computation

Begin with a checkpoint-specific graph and adapter declaration, not a remembered architecture name. Identify logical sites, storage axes, routing, recurrent states, shared weights, normalization, and the exact point at which an observation is captured. Structural coverage and decode coverage are different evidence.

A state dictionary alone does not encode the complete forward computation. Unknown semantics remain unsupported.

## 2. Derive a candidate, not a conclusion

Mean differences, linear probes, and related statistics produce `DirectionCandidate` records. These are correlational: they may contain topic, wording, length, style, source, or other confounds. A probe can recover information that the model does not use.

Use separate capture and calibration collections, balanced controls, full counts, and bank digests. Preserve held-out evaluation sources and never turn a private release holdout into a tuning signal.

A controlled weight intervention can test a specific causal hypothesis. Its conclusion remains scoped to that checkpoint, intervention, distribution, and evaluation setup; it does not validate a universal mechanism.

## 3. Bind a direction to a writer

A direction needs a source coordinate and an explicit destination output space. Normalize once, persist the unit vector and source-vector digest, and bind the exact baseline, graph, tensor, shape, dtype, layout, writer, bank, and algorithm identities.

Do not copy a residual direction into an attention writer, latent expert, recurrent state, or another model because the dimensions match. Cross-space transfer needs a declared coordinate map, calibrated geometry, and independent reconstruction/intervention validation.

## 4. Make one bounded change

The v0 reference operation is:

```text
W′ = W − αd(dᵀW)
```

`W` is an `out × in` matrix, `d` is unit length in the declared output space, and `α` is between zero and one. `α=0` is a control; `α=1` removes the `d` component. This is not a promise of useful behavioral modification.

Multiple non-orthogonal updates are order-dependent. v0 therefore supports only one coordinate per intervention. It does not silently sum or compose directions.

## 5. Compare base and candidate

Keep decoding, rendering, tokenization, tools, serving versions, and evaluation inputs fixed. Start with an analytic `α=0` control, unchanged non-target tensors, and independent recomputation. Add sham or random-direction controls in behavioral research.

Direct-logit attribution and teacher-forced ranks are localization diagnostics: they omit indirect effects, do not establish sequence quality, and do not prove task success or selective behavior. Use interventions and held-out behavioral evidence for stronger claims.

## 6. Preserve every outcome

Rejected, failed, incomplete, and negative trials remain in the lineage. Promotion appends a record; it does not rewrite history. A structural result never stands in for serving, capability, or red-line evidence.

See `measurement.md` for evidence states and `architecture-navigation.md` for adapter-specific obligations.
