# From refusal removal to Domain Weight Modification

**Ablagent grows out of false-refusal removal and abliteration research. Domain Weight Modification (DWM) is the broader research framework that emerged from it.** Refusal removal is the originating application, not a side example; the purpose of DWM is to carry the underlying intervention approach into a wider, more rigorous study of learned behavior.

## The research lineage

Researchers investigating unwanted refusals discovered an important question: when a model refuses a request that it could answer, can the relevant behavior be represented and modified as a direction in the model's computation? That led to work on **refusal directions**, **directional ablation**, and **abliteration**—modifying weights so that a particular component of the representation no longer drives the selected writer.

That research motivates more general questions:

- If a behavioral direction can be isolated for refusal, what other behaviors can be represented directionally?
- When does a direction reflect a behavior rather than a correlated feature of its capture data?
- How does an intervention change when it crosses a projection, normalization boundary, routed expert, recurrent state, or shared weight?
- Can a direction move between architectures, and what evidence would establish that transfer?
- How can a targeted change be evaluated without hiding collateral capability or red-line effects?

**DWM is the framework for investigating those questions.** It extends beyond refusal removal, but it does not discard or rebrand its origins. The objective is a general discipline for representing, testing, and retaining knowledge about targeted weight-space interventions.

## Where the mathematics connects

The v0 reference operation states the familiar rank-one directional removal explicitly:

```text
W′ = W − αd(dᵀW)
```

Here `d` is a unit direction. At `α = 1`, the transformation removes its pre-normalization component from the declared writer; `α = 0` is the control. For refusal research, that direction may be a candidate refusal direction. For broader DWM research, it may represent another explicitly studied feature or behavior.

The mathematics alone does not establish that a direction is a refusal mechanism, that its intervention generalizes, or that a modification is desirable. Ablagent's contribution is to make the surrounding research claims explicit and inspectable.

| Refusal-research concept | DWM / Ablagent representation |
|---|---|
| Candidate refusal direction | `DirectionCandidate`, with activation source and preprocessing declared |
| Validated direction for a specific baseline | `Direction`, bound to space, baseline, normalization, and writer |
| Refusal-on / refusal-off observations | A `capture` bank with paired, controlled capture records |
| Full-strength directional removal | `α = 1` in the v0 reference intervention |
| Refusal-behavior benchmark | A separate `evaluation` bank with a behavioral measurement contract |
| Safety or red-line holdout | A separate `release_redline` bank with its own evidence |
| "What else did the edit change?" | Baseline-versus-candidate comparisons and retained collateral measurements |

A rank-one direction is a hypothesis to test, not a behavioral conclusion. An activation-space direction does not automatically transfer to another writer, coordinate space, or architecture.

## Why DWM requires more than abliteration

A small dense linear layer provides a clean reference case. Real model research immediately raises questions that a filename-based weight-editing script cannot answer:

- **Representation:** Was the direction captured in residual, normalized, projected, or another space?
- **Writer semantics:** Does this parameter write into that space, transform it, or only consume it?
- **Architecture:** Do routing, recurrence, shared weights, fused parameters, or quantization change the intervention?
- **Evidence:** Are capture, evaluation, red-lines, and capability separated, or is one benchmark being asked to prove everything?
- **Reproducibility:** Can another researcher replay the selected intervention and inspect rejected trials?

DWMIP v0 makes those relationships into explicit artifacts: graph and writer identities, spaces and axes, banks and observations, bound directions, selected interventions, trial lineage, and evidence bundles. This is how the project extends refusal research into architecture-aware DWM rather than treating every parameter as an interchangeable matrix.

## What better refusal research looks like

Ablagent is designed to support a research workflow in which:

1. **A direction remains a candidate until its evidence supports promotion.** A mean difference between refusal-on and refusal-off activations can also encode topic, wording, response length, or style. Capture and calibration need separate, controlled observations.
2. **The baseline and writer are part of the claim.** A direction is not declared portable merely because another tensor has a compatible dimension.
3. **Negative results remain available.** Rejected, failed, and superseded trials are part of the research record, rather than disappearing behind the one successful parameter choice.
4. **Refusal behavior, capability, and red-lines are measured separately.** A capability improvement cannot compensate for a missing red-line measurement, and an intervention metric cannot establish behavioral change.
5. **Architecture transfer requires new evidence.** A direction or scale from one checkpoint is not a recipe for another architecture.

The repository's six mock red-line categories—biological, chemical, explosives/incendiary, strategic WMD, cyber, and loss of control—exercise the structure of this separate evidence requirement. They are not measurements of real model behavior.

## Current scope and the next research layer

The executable reference runtime is a synthetic, CPU-only implementation of one rank-one intervention on one declared dense writer. It demonstrates the protocol, artifact identities, replay checks, and fail-closed policy behavior.

**It does not currently remove refusals from a real checkpoint, capture neural activations, evaluate real refusal behavior, or establish that an intervention is safe or effective.** Those capabilities require separately qualified real-model adapters, capture pipelines, behavioral evaluation, and serving verification.

The research direction is to carry the lessons of refusal removal into that broader DWM workflow: explicit hypotheses, architecture-specific evidence, reproducible experiments, and honest negative results. The name is broader because the research is broader; the foundation remains refusal removal.

Continue with [methodology](methodology.md), [architecture navigation](architecture-navigation.md), [bank design](banks.md), and [measurement contracts](measurement.md).
