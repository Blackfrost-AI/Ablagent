# Refusal removal research on top of Ablagent

Most people arrive here for one reason: a model refuses requests it should answer. The community vocabulary for that work is **false-refusal removal**, **refusal-direction research**, and **abliteration**. Ablagent is not a competing technique — it is the research discipline that work has been missing: explicit spaces, bound directions, retained trials, and red-line evidence that survives contact with reviewers.

## The mapping is already real

The v0 reference operation is the classic rank-one removal, stated neutrally:

```text
W′ = W − αd(dᵀW)
```

`d` is the candidate direction (a refusal direction, in this application), `α = 1` removes its component from the writer, and `α = 0` is the control. That is the same mathematics the refusal-removal literature uses, with the writer, baseline, and normalization made explicit and digest-bound instead of implied.

| Community concept | Ablagent concept |
|---|---|
| Refusal direction | `DirectionCandidate`, promoted to a bound `Direction` |
| Refusal-on / refusal-off prompt pairs | `capture` bank paired observations |
| Ablation at full strength | `α = 1` reference intervention |
| Refusal / harmful-instruction benchmark | `evaluation` bank, held out |
| Safety holdout ("did we break red-lines?") | `release_redline` bank — private, never bundled |
| "It still answers hard prompts" | `redline` metrics gating eligibility |

The six mock red-line categories — biological, chemical, explosives/incendiary, strategic WMD, cyber, loss of control — mirror the domains that refusal research already evaluates.

## What Ablagent adds to that workflow

- **Candidates are not conclusions.** A mean difference between refusal-on and refusal-off activations can carry topic, wording, length, or style confounds. Ablagent keeps the candidate a candidate, with capture/calibration separation and balanced controls.
- **A direction is bound to a writer.** A refusal direction derived in one coordinate space does not authorize edits to a differently-shaped writer just because dimensions match. Source space, destination space, and normalization are explicit and replayable.
- **Negative results stay.** Rejected, failed, and superseded trials remain in the lineage. The field's current norm — publish the one working α, discard the rest — is what Ablagent is designed to prevent.
- **Removal is not release.** Removing a direction says nothing about red-lines or capability. Those are separate evidence categories with separate gates; a passing capability score cannot compensate for a missing red-line measurement.

## What this repo does not claim

Today's runtime proves the contract on synthetic fixtures only. It does not remove refusals from a real checkpoint, does not measure real refusal behavior, and does not certify safety. A real-model adapter — graph declaration, activation capture, behavioral evaluation — is the future extension this structure was built to qualify. When it lands, "abliterated" will be a marketing word; the evidence bundle will be the claim.
