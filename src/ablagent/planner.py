"""Deterministic planner. Its output is a proposal, never authorization."""

from typing import Literal

from pydantic import Field

from ablagent.canonical import digest_model
from ablagent.contracts import Contract, Digest, Identifier
from ablagent.errors import ValidationError
from ablagent.interventions import BoundDirection, InterventionSpec

MAX_PLAN_STEPS = 8


class ProposedPlan(Contract):
    schema_version: Literal["dwmip.proposed-plan.v0"] = "dwmip.proposed-plan.v0"
    planner_id: Literal["bounded-rules-v0"] = "bounded-rules-v0"
    baseline_digest: Digest
    direction_digest: Digest
    interventions: tuple[InterventionSpec, ...] = Field(min_length=1, max_length=MAX_PLAN_STEPS)
    controls: tuple[Literal["alpha_zero", "unchanged_non_targets"], ...] = (
        "alpha_zero",
        "unchanged_non_targets",
    )
    execution_authority: Literal["none"] = "none"
    approval_authority: Literal["none"] = "none"
    claim_scope: Literal["synthetic_structural_and_diagnostic_only"] = (
        "synthetic_structural_and_diagnostic_only"
    )

    def digest(self) -> str:
        return digest_model("proposed-plan", self)


class ExperimentRequest(Contract):
    request_id: Identifier
    direction: BoundDirection
    alpha_values: tuple[float, ...] = Field(min_length=1, max_length=4)


def propose_experiment(request: ExperimentRequest) -> ProposedPlan:
    """Pure bounded planning: no filesystem, network, executor, or release API."""
    values = request.alpha_values
    if len(set(values)) != len(values):
        raise ValidationError("planner alpha values must be unique")
    if any(value != value or not 0 <= value <= 1 for value in values):
        raise ValidationError("planner alpha values must be finite and between zero and one")
    interventions = tuple(
        InterventionSpec(direction=request.direction, alpha=alpha) for alpha in values
    )
    return ProposedPlan(
        baseline_digest=request.direction.baseline_digest,
        direction_digest=request.direction.direction_digest(),
        interventions=interventions,
    )
