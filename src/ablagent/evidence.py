"""Independent file-backed trial and evidence records with explicit result states."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from ablagent.artifacts import ArtifactStore, load_tensors
from ablagent.canonical import digest_model, read_json, write_json
from ablagent.contracts import Contract, Digest, Identifier
from ablagent.errors import ValidationError
from ablagent.interventions import CompiledArtifact
from ablagent.synthetic import SyntheticObservation, observation_digest, observe_synthetic

Outcome = Literal["PASS", "FAIL", "MISSING", "NOT_RUN", "ERROR", "NOT_APPLICABLE"]
Category = Literal[
    "structural",
    "serving",
    "calibration",
    "evaluation",
    "redline",
    "tool_calling",
    "cyber",
    "latent",
    "diagnostic",
]
REQUIRED_REDLINE = frozenset(
    {
        "biological",
        "chemical",
        "explosives_incendiary",
        "strategic_wmd",
        "cyber",
        "loss_of_control",
    }
)


class ArtifactBinding(Contract):
    manifest_path: str
    manifest_digest: Digest
    baseline_file_digest: Digest
    baseline_tensor_digest: Digest
    candidate_file_digest: Digest
    candidate_tensor_digest: Digest
    graph_digest: Digest
    direction_digest: Digest
    intervention_digest: Digest


class MetricResult(Contract):
    category: Category
    name: Identifier
    outcome: Outcome
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    observed: int | None = Field(default=None, ge=0)
    eligible: int | None = Field(default=None, ge=0)
    evidence_digest: Digest
    details: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def complete_observations(self):
        if (self.observed is None) != (self.eligible is None):
            raise ValueError("observed and eligible must be supplied together")
        if self.outcome == "PASS" and (self.eligible is None or self.eligible <= 0):
            raise ValueError("a passing metric requires a positive eligible denominator")
        if self.observed is not None and self.observed > self.eligible:
            raise ValueError("observed cannot exceed eligible")
        if self.outcome == "PASS" and (self.score is None or self.observed != self.eligible):
            raise ValueError("a passing metric requires a score and complete observations")
        if self.outcome != "PASS" and self.score is not None:
            raise ValueError("non-passing outcomes cannot carry a passing score")
        return self


class TrialRecord(Contract):
    schema_version: Literal["dwmip.trial.v0"] = "dwmip.trial.v0"
    trial_id: Identifier
    parent_trial_id: Identifier | None = None
    status: Literal["completed", "rejected", "failed", "promoted"]
    binding: ArtifactBinding
    plan_digest: Digest | None = None
    reason: str = Field(min_length=1, max_length=4000)
    metrics: tuple[MetricResult, ...] = ()
    retained_outputs: tuple[str, ...] = ()

    def eligible_for_promotion(self) -> bool:
        return self.status == "completed" and any(
            metric.category == "structural" and metric.outcome == "PASS" for metric in self.metrics
        )

    def digest(self) -> str:
        return digest_model("trial", self)


class EvidenceBundle(Contract):
    schema_version: Literal["dwmip.evidence.v0"] = "dwmip.evidence.v0"
    binding: ArtifactBinding
    observations: tuple[SyntheticObservation, ...] = ()
    metrics: tuple[MetricResult, ...]
    trials: tuple["TrialReference", ...] = Field(min_length=1, max_length=4096)
    environment: dict[str, str]
    decoding: dict[str, str]
    claim_scope: Literal["synthetic_reference_only"] = "synthetic_reference_only"

    @model_validator(mode="after")
    def identities(self):
        if not self.trials:
            raise ValueError("at least one retained trial is required")
        identities = [(metric.category, metric.name) for metric in self.metrics]
        if len(set(identities)) != len(identities):
            raise ValueError("metric category/name identities must be unique")
        return self


class TrialReference(Contract):
    path: str
    digest: Digest


def bind_compiled(manifest_path: str, compiled: CompiledArtifact) -> ArtifactBinding:
    return ArtifactBinding(
        manifest_path=manifest_path,
        manifest_digest=digest_model("compiled-manifest", compiled),
        baseline_file_digest=compiled.baseline.file_digest,
        baseline_tensor_digest=compiled.baseline.tensor_digest,
        candidate_file_digest=compiled.candidate.file_digest,
        candidate_tensor_digest=compiled.candidate.tensor_digest,
        graph_digest=compiled.candidate.graph_digest,
        direction_digest=compiled.direction_digest,
        intervention_digest=digest_model("intervention", compiled.intervention),
    )


def write_evidence(store: ArtifactStore, bundle: EvidenceBundle, artifact_id: str) -> str:
    if artifact_id.endswith(".json"):
        raise ValidationError("evidence artifact id must omit its extension")
    relative = f"{artifact_id}.evidence.json"
    write_json(store.path(relative), bundle)
    return relative


def verify_evidence(store: ArtifactStore, relative: str) -> EvidenceBundle:
    from pydantic import ValidationError as PydanticValidationError

    try:
        bundle = EvidenceBundle.model_validate(read_json(store.path(relative)))
        compiled = store.verify_compiled(bundle.binding.manifest_path)
    except PydanticValidationError as exc:
        raise ValidationError(f"invalid evidence bundle: {exc}") from exc
    if bind_compiled(bundle.binding.manifest_path, compiled) != bundle.binding:
        raise ValidationError("evidence does not bind to the verified compiled artifact")
    for observation in bundle.observations:
        graph = store.load_graph(f"{compiled.baseline.artifact_id}.graph.json")
        baseline = load_tensors(store.path(compiled.baseline.path))
        candidate = load_tensors(store.path(compiled.candidate.path))
        expected = observe_synthetic(graph, baseline, candidate, observation.observation_id)
        if observation != expected:
            raise ValidationError(
                "observation does not reproduce from the bound baseline and candidate"
            )
    for metric in bundle.metrics:
        if (
            metric.category == "structural"
            and metric.evidence_digest != compiled.structural_check.target_digest_after
        ):
            raise ValidationError("structural metric is not bound to the candidate target tensor")
        if metric.category in {"evaluation", "diagnostic"} and metric.evidence_digest not in {
            observation_digest(item) for item in bundle.observations
        }:
            raise ValidationError("diagnostic metric is not bound to a retained observation")
    for reference in bundle.trials:
        try:
            trial = TrialRecord.model_validate(read_json(store.path(reference.path)))
        except PydanticValidationError as exc:
            raise ValidationError(f"invalid retained trial: {exc}") from exc
        if reference.digest != trial.digest() or trial.binding != bundle.binding:
            raise ValidationError("retained trial identity does not match the evidence bundle")
        if trial.status not in {"completed", "promoted"}:
            raise ValidationError("an evidence bundle must retain its completed trial lineage")
        if trial.metrics != bundle.metrics:
            raise ValidationError("retained trial metrics do not match the evidence bundle")
    return bundle


def record_trial(store: ArtifactStore, trial: TrialRecord) -> str:
    relative = f"{trial.trial_id}.trial.json"
    write_json(store.path(relative), trial)
    return relative


def promote_trial(trial: TrialRecord, new_trial_id: str) -> TrialRecord:
    """Promotion preserves the original trial and appends a new lineage record."""
    if not trial.eligible_for_promotion():
        raise ValidationError(
            "failed, rejected, incomplete, or unaudited trials cannot be promoted"
        )
    return TrialRecord(
        trial_id=new_trial_id,
        parent_trial_id=trial.trial_id,
        status="promoted",
        binding=trial.binding,
        plan_digest=trial.plan_digest,
        reason="Explicit promotion request; eligibility still requires separate gate evaluation.",
        metrics=trial.metrics,
        retained_outputs=trial.retained_outputs,
    )


def timestamp() -> str:
    return datetime.now(UTC).isoformat()
