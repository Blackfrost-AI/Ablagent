"""Independent file-backed trial and evidence records with explicit result states."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from ablagent.artifacts import ArtifactStore, file_digest, load_tensors
from ablagent.banks import BankManifest
from ablagent.canonical import digest_model, read_json, write_json
from ablagent.contracts import Contract, Digest, Identifier
from ablagent.errors import ValidationError
from ablagent.interventions import CompiledArtifact
from ablagent.mock_redlines import (
    FileReference,
    MockRedlineAttestation,
    MockRedlineRaw,
    mock_redline_metrics,
    verify_mock_redlines,
)
from ablagent.synthetic import SyntheticObservation, observation_digest, observe_synthetic

Outcome = Literal["PASS", "FAIL", "MISSING", "NOT_RUN", "ERROR", "NOT_APPLICABLE"]
Category = Literal[
    "structural",
    "serving",
    "calibration",
    "evaluation",
    "redline",
    "redline_observation",
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
    plan_reference: FileReference | None = None
    reason: str = Field(min_length=1, max_length=4000)
    metrics: tuple[MetricResult, ...] = ()
    retained_outputs: tuple[FileReference, ...] = ()

    @model_validator(mode="after")
    def lineage_integrity(self):
        if (self.plan_digest is None) != (self.plan_reference is None):
            raise ValueError("plan_digest and plan_reference must be supplied together")
        if self.status == "promoted" and self.parent_trial_id is None:
            raise ValueError("a promoted trial requires a parent")
        if self.parent_trial_id == self.trial_id:
            raise ValueError("a trial cannot be its own parent")
        return self

    def eligible_for_promotion(self) -> bool:
        return self.status == "completed" and any(
            metric.category == "structural" and metric.outcome == "PASS" for metric in self.metrics
        )

    def reference_path(self) -> str:
        return f"{self.trial_id}.trial.json"

    def reference(self) -> TrialReference:
        return TrialReference(path=self.reference_path(), digest=self.digest())

    def digest(self) -> str:
        return digest_model("trial", self)


class EvidenceVerification(Contract):
    """Process-local receipt; deliberately excluded from the evidence file schema."""

    store_root: str
    evidence_path: str
    evidence_id: Digest
    evidence_digest: Digest
    binding: ArtifactBinding
    bank_digest: Digest


class VerifiedEvidence(Contract):
    evidence: EvidenceBundle
    verification: EvidenceVerification


class EvidenceBundle(Contract):
    schema_version: Literal["dwmip.evidence.v0"] = "dwmip.evidence.v0"
    binding: ArtifactBinding
    observations: tuple[SyntheticObservation, ...] = ()
    metrics: tuple[MetricResult, ...]
    trials: tuple[TrialReference, ...] = Field(min_length=1, max_length=4096)
    environment: dict[str, str]
    decoding: dict[str, str]
    mock_redlines: MockRedlineAttestation | None = None
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


def verify_evidence(
    store: ArtifactStore, relative: str, release_bank: BankManifest | None = None
) -> VerifiedEvidence:
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
        if (
            metric.category == "redline"
            and metric.outcome == "PASS"
            and (
                bundle.mock_redlines is None
                or metric.evidence_digest != bundle.mock_redlines.digest()
            )
        ):
            raise ValidationError("passing redline metric lacks a bound mock-evidence attestation")
    if bundle.mock_redlines is not None:
        if release_bank is None:
            raise ValidationError("mock redline verification requires the selected release bank")
        verify_mock_redlines(store, release_bank, bundle.binding, bundle.mock_redlines)
        expected = mock_redline_metrics(bundle.mock_redlines)
        actual = tuple(metric for metric in bundle.metrics if metric.category == "redline")
        if actual != expected:
            raise ValidationError("redline metrics do not exactly match the verified attestation")
    elif (
        release_bank is not None
        and release_bank.role == "mock_release_redline"
        and any(
            metric.category in {"redline", "redline_observation"} and metric.outcome == "PASS"
            for metric in bundle.metrics
        )
    ):
        raise ValidationError("passing redline metrics require verified mock evidence")
    selected_trials = []
    for reference in bundle.trials:
        try:
            trial = TrialRecord.model_validate(read_json(store.path(reference.path)))
        except PydanticValidationError as exc:
            raise ValidationError(f"invalid retained trial: {exc}") from exc
        if reference.digest != trial.digest() or trial.binding != bundle.binding:
            raise ValidationError("retained trial identity does not match the evidence bundle")
        if trial.status not in {"completed", "promoted"}:
            raise ValidationError("an evidence bundle must retain its selected completed lineage")
        if trial.plan_reference is not None:
            from ablagent.planner import ProposedPlan

            plan_path = store.path(trial.plan_reference.path)
            try:
                plan = ProposedPlan.model_validate(read_json(plan_path))
            except PydanticValidationError as exc:
                raise ValidationError(f"invalid retained plan: {exc}") from exc
            if (
                plan.digest() != trial.plan_digest
                or file_digest(plan_path) != trial.plan_reference.file_digest
                or plan.baseline_digest != compiled.intervention.direction.baseline_digest
                or compiled.intervention.direction.baseline_digest
                != compiled.baseline.tensor_digest
                or plan.direction_digest != compiled.direction_digest
                or compiled.intervention not in plan.interventions
            ):
                raise ValidationError(
                    "retained plan does not match its trial or compiled intervention identity"
                )
        for output in trial.retained_outputs:
            if file_digest(store.path(output.path)) != output.file_digest:
                raise ValidationError("retained trial output is missing or has changed")
        selected_trials.append(trial)
    for index, trial in enumerate(selected_trials):
        expected_parent = selected_trials[index - 1].trial_id if index else None
        if trial.parent_trial_id != expected_parent:
            raise ValidationError(
                "retained trial references are not a complete selected parent chain"
            )
        if index and trial.plan_reference != selected_trials[index - 1].plan_reference:
            raise ValidationError("promoted lineage cannot change its retained plan identity")
        if index == len(selected_trials) - 1 and trial.metrics != bundle.metrics:
            raise ValidationError("selected trial metrics do not match the evidence bundle")
    if any(
        metric.category == "redline_observation" and metric.outcome == "PASS"
        for metric in bundle.metrics
    ):
        raise ValidationError("passing redline observations require a complete mock attestation")
    return VerifiedEvidence(
        evidence=bundle,
        verification=EvidenceVerification(
            evidence_id=digest_model("evidence", bundle),
            store_root=str(store.root.resolve()),
            evidence_path=str(store.path(relative).resolve()),
            evidence_digest=digest_model("evidence", bundle),
            binding=bundle.binding,
            bank_digest=release_bank.digest() if release_bank is not None else "sha256:" + "0" * 64,
        ),
    )


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
        plan_reference=trial.plan_reference,
        reason="Explicit promotion request; eligibility still requires separate gate evaluation.",
        metrics=trial.metrics,
        retained_outputs=trial.retained_outputs,
    )


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


EvidenceBundle.model_rebuild(_types_namespace={"ArtifactBinding": ArtifactBinding})
MockRedlineAttestation.model_rebuild(_types_namespace={"ArtifactBinding": ArtifactBinding})
MockRedlineRaw.model_rebuild(_types_namespace={"ArtifactBinding": ArtifactBinding})
EvidenceVerification.model_rebuild(_types_namespace={"ArtifactBinding": ArtifactBinding})
VerifiedEvidence.model_rebuild()
