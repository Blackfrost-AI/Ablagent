"""Direction candidates, exact bindings, and one-coordinate interventions."""

from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from ablagent.banks import BankReference
from ablagent.canonical import digest_model
from ablagent.contracts import Contract, Digest, Identifier
from ablagent.errors import ValidationError
from ablagent.identity import SiteCoordinate

ALGORITHM_ID = "synthetic.contrast.unit.v0"
# Additive widening: real torch-forward captures declare their own provenance
# id. Existing synthetic instances keep their exact string, so every existing
# digest is unchanged.
HF_ALGORITHM_ID = "hf.contrast.unit.v0"
UPDATE_ALGORITHM_ID = "dense.rank_one_output.v0"
NORM_TOLERANCE = 1e-6


class DirectionCandidate(Contract):
    """A statistical contrast. No causal or behavioral claim is implied."""

    schema_version: Literal["dwmip.direction-candidate.v0"] = "dwmip.direction-candidate.v0"
    candidate_id: Identifier
    baseline_digest: Digest
    graph_digest: Digest
    coordinate: SiteCoordinate
    space_id: Identifier
    capture_bank: BankReference
    algorithm_id: Literal[ALGORITHM_ID, HF_ALGORITHM_ID] = ALGORITHM_ID
    statistic: Literal["mean_difference"] = "mean_difference"
    vector: tuple[float, ...] = Field(min_length=1, max_length=1_000_000)
    benign_count: int = Field(ge=4)
    hostile_count: int = Field(ge=4)
    evidence: Literal["correlational"] = "correlational"

    @model_validator(mode="after")
    def finite_vector(self):
        values = np.asarray(self.vector, dtype=np.float64)
        if not np.isfinite(values).all() or np.linalg.norm(values) == 0:
            raise ValueError("candidate vector must be finite and nonzero")
        if (
            self.benign_count != self.hostile_count
            or self.capture_bank.count != self.benign_count * 2
        ):
            raise ValueError("candidate must use the complete balanced capture bank")
        if self.capture_bank.role != "capture":
            raise ValueError("direction candidates require a capture-role bank")
        return self

    def array(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float64)

    def bind(
        self, writer_id: str, tensor_name: str, dtype: str, shape: tuple[int, int]
    ) -> "BoundDirection":
        values = self.array()
        norm = float(np.linalg.norm(values))
        unit = values / norm
        if dtype == "float32" and not np.isclose(
            np.linalg.norm(unit), 1.0, atol=NORM_TOLERANCE, rtol=0
        ):
            raise ValidationError("cannot bind a stable float32 unit direction")
        binding = BoundDirection(
            candidate_id=self.candidate_id,
            baseline_digest=self.baseline_digest,
            graph_digest=self.graph_digest,
            coordinate=self.coordinate,
            space_id=self.space_id,
            capture_bank=self.capture_bank,
            writer_id=writer_id,
            tensor_name=tensor_name,
            dtype=dtype,
            shape=shape,
            source_algorithm_id=self.algorithm_id,
            source_vector_digest=digest_model("direction-candidate-vector", list(self.vector)),
            unit_vector=tuple(float(x) for x in unit),
            unit_norm_tolerance=NORM_TOLERANCE,
        )
        binding.direction_digest()
        return binding


class BoundDirection(Contract):
    schema_version: Literal["dwmip.bound-direction.v0"] = "dwmip.bound-direction.v0"
    candidate_id: Identifier
    baseline_digest: Digest
    graph_digest: Digest
    coordinate: SiteCoordinate
    space_id: Identifier
    capture_bank: BankReference
    writer_id: Identifier
    tensor_name: Identifier
    dtype: Literal["float32", "float64"]
    shape: tuple[int, int]
    source_algorithm_id: Literal[ALGORITHM_ID, HF_ALGORITHM_ID]
    source_vector_digest: Digest
    unit_vector: tuple[float, ...] = Field(min_length=1, max_length=1_000_000)
    unit_norm_tolerance: float = Field(default=NORM_TOLERANCE, gt=0, le=1e-3)

    @model_validator(mode="after")
    def validate_binding(self):
        if len(self.unit_vector) != self.shape[0]:
            raise ValueError("unit direction must have the writer output dimension")
        values = np.asarray(self.unit_vector, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("unit direction must be finite")
        norm = float(np.linalg.norm(values))
        if not np.isclose(norm, 1.0, atol=self.unit_norm_tolerance, rtol=0):
            raise ValueError("direction is not unit-normalized; normalize it explicitly")
        return self

    def array(self) -> np.ndarray:
        return np.asarray(self.unit_vector, dtype=np.float64)

    def identity_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"unit_vector"})

    def direction_digest(self) -> str:
        # Domain separation also prevents a candidate's arbitrary float vector from
        # being interpreted as an already bound normalized direction.
        return digest_model("bound-direction", self)


class InterventionSpec(Contract):
    """Exactly one coordinate; composition semantics are intentionally not implicit."""

    schema_version: Literal["dwmip.intervention.v0"] = "dwmip.intervention.v0"
    operation: Literal["weight_rank_one"] = "weight_rank_one"
    algorithm_id: Literal[UPDATE_ALGORITHM_ID] = UPDATE_ALGORITHM_ID
    direction: BoundDirection
    alpha: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def single_coordinate(self):
        if not isinstance(self.coordinate, SiteCoordinate):
            raise ValueError("one explicit site coordinate is required")
        return self

    @property
    def coordinate(self) -> SiteCoordinate:
        return self.direction.coordinate


class ArtifactReference(Contract):
    artifact_id: Identifier
    role: Literal["baseline", "candidate"]
    path: str = Field(min_length=1, max_length=512)
    codec: Literal["safetensors"] = "safetensors"
    file_digest: Digest
    tensor_digest: Digest
    checkpoint_id: Identifier
    graph_digest: Digest


class StructuralCheck(Contract):
    baseline: ArtifactReference
    candidate: ArtifactReference
    operation: Literal["weight_rank_one"]
    algorithm_id: Literal[UPDATE_ALGORITHM_ID]
    direction_digest: Digest
    alpha: float = Field(ge=0, le=1, allow_inf_nan=False)
    expected_tensor_digests: dict[Identifier, Digest]
    candidate_tensor_digests: dict[Identifier, Digest]
    target_digest_before: Digest
    target_digest_after: Digest
    maximum_deviation: float = Field(ge=0, allow_inf_nan=False)
    tolerance: float = Field(gt=0, allow_inf_nan=False)
    passed: bool


class CompiledArtifact(Contract):
    """Applicability manifest. It is not an efficacy, safety, or release certificate."""

    schema_version: Literal["dwmip.artifact.v0"] = "dwmip.artifact.v0"
    adapter_id: Identifier
    baseline: ArtifactReference
    candidate: ArtifactReference
    intervention: InterventionSpec
    direction_digest: Digest
    structural_check: StructuralCheck
    codec_capabilities: tuple[str, ...] = ("read", "write", "structural_audit")
    efficacy: Literal["unmeasured"] = "unmeasured"
    safety: Literal["unmeasured"] = "unmeasured"
    release_eligibility: Literal["not_evaluated"] = "not_evaluated"

    @model_validator(mode="after")
    def exact_chain(self):
        if self.direction_digest != self.intervention.direction.direction_digest():
            raise ValueError("compiled direction identity mismatch")
        if self.baseline.artifact_id == self.candidate.artifact_id:
            raise ValueError("baseline and candidate must have distinct identities")
        if self.baseline.file_digest == self.candidate.file_digest and self.intervention.alpha != 0:
            raise ValueError("a nonzero edit must produce a different candidate file")
        return self
