"""Analytic synthetic backend. It is not a neural-network or real-checkpoint adapter."""

import hashlib
from typing import Literal

import numpy as np

from ablagent.artifacts import _tensor_digest
from ablagent.banks import BankManifest, BankRecord, LicenseTerms
from ablagent.canonical import digest_model
from ablagent.contracts import Contract
from ablagent.errors import ValidationError
from ablagent.identity import (
    AdapterCapabilities,
    GraphSpec,
    SiteCoordinate,
    SiteSpec,
    SpaceSpec,
    WriterSpec,
)
from ablagent.interventions import DirectionCandidate


class SyntheticObservation(Contract):
    """Reference output before and after a pure matrix transformation."""

    schema_version: Literal["dwmip.synthetic-observation.v0"] = "dwmip.synthetic-observation.v0"
    observation_id: str
    graph_digest: str
    baseline_tensor_digest: str
    candidate_tensor_digest: str
    coordinate: SiteCoordinate
    writer_id: str
    input_vector: tuple[float, ...]
    baseline_output: tuple[float, ...]
    candidate_output: tuple[float, ...]
    behavior_class: Literal["diagnostic_only"] = "diagnostic_only"


def reference_graph(dtype: str = "float64") -> GraphSpec:
    if dtype not in {"float32", "float64"}:
        raise ValidationError("synthetic reference dtype must be float32 or float64")
    coordinate = SiteCoordinate(site_id="reference-layer-0", logical_layer=0)
    output = SpaceSpec(
        space_id="reference-output",
        kind="synthetic_output",
        dimension=3,
        basis="Declared synthetic matrix output coordinates; no cross-space transfer.",
        normalization=(
            "No implicit normalization; the bank declaration is part of the capture contract."
        ),
    )
    input_space = SpaceSpec(
        space_id="reference-input",
        kind="synthetic_input",
        dimension=2,
        basis="Declared synthetic matrix input coordinates.",
        normalization="Unit input probes are explicit inputs, not language activations.",
    )
    return GraphSpec(
        adapter_id="synthetic-cpu-v0",
        checkpoint_id="synthetic-reference-v0",
        topology="dense_linear",
        execution_supported=True,
        spaces=(output, input_space),
        sites=(
            SiteSpec(
                coordinate=coordinate,
                space_id=output.space_id,
                capture_timing=(
                    "Immediately after the declared matrix operation, before any other operation."
                ),
            ),
        ),
        writers=(
            WriterSpec(
                writer_id="reference-writer-0",
                tensor_name="writer0",
                coordinate=coordinate,
                output_space_id=output.space_id,
                input_space_id=input_space.space_id,
                dtype=dtype,
                shape=(3, 2),
                operations=("rank_one_output",),
            ),
        ),
    )


def reference_tensors(dtype: str = "float64") -> dict[str, np.ndarray]:
    reference_graph(dtype)
    return {"writer0": np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=dtype)}


def synthetic_capture_bank() -> BankManifest:
    records = []
    for index, hostile in enumerate(([0, 0], [2, 0], [2, 1], [4, 0]), start=1):
        for group, values in (("benign", [0, 1]), ("hostile", hostile)):
            import json

            records.append(
                BankRecord(
                    record_id=f"capture-{index}-{group}",
                    prompt=json.dumps({"x": values}),
                    group=group,
                    domain="synthetic",
                    pair_id=f"pair-{index}",
                )
            )
    return BankManifest(
        bank_id="synthetic-capture",
        version="v0",
        role="capture",
        visibility="public",
        records=records,
        license=LicenseTerms(
            spdx_id="CC-BY-4.0",
            copyright="2026 Ablagent contributors",
            source="Original deterministic synthetic numeric records.",
            license_text="banks/LICENSE-CC-BY-4.0",
            redistribution_allowed=True,
            transformation_lineage=(
                "Generated in this package; no third-party records or model outputs."
            ),
            generator_terms="No external generator or terms apply.",
        ),
        split_policy="Public synthetic capture fixture; never a release holdout.",
        notes="Balanced benign/hostile numeric probes; not language data.",
    )


def capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        adapter_id="synthetic-cpu-v0",
        executable_topologies=("dense_linear",),
        supported_operations=("rank_one_output",),
        supported_codecs=("safetensors",),
    )


def derive_contrast(
    graph: GraphSpec, tensors: dict[str, np.ndarray], bank: BankManifest, candidate_id: str
) -> DirectionCandidate:
    """Derive a contrast from paired, numeric synthetic bank prompts.

    Each record's `prompt` is a JSON object `{"x": [x0, x1]}`. This intentionally
    narrow fixture avoids pretending that text parsing or language behavior exists.
    The benign and hostile mean-input contrast is transformed by the declared writer;
    it remains correlational even after subsequent controlled interventions.
    """
    import json

    if bank.role != "capture":
        raise ValidationError("synthetic direction derivation requires a capture bank")
    writer = graph.writers[0]
    matrix = tensors[writer.tensor_name]
    if graph.digest() == "" or matrix.shape != (3, 2):  # pragma: no cover - defensive guard
        raise ValidationError("unknown synthetic geometry")
    coordinates: dict[str, dict[str, np.ndarray]] = {}
    for record in bank.records:
        try:
            raw = json.loads(record.prompt)
            if not isinstance(raw, dict) or set(raw) != {"x"}:
                raise ValueError("expected only an x field")
            vector = np.asarray(raw["x"], dtype=np.float64)
            if vector.shape != (2,) or not np.isfinite(vector).all():
                raise ValueError("x must contain two finite numbers")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValidationError(
                f"invalid synthetic bank record {record.record_id}: {exc}"
            ) from exc
        if record.group not in {"benign", "hostile"}:
            raise ValidationError("capture records must be benign or hostile")
        coordinates.setdefault(record.pair_id, {})[record.group] = vector
    if any(set(pair) != {"benign", "hostile"} for pair in coordinates.values()):
        raise ValidationError("incomplete paired synthetic observations")
    benign = np.stack([pair["benign"] for pair in coordinates.values()])
    hostile = np.stack([pair["hostile"] for pair in coordinates.values()])
    contrast = (hostile.mean(axis=0) - benign.mean(axis=0)) @ matrix.T
    return DirectionCandidate(
        candidate_id=candidate_id,
        baseline_digest=_tensor_digest(tensors),
        graph_digest=graph.digest(),
        coordinate=writer.coordinate,
        space_id=writer.output_space_id,
        capture_bank=bank.reference(),
        vector=tuple(float(value) for value in contrast),
        benign_count=len(benign),
        hostile_count=len(hostile),
    )


def observe_synthetic(
    graph: GraphSpec,
    baseline_tensors: dict[str, np.ndarray],
    candidate_tensors: dict[str, np.ndarray],
    observation_id: str,
) -> SyntheticObservation:
    writer = graph.writers[0]
    baseline = baseline_tensors[writer.tensor_name]
    candidate = candidate_tensors[writer.tensor_name]
    inputs = np.zeros(baseline.shape[1], dtype=np.float64)
    inputs[0] = 1.0
    before = baseline.astype(np.float64) @ inputs
    after = candidate.astype(np.float64) @ inputs
    return SyntheticObservation(
        observation_id=observation_id,
        graph_digest=graph.digest(),
        baseline_tensor_digest=_tensor_digest(baseline_tensors),
        candidate_tensor_digest=_tensor_digest(candidate_tensors),
        coordinate=writer.coordinate,
        writer_id=writer.writer_id,
        input_vector=tuple(float(x) for x in inputs),
        baseline_output=tuple(float(x) for x in before),
        candidate_output=tuple(float(x) for x in after),
    )


def observation_digest(observation: SyntheticObservation) -> str:
    return digest_model("synthetic-observation", observation)


def vector_digest(vector: tuple[float, ...]) -> str:
    return "sha256:" + hashlib.sha256(repr(vector).encode("utf-8")).hexdigest()
