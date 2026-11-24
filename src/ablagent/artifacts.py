"""A small, content-addressed safetensors checkpoint store with safe loading."""

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from pydantic import ValidationError as PydanticValidationError
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from ablagent.canonical import digest_bytes, digest_model, read_json, write_json
from ablagent.errors import ValidationError
from ablagent.identity import GraphSpec, WriterSpec
from ablagent.interventions import (
    ArtifactReference,
    BoundDirection,
    CompiledArtifact,
    DirectionCandidate,
    InterventionSpec,
    StructuralCheck,
)

_SUPPORTED_DTYPES = {"float32", "float64"}
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_HEADER_BYTES = 1024 * 1024
_MAX_FILE_BYTES = 1024 * 1024 * 1024


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValidationError(f"cannot hash artifact {path}: {exc}") from exc
    return "sha256:" + digest.hexdigest()


def _tensor_digest(tensors: dict[str, np.ndarray]) -> str:
    payloads = []
    for name in sorted(tensors):
        value = tensors[name]
        contiguous = np.ascontiguousarray(value)
        payloads.append(
            {
                "name": name,
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "data_sha256": "sha256:"
                + hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
            }
        )
    return digest_model("checkpoint-tensors", payloads)


def _check_safetensors_file(path: Path, max_bytes: int) -> None:
    if path.stat().st_size > max_bytes:
        raise ValidationError(f"tensor file exceeds the {max_bytes}-byte resource limit")
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValidationError("truncated safetensors header")
        header_length = int.from_bytes(raw_length, "little")
        if header_length > _MAX_HEADER_BYTES:
            raise ValidationError("safetensors header exceeds the resource limit")
        if header_length + 8 > path.stat().st_size:
            raise ValidationError("safetensors header length exceeds the file size")


def load_tensors(path: Path, max_bytes: int = _MAX_FILE_BYTES) -> dict[str, np.ndarray]:
    try:
        _check_safetensors_file(path, max_bytes)
        tensors = load_file(str(path))
        for name, tensor in tensors.items():
            if str(tensor.dtype) not in _SUPPORTED_DTYPES:
                raise ValidationError(f"unsupported tensor dtype: {name}: {tensor.dtype}")
            if not np.isfinite(tensor).all():
                raise ValidationError(f"non-finite tensor value: {name}")
        return dict(tensors)
    except ValidationError:
        raise
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise ValidationError(f"invalid safetensors input {path}: {exc}") from exc


def validate_graph_tensors(graph: GraphSpec, tensors: dict[str, np.ndarray]) -> None:
    if set(tensors) != {writer.tensor_name for writer in graph.writers}:
        raise ValidationError("checkpoint tensors must exactly match the declared graph writers")
    for writer in graph.writers:
        tensor = tensors[writer.tensor_name]
        if tensor.shape != writer.shape or str(tensor.dtype) != writer.dtype:
            raise ValidationError(f"tensor identity mismatch: {writer.tensor_name}")
        if tensor.ndim != 2:
            raise ValidationError(f"writer must be a dense matrix: {writer.tensor_name}")


class ArtifactStore:
    """File artifacts are authoritative; no database is needed for verification."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def path(self, relative: str) -> Path:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValidationError(f"artifact path escapes the run: {relative}")
        return self.root.joinpath(*pure.parts)

    def _unique(self, stem: str, suffix: str) -> Path:
        if not _SAFE_NAME.fullmatch(stem):
            raise ValidationError("unsafe artifact filename")
        path = self.root / f"{stem}{suffix}"
        if path.exists():
            raise ValidationError(f"refusing to overwrite immutable artifact: {path.name}")
        return path

    def write_graph(self, graph: GraphSpec, artifact_id: str) -> str:
        relative = f"{artifact_id}.graph.json"
        write_json(self._unique(artifact_id, ".graph.json"), graph)
        return relative

    def load_graph(self, relative: str) -> GraphSpec:
        return _parse(GraphSpec, read_json(self.path(relative)))

    def write_direction(
        self, direction: DirectionCandidate | BoundDirection, artifact_id: str
    ) -> str:
        write_json(self._unique(artifact_id, ".direction.json"), direction)
        return f"{artifact_id}.direction.json"

    def load_direction(self, relative: str) -> DirectionCandidate | BoundDirection:
        raw = read_json(self.path(relative))
        if not isinstance(raw, dict):
            raise ValidationError("direction must be a JSON object")
        if raw.get("schema_version") == "dwmip.bound-direction.v0":
            return _parse(BoundDirection, raw)
        if raw.get("schema_version") == "dwmip.direction-candidate.v0":
            return _parse(DirectionCandidate, raw)
        raise ValidationError("unknown direction schema")

    def write_bank(self, bank: Any, artifact_id: str) -> str:
        write_json(self._unique(artifact_id, ".bank.json"), bank)
        return f"{artifact_id}.bank.json"

    def save_checkpoint(
        self, tensors: dict[str, np.ndarray], graph: GraphSpec, artifact_id: str
    ) -> ArtifactReference:
        validate_graph_tensors(graph, tensors)
        path = self._unique(artifact_id, ".safetensors")
        metadata = {
            "dwmip_version": "0",
            "checkpoint_id": graph.checkpoint_id,
            "graph_digest": graph.digest(),
            "artifact_id": artifact_id,
            "dtype_contract": "float32-or-float64",
        }
        try:
            save_file(tensors, str(path), metadata=metadata)
            loaded = load_tensors(path)
        except Exception as exc:
            path.unlink(missing_ok=True)
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(f"could not write safetensors checkpoint: {exc}") from exc
        if _tensor_digest(loaded) != _tensor_digest(tensors):
            path.unlink(missing_ok=True)
            raise ValidationError("checkpoint readback did not preserve tensors")
        return self.reference(path.name, "baseline", graph, loaded)

    def reference(
        self,
        relative: str,
        role: str,
        graph: GraphSpec,
        tensors: dict[str, np.ndarray] | None = None,
    ) -> ArtifactReference:
        path = self.path(relative)
        loaded = load_tensors(path) if tensors is None else tensors
        validate_graph_tensors(graph, loaded)
        return ArtifactReference(
            artifact_id=path.name.removesuffix(".safetensors"),
            role=role,
            path=relative,
            file_digest=file_digest(path),
            tensor_digest=_tensor_digest(loaded),
            checkpoint_id=graph.checkpoint_id,
            graph_digest=graph.digest(),
        )

    def derive_candidate(
        self,
        candidate: DirectionCandidate,
        graph: GraphSpec,
        tensors: dict[str, np.ndarray],
        writer_id: str,
        artifact_id: str,
    ) -> BoundDirection:
        writer = _find_writer(graph, writer_id)
        baseline = ArtifactReference(
            artifact_id="in-memory-baseline",
            role="baseline",
            path="in-memory",
            file_digest=digest_bytes("in-memory", _tensor_digest(tensors).encode()),
            tensor_digest=_tensor_digest(tensors),
            checkpoint_id=graph.checkpoint_id,
            graph_digest=graph.digest(),
        )
        if (
            candidate.baseline_digest != baseline.tensor_digest
            or candidate.graph_digest != graph.digest()
        ):
            raise ValidationError("candidate is not bound to this baseline and graph")
        coordinate_sites = [site for site in graph.sites if site.coordinate == candidate.coordinate]
        if len(coordinate_sites) != 1 or coordinate_sites[0].space_id != candidate.space_id:
            raise ValidationError("candidate capture coordinate/space is not declared by the graph")
        if candidate.space_id != writer.output_space_id:
            raise ValidationError(
                "candidate capture space is not the writer's declared output space"
            )
        bound = candidate.bind(writer.writer_id, writer.tensor_name, writer.dtype, writer.shape)
        self.write_direction(bound, artifact_id)
        return bound

    def compile(
        self,
        graph: GraphSpec,
        baseline: ArtifactReference,
        intervention: InterventionSpec,
        artifact_id: str,
        direction_path: str | None = None,
    ) -> CompiledArtifact:
        _validate_execution(graph)
        if direction_path is not None:
            persisted = self.load_direction(direction_path)
            if not isinstance(persisted, BoundDirection) or persisted != intervention.direction:
                raise ValidationError(
                    "intervention does not match the persisted bound direction, "
                    "including capture-bank identity"
                )
        baseline_path = self.path(baseline.path)
        observed_baseline = self.reference(baseline.path, "baseline", graph)
        if observed_baseline != baseline:
            raise ValidationError("baseline identity, content, or graph has changed")
        tensors = load_tensors(baseline_path)
        direction = intervention.direction
        if baseline.role != "baseline" or baseline.artifact_id != baseline_path.name.removesuffix(
            ".safetensors"
        ):
            raise ValidationError("baseline role and artifact identity do not match its path")
        if direction.baseline_digest != baseline.tensor_digest:
            raise ValidationError("direction baseline digest mismatch")
        if direction.graph_digest != graph.digest():
            raise ValidationError("direction graph digest mismatch")
        writer = _find_writer(graph, direction.writer_id)
        _validate_writer_binding(writer, direction)
        original = tensors[writer.tensor_name]
        unit = direction.array()
        # The single supported update is W' = W - alpha*d*(d^T W).
        projection = unit @ original.astype(np.float64)
        delta = intervention.alpha * np.outer(unit, projection)
        edited = original.astype(np.float64) - delta
        if not np.isfinite(edited).all():
            raise ValidationError("edit produced non-finite tensor values")
        result = edited.astype(original.dtype)
        updated = dict(tensors)
        updated[writer.tensor_name] = result
        candidate_path = self._unique(artifact_id, ".safetensors")
        metadata = {
            "dwmip_version": "0",
            "checkpoint_id": graph.checkpoint_id,
            "graph_digest": graph.digest(),
            "artifact_id": artifact_id,
            "baseline_tensor_digest": baseline.tensor_digest,
            "baseline_file_digest": baseline.file_digest,
            "direction_digest": direction.direction_digest(),
            "intervention_digest": digest_model("intervention", intervention),
            "algorithm_id": intervention.algorithm_id,
        }
        try:
            save_file(updated, str(candidate_path), metadata=metadata)
            candidate_tensors = load_tensors(candidate_path)
        except Exception as exc:
            candidate_path.unlink(missing_ok=True)
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(f"could not write candidate checkpoint: {exc}") from exc
        observed_digests = _per_tensor_digests(candidate_tensors)
        # Expected identity is computed from original tensors, never candidate metadata.
        expected_digests = _per_tensor_digests({**tensors, writer.tensor_name: result})
        maximum_deviation = (
            float(np.max(np.abs(result.astype(np.float64) - edited))) if result.size else 0.0
        )
        tolerance = (
            float(np.finfo(original.dtype).eps) * max(1.0, float(np.max(np.abs(original)))) * 4
        )
        unchanged = all(
            observed_digests[name] == _one_tensor_digest(value)
            for name, value in tensors.items()
            if name != writer.tensor_name
        )
        passed = (
            observed_digests == expected_digests
            and unchanged
            and maximum_deviation <= tolerance
            and candidate_path.name != baseline_path.name
        )
        if not passed:
            candidate_path.unlink(missing_ok=True)
            raise ValidationError(
                "candidate structural or numerical audit failed; artifact removed"
            )
        candidate = self.reference(candidate_path.name, "candidate", graph, candidate_tensors)
        manifest_relative = f"{artifact_id}.manifest.json"
        if self.path(manifest_relative).exists():
            candidate_path.unlink(missing_ok=True)
            raise ValidationError("refusing to overwrite an existing compiled manifest")
        expected_metadata = {
            "dwmip_version": "0",
            "checkpoint_id": graph.checkpoint_id,
            "graph_digest": graph.digest(),
            "artifact_id": artifact_id,
            "baseline_tensor_digest": baseline.tensor_digest,
            "baseline_file_digest": baseline.file_digest,
            "direction_digest": direction.direction_digest(),
            "intervention_digest": digest_model("intervention", intervention),
            "algorithm_id": intervention.algorithm_id,
        }
        if tensor_metadata(candidate_path) != expected_metadata:
            candidate_path.unlink(missing_ok=True)
            raise ValidationError(
                "candidate metadata did not preserve the exact transformation chain"
            )
        check = _structural_check(
            graph=graph,
            baseline=baseline,
            candidate=candidate,
            intervention=intervention,
            original_tensors=tensors,
            candidate_tensors=candidate_tensors,
            original=original,
            edited=edited,
            expected=result,
        )
        compiled = CompiledArtifact(
            adapter_id=graph.adapter_id,
            baseline=baseline,
            candidate=candidate,
            intervention=intervention,
            direction_digest=direction.direction_digest(),
            structural_check=check,
        )
        write_json(self.path(manifest_relative), compiled)
        return compiled

    def verify_compiled(self, relative: str) -> CompiledArtifact:
        compiled = _parse(CompiledArtifact, read_json(self.path(relative)))
        graph = self.load_graph(f"{compiled.baseline.artifact_id}.graph.json")
        if graph.adapter_id != compiled.adapter_id:
            raise ValidationError("compiled adapter identity mismatch")
        if graph.digest() != compiled.baseline.graph_digest:
            raise ValidationError("compiled graph identity mismatch")
        _validate_execution(graph)
        for reference in (compiled.baseline, compiled.candidate):
            if self.reference(reference.path, reference.role, graph) != reference:
                raise ValidationError(f"{reference.role} artifact identity mismatch")
        # Independently consume candidate tensors in addition to verifying its file hash.
        base = load_tensors(self.path(compiled.baseline.path))
        candidate_tensors = load_tensors(self.path(compiled.candidate.path))
        expected_metadata = {
            "dwmip_version": "0",
            "checkpoint_id": graph.checkpoint_id,
            "graph_digest": graph.digest(),
            "artifact_id": compiled.candidate.artifact_id,
            "baseline_tensor_digest": compiled.baseline.tensor_digest,
            "baseline_file_digest": compiled.baseline.file_digest,
            "direction_digest": compiled.direction_digest,
            "intervention_digest": digest_model("intervention", compiled.intervention),
            "algorithm_id": compiled.intervention.algorithm_id,
        }
        if tensor_metadata(self.path(compiled.candidate.path)) != expected_metadata:
            raise ValidationError("candidate metadata does not match the manifest")
        direction = compiled.intervention.direction
        writer = _find_writer(graph, direction.writer_id)
        _validate_writer_binding(writer, direction)
        if direction.baseline_digest != compiled.baseline.tensor_digest:
            raise ValidationError("compiled direction baseline mismatch")
        if direction.graph_digest != graph.digest():
            raise ValidationError("compiled direction graph mismatch")
        original = base[writer.tensor_name]
        vector = np.array(direction.unit_vector, dtype=np.float64)
        expected = original.astype(np.float64) - compiled.intervention.alpha * np.outer(
            vector, vector @ original.astype(np.float64)
        )
        observed = candidate_tensors[writer.tensor_name]
        expected_check = _structural_check(
            graph=graph,
            baseline=compiled.baseline,
            candidate=compiled.candidate,
            intervention=compiled.intervention,
            original_tensors=base,
            candidate_tensors=candidate_tensors,
            original=original,
            edited=observed,
            expected=expected,
        )
        if compiled.structural_check != expected_check:
            raise ValidationError("structural audit does not match independent recomputation")
        return compiled


def tensor_metadata(path: Path) -> dict[str, str]:
    """Read the small UTF-8 metadata header without materializing tensors."""
    try:
        _check_safetensors_file(path, _MAX_FILE_BYTES)
        with safe_open(str(path), framework="numpy") as handle:
            return handle.metadata() or {}
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise ValidationError(f"invalid safetensors metadata {path}: {exc}") from exc


def _parse(model: Any, raw: Any) -> Any:
    try:
        return model.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValidationError(f"schema validation failed: {exc}") from exc


def _find_writer(graph: GraphSpec, writer_id: str) -> WriterSpec:
    try:
        return graph.writer(writer_id)
    except KeyError as exc:
        raise ValidationError(f"unknown writer: {writer_id}") from exc


def _validate_execution(graph: GraphSpec) -> None:
    if graph.adapter_id != "synthetic-cpu-v0":
        raise ValidationError("only the declared synthetic-cpu-v0 adapter can execute in v0")
    if graph.topology != "dense_linear" or not graph.execution_supported:
        raise ValidationError("graph is descriptive but not executable by the synthetic adapter")
    if any(
        coordinate.recurrent_step is not None or coordinate.loop_iteration is not None
        for coordinate in [
            *(site.coordinate for site in graph.sites),
            *(writer.coordinate for writer in graph.writers),
        ]
    ) or any(writer.shared_group is not None for writer in graph.writers):
        raise ValidationError(
            "recurrent, looped, or shared execution is unsupported",
            code="unsupported_execution_coordinate",
        )
    sites = [site.coordinate for site in graph.sites]
    for writer in graph.writers:
        if writer.shared_group is not None or writer.operations != ("rank_one_output",):
            raise ValidationError("shared or unsupported writers cannot be edited by this adapter")
        if sites.count(writer.coordinate) != 1:
            raise ValidationError("writer coordinate must map to exactly one declared site")
    if any(sum(w.coordinate == site for w in graph.writers) != 1 for site in sites):
        raise ValidationError("synthetic execution requires one writer per site")


def _validate_writer_binding(writer: WriterSpec, direction: BoundDirection) -> None:
    actual = (writer.tensor_name, writer.dtype, writer.shape, writer.output_space_id)
    expected = (direction.tensor_name, direction.dtype, direction.shape, direction.space_id)
    if actual != expected:
        raise ValidationError("direction writer, tensor, dtype, shape, or space binding mismatch")


def _structural_check(
    graph: GraphSpec,
    baseline: ArtifactReference,
    candidate: ArtifactReference,
    intervention: InterventionSpec,
    original_tensors: dict[str, np.ndarray],
    candidate_tensors: dict[str, np.ndarray],
    original: np.ndarray,
    edited: np.ndarray,
    expected: np.ndarray,
) -> StructuralCheck:
    expected_tensors = {**original_tensors, intervention.direction.tensor_name: expected}
    maximum_deviation = (
        float(np.max(np.abs(expected.astype(np.float64) - edited))) if expected.size else 0.0
    )
    tolerance = float(np.finfo(original.dtype).eps) * max(1.0, float(np.max(np.abs(original)))) * 4
    expected_digests = _per_tensor_digests(expected_tensors)
    observed_digests = _per_tensor_digests(candidate_tensors)
    unchanged = all(
        observed_digests.get(name) == _one_tensor_digest(value)
        for name, value in original_tensors.items()
        if name != intervention.direction.tensor_name
    )
    passed = (
        observed_digests == expected_digests
        and unchanged
        and maximum_deviation <= tolerance
        and candidate.path != baseline.path
    )
    if not passed:
        raise ValidationError("structural audit failed independent tensor comparison")
    return StructuralCheck(
        baseline=baseline,
        candidate=candidate,
        operation=intervention.operation,
        algorithm_id=intervention.algorithm_id,
        direction_digest=intervention.direction.direction_digest(),
        alpha=intervention.alpha,
        expected_tensor_digests=expected_digests,
        candidate_tensor_digests=observed_digests,
        target_digest_before=_one_tensor_digest(original),
        target_digest_after=_one_tensor_digest(expected),
        maximum_deviation=maximum_deviation,
        tolerance=tolerance,
        passed=passed,
    )


def _one_tensor_digest(tensor: np.ndarray) -> str:
    return digest_model(
        "single-tensor",
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "data_sha256": "sha256:"
            + hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest(),
        },
    )


def _per_tensor_digests(tensors: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: _one_tensor_digest(tensor) for name, tensor in tensors.items()}
