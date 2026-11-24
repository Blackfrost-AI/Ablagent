"""Executable, file-driven conformance for the synthetic DWMIP reference adapter."""

from pathlib import Path

import numpy as np
from pydantic import ValidationError as PydanticValidationError

from ablagent.artifacts import ArtifactStore
from ablagent.canonical import read_json, write_json
from ablagent.errors import AblagentError
from ablagent.identity import GraphSpec
from ablagent.interventions import InterventionSpec
from ablagent.synthetic import (
    derive_contrast,
    reference_graph,
    reference_tensors,
    synthetic_capture_bank,
)


def _load_cases(fixtures: Path, name: str) -> list[dict]:
    document = read_json(fixtures / f"{name}.json", max_bytes=1024 * 1024)
    expected_schema = f"dwmip.conformance.{name}.v0"
    if not isinstance(document, dict) or document.get("schema_version") != expected_schema:
        raise ValueError(f"{name} conformance document has an invalid schema version")
    if not isinstance(document.get("cases"), list):
        raise ValueError(f"{name} conformance document must contain a cases list")
    cases = document["cases"]
    if not cases or any(not isinstance(case, dict) for case in cases):
        raise ValueError(f"{name} conformance cases must be objects")
    for case in cases:
        if case.get("outcome") not in {"accept", "reject"}:
            raise ValueError("every conformance case requires an accept or reject outcome")
        if not isinstance(case.get("case_id"), str) or not case["case_id"]:
            raise ValueError("every conformance case requires a nonempty case_id")
    return cases


def _numeric_array(value: list) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("conformance vectors must be finite")
    return array


def _matrix_case(case: dict) -> None:
    baseline = _numeric_array(case["baseline"])
    expected_direction = _numeric_array(case["expected_unit_vector"])
    if baseline.ndim != 2 or expected_direction.shape != (baseline.shape[0],):
        raise ValueError("invalid synthetic conformance geometry")
    norm = np.linalg.norm(expected_direction)
    if abs(norm - 1) > 1e-12:
        raise ValueError("expected direction is not unit normalized")
    direction = expected_direction / norm
    observed = baseline - case["alpha"] * np.outer(direction, direction @ baseline)
    np.testing.assert_allclose(observed, case["expected_writer"], rtol=0, atol=1e-15)


def _prepared(store: ArtifactStore):
    graph = reference_graph()
    tensors = reference_tensors()
    store.write_graph(graph, "conformance-baseline")
    baseline = store.save_checkpoint(tensors, graph, "conformance-baseline")
    candidate = derive_contrast(graph, tensors, synthetic_capture_bank(), "conformance")
    direction = store.derive_candidate(
        candidate, graph, tensors, "reference-writer-0", "conformance-bound"
    )
    return graph, baseline, direction


def _manifest_case(case: dict, store: ArtifactStore) -> None:
    graph = reference_graph()
    operation = case["operation"]
    if operation == "reference_graph":
        GraphSpec.model_validate(graph.model_dump(mode="json"))
        return
    if operation == "missing_space":
        raw = graph.model_dump(mode="json")
        raw["writers"][0]["output_space_id"] = "missing-space"
        GraphSpec.model_validate(raw)
        return
    graph, baseline, direction = _prepared(store)
    if operation == "execute_recurrent":
        raw = graph.model_dump(mode="json")
        raw["sites"][0]["coordinate"].update({"recurrent_step": 0, "state_component": "hidden"})
        raw["writers"][0]["coordinate"].update({"recurrent_step": 0, "state_component": "hidden"})
        graph = GraphSpec.model_validate(raw)
        direction = direction.model_copy(
            update={"graph_digest": graph.digest(), "coordinate": graph.writers[0].coordinate}
        )
        store.write_direction(direction, "recurrent-bound")
        direction_path = "recurrent-bound.direction.json"
    elif operation == "execute_unknown_adapter":
        graph = graph.model_copy(update={"adapter_id": "unqualified-conformance-adapter"})
        direction_path = "conformance-bound.direction.json"
    else:
        raise ValueError(f"unknown manifest conformance operation: {operation}")
    store.compile(
        graph,
        baseline,
        InterventionSpec(direction=direction, alpha=1),
        "manifest-candidate",
        direction_path=direction_path,
    )


def _identity_case(case: dict, store: ArtifactStore) -> None:
    graph, baseline, direction = _prepared(store)
    operation = case["operation"]
    if operation == "matching_direction":
        pass
    elif operation == "unknown_writer":
        direction = direction.model_copy(update={"writer_id": "unknown-writer"})
    elif operation == "wrong_space":
        direction = direction.model_copy(update={"space_id": "unknown-space"})
    elif operation == "non_unit_direction":
        # Revalidate explicitly: model_copy intentionally does not revalidate a model.
        direction = direction.model_validate(
            {**direction.model_dump(mode="json"), "unit_vector": (2.0, 0.0, 0.0)}
        )
    else:
        raise ValueError(f"unknown identity conformance operation: {operation}")
    if direction != store.load_direction("conformance-bound.direction.json"):
        identity = f"identity-{operation}"
        store.write_direction(direction, identity)
        direction_path = f"{identity}.direction.json"
    else:
        direction_path = "conformance-bound.direction.json"
    store.compile(
        graph,
        baseline,
        InterventionSpec(direction=direction, alpha=1),
        "identity-candidate",
        direction_path=direction_path,
    )


def _verification_case(case: dict, store: ArtifactStore) -> None:
    graph, baseline, direction = _prepared(store)
    compiled = store.compile(
        graph,
        baseline,
        InterventionSpec(direction=direction, alpha=1),
        "conformance-candidate",
        direction_path="conformance-bound.direction.json",
    )
    operation = case["operation"]
    if operation == "valid_compile":
        store.verify_compiled("conformance-candidate.manifest.json")
    elif operation == "tampered_candidate":
        candidate_path = store.path(compiled.candidate.path)
        candidate_path.write_bytes(candidate_path.read_bytes()[:-1] + bytes([0]))
        store.verify_compiled("conformance-candidate.manifest.json")
    elif operation == "tampered_audit":
        raw = read_json(store.path("conformance-candidate.manifest.json"))
        raw["structural_check"]["alpha"] = 0.0
        write_json(store.path("conformance-tampered.manifest.json"), raw)
        store.verify_compiled("conformance-tampered.manifest.json")
    else:
        raise ValueError(f"unknown verification conformance operation: {operation}")


def run_conformance(fixtures: Path, run_dir: Path) -> dict:
    """Report actual accept/reject observations; a rejection must be expected."""
    results = []
    for name, handler in (
        ("synthetic", lambda case, _store: _matrix_case(case)),
        ("manifests", _manifest_case),
        ("identity", _identity_case),
        ("verification", _verification_case),
    ):
        for case in _load_cases(fixtures, name):
            observed = "accept"
            error_code = None
            detail = "accepted"
            try:
                handler(case, ArtifactStore(run_dir / name / case["case_id"]))
            except (
                AblagentError,
                PydanticValidationError,
                ValueError,
                KeyError,
                TypeError,
                AssertionError,
            ) as exc:
                observed = "reject"
                error_code = getattr(exc, "code", type(exc).__name__)
                detail = str(exc)
            expected = case["outcome"]
            passed = observed == expected and (
                case.get("error_code") is None or case["error_code"] == error_code
            )
            results.append(
                {
                    "case_id": case["case_id"],
                    "suite": name,
                    "expected": expected,
                    "observed": observed,
                    "error_code": error_code,
                    "passed": passed,
                    "detail": detail if not passed else "conformance expectation satisfied",
                }
            )
    return {
        "schema_version": "dwmip.conformance-report.v0",
        "adapter_id": "synthetic-cpu-v0",
        "claim_scope": "synthetic_protocol_only",
        "passed": all(result["passed"] for result in results),
        "count": len(results),
        "results": results,
    }
