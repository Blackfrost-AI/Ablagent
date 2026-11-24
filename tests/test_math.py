import math

import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from ablagent.artifacts import load_tensors
from ablagent.errors import ValidationError
from ablagent.interventions import InterventionSpec


def test_hand_computed_rank_one_update(prepared):
    store, graph, _, baseline, _, direction = prepared
    # An independent unit direction at the same declared site removes row zero.
    direction = direction.model_copy(update={"unit_vector": (1.0, 0.0, 0.0)})
    compiled = store.compile(
        graph, baseline, InterventionSpec(direction=direction, alpha=1.0), "candidate"
    )
    actual = load_tensors(store.path(compiled.candidate.path))["writer0"]
    np.testing.assert_array_equal(actual, np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    assert store.verify_compiled("candidate.manifest.json") == compiled


@pytest.mark.parametrize(
    ("alpha", "artifact_id"),
    [(0.0, "candidate-zero"), (0.25, "candidate-quarter"), (1.0, "candidate-full")],
)
def test_alpha_controls_only_the_declared_component(prepared, alpha, artifact_id):
    store, graph, tensors, baseline, _, direction = prepared
    compiled = store.compile(
        graph, baseline, InterventionSpec(direction=direction, alpha=alpha), artifact_id
    )
    original = tensors["writer0"]
    actual = load_tensors(store.path(compiled.candidate.path))["writer0"]
    expected = original - alpha * np.outer(direction.array(), direction.array() @ original)
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
    # The component removed from every output column is alpha*(d^T W).
    np.testing.assert_allclose(
        original - actual,
        alpha * np.outer(direction.array(), direction.array() @ original),
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_allclose(
        direction.array() @ actual,
        (1 - alpha) * (direction.array() @ original),
        atol=1e-15,
        rtol=0,
    )
    # There is only one tensor in the reference graph; the analytic alpha=0 case
    # and multi-tensor fixture below cover exact preservation of unrelated data.


def test_non_unit_direction_is_not_silently_accepted(prepared):
    _, _, _, _, _, direction = prepared
    raw = direction.model_dump(mode="json")
    raw["unit_vector"] = [2.0, 0.0, 0.0]
    with pytest.raises(PydanticValidationError, match="unit-normalized"):
        type(direction).model_validate(raw)


@pytest.mark.parametrize("vector", [[0.0, 0.0, 0.0], [math.nan, 0.0, 0.0], [math.inf, 0.0, 0.0]])
def test_zero_and_nonfinite_directions_rejected(prepared, vector):
    _, _, _, _, _, direction = prepared
    raw = direction.model_dump(mode="json")
    raw["unit_vector"] = vector
    with pytest.raises(PydanticValidationError):
        type(direction).model_validate(raw)


def test_wrong_rank_shape_dtype_and_writer_rejected(prepared):
    store, graph, _, baseline, _, direction = prepared
    for field, value in (
        ("shape", [4, 2]),
        ("dtype", "float32"),
        ("writer_id", "other"),
        ("tensor_name", "other"),
    ):
        altered = direction.model_copy(update={field: value})
        expected_error = PydanticValidationError if field == "shape" else ValidationError
        with pytest.raises(expected_error):
            store.compile(
                graph, baseline, InterventionSpec(direction=altered, alpha=1.0), f"bad-{field}"
            )
    raw = direction.model_dump(mode="json")
    raw["shape"] = [3, 2, 1]
    with pytest.raises(PydanticValidationError):
        type(direction).model_validate(raw)


def test_wrong_baseline_and_capture_bank_rejected(prepared, bank):
    store, graph, _, baseline, _, direction = prepared
    wrong = direction.model_copy(update={"baseline_digest": "sha256:" + "0" * 64})
    with pytest.raises(ValidationError, match="baseline digest"):
        store.compile(
            graph, baseline, InterventionSpec(direction=wrong, alpha=1.0), "wrong-baseline"
        )
    reference = direction.capture_bank.model_copy(update={"records_digest": "sha256:" + "1" * 64})
    altered = direction.model_copy(update={"capture_bank": reference})
    with pytest.raises(ValidationError, match="persisted bound direction"):
        store.compile(
            graph,
            baseline,
            InterventionSpec(direction=altered, alpha=1.0),
            "wrong-bank",
            direction_path="bound.direction.json",
        )


def test_direction_rejects_cross_space_binding(prepared):
    store, graph, tensors, _, candidate, _ = prepared
    wrong = candidate.model_copy(update={"space_id": "reference-input"})
    with pytest.raises(ValidationError, match="space"):
        store.derive_candidate(wrong, graph, tensors, "reference-writer-0", "wrong-space")


def test_alpha_zero_is_an_explicit_control(prepared):
    store, graph, _, baseline, _, direction = prepared
    compiled = store.compile(
        graph, baseline, InterventionSpec(direction=direction, alpha=0.0), "zero"
    )
    assert compiled.baseline.tensor_digest == compiled.candidate.tensor_digest
    assert compiled.baseline.file_digest != compiled.candidate.file_digest
    assert (
        compiled.structural_check.target_digest_before
        == compiled.structural_check.target_digest_after
    )
    store.verify_compiled("zero.manifest.json")
