import pytest
from pydantic import ValidationError as PydanticValidationError

from ablagent.errors import ValidationError
from ablagent.planner import ExperimentRequest, propose_experiment


def test_planner_is_bounded_and_proposal_only(prepared):
    direction = prepared[-1]
    request = ExperimentRequest(
        request_id="experiment", direction=direction, alpha_values=(0, 0.5, 1)
    )
    first = propose_experiment(request)
    second = propose_experiment(request)
    assert first == second
    assert first.execution_authority == "none"
    assert first.approval_authority == "none"
    assert len(first.interventions) == 3
    assert first.digest() == second.digest()


def test_planner_rejects_unbounded_or_nonfinite_requests(prepared):
    direction = prepared[-1]
    with pytest.raises(PydanticValidationError):
        ExperimentRequest(
            request_id="too-many", direction=direction, alpha_values=(0, 0.1, 0.2, 0.3, 0.4)
        )
    with pytest.raises(ValidationError, match="unique"):
        propose_experiment(
            ExperimentRequest(request_id="duplicate", direction=direction, alpha_values=(1, 1))
        )
    with pytest.raises(ValidationError, match="finite"):
        propose_experiment(
            ExperimentRequest(
                request_id="infinite", direction=direction, alpha_values=(float("nan"),)
            )
        )
    with pytest.raises(PydanticValidationError):
        ExperimentRequest(
            request_id="unknown-authority", direction=direction, alpha_values=(1,), authorized=True
        )


def test_multiple_direction_composition_is_not_available(prepared, tmp_path):
    store, _, _, _, _, direction = prepared
    from ablagent.canonical import write_json
    from ablagent.cli import main

    plan = propose_experiment(
        ExperimentRequest(request_id="single", direction=direction, alpha_values=(1,))
    )
    # No intervention has a list of directions or coordinates. A separate proposal
    # can contain trials, but it is rejected by the single-step compile boundary.
    write_json(tmp_path / "one.plan.json", plan)
    assert (
        main(
            [
                "compile",
                "bound.direction.json",
                "--run-dir",
                str(tmp_path),
                "--plan",
                "one.plan.json",
                "--artifact-id",
                "explicit",
            ]
        )
        == 0
    )
    assert store.path("explicit.manifest.json").exists()
    two = propose_experiment(
        ExperimentRequest(request_id="two", direction=direction, alpha_values=(0, 1))
    )
    write_json(tmp_path / "two.plan.json", two)
    assert (
        main(
            [
                "compile",
                "bound.direction.json",
                "--run-dir",
                str(tmp_path),
                "--plan",
                "two.plan.json",
                "--artifact-id",
                "implicit-composition",
            ]
        )
        == 2
    )
    assert not store.path("implicit-composition.safetensors").exists()
