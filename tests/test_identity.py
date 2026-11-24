import pytest
from pydantic import ValidationError as PydanticValidationError

from ablagent.errors import ValidationError
from ablagent.identity import GraphSpec, SiteCoordinate, SiteSpec, SpaceSpec, WriterSpec


def complex_graph(topology="mo_e", execution_supported=False):
    coordinate = SiteCoordinate(
        site_id="layer-0-expert-3-route-a",
        logical_layer=0,
        expert_id="expert-3",
        route_id="route-a",
        recurrent_step=1 if topology == "recurrent" else None,
        state_component="memory" if topology == "recurrent" else None,
        loop_iteration=2 if topology == "looped" else None,
    )
    output = SpaceSpec(
        space_id="latent-output",
        kind="routed_latent",
        dimension=3,
        basis="Explicit synthetic latent coordinates; not a residual basis.",
        normalization="Declared test normalization.",
    )
    input_space = SpaceSpec(
        space_id="latent-input",
        kind="synthetic_input",
        dimension=2,
        basis="Explicit test input coordinates.",
        normalization="Declared test normalization.",
    )
    return GraphSpec(
        adapter_id="descriptive-fixture",
        checkpoint_id="no-real-checkpoint",
        topology=topology,
        execution_supported=execution_supported,
        sites=(
            SiteSpec(
                coordinate=coordinate,
                space_id="latent-output",
                capture_timing="Explicit test site.",
            ),
        ),
        spaces=(output, input_space),
        writers=(
            WriterSpec(
                writer_id="latent-writer",
                tensor_name="latent_writer",
                coordinate=coordinate,
                output_space_id="latent-output",
                input_space_id="latent-input",
                dtype="float64",
                shape=(3, 2),
                operations=("rank_one_output",),
            ),
        ),
    )


@pytest.mark.parametrize("topology", ["mo_e", "recurrent", "looped"])
def test_complex_graph_is_representable_but_not_executable(topology):
    graph = complex_graph(topology)
    assert graph.digest().startswith("sha256:")
    from ablagent.artifacts import _validate_execution
    from ablagent.errors import ValidationError

    with pytest.raises(ValidationError, match="synthetic-cpu-v0"):
        _validate_execution(graph)


def test_graph_rejects_duplicate_sites_and_unknown_spaces():
    graph = complex_graph()
    raw = graph.model_dump(mode="json")
    raw["sites"].append(raw["sites"][0])
    with pytest.raises(PydanticValidationError, match="duplicate"):
        GraphSpec.model_validate(raw)
    raw = graph.model_dump(mode="json")
    raw["writers"][0]["output_space_id"] = "unknown"
    with pytest.raises(PydanticValidationError, match="undeclared"):
        GraphSpec.model_validate(raw)


def test_incomplete_recurrent_coordinate_rejected():
    with pytest.raises(PydanticValidationError, match="together"):
        SiteCoordinate(site_id="memory", recurrent_step=1)


def test_same_shape_does_not_make_writers_interchangeable(prepared):
    store, graph, tensors, _, candidate, _ = prepared
    raw = graph.model_dump(mode="json")
    raw["writers"][0]["writer_id"] = "same-shape-other-writer"
    changed = GraphSpec.model_validate(raw)
    with pytest.raises(ValidationError, match=r"baseline|graph"):
        store.derive_candidate(candidate, changed, tensors, "same-shape-other-writer", "wrong")
