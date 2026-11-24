"""Explicit architecture semantics; names alone never establish compatibility."""

from typing import Literal

from pydantic import Field, model_validator

from ablagent.canonical import digest_model
from ablagent.contracts import Contract, Identifier

SpaceKind = Literal[
    "synthetic_output",
    "synthetic_input",
    "residual_stream",
    "attention_output",
    "mlp_output",
    "recurrent_state",
    "routed_latent",
]


class SiteCoordinate(Contract):
    site_id: Identifier
    logical_layer: int | None = Field(default=None, ge=0)
    expert_id: Identifier | None = None
    route_id: Identifier | None = None
    recurrent_step: int | None = Field(default=None, ge=0)
    loop_iteration: int | None = Field(default=None, ge=0)
    state_component: Identifier | None = None

    @model_validator(mode="after")
    def recurrent_semantics(self):
        if (self.recurrent_step is None) != (self.state_component is None):
            raise ValueError("recurrent_step and state_component must be declared together")
        if self.loop_iteration is not None and self.logical_layer is None:
            raise ValueError("a loop iteration requires a logical layer")
        return self


class SpaceSpec(Contract):
    space_id: Identifier
    kind: SpaceKind
    dimension: int = Field(gt=0, le=1_000_000)
    basis: str = Field(min_length=1, max_length=1000)
    normalization: str = Field(min_length=1, max_length=1000)
    inner_product: Literal["euclidean"] = "euclidean"


class SiteSpec(Contract):
    coordinate: SiteCoordinate
    space_id: Identifier
    capture_timing: str = Field(min_length=1, max_length=1000)


class WriterSpec(Contract):
    writer_id: Identifier
    tensor_name: Identifier
    coordinate: SiteCoordinate
    output_space_id: Identifier
    input_space_id: Identifier
    layout: Literal["dense_out_in"] = "dense_out_in"
    packing: Literal["dense_row_major"] = "dense_row_major"
    dtype: Literal["float32", "float64"]
    shape: tuple[int, int]
    operations: tuple[Literal["rank_one_output"], ...] = ()
    shared_group: Identifier | None = None

    @model_validator(mode="after")
    def dimensions(self):
        if any(dimension <= 0 or dimension > 1_000_000 for dimension in self.shape):
            raise ValueError("writer dimensions must be in 1..1,000,000")
        return self


class GraphSpec(Contract):
    """A descriptive graph. Representation does not confer execution support."""

    schema_version: Literal["dwmip.graph.v0"] = "dwmip.graph.v0"
    adapter_id: Identifier
    checkpoint_id: Identifier
    topology: Literal["dense_linear", "mo_e", "recurrent", "looped", "hybrid"]
    sites: tuple[SiteSpec, ...] = Field(min_length=1, max_length=4096)
    spaces: tuple[SpaceSpec, ...] = Field(min_length=1, max_length=4096)
    writers: tuple[WriterSpec, ...] = Field(min_length=1, max_length=4096)
    execution_supported: bool = False

    @model_validator(mode="after")
    def graph_integrity(self):
        for label, values in (
            ("site", [s.coordinate.site_id for s in self.sites]),
            ("space", [s.space_id for s in self.spaces]),
            ("writer", [w.writer_id for w in self.writers]),
            ("tensor", [w.tensor_name for w in self.writers]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} identifier")
        coordinates = [s.coordinate for s in self.sites]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("duplicate site coordinate")
        spaces = {space.space_id: space for space in self.spaces}
        for site in self.sites:
            if site.space_id not in spaces:
                raise ValueError(f"unknown capture space: {site.space_id}")
        for writer in self.writers:
            if writer.coordinate not in coordinates:
                raise ValueError(f"writer {writer.writer_id} has an undeclared coordinate")
            if writer.output_space_id not in spaces or writer.input_space_id not in spaces:
                raise ValueError(f"writer {writer.writer_id} has an undeclared space")
            if spaces[writer.output_space_id].dimension != writer.shape[0]:
                raise ValueError(f"writer {writer.writer_id} output dimension mismatch")
            if spaces[writer.input_space_id].dimension != writer.shape[1]:
                raise ValueError(f"writer {writer.writer_id} input dimension mismatch")
        return self

    def writer(self, writer_id: str) -> WriterSpec:
        for writer in self.writers:
            if writer.writer_id == writer_id:
                return writer
        raise KeyError(writer_id)

    def digest(self) -> str:
        return digest_model("graph", self)


class AdapterCapabilities(Contract):
    adapter_id: Identifier
    supported_schema: Literal["dwmip.graph.v0"] = "dwmip.graph.v0"
    executable_topologies: tuple[Literal["dense_linear"], ...] = ()
    supported_operations: tuple[Literal["rank_one_output"], ...] = ()
    supported_codecs: tuple[Literal["safetensors"], ...] = ()
    # Additive widening (was Literal[False]): the hf-dense-linear-v0 adapter
    # declares True. Default stays False; existing payloads are unchanged.
    real_checkpoint_support: bool = False
    execution_requires: tuple[str, ...] = (
        "execution_supported=true",
        "one writer per site",
        "unshared dense_out_in tensors",
        "declared output capture space",
    )
