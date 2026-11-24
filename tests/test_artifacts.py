import numpy as np
import pytest
from safetensors.numpy import save_file

from ablagent.artifacts import load_tensors, tensor_metadata
from ablagent.canonical import canonical_bytes, digest_bytes, read_json
from ablagent.errors import ValidationError


def test_golden_canonical_json_and_domain_separation():
    value = {"z": [True, None, -2], "a": "café", "n": 1.25}
    assert canonical_bytes(value) == '{"a":"café","n":1.25,"z":[true,null,-2]}'.encode()
    assert digest_bytes("one", b"x") != digest_bytes("two", b"x")
    for invalid in (float("nan"), float("inf"), -0.0, {"x": object()}):
        with pytest.raises(ValidationError):
            canonical_bytes(invalid)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate"):
        read_json(path)


def test_exact_tensor_roundtrip_and_metadata(prepared):
    store, graph, tensors, baseline, _, _ = prepared
    actual = load_tensors(store.path(baseline.path))
    np.testing.assert_array_equal(actual["writer0"], tensors["writer0"])
    metadata = tensor_metadata(store.path(baseline.path))
    assert metadata["graph_digest"] == graph.digest()
    assert metadata["artifact_id"] == "baseline"


def test_malformed_and_truncated_safetensors_rejected(tmp_path):
    for content in (b"", b"abc", (1000).to_bytes(8, "little") + b"{}"):
        path = tmp_path / "bad.safetensors"
        path.write_bytes(content)
        with pytest.raises(ValidationError):
            load_tensors(path)


def test_unsupported_dtype_rejected(tmp_path):
    path = tmp_path / "integer.safetensors"
    save_file({"writer0": np.ones((3, 2), dtype=np.int64)}, str(path))
    with pytest.raises(ValidationError, match="dtype"):
        load_tensors(path)


def test_declared_header_limit_checked_before_tensor_allocation(tmp_path):
    path = tmp_path / "large-header.safetensors"
    path.write_bytes((1024 * 1024 + 1).to_bytes(8, "little") + b"x")
    with pytest.raises(ValidationError, match="resource limit"):
        load_tensors(path)


def test_file_resource_limit_rejected(tmp_path):
    path = tmp_path / "oversized.safetensors"
    save_file({"writer0": np.ones((3, 2), dtype=np.float64)}, str(path))
    with pytest.raises(ValidationError, match="resource limit"):
        load_tensors(path, max_bytes=10)


def test_json_resource_limit_rejected(tmp_path):
    path = tmp_path / "large.json"
    path.write_text('{"value":"' + "x" * 100 + '"}', encoding="utf-8")
    with pytest.raises(ValidationError, match="limit"):
        read_json(path, max_bytes=32)


def test_compiled_structural_audit_is_recomputed(prepared):
    store, _, _, _, _, _ = prepared
    from tests.conftest import compiled_case, dump_json

    compiled_case(prepared)
    path = store.path("candidate.manifest.json")
    raw = read_json(path)
    raw["structural_check"]["tolerance"] = 2.0
    dump_json(path, raw)
    with pytest.raises(ValidationError, match="structural audit"):
        store.verify_compiled("candidate.manifest.json")


def test_adapter_name_does_not_confer_execution_support(prepared):
    store, graph, _, baseline, _, direction = prepared
    from ablagent.interventions import InterventionSpec

    changed = graph.model_copy(update={"adapter_id": "unqualified-adapter"})
    with pytest.raises(ValidationError, match="synthetic-cpu-v0"):
        store.compile(
            changed,
            baseline,
            InterventionSpec(direction=direction, alpha=1),
            "unqualified",
        )


def test_safetensors_missing_file_is_a_validation_error(tmp_path):
    with pytest.raises(ValidationError, match="invalid safetensors"):
        load_tensors(tmp_path / "missing.safetensors")


def test_structural_audit_compares_observed_candidate_tensors(prepared):
    from ablagent.artifacts import _structural_check
    from ablagent.interventions import InterventionSpec

    store, graph, tensors, baseline, _, direction = prepared
    save_file(tensors, str(store.path("observed.safetensors")))
    reference = store.reference("observed.safetensors", "candidate", graph, tensors)
    original = tensors["writer0"]
    expected = original - np.outer(direction.array(), direction.array() @ original)
    observed = original.copy()
    with pytest.raises(ValidationError, match="independent tensor comparison"):
        _structural_check(
            graph=graph,
            baseline=baseline,
            candidate=reference,
            intervention=InterventionSpec(direction=direction, alpha=1),
            original_tensors=tensors,
            candidate_tensors={**tensors, "writer0": observed},
            original=original,
            edited=observed,
            expected=expected,
        )
