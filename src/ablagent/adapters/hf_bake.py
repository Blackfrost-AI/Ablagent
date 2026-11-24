"""S3 real bake: rank-one edit through the DWMIP compiler, then HF export.

The edit itself is executed by ArtifactStore.compile (W' = W - alpha*d(d^T W) in
float64, recast to the checkpoint dtype, independent structural audit). This
module adds:

- selection of one declared candidate + writer (operator parameter),
- ``verify_compiled`` re-verification of the persisted manifest,
- export: copy the pinned HF snapshot, replace ONLY the edited tensor in its
  original shard, recast to the ORIGINAL dtype (bf16/f16/f32), byte-identical
  checks for every other shard and file, and an export receipt binding
  candidate digests to the exported bytes. Conv1D (in, out) layouts fail
  closed at intake, so no transpose handling exists here; a transposed flag
  in provenance is rejected rather than silently applied,
- a numeric sign test: projection of the edited weight rows onto d shrinks by
  exactly (1 - alpha) up to dtype tolerance.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ablagent.artifacts import ArtifactStore, _tensor_digest, load_tensors
from ablagent.errors import ValidationError
from ablagent.identity import GraphSpec
from ablagent.interventions import ArtifactReference, InterventionSpec


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def bake(
    store: ArtifactStore,
    graph: GraphSpec,
    baseline: ArtifactReference,
    *,
    candidate_path: str,
    writer_id: str,
    alpha: float,
    artifact_id: str = "candidate",
) -> dict[str, Any]:
    """Bind the selected candidate, compile the rank-one edit, verify the manifest."""
    if not 0.0 < alpha <= 1.0:
        raise ValidationError("alpha must be in (0, 1] for a real bake")
    candidate = store.load_direction(candidate_path)
    tensors = load_tensors(store.path(baseline.path))
    bound = store.derive_candidate(candidate, graph, tensors, writer_id, f"{artifact_id}-bound")
    direction_path = f"{artifact_id}-bound.direction.json"
    intervention = InterventionSpec(direction=bound, alpha=float(alpha))
    compiled = store.compile(
        graph, baseline, intervention, artifact_id, direction_path=direction_path
    )
    verified = store.verify_compiled(f"{artifact_id}.manifest.json")
    if verified.candidate.tensor_digest != compiled.candidate.tensor_digest:
        raise ValidationError("verify_compiled disagrees with compile output")
    # Numeric sign test on the target rows: projection must shrink by (1-alpha).
    import numpy as np

    original = tensors[bound.tensor_name].astype(np.float64)
    edited = load_tensors(store.path(compiled.candidate.path))[bound.tensor_name].astype(np.float64)
    unit = bound.array()
    before = unit @ original
    after = unit @ edited
    expected = (1.0 - alpha) * before
    residual = float(np.max(np.abs(after - expected))) if before.size else 0.0
    scale = max(1.0, float(np.max(np.abs(before))))
    sign_ok = residual <= 1e-5 * scale
    if not sign_ok:
        raise ValidationError(
            f"rank-one sign test failed: residual {residual:.3e} > tol {1e-5 * scale:.3e}"
        )
    return {
        "manifest": f"{artifact_id}.manifest.json",
        "candidate": compiled.candidate.model_dump(mode="json"),
        "direction_digest": compiled.direction_digest,
        "structural_check_passed": compiled.structural_check.passed,
        "sign_test": {
            "passed": sign_ok,
            "max_residual": residual,
            "tolerance": 1e-5 * scale,
            "identity": "d^T W' == (1 - alpha) * d^T W",
        },
        "alpha": float(alpha),
        "writer_id": writer_id,
        "tensor_name": bound.tensor_name,
    }


def export_hf(
    store: ArtifactStore,
    graph: GraphSpec,
    *,
    candidate_path: str,
    snapshot_dir: Path,
    export_dir: Path,
    intake_receipt: dict[str, Any],
    tensor_name: str,
) -> dict[str, Any]:
    """Write an HF-loadable copy of the snapshot with exactly one tensor replaced.

    Every file except the edited tensor's shard is copied byte-for-byte and
    hash-verified. The edited shard is re-serialized with the original tensor
    set, original dtypes, original metadata; the edited tensor is recast from
    the float32 candidate to its original dtype. Transposed layouts are not
    supported in v0 (Conv1D families fail closed at intake) and are rejected.
    """
    import torch
    from safetensors.torch import load_file, save_file

    if export_dir.exists():
        raise ValidationError(f"export dir already exists: {export_dir}")
    provenance = intake_receipt["provenance"]
    original_dtype = provenance["original_dtypes"].get(tensor_name)
    if original_dtype is None:
        raise ValidationError(f"{tensor_name!r} is not part of the declared editable surface")
    shard_name = intake_receipt["shard_of"][tensor_name]
    was_transposed = bool(provenance.get("transposed", {}).get(tensor_name, False))
    if was_transposed:
        raise ValidationError(
            f"{tensor_name!r} is recorded as transposed; transposed layouts are "
            "not qualified in v0 (Conv1D families fail closed at intake)"
        )

    candidate_tensors = load_tensors(store.path(candidate_path))
    edited_f32 = candidate_tensors[tensor_name]

    export_dir.mkdir(parents=True)
    snapshot_dir = Path(snapshot_dir)
    shard_files = set(intake_receipt["shards"])
    copied: dict[str, str] = {}
    for source in sorted(snapshot_dir.iterdir()):
        if not source.is_file() or source.name in shard_files:
            continue
        target = export_dir / source.name
        shutil.copyfile(source, target)
        if _sha256(source) != _sha256(target):
            raise ValidationError(f"copy verification failed for {source.name}")
        copied[source.name] = _sha256(target)

    exported_shards: dict[str, str] = {}
    for shard in sorted(shard_files):
        source = snapshot_dir / shard
        target = export_dir / shard
        if shard != shard_name:
            shutil.copyfile(source, target)
            if source.read_bytes() != target.read_bytes():
                raise ValidationError(f"non-edited shard changed: {shard}")
            exported_shards[shard] = _sha256(target)
            continue
        from safetensors import safe_open

        with safe_open(str(source), framework="pt") as handle:
            metadata = handle.metadata() or {}
        shard_tensors = load_file(str(source))
        new = torch.from_numpy(edited_f32.copy())
        new = new.to(getattr(torch, original_dtype))
        if tuple(new.shape) != tuple(shard_tensors[tensor_name].shape):
            raise ValidationError("exported tensor shape does not match the original shard")
        shard_tensors[tensor_name] = new
        save_file(shard_tensors, str(target), metadata=metadata)
        # Readback: every OTHER tensor in the edited shard must be bit-identical.
        original_shard = load_file(str(source))
        readback = load_file(str(target))
        if set(readback) != set(original_shard):
            raise ValidationError("exported shard tensor set changed")
        for name, tensor in readback.items():
            if name == tensor_name:
                continue
            if not torch.equal(tensor, original_shard[name]):
                raise ValidationError(f"non-edited tensor changed in shard: {name}")
        # Numeric receipt for the EDITED tensor: bound the dtype-recast error
        # between the float32 candidate and the exported original-dtype bytes.
        import numpy as np

        exported_edited = readback[tensor_name].to(torch.float32).cpu().numpy()
        recast_deviation = float(
            np.max(np.abs(exported_edited.astype(np.float64) - edited_f32.astype(np.float64)))
        )
        exported_shards[shard] = _sha256(target)

    receipt = {
        "tensor_name": tensor_name,
        "shard": shard_name,
        "original_dtype": original_dtype,
        "transposed_back": was_transposed,
        "candidate_tensor_digest": _tensor_digest(candidate_tensors),
        "exported_shard_digests": exported_shards,
        "copied_file_digests": copied,
        "non_edited_shards_byte_identical": True,
        "edited_tensor_recast_max_deviation": recast_deviation,
        "export_dir": str(export_dir),
        "source_revision": intake_receipt["revision"],
        "claim": "hf_loadable_export_not_a_release",
    }
    (export_dir / "ABLAGENT-EXPORT-RECEIPT.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return receipt
