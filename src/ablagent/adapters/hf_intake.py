"""S0 real intake: pin a dense HuggingFace checkpoint and declare its editable surface.

CPU-only. torch is imported lazily and only by the explicit real path. The
DWMIP "editable surface" checkpoint contains every 2-D float weight (each
declared as a writer + site + output/input spaces); 1-D norms/biases and
non-float tensors stay untouched in the HF snapshot. bf16/f16 are widened to
float32 here and the original dtype is recorded so S3 can restore it on export.

Tied weights: only the canonical stored tensor (the one present in the
safetensors shards) is declared as a writer; the alias (e.g. lm_head.weight
when tie_word_embeddings=True) is excluded and the tie recorded in provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ablagent.artifacts import ArtifactStore, _tensor_digest
from ablagent.constants import HF_ADAPTER_ID
from ablagent.contracts import _identifier
from ablagent.errors import ValidationError
from ablagent.identity import (
    GraphSpec,
    SiteCoordinate,
    SiteSpec,
    SpaceSpec,
    WriterSpec,
)

# The DWMIP single-file checkpoint is capped at 1 GiB (artifacts._MAX_FILE_BYTES).
# Leave headroom for the safetensors header; fail closed at intake, not mid-compile.
_MAX_SURFACE_BYTES = 900 * 1024 * 1024
_MAX_TENSORS = 4096

_HF_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?huggingface\.co/(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"(?:/(?:tree|resolve|blob)/(?P<rev>[A-Za-z0-9_.-]+))?(?:/.*)?$"
)


def require_real_deps() -> dict[str, str]:
    """Fail closed with a clear install hint when the [real] extra is absent."""
    missing = []
    versions: dict[str, str] = {}
    for mod in ("torch", "transformers", "huggingface_hub", "safetensors"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            missing.append(mod)
    if missing:
        raise ValidationError(
            "real executor requires the [real] extra: pip install 'ablagent[real]' "
            f"(missing: {', '.join(missing)})"
        )
    return versions


def parse_model_ref(model: str) -> tuple[str, str | None]:
    """Accept an HF URL or a repo id; return (repo_id, revision|None)."""
    model = model.strip()
    if model.startswith(("mock://", "synthetic://")):
        raise ValidationError(f"{model!r} is not a real checkpoint reference")
    if not model.startswith(("http://", "https://", "www.")) and Path(model).is_dir():
        return str(Path(model).resolve()), None  # local HF-format directory (offline/CI)
    match = _HF_URL_RE.match(model)
    if match:
        return match.group("repo"), match.group("rev")
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model):
        return model, None
    raise ValidationError(
        f"cannot parse model reference {model!r}; expected an HF URL or 'owner/name'"
    )


def _tensor_id(tensor_name: str) -> str:
    """Map a safetensors tensor name to a valid DWMIP Identifier (dots allowed)."""
    slug = re.sub(r"[^a-z0-9_.:-]", "-", tensor_name.lower())
    slug = re.sub(r"^[^a-z0-9]+", "", slug)[:120]
    if not slug:
        raise ValidationError(f"tensor name {tensor_name!r} cannot form an identifier")
    return _identifier(slug)


def download_snapshot(repo_id: str, revision: str | None, dest: Path) -> dict[str, Any]:
    """Pin and download the exact revision; record per-file hashes. Network only here.

    A local directory path is also accepted (offline runs / CI fixtures); it is
    pinned by a deterministic digest over its file hashes instead of a commit.
    """
    local_path = Path(repo_id)
    if local_path.is_dir():
        return _pin_local_snapshot(local_path, dest)
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    pinned = info.sha
    if not pinned:
        raise ValidationError(f"could not resolve a commit sha for {repo_id}")
    _preflight_shard_sizes(info)
    local = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=pinned,
            local_dir=str(dest),
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja"],
        )
    )
    shards = sorted(local.glob("*.safetensors"))
    if not shards:
        raise ValidationError(
            f"{repo_id}@{pinned[:12]} has no safetensors shards; "
            "the v0 real adapter supports safetensors dense checkpoints only"
        )
    files = {
        p.name: "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(local.iterdir())
        if p.is_file()
    }
    return {
        "repo_id": repo_id,
        "revision": pinned,
        "snapshot_dir": str(local),
        "shards": [p.name for p in shards],
        "file_digests": files,
    }


def _preflight_shard_sizes(info: Any) -> None:
    """Fail closed BEFORE downloading when shards exceed the v0 envelope.

    Uses remote file metadata; a float32 surface is at most 2x the bf16 shard
    bytes, so shard bytes > cap is a definite oversize (conservative, no OOM).
    """
    total = 0
    for sibling in getattr(info, "siblings", []) or []:
        name = getattr(sibling, "rfilename", "")
        size = getattr(sibling, "size", None)
        if name.endswith(".safetensors") and size:
            total += int(size)
    if total > _MAX_SURFACE_BYTES:
        raise ValidationError(
            "model exceeds the v0 real-adapter resource envelope "
            f"(safetensors shards total > {_MAX_SURFACE_BYTES // (1024 * 1024)} MiB); "
            "use a smaller dense model"
        )


def _pin_local_snapshot(local_path: Path, dest: Path) -> dict[str, Any]:
    """Copy a local HF-format directory into the run and pin its content digest."""
    import shutil

    shards = sorted(local_path.glob("*.safetensors"))
    if not shards:
        raise ValidationError(
            f"{local_path} has no safetensors shards; "
            "the v0 real adapter supports safetensors dense checkpoints only"
        )
    total = sum(p.stat().st_size for p in shards)
    if total > _MAX_SURFACE_BYTES:
        raise ValidationError(
            "model exceeds the v0 real-adapter resource envelope "
            f"(safetensors shards total > {_MAX_SURFACE_BYTES // (1024 * 1024)} MiB); "
            "use a smaller dense model"
        )
    dest.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for source in sorted(local_path.iterdir()):
        allowed = {".safetensors", ".json", ".txt", ".model", ".jinja"}
        if source.is_file() and source.suffix in allowed:
            target = dest / source.name
            shutil.copyfile(source, target)
            # Hash the copied bytes (the pinned artifact), not the source.
            files[source.name] = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "repo_id": str(local_path),
        "revision": "local-" + manifest_digest[:40],
        "snapshot_dir": str(dest),
        "shards": [p.name for p in shards],
        "file_digests": files,
    }


def _load_shard_tensors(shards: list[Path]) -> dict[str, Any]:
    """Load every shard via safetensors.torch (handles bf16/f16/f32/f64)."""
    from safetensors.torch import load_file

    tensors: dict[str, Any] = {}
    shard_of: dict[str, str] = {}
    for shard in shards:
        loaded = load_file(str(shard))
        for name, tensor in loaded.items():
            if name in tensors:
                raise ValidationError(f"tensor {name!r} appears in multiple shards")
            tensors[name] = tensor
            shard_of[name] = shard.name
    return {"tensors": tensors, "shard_of": shard_of}


def build_editable_surface(
    tensors: dict[str, Any], config: Any, checkpoint_id: str
) -> tuple[GraphSpec, dict[str, Any], dict[str, Any]]:
    """Declare every 2-D float weight as a writer; return (graph, f32_tensors, provenance).

    ``tensors`` maps name -> torch tensor. Non-2-D / non-float tensors are left
    out of the surface (they remain in the HF snapshot). Tied aliases are excluded.
    Conv1D (in, out) families (gpt2/gpt_neo/gptj) fail closed: shape-only
    layout inference could edit the wrong axis on square projections.
    """
    import numpy as np
    import torch

    tie = bool(getattr(config, "tie_word_embeddings", False))
    stored_names = set(tensors)
    excluded_aliases: dict[str, str] = {}
    if tie:
        for alias, canonical in (
            ("lm_head.weight", "model.embed_tokens.weight"),
            ("lm_head.weight", "transformer.wte.weight"),
        ):
            if alias in stored_names and canonical in stored_names:
                excluded_aliases[alias] = canonical

    # Conv1D families (GPT-2/Neo/J) store 2-D linears as (in, out). Their layout
    # cannot be inferred from shapes alone (square projections are ambiguous),
    # so v0 fails closed rather than risk a wrong-axis edit. Support requires a
    # module-class-qualified adapter revision.
    model_type = getattr(config, "model_type", "")
    if model_type in {"gpt2", "gpt_neo", "gptj"}:
        raise ValidationError(
            f"model_type {model_type!r} uses Conv1D (in, out) weight layout; "
            "the v0 real adapter supports dense_out_in checkpoints only "
            "(llama/qwen/mistral/gemma-style). Failing closed: a shape-only "
            "transpose heuristic could edit the wrong axis on square weights."
        )
    spaces: list[SpaceSpec] = []
    sites: list[SiteSpec] = []
    writers: list[WriterSpec] = []
    surface: dict[str, Any] = {}
    original_dtypes: dict[str, str] = {}
    transposed: dict[str, bool] = {}
    seen_ids: set[str] = set()
    total_bytes = 0

    for name in sorted(stored_names):
        tensor = tensors[name]
        if name in excluded_aliases:
            continue
        if tensor.ndim != 2 or not tensor.is_floating_point():
            continue
        if len(writers) >= _MAX_TENSORS:
            raise ValidationError(f"editable surface exceeds {_MAX_TENSORS} tensors (v0 envelope)")
        original_dtype = str(tensor.dtype).removeprefix("torch.")
        transposed[name] = False  # dense_out_in only; Conv1D families fail closed above
        arr = tensor.detach().to(torch.float32).cpu().numpy()
        arr = np.array(arr, dtype=np.float32, copy=True)
        if not np.isfinite(arr).all():
            raise ValidationError(f"tensor {name!r} contains non-finite values")
        total_bytes += arr.nbytes
        if total_bytes > _MAX_SURFACE_BYTES:
            raise ValidationError(
                "model exceeds the v0 real-adapter resource envelope "
                f"(editable surface > {_MAX_SURFACE_BYTES // (1024 * 1024)} MiB float32); "
                "use a smaller dense model"
            )
        out_dim, in_dim = int(arr.shape[0]), int(arr.shape[1])
        tid = _tensor_id(name)
        if tid in seen_ids:
            raise ValidationError(f"tensor identifier collision: {tid}")
        seen_ids.add(tid)
        space_id = f"{tid}-out"
        input_space_id = f"{tid}-in"
        site_id = f"{tid}-site"
        logical_layer = len(sites)
        lowered = name.lower()
        if "attn" in lowered or "attention" in lowered:
            out_kind = "attention_output"
        elif "mlp" in lowered or "feed_forward" in lowered or "ffn" in lowered:
            out_kind = "mlp_output"
        else:
            out_kind = "residual_stream"  # embeddings/heads write into the residual basis
        spaces.append(
            SpaceSpec(
                space_id=space_id,
                kind=out_kind,
                dimension=out_dim,
                basis=f"hf:{name}:rows (declared output coordinates; no cross-space transfer)",
                normalization="none; unit direction is explicit at bind time",
            )
        )
        spaces.append(
            SpaceSpec(
                space_id=input_space_id,
                kind="residual_stream",
                dimension=in_dim,
                basis=f"hf:{name}:cols (declared input coordinates)",
                normalization="none",
            )
        )
        sites.append(
            SiteSpec(
                coordinate=SiteCoordinate(site_id=site_id, logical_layer=logical_layer),
                space_id=space_id,
                capture_timing="post",
            )
        )
        writers.append(
            WriterSpec(
                writer_id=f"{tid}-writer",
                tensor_name=name,
                coordinate=SiteCoordinate(site_id=site_id, logical_layer=logical_layer),
                output_space_id=space_id,
                input_space_id=input_space_id,
                dtype="float32",
                shape=(out_dim, in_dim),
                operations=("rank_one_output",),
            )
        )
        surface[name] = arr
        original_dtypes[name] = original_dtype

    if not writers:
        raise ValidationError("checkpoint exposes no 2-D float editable surface")

    graph = GraphSpec(
        adapter_id=HF_ADAPTER_ID,
        checkpoint_id=checkpoint_id,
        topology="dense_linear",
        execution_supported=True,
        spaces=tuple(spaces),
        sites=tuple(sites),
        writers=tuple(writers),
    )
    provenance = {
        "tie_word_embeddings": tie,
        "excluded_aliases": excluded_aliases,
        "original_dtypes": original_dtypes,
        "transposed": {k: v for k, v in transposed.items() if v},
        "surface_tensor_count": len(surface),
        "surface_bytes": total_bytes,
    }
    return graph, surface, provenance


def intake(model: str, *, snapshot_dir: Path, store_root: Path) -> dict[str, Any]:
    """S0: pin + download + declare editable surface + save baseline checkpoint.

    The baseline artifact id is "baseline" so the graph (baseline.graph.json) and
    checkpoint (baseline.safetensors) share a stem (verify_compiled loads the
    graph from the baseline artifact id).
    """
    import torch
    from transformers import AutoConfig

    versions = require_real_deps()
    repo_id, revision = parse_model_ref(model)
    pin = download_snapshot(repo_id, revision, snapshot_dir)
    config = AutoConfig.from_pretrained(pin["snapshot_dir"])
    loaded = _load_shard_tensors([Path(pin["snapshot_dir"]) / s for s in pin["shards"]])
    checkpoint_id = _identifier(f"hf-{repo_id.replace('/', '-')}-{pin['revision'][:12]}")
    graph, surface, provenance = build_editable_surface(loaded["tensors"], config, checkpoint_id)

    store = ArtifactStore(store_root)
    graph_path = store.write_graph(graph, "baseline")
    baseline = store.save_checkpoint(surface, graph, "baseline")
    return {
        "stage": "S0_INTAKE",
        "adapter_id": HF_ADAPTER_ID,
        "model": model,
        "repo_id": repo_id,
        "revision": pin["revision"],
        "checkpoint_id": checkpoint_id,
        "graph_path": graph_path,
        "baseline": baseline.model_dump(mode="json"),
        "surface_tensor_digest": _tensor_digest(surface),
        "provenance": provenance,
        "shard_of": loaded["shard_of"],
        "file_digests": pin["file_digests"],
        "shards": pin["shards"],
        "config": {
            "model_type": getattr(config, "model_type", None),
            "architectures": getattr(config, "architectures", None),
            "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
        },
        "dependency_versions": versions,
        "torch_dtype_default": str(torch.get_default_dtype()),
    }
