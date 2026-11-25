#!/usr/bin/env python3
"""Build the stock Qwen3.6-35B alpha-1.0 x3 BF16 DWM master.

The Transformers wrapper omits the upstream ``mtp.*`` payload.  The final
checkpoint is therefore produced by copying the immutable source package and
transplanting only the declared attention-output tensors into their original
shards.  This preserves every non-target tensor, including MTP, exactly.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from capture_directions import capture, direction_stats, layers_of, load_bank
from locked_source import verify_source


REPO_ID = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
CHECKPOINT_DIGEST = "sha256:88760187ccb2560500b1d51f0e75610658c956be010b4477bc11bb8265c7413b"
BANK_SHA256 = "902ac2472d572ff5724e0b70012efe687c87d19c655f09cd528d8a9bca60f7a8"
S2_DIRECTION_SHA256 = "fc2e01feace4215a2c27bc84bc7a7c6d576013ab358671870db7357ccc07d9b2"
ALPHA = 1.0
PASSES = 3
SKIP_EARLY = 2
EXPECTED_LAYERS = 40
EXPECTED_HIDDEN_SIZE = 2048
EXPECTED_TARGETS = 38
EXPECTED_MTP_TENSORS = 19


TARGET = re.compile(
    r"^model\.language_model\.layers\.(\d+)\."
    r"(self_attn\.o_proj|linear_attn\.out_proj)$"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    """Hash tensor content in bounded chunks without one giant bytes copy."""
    raw = value.detach().contiguous().cpu().view(torch.uint8).reshape(-1)
    h = hashlib.sha256()
    step = 8 * 1024 * 1024
    for offset in range(0, raw.numel(), step):
        h.update(raw[offset:offset + step].numpy().tobytes())
    return h.hexdigest()


def tensor_inventory(model_dir: Path) -> dict[str, dict]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(key)
    result: dict[str, dict] = {}
    for number, shard in enumerate(sorted(by_shard), 1):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for key in sorted(by_shard[shard]):
                value = handle.get_tensor(key)
                result[key] = {
                    "sha256": tensor_digest(value),
                    "dtype": str(value.dtype).replace("torch.", ""),
                    "shape": list(value.shape),
                    "shard": shard,
                }
                del value
        print(f"INVENTORY {number}/{len(by_shard)} {shard}", flush=True)
    return result


def copy_source(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    shutil.copytree(source, output, symlinks=False, ignore=shutil.ignore_patterns(".cache"))


def transplant(output: Path, target_values: dict[str, torch.Tensor]) -> list[str]:
    index = json.loads((output / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for key in target_values:
        by_shard.setdefault(index["weight_map"][key], []).append(key)
    rewritten = []
    for shard, keys in sorted(by_shard.items()):
        path = output / shard
        metadata = None
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        tensors = load_file(path, device="cpu")
        for key in keys:
            tensors[key] = target_values[key].contiguous()
        tmp = path.with_name(path.name + ".tmp")
        save_file(tensors, tmp, metadata=metadata)
        os.replace(tmp, path)
        rewritten.append(shard)
        print(f"TRANSPLANT shard={shard} targets={len(keys)}", flush=True)
        del tensors
        gc.collect()
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bank = root / "assets" / "v1_corpus_hard32.jsonl"
    s2_directions = root / "assets" / "directions_on.safetensors"
    source_lock = root / "assets" / "source-lock.json"
    if sha256(bank) != BANK_SHA256:
        raise ValueError("pair-bank hash mismatch")
    if sha256(s2_directions) != S2_DIRECTION_SHA256:
        raise ValueError("pinned S2 direction hash mismatch")
    verified_source = verify_source(args.source, source_lock)
    if verified_source["checkpoint_digest"] != CHECKPOINT_DIGEST:
        raise RuntimeError("verified source identity changed")
    args.evidence.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    harmful, safe = load_bank(bank, BANK_SHA256)

    print("INVENTORY immutable source tensors", flush=True)
    source_inventory = tensor_inventory(args.source)
    mtp_keys = sorted(key for key in source_inventory if key.startswith("mtp."))
    if len(mtp_keys) != EXPECTED_MTP_TENSORS:
        raise RuntimeError(f"expected {EXPECTED_MTP_TENSORS} MTP tensors, found {len(mtp_keys)}")

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.source,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device).eval()
    if len(layers_of(model)) != EXPECTED_LAYERS or int(model.config.text_config.hidden_size) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("unexpected Qwen3.6-35B language architecture")

    modules = []
    surface_counts = {"linear_attn.out_proj": 0, "self_attn.o_proj": 0}
    for name, module in model.named_modules():
        match = TARGET.match(name)
        if match and int(match.group(1)) >= SKIP_EARLY:
            surface = match.group(2)
            modules.append((name, int(match.group(1)), surface, module))
            surface_counts[surface] += 1
    modules.sort(key=lambda item: item[0])
    if len(modules) != EXPECTED_TARGETS or surface_counts != {
        "linear_attn.out_proj": 28, "self_attn.o_proj": 10
    }:
        raise RuntimeError(f"unexpected attention target set: {len(modules)} {surface_counts}")
    target_keys = [name + ".weight" for name, _, _, _ in modules]
    if any(key not in source_inventory for key in target_keys):
        raise RuntimeError("one or more target keys are absent from the source index")
    target_set_sha256 = hashlib.sha256(("\n".join(target_keys) + "\n").encode()).hexdigest()
    originals = {name: module.weight.detach().cpu().clone() for name, _, _, module in modules}

    pinned = load_file(s2_directions)
    pinned_dirs = torch.stack([pinned[f"layer_{i}"].float() for i in range(EXPECTED_LAYERS)])
    traces = []
    pass_directions = []
    for pass_index in range(1, PASSES + 1):
        t0 = time.monotonic()
        harmful_acts = capture(model, tokenizer, harmful, device, f"pass{pass_index}-harmful")
        safe_acts = capture(model, tokenizer, safe, device, f"pass{pass_index}-safe")
        directions, report = direction_stats(harmful_acts, safe_acts)
        cosine_to_s2 = torch.nn.functional.cosine_similarity(directions, pinned_dirs, dim=1)
        if pass_index == 1 and cosine_to_s2[39].item() < 0.99:
            raise RuntimeError(
                f"round-one L39 direction did not reproduce S2: {cosine_to_s2[39].item()}"
            )
        save_file(
            {f"layer_{i}": directions[i].to(torch.bfloat16) for i in range(40)},
            args.evidence / f"directions-pass{pass_index}.safetensors",
            metadata={"pass": str(pass_index), "thinking": "on"},
        )
        with torch.no_grad():
            for _name, layer, _surface, module in modules:
                r = directions[layer].to(device)
                weight = module.weight.detach().float()
                before_norm = weight.norm()
                edited = weight - ALPHA * r[:, None] * (r @ weight)[None, :]
                edited *= before_norm / edited.norm().clamp_min(1.0e-12)
                module.weight.copy_(edited.to(module.weight.dtype))
        trace = {
            "pass": pass_index,
            "selected_layer": report["selected_layer"],
            "separation_at_selected": report["separation_at_selected"],
            "mean_separation_after_skip": sum(report["separation_by_layer"][SKIP_EARLY:]) /
                len(report["separation_by_layer"][SKIP_EARLY:]),
            "split_half_cosine_selected": report["split_half_cosine_selected"],
            "s2_cosine_selected_l39": cosine_to_s2[39].item(),
            "s2_cosine_mean": cosine_to_s2.mean().item(),
            "elapsed_seconds": time.monotonic() - t0,
        }
        if not all(torch.isfinite(module.weight).all() for _, _, _, module in modules):
            raise RuntimeError(f"non-finite target after pass {pass_index}")
        traces.append(trace)
        pass_directions.append(directions.cpu())
        print("PASS " + json.dumps(trace, sort_keys=True), flush=True)
        del harmful_acts, safe_acts, directions
        torch.cuda.empty_cache()

    target_metrics = []
    target_values: dict[str, torch.Tensor] = {}
    last_dirs = pass_directions[-1]
    for name, layer, surface, module in modules:
        key = name + ".weight"
        before = originals[name].float()
        after = module.weight.detach().float().cpu()
        r = last_dirs[layer].float()
        projection_before = r @ before
        projection_after = r @ after
        target_metrics.append({
            "name": key,
            "layer": layer,
            "surface": surface,
            "relative_l2_change": ((after - before).norm() / before.norm().clamp_min(1e-12)).item(),
            "norm_ratio": (after.norm() / before.norm().clamp_min(1e-12)).item(),
            "projection_before_norm": projection_before.norm().item(),
            "projection_after_norm": projection_after.norm().item(),
            "projection_residual_ratio": (
                projection_after.norm() / projection_before.norm().clamp_min(1e-12)
            ).item(),
        })
        target_values[key] = module.weight.detach().cpu().clone()

    del model, originals, pass_directions
    gc.collect(); torch.cuda.empty_cache()
    copy_source(args.source, args.output)
    rewritten = transplant(args.output, target_values)
    del target_values
    gc.collect()

    print("INVENTORY persisted candidate tensors", flush=True)
    candidate_inventory = tensor_inventory(args.output)
    source_keys, candidate_keys = set(source_inventory), set(candidate_inventory)
    if source_keys != candidate_keys:
        raise RuntimeError("candidate tensor key set differs from source")
    changed_targets = [k for k in target_keys
                       if source_inventory[k]["sha256"] != candidate_inventory[k]["sha256"]]
    unchanged_targets = [k for k in target_keys
                         if source_inventory[k]["sha256"] == candidate_inventory[k]["sha256"]]
    changed_nontargets = [
        k for k in sorted(source_keys - set(target_keys))
        if source_inventory[k]["sha256"] != candidate_inventory[k]["sha256"]
        or source_inventory[k]["dtype"] != candidate_inventory[k]["dtype"]
        or source_inventory[k]["shape"] != candidate_inventory[k]["shape"]
    ]
    mtp_changed = [k for k in mtp_keys if source_inventory[k] != candidate_inventory[k]]
    nonfinite = 0
    for key in target_keys:
        shard = candidate_inventory[key]["shard"]
        with safe_open(args.output / shard, framework="pt", device="cpu") as handle:
            nonfinite += int(not torch.isfinite(handle.get_tensor(key).float()).all())

    structural = {
        "schema": "ablagent.qwen36.structural.v1",
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "model_revision": REVISION,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "source_modified": False,
        "alpha": ALPHA,
        "passes": PASSES,
        "nominal_retention": (1 - ALPHA) ** PASSES,
        "regime": "alpha_eq_1",
        "skip_early": SKIP_EARLY,
        "surface_counts": surface_counts,
        "eligible_count": len(target_keys),
        "selected_count": len(target_keys),
        "coverage_ratio": 1.0,
        "target_set_sha256": target_set_sha256,
        "changed_target_count": len(changed_targets),
        "missed_target_count": len(unchanged_targets),
        "changed_nontarget_count": len(changed_nontargets),
        "changed_nontargets": changed_nontargets,
        "nonfinite_target_count": nonfinite,
        "norm_restoration": "single global scalar per matrix per pass",
        "arithmetic_precision": "float32",
        "serialization_precision": "bfloat16",
        "rewritten_shards": rewritten,
        "source_tensor_count": len(source_inventory),
        "candidate_tensor_count": len(candidate_inventory),
        "mtp": {
            "source_count": len(mtp_keys),
            "candidate_count": sum(k.startswith("mtp.") for k in candidate_inventory),
            "restored_count": len(mtp_keys) - len(mtp_changed),
            "changed_common_count": len(mtp_changed),
            "changed_keys": mtp_changed,
            "edit_surface_included": False,
            "package_complete": len(mtp_changed) == 0,
        },
        "iteration_trace": traces,
        "targets": target_metrics,
    }
    (args.evidence / "structural-metrics.json").write_text(
        json.dumps(structural, sort_keys=True, indent=2) + "\n"
    )
    build = {
        "schema": "ablagent.qwen36.build.v1",
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "revision": REVISION,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "bank": str(bank.resolve()),
        "bank_sha256": sha256(bank),
        "s2_direction_sha256": sha256(s2_directions),
        "alpha": ALPHA,
        "passes": PASSES,
        "target_set_sha256": target_set_sha256,
        "source_tensor_count": len(source_inventory),
        "candidate_tensor_count": len(candidate_inventory),
        "source_mtp_count": len(mtp_keys),
        "candidate_mtp_count": sum(k.startswith("mtp.") for k in candidate_inventory),
        "mtp_byte_identical": len(mtp_changed) == 0,
        "source_immutable": True,
        "save_method": "copy immutable snapshot then attention-target-only safetensors transplant",
    }
    (args.evidence / "build-record.json").write_text(
        json.dumps(build, sort_keys=True, indent=2) + "\n"
    )
    summary = {
        "targets": len(target_keys),
        "surface_counts": surface_counts,
        "changed_targets": len(changed_targets),
        "changed_nontargets": len(changed_nontargets),
        "mtp_count": len(mtp_keys),
        "mtp_changed": len(mtp_changed),
        "nonfinite_targets": nonfinite,
        "rounds": traces,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if unchanged_targets or changed_nontargets or mtp_changed or nonfinite:
        raise RuntimeError("structural gate failed")
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
