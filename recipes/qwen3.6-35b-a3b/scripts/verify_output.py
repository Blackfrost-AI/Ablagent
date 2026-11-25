#!/usr/bin/env python3
"""Independently verify a Qwen3.6-35B-A3B alpha-1.0 x3 output."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from locked_source import verify_source


REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
CHECKPOINT_DIGEST = "sha256:88760187ccb2560500b1d51f0e75610658c956be010b4477bc11bb8265c7413b"
S2_DIRECTION_SHA256 = "fc2e01feace4215a2c27bc84bc7a7c6d576013ab358671870db7357ccc07d9b2"
TARGET_SET_SHA256 = "7e167ee01961a73f1ff67b69c7ac92919e03acd7bfca8fe717a066047b62383f"
TARGET = re.compile(
    r"^model\.language_model\.layers\.(\d+)\."
    r"(self_attn\.o_proj|linear_attn\.out_proj)\.weight$"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    step = 8 * 1024 * 1024
    for offset in range(0, raw.numel(), step):
        digest.update(raw[offset : offset + step].numpy().tobytes())
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, dict]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(key)
    result: dict[str, dict] = {}
    for number, (shard, keys) in enumerate(sorted(by_shard.items()), 1):
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for key in sorted(keys):
                value = handle.get_tensor(key)
                result[key] = {
                    "sha256": tensor_digest(value),
                    "dtype": str(value.dtype).replace("torch.", ""),
                    "shape": list(value.shape),
                    "norm": value.float().norm().item(),
                    "shard": shard,
                }
                del value
        print(f"INVENTORY {number}/{len(by_shard)} {shard}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-record", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    require(source != output, "source and output resolve to the same path")
    require(source.is_dir() and output.is_dir(), "source or output directory is missing")
    root = Path(__file__).resolve().parents[1]
    verified_source = verify_source(source, root / "assets" / "source-lock.json")
    require(verified_source["checkpoint_digest"] == CHECKPOINT_DIGEST, "verified source identity changed")
    build = json.loads(args.build_record.read_text())
    expected_build = {
        "revision": REVISION,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "s2_direction_sha256": S2_DIRECTION_SHA256,
        "alpha": 1.0,
        "passes": 3,
        "target_set_sha256": TARGET_SET_SHA256,
        "source_tensor_count": 1045,
        "candidate_tensor_count": 1045,
        "source_mtp_count": 19,
        "candidate_mtp_count": 19,
        "mtp_byte_identical": True,
        "source_immutable": True,
    }
    for key, wanted in expected_build.items():
        require(build.get(key) == wanted, f"build record mismatch: {key}")

    source_inventory = inventory(source)
    output_inventory = inventory(output)
    require(set(source_inventory) == set(output_inventory), "tensor key sets differ")

    targets = []
    surfaces = {"linear_attn.out_proj": 0, "self_attn.o_proj": 0}
    for key in sorted(source_inventory):
        match = TARGET.fullmatch(key)
        if match and 2 <= int(match.group(1)) <= 39:
            targets.append(key)
            surfaces[match.group(2)] += 1
    require(len(targets) == 38, f"expected 38 targets, found {len(targets)}")
    require(
        surfaces == {"linear_attn.out_proj": 28, "self_attn.o_proj": 10},
        f"target surface mismatch: {surfaces}",
    )
    target_set = hashlib.sha256(("\n".join(targets) + "\n").encode()).hexdigest()
    require(target_set == TARGET_SET_SHA256, "target-set digest mismatch")

    target_names = set(targets)
    changed_targets = []
    changed_nontargets = []
    nonfinite_targets = []
    norm_ratios = []
    for key in sorted(source_inventory):
        before = source_inventory[key]
        after = output_inventory[key]
        same = (
            before["sha256"] == after["sha256"]
            and before["dtype"] == after["dtype"]
            and before["shape"] == after["shape"]
        )
        if key in target_names:
            if not same:
                changed_targets.append(key)
            if after["dtype"] != "bfloat16":
                nonfinite_targets.append(key + ":unexpected-dtype")
            ratio = after["norm"] / max(before["norm"], 1.0e-12)
            norm_ratios.append(ratio)
            if not 0.998 <= ratio <= 1.002:
                nonfinite_targets.append(key + ":norm-restore-failed")
        elif not same:
            changed_nontargets.append(key)

    mtp_keys = sorted(key for key in source_inventory if key.startswith("mtp."))
    mtp_changed = [key for key in mtp_keys if source_inventory[key] != output_inventory[key]]
    require(len(mtp_keys) == 19, f"expected 19 MTP tensors, found {len(mtp_keys)}")
    require(len(changed_targets) == 38, f"changed target count is {len(changed_targets)}")
    require(not changed_nontargets, f"non-target tensors changed: {changed_nontargets[:8]}")
    require(not mtp_changed, f"MTP tensors changed: {mtp_changed[:8]}")
    require(not nonfinite_targets, f"target validation failures: {nonfinite_targets[:8]}")

    result = {
        "schema": "blackfrost.qwen36.output-verification.v1",
        "passed": True,
        "source_tensors": len(source_inventory),
        "output_tensors": len(output_inventory),
        "target_set_sha256": target_set,
        "changed_targets": len(changed_targets),
        "changed_nontargets": len(changed_nontargets),
        "surface_counts": surfaces,
        "mtp_tensors": len(mtp_keys),
        "mtp_changed": len(mtp_changed),
        "target_norm_ratio_min": min(norm_ratios),
        "target_norm_ratio_max": max(norm_ratios),
        "build_record": expected_build,
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

