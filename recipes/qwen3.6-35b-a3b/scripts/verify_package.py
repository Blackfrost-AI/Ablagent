#!/usr/bin/env python3
"""Fail-closed verification for the Qwen3.6-35B-A3B alpha-1.0 x3 package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import re
import struct


ROOT = pathlib.Path(__file__).resolve().parents[1]
RECIPE_ID = "qwen3.6-35b-a3b-stock-bf16-a1p0x3"
SOURCE_REPO = "Qwen/Qwen3.6-35B-A3B"
SOURCE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
CHECKPOINT_DIGEST = "sha256:88760187ccb2560500b1d51f0e75610658c956be010b4477bc11bb8265c7413b"
SOURCE_LOCK_SHA256 = "07cc4141def9c9ad77251badd690efa5542db478ba78ce89417ea3e233d437e7"
BANK_SHA256 = "902ac2472d572ff5724e0b70012efe687c87d19c655f09cd528d8a9bca60f7a8"
HELDOUT_SHA256 = "8b9c4695422caaac2854c250c3e874f263b2ac820c08c5c3ac8ef9ed2d237309"
DIRECTION_SHA256 = "fc2e01feace4215a2c27bc84bc7a7c6d576013ab358671870db7357ccc07d9b2"
DIRECTION_BYTES = 167_128
ACTIVATIONS_SHA256 = "2ef775b19e30a90b73116baba69942549ef176a232d7b38062b6acde3fa95d00"
ACTIVATIONS_BYTES = 10_486_000
TARGET_SET_SHA256 = "7e167ee01961a73f1ff67b69c7ac92919e03acd7bfca8fe717a066047b62383f"
IMAGE_DIGEST = "sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
RUNTIME_EXCLUSIONS = {"SHA256SUMS", "config.local.json", ".DS_Store"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def value_at(document, dotted: str):
    value = document
    for part in dotted.split("."):
        value = value[part]
    return value


def verify_lock() -> dict:
    lock = read_json(ROOT / "recipe.lock.json")
    expected = {
        "schema": "blackfrost.dwm.locked-method.v1",
        "recipe_id": RECIPE_ID,
        "version": "1.0.0",
        "source.repo": SOURCE_REPO,
        "source.revision": SOURCE_REVISION,
        "source.precision": "bfloat16",
        "source.quantized": False,
        "source.checkpoint_digest": CHECKPOINT_DIGEST,
        "source.source_lock_sha256": SOURCE_LOCK_SHA256,
        "source.payload_files": 40,
        "source.weight_shards": 26,
        "source.payload_bytes": 71926865825,
        "source.indexed_tensors": 1045,
        "directions.prompt_bank_sha256": BANK_SHA256,
        "directions.harmful_count": 32,
        "directions.safe_count": 32,
        "directions.thinking": True,
        "directions.system_prompt": None,
        "directions.layers": 40,
        "directions.hidden_size": 2048,
        "directions.tensor_count": 40,
        "directions.initial_direction_sha256": DIRECTION_SHA256,
        "directions.initial_direction_bytes": DIRECTION_BYTES,
        "directions.initial_raw_activations_sha256": ACTIVATIONS_SHA256,
        "directions.initial_raw_activations_bytes": ACTIVATIONS_BYTES,
        "directions.fresh_recapture_before_each_pass": True,
        "directions.round_one_minimum_cosine_to_initial_l39": 0.99,
        "operation.alpha": 1.0,
        "operation.passes": 3,
        "operation.rank": 1,
        "operation.skip_early": 2,
        "operation.first_layer": 2,
        "operation.last_layer": 39,
        "operation.targets.total": 38,
        "operation.targets.target_set_sha256": TARGET_SET_SHA256,
        "operation.mtp.expected_tensors": 19,
        "operation.mtp.edit_surface_included": False,
        "operation.mtp.require_all_byte_identical": True,
        "measurement.heldout_sha256": HELDOUT_SHA256,
        "measurement.prompt_count": 8,
        "recorded_candidate.structural_gate_passed": True,
        "recorded_candidate.automatic_distribution_gate_passed": True,
        "recorded_candidate.served_smoke_passed": True,
        "recorded_candidate.operator_hitl_approved": True,
        "qualification.release_qualified": False,
        "qualification.weights_included": False,
        "reproducibility.byte_identical_output_promised": False,
    }
    for key, wanted in expected.items():
        actual = value_at(lock, key)
        require(actual == wanted, f"recipe lock mismatch: {key}: {actual!r} != {wanted!r}")
    require(
        lock["operation"]["norm_restoration"]
        == "one global Frobenius scalar per matrix after every pass",
        "norm restoration changed",
    )
    require(lock["operation"]["targets"]["linear_attn.out_proj"] == 28, "linear target count changed")
    require(lock["operation"]["targets"]["self_attn.o_proj"] == 10, "full-attention target count changed")
    return lock


def verify_jsonl() -> dict:
    bank = ROOT / "assets" / "v1_corpus_hard32.jsonl"
    heldout = ROOT / "assets" / "heldout_distribution.jsonl"
    require(sha256(bank) == BANK_SHA256, "pair-bank hash mismatch")
    require(sha256(heldout) == HELDOUT_SHA256, "held-out hash mismatch")
    rows = [json.loads(line) for line in bank.read_text().splitlines() if line.strip()]
    harmful = [row for row in rows if row.get("label") == "harmful"]
    safe = [row for row in rows if row.get("label") == "safe"]
    require(len(rows) == 64 and len(harmful) == len(safe) == 32, "expected 32+32 bank")
    require(all(isinstance(row.get("prompt"), str) and row["prompt"] for row in rows), "empty prompt")
    heldout_rows = [json.loads(line) for line in heldout.read_text().splitlines() if line.strip()]
    require(len(heldout_rows) == 8, "expected eight held-out prompts")
    require(len({row["id"] for row in heldout_rows}) == 8, "duplicate held-out id")
    return {"bank_rows": 64, "harmful": 32, "safe": 32, "heldout": 8}


def safetensors_header(path: pathlib.Path) -> dict:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size))


def verify_assets() -> dict:
    direction = ROOT / "assets" / "directions_on.safetensors"
    activations = ROOT / "assets" / "capture-activations.safetensors"
    require(direction.stat().st_size == DIRECTION_BYTES, "direction size mismatch")
    require(sha256(direction) == DIRECTION_SHA256, "direction hash mismatch")
    require(activations.stat().st_size == ACTIVATIONS_BYTES, "activation size mismatch")
    require(sha256(activations) == ACTIVATIONS_SHA256, "activation hash mismatch")
    d_header = safetensors_header(direction)
    d_keys = {key for key in d_header if key != "__metadata__"}
    require(d_keys == {f"layer_{i}" for i in range(40)}, "direction key set mismatch")
    for key in d_keys:
        require(d_header[key]["dtype"] == "BF16", f"direction dtype mismatch: {key}")
        require(d_header[key]["shape"] == [2048], f"direction shape mismatch: {key}")
    a_header = safetensors_header(activations)
    require(sorted(key for key in a_header if key != "__metadata__") == ["harmful", "safe"], "activation keys mismatch")
    for key in ("harmful", "safe"):
        require(a_header[key]["dtype"] == "BF16", f"activation dtype mismatch: {key}")
        require(a_header[key]["shape"] == [32, 40, 2048], f"activation shape mismatch: {key}")
    return {"direction_tensors": 40, "direction_shape": [2048], "activation_shape": [32, 40, 2048]}


def assignments(path: pathlib.Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return result


def verify_scripts() -> dict:
    run_path = ROOT / "scripts" / "run_locked.py"
    capture_path = ROOT / "scripts" / "capture_directions.py"
    measure_path = ROOT / "scripts" / "measure_compare.py"
    source_path = ROOT / "scripts" / "locked_source.py"
    run_text = run_path.read_text()
    capture_text = capture_path.read_text()
    measure_text = measure_path.read_text()
    source_text = source_path.read_text()
    for text in (run_text, capture_text, measure_text, source_text):
        ast.parse(text)
    run = assignments(run_path)
    capture = assignments(capture_path)
    measure = assignments(measure_path)
    source = assignments(source_path)
    expected_run = {
        "REPO_ID": SOURCE_REPO,
        "REVISION": SOURCE_REVISION,
        "CHECKPOINT_DIGEST": CHECKPOINT_DIGEST,
        "BANK_SHA256": BANK_SHA256,
        "S2_DIRECTION_SHA256": DIRECTION_SHA256,
        "ALPHA": 1.0,
        "PASSES": 3,
        "SKIP_EARLY": 2,
        "EXPECTED_LAYERS": 40,
        "EXPECTED_HIDDEN_SIZE": 2048,
        "EXPECTED_TARGETS": 38,
        "EXPECTED_MTP_TENSORS": 19,
    }
    for key, wanted in expected_run.items():
        require(run.get(key) == wanted, f"executor constant mismatch: {key}")
    for key, wanted in {
        "REPO_ID": SOURCE_REPO,
        "REVISION": SOURCE_REVISION,
        "CHECKPOINT_DIGEST": CHECKPOINT_DIGEST,
        "BANK_SHA256": BANK_SHA256,
        "EXPECTED_LAYERS": 40,
        "EXPECTED_HIDDEN_SIZE": 2048,
    }.items():
        require(capture.get(key) == wanted, f"capture constant mismatch: {key}")
    for key, wanted in {
        "BANK_SHA256": BANK_SHA256,
        "HELDOUT_SHA256": HELDOUT_SHA256,
        "BASELINE_ACTIVATIONS_SHA256": ACTIVATIONS_SHA256,
        "BASELINE_DIRECTIONS_SHA256": DIRECTION_SHA256,
    }.items():
        require(measure.get(key) == wanted, f"measurement constant mismatch: {key}")
    for key, wanted in {
        "SOURCE_REPO": SOURCE_REPO,
        "SOURCE_REVISION": SOURCE_REVISION,
        "CHECKPOINT_DIGEST": CHECKPOINT_DIGEST,
        "SOURCE_LOCK_SHA256": SOURCE_LOCK_SHA256,
    }.items():
        require(source.get(key) == wanted, f"source verifier constant mismatch: {key}")
    for forbidden in ("--alpha", "--passes", "--skip-early", "--bank", "--s2-directions", "--revision", "--checkpoint-digest"):
        require(f'add_argument("{forbidden}"' not in run_text, f"executor exposes {forbidden}")
    for forbidden in ("--bank", "--revision", "--checkpoint-digest", "--expected-bank-sha256", "--expected-layers", "--expected-hidden-size"):
        require(f'add_argument("{forbidden}"' not in capture_text, f"capture exposes {forbidden}")
    require("for pass_index in range(1, PASSES + 1)" in run_text, "fresh pass loop changed")
    require("capture(model, tokenizer, harmful" in run_text, "harmful recapture missing")
    require("capture(model, tokenizer, safe" in run_text, "safe recapture missing")
    require("edited = weight - ALPHA * r[:, None] * (r @ weight)[None, :]" in run_text, "operator changed")
    require("edited *= before_norm / edited.norm().clamp_min(1.0e-12)" in run_text, "norm restoration changed")
    require("verify_source(args.source, source_lock)" in run_text, "executor source verification missing")
    require("verify_source(args.model, source_lock)" in capture_text, "capture source verification missing")
    require("verify_source(args.base, source_lock)" in measure_text, "measurement source verification missing")
    return {"alpha": 1.0, "passes": 3, "method_overrides": False}


def verify_evidence() -> dict:
    capture = read_json(ROOT / "evidence" / "capture-receipt.json")
    result = read_json(ROOT / "evidence" / "verification-summary.json")
    hitl = read_json(ROOT / "evidence" / "hitl-receipt.json")
    require(capture["captured_verified"] is True, "capture is not verified")
    require(capture["direction"]["sha256"] == DIRECTION_SHA256, "capture direction mismatch")
    require(result["automatic_gate"]["passed"] is True, "recorded automatic gate is not a pass")
    require(result["structure"] == {"changed_nontargets": 0, "changed_targets": 38, "missed_targets": 0, "mtp_exact": 19, "mtp_total": 19, "nonfinite_targets": 0}, "recorded structure mismatch")
    require(result["qualification_boundary"]["release_qualified"] is False, "release boundary changed")
    require(result["qualification_boundary"]["missing_pipeline_stages"] == ["S1", "S6", "S7", "SR"], "missing-stage boundary changed")
    require(hitl["status"] == "hitl_active" and hitl["does_not_grant_release_qualification"] is True, "HITL boundary changed")
    return {"capture": "verified", "automatic_gate": "pass", "hitl": "approved", "release_qualified": False}


def verify_harness_plan() -> dict:
    plan = read_json(ROOT / "plans" / "ablagent-read-only-audit.json")
    require(plan["name"] == "qwen36-35b-a3b-dwm-a1p0x3-audit", "unexpected plan name")
    require(len(plan["stages"]) == 8, "unexpected plan stage count")
    allowed = {"read_file", "grep", "glob"}
    for stage in plan["stages"]:
        require(set(stage["allowedTools"]).issubset(allowed), f"write-capable tool in {stage['id']}")
        for relative in stage.get("exactPaths", []):
            pure = pathlib.PurePosixPath(relative)
            require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe exact path: {relative}")
            require((ROOT / relative).is_file(), f"missing exact path: {relative}")
    require(plan["stages"][-1]["allowedTools"] == [], "synthesis stage must use no tools")
    return {"name": plan["name"], "stages": 8, "tools": sorted(allowed)}


def verify_source(path: pathlib.Path) -> dict:
    require(path.is_dir(), f"source directory not found: {path}")
    source_lock_path = ROOT / "assets" / "source-lock.json"
    require(sha256(source_lock_path) == SOURCE_LOCK_SHA256, "source-lock hash mismatch")
    source_lock = read_json(source_lock_path)
    require(source_lock["repo"] == SOURCE_REPO, "source-lock repository mismatch")
    require(source_lock["revision"] == SOURCE_REVISION, "source-lock revision mismatch")
    require(source_lock["checkpoint_digest"] == CHECKPOINT_DIGEST, "source-lock checkpoint mismatch")
    require(len(source_lock["files"]) == 40, "source-lock file count mismatch")
    observed = []
    for item in source_lock["files"]:
        target = path / item["path"]
        require(target.is_file() and not target.is_symlink(), f"source file missing: {item['path']}")
        require(target.stat().st_size == item["size"], f"source size mismatch: {item['path']}")
        actual = sha256(target)
        require(actual == item["sha256"], f"source hash mismatch: {item['path']}")
        observed.append({"path": item["path"], "digest": "sha256:" + actual, "size": item["size"]})
    canonical = "sha256:" + hashlib.sha256(json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(canonical == CHECKPOINT_DIGEST, "canonical source identity mismatch")
    config = read_json(path / "config.json")
    text = config["text_config"]
    require(text["num_hidden_layers"] == 40 and text["hidden_size"] == 2048, "source geometry mismatch")
    index = read_json(path / "model.safetensors.index.json")
    weights = index["weight_map"]
    require(len(weights) == 1045, "source tensor count mismatch")
    require(len(set(weights.values())) == 26, "source shard count mismatch")
    mtp = [key for key in weights if key.startswith("mtp.")]
    require(len(mtp) == 19, "source MTP count mismatch")
    target = re.compile(r"^model\.language_model\.layers\.(\d+)\.(self_attn\.o_proj|linear_attn\.out_proj)\.weight$")
    targets = [key for key in sorted(weights) if (match := target.fullmatch(key)) and 2 <= int(match.group(1)) <= 39]
    require(len(targets) == 38, "source target count mismatch")
    digest = hashlib.sha256(("\n".join(targets) + "\n").encode()).hexdigest()
    require(digest == TARGET_SET_SHA256, "source target-set digest mismatch")
    return {"checkpoint_digest": canonical, "files": 40, "shards": 26, "tensors": 1045, "targets": 38, "mtp": 19}


def verify_manifest() -> dict:
    path = ROOT / "SHA256SUMS"
    require(path.is_file() and not path.is_symlink(), "SHA256SUMS is missing")
    seen = set()
    for line in path.read_text().splitlines():
        if not line:
            continue
        wanted, relative = line.split("  ", 1)
        pure = pathlib.PurePosixPath(relative)
        require(re.fullmatch(r"[0-9a-f]{64}", wanted) is not None, f"invalid hash: {relative}")
        require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe manifest path: {relative}")
        require(relative not in seen, f"duplicate manifest path: {relative}")
        seen.add(relative)
        target = ROOT / relative
        require(target.is_file() and not target.is_symlink(), f"invalid manifest target: {relative}")
        require(sha256(target) == wanted, f"manifest mismatch: {relative}")
    expected = set()
    for target in ROOT.rglob("*"):
        if not target.is_file() or target.is_symlink():
            continue
        relative = target.relative_to(ROOT).as_posix()
        parts = pathlib.PurePosixPath(relative).parts
        if relative in RUNTIME_EXCLUSIONS or ".git" in parts or "__pycache__" in parts:
            continue
        if relative.endswith(".pyc") or relative.startswith(("artifacts/", "output/", "run/")):
            continue
        expected.add(relative)
    require(seen == expected, f"manifest coverage mismatch: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    return {"files": len(seen), "complete": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path)
    args = parser.parse_args()
    for path in ROOT.rglob("*"):
        require(not path.is_symlink(), f"symlink is not allowed: {path.relative_to(ROOT)}")
    lock = verify_lock()
    environment = read_json(ROOT / "environment.lock.json")
    require(environment["capture_edit_and_serve_image_digest"] == IMAGE_DIGEST, "environment image digest mismatch")
    require((ROOT / "VERSION").read_text().strip() == "1.0.0", "VERSION mismatch")
    result = {
        "passed": True,
        "lock": lock["recipe_id"],
        "alpha": lock["operation"]["alpha"],
        "passes": lock["operation"]["passes"],
        "inputs": verify_jsonl(),
        "assets": verify_assets(),
        "scripts": verify_scripts(),
        "evidence": verify_evidence(),
        "harness": verify_harness_plan(),
        "manifest": verify_manifest(),
        "source": verify_source(args.source.resolve()) if args.source else "not supplied",
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

