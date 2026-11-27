#!/usr/bin/env python3
"""Fail-closed verification for the Inkling-2-Small alpha-3.8 recipe package.

Checks, in order:
  1. recipe.lock.json against the pinned method constants below.
  2. PROVENANCE.json redaction table integrity (original != vendored digests).
  3. Vendored kit files against the HF-published manifests/kit-sha256.json,
     resolving the 7 documented endpoint redactions through PROVENANCE.json
     and exempting the manifest's self-referential entry.
  4. MANIFEST.sha256 (kit-scoped) with the same redaction resolution; entries
     for non-vendored files (weights, banner, model card) are reported as
     absent-by-design, never as failures.
  5. Corpus / direction / battery digests against the lock.
  6. Optional --raw-capture <path>: exact bytes+SHA-256 of the fetch-only
     89 MB capture tensor (audit input; not required for execution).
  7. Private-material scan over package text files (fails on operator
     endpoints, home paths, key material, tokens).

The raw capture is fetch-only (assets/raw-capture.asset.json); its absence is
reported honestly and does not fail package integrity. No network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

SOURCE_REPO = "Blackfrost-Research/Inkling-2-Small-Derisked"
SOURCE_REVISION = "c75bc4f131c2db9aba463e0c9ba6aead92ea4428"
UPSTREAM_REPO = "thinkingmachines/Inkling-2-Small"
UPSTREAM_REVISION = "885d78bfca61a7e9a498e33cfa7ad35850533b8c"
HARD32_SHA256 = "902ac2472d572ff5724e0b70012efe687c87d19c655f09cd528d8a9bca60f7a8"
DIRECTIONS_SHA256 = "95a6c68f23774cba65bee0aacea9fd252965a78b06c095020d5560fc6b8e8a47"
BATTERY_SHA256 = "b591b177376c8b83840a2bac14653c4a47fd4c13cd92ca1c0393728d470daaaf"
RAW_CAPTURE_BYTES = 89_421_956
RAW_CAPTURE_SHA256 = "63a559079f9f39da84da731561c63d879c9c69de36f5ff8121a253bb0468cd88"
MTP_SHA256 = "3b265640ec84ba66e928a2844c4c022c94be731336b0f63755a20a247bfd2a36"
SERVE_IMAGE = "vllm/vllm-openai@sha256:e72944ef5ab9d1b91228d066117595847e94bf95321018a2119e70e8ba643ff6"
SELF_MANIFEST_ENTRY = "manifests/kit-sha256.json"
# Package wrapper files intentionally replace HF release presentation files.
WRAPPER_REPLACED = {"README.md"}
NOT_VENDORED_BY_DESIGN = {
    "DWM_RECIPE/evidence/s2-capture/raw-capture.npz",
    "assets/BLACKFROST-AI-BANNER.png",
}
# HF repo-root MANIFEST.sha256 also covers release presentation and model
# artifacts that are deliberately not vendored. Anything else must be present.
NOT_VENDORED_ROOT = NOT_VENDORED_BY_DESIGN | {
    ".gitattributes", "ASSETS/BLACKFROST-AI-BANNER.png",
    "chat_template.jinja", "config.json", "generation_config.json",
    "hf_quant_config.json", "model.safetensors.index.json", "mtp.safetensors",
    "processor_config.json", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json",
} | {f"model-{i:05d}-of-00024.safetensors" for i in range(1, 25)}
R443_CORPUS_SHA256 = "ba6431ca900cf41eb4f621cbf04d7f084187a16ec1ba6d1b8b6663967302d2e6"
CYBER10_CORPUS_SHA256 = "c8c27c8cee1b8e6de3c9e52cc180879eda8838bb8b7fccb3a22c332d0a69f374"

PRIVATE_PATTERNS = {
    "macOS home path": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    "operator Linux home path": re.compile(r"/home/(craigmac|blackfrost_ai)/"),
    "operator WireGuard subnet": re.compile(r"10\.80\.10\."),
    "research pod public IP": re.compile(r"\b(16\.112\.99\.36|40\.192\.103\.113|34\.66\.10\.57)\b"),
    "private key marker": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
}
# Model-output evidence rows may contain arbitrary generated prose; scan them
# only for credential/key patterns, not endpoint prose false-positives.
OUTPUT_ROW_FILES = ("rows.jsonl", "ledger-")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def verify_lock() -> dict:
    lock = read_json("recipe.lock.json")
    expected = {
        "schema": "blackfrost.dwm.locked-method.v1",
        "recipe_id": "inkling-2-small-a3p8",
        "version": "1.0.0",
        "source.repo": SOURCE_REPO,
        "source.revision": SOURCE_REVISION,
        "source.upstream_repo": UPSTREAM_REPO,
        "source.upstream_revision": UPSTREAM_REVISION,
        "method.alpha": 3.8,
        "method.passes": 1,
        "method.layers": "20-37",
        "method.target_surface": "wo_ud",
        "method.space": "OUT",
        "method.arithmetic": "W := W - alpha * W @ (q q^T) on transposed wo_ud [out 4608, in 6144], BF16, no whitening, no requantization",
        "method.mtp": "copied byte-exact from upstream (not edited)",
        "directions.artifact": "DWM_RECIPE/evidence/s2-battery/directions.npz",
        "directions.sha256": DIRECTIONS_SHA256,
        "directions.battery_sha256": BATTERY_SHA256,
        "directions.corpus": "DWM_RECIPE/input/v1_corpus_hard32.jsonl",
        "directions.corpus_sha256": HARD32_SHA256,
        "directions.capture_pairs": 32,
        "raw_capture.bytes": RAW_CAPTURE_BYTES,
        "raw_capture.sha256": RAW_CAPTURE_SHA256,
        "raw_capture.vendored": False,
        "candidate.mtp_sha256": MTP_SHA256,
        "serve.image": SERVE_IMAGE,
        "serve.tensor_parallel": 4,
        "serve.max_model_len": 262144,
        "serve.speculative_method": "dspark",
        "serve.num_speculative_tokens": 7,
        "eval.r443_corpus_sha256": R443_CORPUS_SHA256,
        "eval.cyber10_corpus_sha256": CYBER10_CORPUS_SHA256,
    }
    for dotted, want in expected.items():
        value = lock
        for part in dotted.split("."):
            value = value[part]
        require(value == want, f"recipe.lock.json {dotted}: {value!r} != {want!r}")
    return {"entries_checked": len(expected)}


def verify_provenance() -> dict:
    prov = read_json("PROVENANCE.json")
    require(prov["source"]["repository"].endswith(SOURCE_REPO), "provenance source repo mismatch")
    require(prov["source"]["revision"] == SOURCE_REVISION, "provenance revision mismatch")
    reds = prov["redactions"]
    require(len(reds) == 7, f"expected 7 documented redactions, found {len(reds)}")
    for row in reds:
        require(re.fullmatch(r"[0-9a-f]{64}", row["original_sha256"]) is not None,
                f"bad original digest: {row['path']}")
        require(re.fullmatch(r"[0-9a-f]{64}", row["vendored_sha256"]) is not None,
                f"bad vendored digest: {row['path']}")
        require(row["original_sha256"] != row["vendored_sha256"],
                f"redaction entry is a no-op: {row['path']}")
        target = ROOT / row["path"]
        require(target.is_file(), f"redacted file missing: {row['path']}")
        require(sha256(target) == row["vendored_sha256"],
                f"redacted file drifted from PROVENANCE: {row['path']}")
    fetch = prov["fetch_only"]
    require(len(fetch) == 1 and fetch[0]["sha256"] == RAW_CAPTURE_SHA256,
            "fetch-only descriptor mismatch")
    return {"redactions_verified": len(reds)}


def verify_kit_manifest() -> dict:
    prov = read_json("PROVENANCE.json")
    red_by_original = {}
    for r in prov["redactions"]:
        red_by_original.setdefault(r["original_sha256"], []).append(r)
    kit = read_json("manifests/kit-sha256.json")
    require(kit["count"] == len(kit["files"]), "kit manifest count mismatch")
    checked = redacted = selfskip = fetchonly = replaced = 0
    for entry in kit["files"]:
        rel = entry["path"]
        if rel == SELF_MANIFEST_ENTRY:
            selfskip += 1
            continue
        if rel in WRAPPER_REPLACED:
            require((ROOT / rel).is_file(), f"wrapper file missing: {rel}")
            replaced += 1
            continue
        target = ROOT / rel
        if not target.is_file():
            require(rel in NOT_VENDORED_BY_DESIGN,
                    f"kit file missing without fetch-only status: {rel}")
            if rel.endswith("raw-capture.npz"):
                require(entry["sha256"] == RAW_CAPTURE_SHA256 and entry["bytes"] == RAW_CAPTURE_BYTES,
                        "raw-capture manifest identity mismatch")
                fetchonly += 1
            else:
                replaced += 1
            continue
        actual = sha256(target)
        if actual == entry["sha256"]:
            checked += 1
        else:
            rows = [r for r in red_by_original.get(entry["sha256"], []) if r["path"] == rel]
            require(any(r["vendored_sha256"] == actual for r in rows),
                    f"kit manifest drift not covered by a documented redaction: {rel}")
            redacted += 1
    require(redacted == 7, f"expected 7 redaction-resolved entries, got {redacted}")
    return {"byte_identical": checked, "redaction_resolved": redacted,
            "self_referential_skipped": selfskip, "fetch_only": fetchonly,
            "wrapper_or_presentation": replaced}


def verify_repo_manifest() -> dict:
    prov = read_json("PROVENANCE.json")
    red_by_original = {}
    for r in prov["redactions"]:
        red_by_original.setdefault(r["original_sha256"], []).append(r)
    checked = redacted = absent = 0
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        want, rel = line.split("  ", 1)
        target = ROOT / rel
        if rel in WRAPPER_REPLACED:
            require(target.is_file(), f"wrapper file missing: {rel}")
            absent += 1
            continue
        if not target.is_file():
            require(rel in NOT_VENDORED_ROOT,
                    f"MANIFEST entry missing without a documented reason: {rel}")
            absent += 1  # weights, banner, raw capture: not vendored by design
            continue
        actual = sha256(target)
        if actual == want:
            checked += 1
        else:
            rows = [r for r in red_by_original.get(want, []) if r["path"] == rel]
            require(any(r["vendored_sha256"] == actual for r in rows),
                    f"MANIFEST.sha256 drift not covered by a documented redaction: {rel}")
            redacted += 1
    return {"byte_identical": checked, "redaction_resolved": redacted, "not_vendored": absent}


def verify_hashes() -> dict:
    pinned = {
        "DWM_RECIPE/input/v1_corpus_hard32.jsonl": HARD32_SHA256,
        "DWM_RECIPE/evidence/s2-battery/directions.npz": DIRECTIONS_SHA256,
        "DWM_RECIPE/evidence/s2-battery/direction-battery.json": BATTERY_SHA256,
    }
    for rel, want in pinned.items():
        require(sha256(ROOT / rel) == want, f"pinned artifact digest mismatch: {rel}")
    # Cross-check against the monorepo benchmark corpora when present.
    crosschecked = []
    bench = ROOT.parents[1] / "benchmarks" / "corpora"
    for name, want in (("R1-FALSE-REFUSAL-443-v1.jsonl", R443_CORPUS_SHA256),
                       ("CYBER-REASONING-10-v1.jsonl", CYBER10_CORPUS_SHA256)):
        candidate = bench / name
        if candidate.is_file():
            require(sha256(candidate) == want, f"monorepo benchmark drift: {name}")
            crosschecked.append(name)
    return {"pinned_artifacts": len(pinned), "benchmark_crosschecks": crosschecked}


def verify_raw_capture(path: pathlib.Path) -> dict:
    require(path.is_file(), f"raw capture not found: {path}")
    size = path.stat().st_size
    require(size == RAW_CAPTURE_BYTES, f"raw capture size {size} != {RAW_CAPTURE_BYTES}")
    digest = sha256(path)
    require(digest == RAW_CAPTURE_SHA256, f"raw capture SHA-256 mismatch: {digest}")
    return {"bytes": size, "sha256": digest, "status": "exact-match"}


def verify_private_scan() -> dict:
    violations = []
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        is_output = path.name.startswith(OUTPUT_ROW_FILES) or "rows.jsonl" in path.name
        for label, pattern in PRIVATE_PATTERNS.items():
            if is_output and label not in ("private key marker", "GitHub token", "Hugging Face token"):
                continue
            if pattern.search(text):
                violations.append({"path": path.relative_to(ROOT).as_posix(), "pattern": label})
    require(not violations, f"private-material scan failed: {violations[:8]}")
    return {"files_scanned": scanned, "patterns": len(PRIVATE_PATTERNS)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-capture", type=pathlib.Path, default=None,
                        help="path to the fetched raw-capture.npz for exact audit verification")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args()
    report = {"package": "inkling-2-small-a3.8", "status": "PASS"}
    try:
        report["lock"] = verify_lock()
        report["provenance"] = verify_provenance()
        report["kit_manifest"] = verify_kit_manifest()
        report["repo_manifest"] = verify_repo_manifest()
        report["hashes"] = verify_hashes()
        report["private_scan"] = verify_private_scan()
        if args.raw_capture is not None:
            report["raw_capture"] = verify_raw_capture(args.raw_capture)
        else:
            report["raw_capture"] = {
                "status": "fetch-only (not vendored; not required for execution)",
                "bytes": RAW_CAPTURE_BYTES, "sha256": RAW_CAPTURE_SHA256,
                "fetch": "see assets/raw-capture.asset.json",
            }
    except (RuntimeError, KeyError, FileNotFoundError, ValueError) as exc:
        report["status"] = "FAIL"
        report["error"] = str(exc)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())