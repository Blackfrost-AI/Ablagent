#!/usr/bin/env python3
"""Verify the vendored Inkling-2-Mini-Derisked example against PROVENANCE.json.

Offline by default: recomputes SHA-256 of every vendored file and compares
against the manifest (which is pinned to HF revision
31ec2eddecfdf93c8626d9e4dfc0a2aca6070900 of
Blackfrost-AI/Inkling-2-Mini-Derisked). Also checks the internal hash chain:
directions.npz digest in s2-summary.json and s3-edit-receipt-a25.json, and the
hard32 corpus digest asserted by the capture metadata.

With --fetch-only, additionally verifies the large raw-capture.npz if it has
been downloaded (see fetch_example_assets.py). Missing fetch-only assets are
reported as absent, not as failures.

Exit codes: 0 = all present files verified; 1 = any mismatch or missing
vendored file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "inkling-2-mini-derisked"
MANIFEST = EXAMPLE_DIR / "PROVENANCE.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_entry(entry: dict, root: Path, required: bool) -> tuple[bool, str]:
    path = root / entry["path"]
    if not path.is_file():
        if required:
            return False, f"MISSING  {entry['path']}"
        return True, f"absent   {entry['path']} (fetch-only; not downloaded)"
    digest = sha256_file(path)
    if digest != entry["sha256"]:
        return False, f"MISMATCH {entry['path']}: got {digest}, expected {entry['sha256']}"
    if path.stat().st_size != entry["size"]:
        got, want = path.stat().st_size, entry["size"]
        return False, f"SIZE     {entry['path']}: got {got}, expected {want}"
    return True, f"ok       {entry['path']}"


def verify_hash_chain(root: Path) -> list[str]:
    """Cross-check digests that the kit itself binds, independent of the manifest."""
    problems: list[str] = []
    directions = root / "DWM_RECIPE" / "s2-capture" / "directions.npz"
    expected = "6674ddc8f6e8762c10f14aa37fdcef914186fda1276804f37d7bd30365fb1b74"
    if directions.is_file():
        actual = sha256_file(directions)
        for rel in (
            "DWM_RECIPE/s2-capture/s2-summary.json",
            "DWM_RECIPE/s3-edit-receipt-a25.json",
        ):
            doc = json.loads((root / rel).read_text())
            bound = None
            for key in ("directions_sha256", "directions_npz_sha256"):
                if key in doc:
                    bound = str(doc[key]).removeprefix("sha256:")
            if bound is not None and bound != actual:
                problems.append(f"hash chain: {rel} binds {bound}, directions.npz is {actual}")
        if actual != expected:
            problems.append(f"directions.npz digest {actual} != canonical {expected}")
    corpus = root / "DWM_RECIPE" / "scripts" / "v1_corpus_hard32.jsonl"
    meta = root / "DWM_RECIPE" / "s2-capture" / "capture-metadata.json"
    if corpus.is_file() and meta.is_file():
        actual = sha256_file(corpus)
        doc = json.loads(meta.read_text())
        for key in ("corpus_sha256", "hard32_corpus_sha256"):
            bound = str(doc.get(key, "")).removeprefix("sha256:")
            if bound and bound != actual:
                problems.append(f"hash chain: capture-metadata {key}={bound}, corpus is {actual}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="also verify fetch-only assets (raw-capture.npz) when present",
    )
    parser.add_argument(
        "--example-dir",
        type=Path,
        default=EXAMPLE_DIR,
        help="example directory (default: repo examples/inkling-2-mini-derisked)",
    )
    args = parser.parse_args(argv)

    root = args.example_dir
    manifest_path = root / "PROVENANCE.json"
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())

    failures = 0
    for entry in manifest["files"]:
        ok, line = verify_entry(entry, root, required=True)
        print(line)
        failures += 0 if ok else 1
    if args.fetch_only:
        for entry in manifest.get("fetch_only", []):
            ok, line = verify_entry(entry, root, required=False)
            print(line)
            failures += 0 if ok else 1

    for problem in verify_hash_chain(root):
        print(f"FAIL     {problem}")
        failures += 1

    if failures:
        print(f"\n{failures} verification failure(s).", file=sys.stderr)
        return 1
    print("\nAll vendored example files verified against PROVENANCE.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
