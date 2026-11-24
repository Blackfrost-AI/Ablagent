#!/usr/bin/env python3
"""Explicitly fetch large Inkling example assets from the pinned HF revision.

Downloads raw-capture.npz (79 MB, required by the direction battery script)
from Blackfrost-AI/Inkling-2-Mini-Derisked at revision
c352150aacc7d0ca4f6e4f4994cfe39b1077daa8, verifies its SHA-256 against
PROVENANCE.json, and only then keeps the file. Stdlib only; no implicit
downloads — this script runs only when you invoke it.

Usage:
    python scripts/fetch_example_assets.py            # fetch raw-capture.npz
    python scripts/fetch_example_assets.py --check    # verify if already present

The destination is gitignored. An existing file with a wrong digest is never
silently overwritten: the script fails and leaves it in place for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "examples" / "inkling-2-mini-derisked"
MANIFEST = EXAMPLE_DIR / "PROVENANCE.json"
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry: dict, source: dict, check_only: bool) -> int:
    dest = EXAMPLE_DIR / entry["path"]
    expected = entry["sha256"]

    if dest.is_file():
        actual = sha256_file(dest)
        if actual == expected:
            print(f"ok: {entry['path']} already present and verified ({entry['size']} bytes)")
            return 0
        print(
            f"ERROR: {dest} exists but has digest {actual}, expected {expected}.\n"
            "Refusing to overwrite; inspect or remove the file manually.",
            file=sys.stderr,
        )
        return 1
    if check_only:
        print(f"absent: {entry['path']} (run without --check to download)")
        return 0

    url = f"{source['repository']}/resolve/{source['revision']}/{entry['hf_path']}"
    print(f"downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — closed via context manager below
        dir=dest.parent, delete=False, suffix=".part"
    )
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp:
            if resp.status != 200:
                print(f"ERROR: HTTP {resp.status}", file=sys.stderr)
                return 1
            shutil.copyfileobj(resp, tmp, CHUNK)
        tmp_path = Path(tmp.name)
        actual = sha256_file(tmp_path)
        if actual != expected:
            print(
                f"ERROR: downloaded digest {actual} != expected {expected}; discarding.",
                file=sys.stderr,
            )
            return 1
        if tmp_path.stat().st_size != entry["size"]:
            print("ERROR: downloaded size mismatch; discarding.", file=sys.stderr)
            return 1
        tmp_path.replace(dest)
        print(f"verified and saved: {dest} ({entry['size']} bytes, sha256 {expected})")
        return 0
    finally:
        leftover = Path(tmp.name)
        if leftover.exists():
            leftover.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only; never download")
    args = parser.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text())
    failures = 0
    for entry in manifest.get("fetch_only", []):
        failures += fetch(entry, manifest["source"], args.check)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
