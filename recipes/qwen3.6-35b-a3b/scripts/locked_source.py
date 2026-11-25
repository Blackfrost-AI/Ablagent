#!/usr/bin/env python3
"""Verify the exact source package shared by the locked executors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SOURCE_REPO = "Qwen/Qwen3.6-35B-A3B"
SOURCE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
CHECKPOINT_DIGEST = "sha256:88760187ccb2560500b1d51f0e75610658c956be010b4477bc11bb8265c7413b"
SOURCE_LOCK_SHA256 = "07cc4141def9c9ad77251badd690efa5542db478ba78ce89417ea3e233d437e7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(root: Path, source_lock_path: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source directory is missing: {root}")
    if sha256(source_lock_path) != SOURCE_LOCK_SHA256:
        raise RuntimeError("bundled source lock digest mismatch")
    source_lock = json.loads(source_lock_path.read_text())
    if source_lock.get("repo") != SOURCE_REPO:
        raise RuntimeError("source-lock repository mismatch")
    if source_lock.get("revision") != SOURCE_REVISION:
        raise RuntimeError("source-lock revision mismatch")
    if source_lock.get("checkpoint_digest") != CHECKPOINT_DIGEST:
        raise RuntimeError("source-lock checkpoint identity mismatch")
    if len(source_lock.get("files", [])) != 40:
        raise RuntimeError("source-lock must contain exactly 40 files")

    observed = []
    for number, item in enumerate(source_lock["files"], 1):
        path = root / item["path"]
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"source file is missing or a symlink: {item['path']}")
        if path.stat().st_size != item["size"]:
            raise RuntimeError(f"source size mismatch: {item['path']}")
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"source digest mismatch: {item['path']}")
        observed.append({
            "path": item["path"],
            "digest": "sha256:" + actual,
            "size": item["size"],
        })
        print(f"SOURCE_LOCK {number}/40 {item['path']}", flush=True)

    digest = "sha256:" + hashlib.sha256(
        json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != CHECKPOINT_DIGEST:
        raise RuntimeError(f"canonical source identity mismatch: {digest}")
    return {
        "repo": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "checkpoint_digest": digest,
        "files": len(observed),
        "bytes": sum(item["size"] for item in source_lock["files"]),
    }

