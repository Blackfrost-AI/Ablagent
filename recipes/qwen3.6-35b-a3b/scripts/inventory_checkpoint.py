#!/usr/bin/env python3
"""Hash a remote checkpoint into a portable artifact identity record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.checkpoint.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and ".cache" not in path.parts:
            files.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
            print(f"HASH {files[-1]['path']} {files[-1]['bytes']}", flush=True)
    identity_input = [
        {"path": item["path"], "digest": "sha256:" + item["sha256"], "size": item["bytes"]}
        for item in files
    ]
    digest = "sha256:" + hashlib.sha256(json.dumps(
        identity_input, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    result = {
        "schema": "ablagent.remote-checkpoint-inventory.v1",
        "artifact_identity": {"id": args.artifact_id, "digest": digest},
        "model_id": args.model_id,
        "model_revision": args.revision,
        "checkpoint_path": str(root),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({
        "artifact_identity": result["artifact_identity"],
        "files": len(files),
        "bytes": result["total_bytes"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
