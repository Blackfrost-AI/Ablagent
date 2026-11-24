"""Run-directory layout and immutable run metadata."""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import secrets
from pathlib import Path
from typing import Any

import yaml

from .control.codec import canonical_bytes, digest
from .control.files import atomic_write, ensure_private_dir
from .packs import DomainPack, verify_corpus_pins, sha256_bytes

RUN_SCHEMA = "ablagent.run.v2"
EVIDENCE_SCHEMA = "ablagent.evidence.v2"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def create_run(pack: DomainPack, *, model: str, executor: str, runs_root: Path,
               options: dict[str, Any] | None = None) -> dict[str, Any]:
    # Re-verify at the execution boundary; the pack object may have crossed
    # an API boundary and its parsed manifest is intentionally not authority.
    manifest_bytes = pack.manifest_path.read_bytes()
    if sha256_bytes(manifest_bytes) != pack.manifest_sha256:
        raise ValueError("pack manifest changed after loading")
    verify_corpus_pins(pack)
    root = ensure_private_dir(Path(runs_root), create=True) if not Path(runs_root).exists() else ensure_private_dir(Path(runs_root))
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
    run_dir = root / run_id
    run_dir.mkdir(mode=0o700)
    run = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "created_at": _now(),
        "run_directory": str(run_dir.resolve()),
        "pack": {"id": pack.pack_id, "version": pack.version, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "model": model,
        "executor": executor,
        "options": options or {},
        "immutable_inputs": {"pack_manifest": pack.manifest_sha256, "corpora": list(pack.pins().items())},
    }
    atomic_write(run_dir / "RUN.yaml", yaml.safe_dump(run, sort_keys=False).encode())
    for name in ("evidence", "logs", "artifacts"):
        (run_dir / name).mkdir(mode=0o700)
    return run


def write_evidence(run_dir: Path, *, stage: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    evidence_dir = Path(run_dir) / "evidence"
    evidence_dir.mkdir(mode=0o700, exist_ok=True)
    body = {"schema": EVIDENCE_SCHEMA, "stage": stage, "kind": kind, "captured_at": _now(), "payload": payload}
    body["evidence_digest"] = digest(body)
    filename = f"{stage.lower()}-{kind}-{len(list(evidence_dir.glob('*.json'))):04d}.json"
    atomic_write(evidence_dir / filename, canonical_bytes(body))
    return body
