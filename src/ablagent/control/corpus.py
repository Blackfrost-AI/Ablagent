"""Pack-driven JSONL corpus intake and registry receipts."""

from __future__ import annotations

import json
from typing import Any

from ..packs import DomainPack, PackError, _resolve_inside, sha256_file

REQUIRED_FIELDS = ("id", "dataset", "label", "prompt")


def read_jsonl(
    pack: DomainPack, entry: dict[str, Any], *, expected_rows: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Verify at the read boundary as well as at load/run creation. This keeps
    # mutable DomainPack objects from becoming an execution authority.
    from ..packs import verify_corpus_pins

    verify_corpus_pins(pack)
    cid = str(entry["corpus_id"])
    target = _resolve_inside(
        pack.pack_dir, str(entry["path"]), str(pack.manifest_path), f"corpus '{cid}'"
    )
    if not target.is_file():
        raise PackError(f"corpus missing: {target}")
    rows: list[dict[str, Any]] = []
    with target.open("rb") as fh:
        for number, raw in enumerate(fh, 1):
            if not raw.strip():
                raise PackError(f"corpus {cid}: blank line at {number}")
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PackError(f"corpus {cid}: invalid JSON at line {number}") from exc
            if not isinstance(row, dict) or any(field not in row for field in REQUIRED_FIELDS):
                raise PackError(f"corpus {cid}: line {number} missing required fields")
            if not all(isinstance(row[field], str) and row[field] for field in REQUIRED_FIELDS):
                raise PackError(f"corpus {cid}: line {number} fields must be non-empty strings")
            rows.append(row)
    if expected_rows is not None and (type(expected_rows) is not int or len(rows) != expected_rows):
        raise PackError(f"corpus {cid}: expected {expected_rows} rows, got {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise PackError(f"corpus {cid}: duplicate row id")
    return rows, {
        "corpus_id": cid,
        "path": str(entry["path"]),
        "sha256": sha256_file(target),
        "rows": len(rows),
        "required_fields": list(REQUIRED_FIELDS),
    }


def registry(pack: DomainPack) -> tuple[dict[str, Any], ...]:
    result = []
    for entry in pack.corpus_entries():
        _rows, receipt = read_jsonl(pack, entry)
        result.append({"pack_id": pack.pack_id, "pack_version": pack.version, **receipt})
    return tuple(result)
