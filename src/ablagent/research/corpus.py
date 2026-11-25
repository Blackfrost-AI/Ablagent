"""Derive deterministic JSONL corpora from evidence-linked research notes."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..control.codec import canonical_bytes
from ..control.errors import InputError
from ..control.files import atomic_write, ensure_private_dir


def derive_jsonl(
    notes: Iterable[dict[str, Any]], output: Path, *, dataset: str, label: str
) -> dict[str, Any]:
    rows = []
    for index, note in enumerate(notes):
        if note.get("schema") != "ablagent.research.note.v1" or not note.get("note_digest"):
            raise InputError("every note must be a verified research note")
        prompt = note.get("title")
        if not isinstance(prompt, str) or not prompt.strip():
            raise InputError("note title is required")
        rows.append(
            {
                "id": f"{dataset}-{index:06d}",
                "dataset": dataset,
                "label": label,
                "prompt": prompt,
                "source_note": note["note_digest"],
            }
        )
    if not rows:
        raise InputError("cannot derive an empty corpus")
    output = Path(output)
    ensure_private_dir(output.parent, create=not output.parent.exists())
    data = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    atomic_write(output, data)
    return {
        "dataset": dataset,
        "rows": len(rows),
        "sha256": __import__("hashlib").sha256(data).hexdigest(),
        "source_notes": [row["source_note"] for row in rows],
    }
