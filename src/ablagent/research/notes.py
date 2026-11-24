"""Research notes bound to source and fetch receipts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..control.codec import canonical_bytes, digest
from ..control.files import atomic_write, ensure_private_dir
from ..control.errors import InputError

NOTE_SCHEMA = "ablagent.research.note.v1"

@dataclass(frozen=True)
class Note:
    note_id: str
    title: str
    claims: tuple[dict[str, Any], ...]
    source_ids: tuple[str, ...]
    fetch_receipt_digests: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        body = {"schema": NOTE_SCHEMA, "note_id": self.note_id, "title": self.title,
                "claims": list(self.claims), "source_ids": list(self.source_ids),
                "fetch_receipt_digests": list(self.fetch_receipt_digests)}
        body["note_digest"] = digest(body)
        return body

def write_note(root: Path, note: Note) -> Path:
    if not note.note_id or not note.title or not note.claims or not note.source_ids:
        raise InputError("note requires id, title, claims, and source_ids")
    target_dir = ensure_private_dir(Path(root), create=not Path(root).exists())
    target = target_dir / f"{note.note_id}.json"
    if target.exists():
        raise InputError(f"note already exists: {note.note_id}")
    atomic_write(target, canonical_bytes(note.to_dict()))
    return target
