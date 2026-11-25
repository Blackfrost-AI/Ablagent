"""Evidence-first research ingestion and corpus derivation."""

from .corpus import derive_jsonl
from .fetch import FetchReceipt, fetch_source
from .notes import Note, write_note
from .sources import Source, SourceRegistry

__all__ = [
    "FetchReceipt",
    "Note",
    "Source",
    "SourceRegistry",
    "derive_jsonl",
    "fetch_source",
    "write_note",
]
