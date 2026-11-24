"""Evidence-first research ingestion and corpus derivation."""
from .sources import Source, SourceRegistry
from .fetch import FetchReceipt, fetch_source
from .notes import Note, write_note
from .corpus import derive_jsonl

__all__ = ["Source", "SourceRegistry", "FetchReceipt", "fetch_source", "Note", "write_note", "derive_jsonl"]
