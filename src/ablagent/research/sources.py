"""Local source registry with explicit URL and license provenance."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from ..control.codec import canonical_bytes, digest, load_json
from ..control.files import atomic_write, ensure_private_dir, read_stable
from ..control.errors import InputError, IntegrityError

SOURCE_SCHEMA = "ablagent.research.source.v1"
_ALLOWED_SCHEMES = {"https"}
_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")


@dataclass(frozen=True)
class Source:
    source_id: str
    url: str
    title: str
    license: str
    allowed_use: str
    citation: str = ""

    def validate(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in _ALLOWED_SCHEMES or parsed.username or parsed.password:
            raise InputError("source URL must be an https URL without embedded credentials")
        if not parsed.hostname or not _HOST_RE.match(parsed.hostname.lower()):
            raise InputError("source URL has invalid host")
        if not self.source_id or not re.match(r"^[a-z0-9][a-z0-9._-]{0,63}$", self.source_id):
            raise InputError("source_id must be lowercase identifier")
        if not self.title or not self.license or not self.allowed_use:
            raise InputError("source title, license, and allowed_use are required")


class SourceRegistry:
    def __init__(self, root: Path):
        self.root = ensure_private_dir(Path(root), create=not Path(root).exists())
        self.path = self.root / "SOURCES.json"

    def register(self, source: Source) -> Source:
        source.validate()
        sources = self.list()
        if any(item.source_id == source.source_id for item in sources):
            raise IntegrityError(f"duplicate source_id: {source.source_id}")
        atomic_write(self.path, canonical_bytes({"schema": SOURCE_SCHEMA, "sources": [asdict(item) for item in sources] + [asdict(source)]}))
        return source

    def list(self) -> tuple[Source, ...]:
        if not self.path.exists():
            return ()
        value = load_json(read_stable(self.path), source=str(self.path))
        if not isinstance(value, dict) or value.get("schema") != SOURCE_SCHEMA or not isinstance(value.get("sources"), list):
            raise IntegrityError("invalid source registry")
        result = []
        for item in value["sources"]:
            source = Source(**item)
            source.validate()
            result.append(source)
        return tuple(result)

    def get(self, source_id: str) -> Source:
        for source in self.list():
            if source.source_id == source_id:
                return source
        raise InputError(f"unknown source: {source_id}")
