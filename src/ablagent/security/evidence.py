"""Audit-only evidence receipt verification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..control.codec import digest
from ..control.errors import IntegrityError


def verify_evidence(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "ablagent.evidence.v2":
        raise IntegrityError(f"unsupported evidence schema: {path}")
    claimed = value.get("evidence_digest")
    body = dict(value)
    body.pop("evidence_digest", None)
    if claimed != digest(body):
        raise IntegrityError(f"evidence digest mismatch: {path}")
    return value
