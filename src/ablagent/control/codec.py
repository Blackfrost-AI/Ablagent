"""Strict bounded canonical JSON used by every control-plane receipt."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .errors import InputError, IntegrityError

MAX_JSON_BYTES = 8 * 1024 * 1024


def _reject_constant(value: str) -> None:
    raise IntegrityError(f"non-finite JSON number is forbidden: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(data: bytes, *, source: str = "JSON") -> Any:
    if len(data) > MAX_JSON_BYTES:
        raise IntegrityError(f"{source} exceeds {MAX_JSON_BYTES} bytes")
    try:
        return json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid JSON in {source}: {exc}") from exc


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InputError(f"value is not canonical JSON: {exc}") from exc


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def bare_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
