"""Canonical JSON and domain-separated digests.

This is Ablagent Canonical JSON v1, a deliberately small deterministic subset,
not a claim of RFC 8785 compliance. See docs/determinism.md.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ablagent.errors import ValidationError


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _check(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value) or (value == 0.0 and math.copysign(1, value) < 0):
            raise ValidationError(f"non-finite or negative-zero number at {path}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValidationError(f"non-string JSON key at {path}")
        for key, item in value.items():
            _check(item, f"{path}.{key}")
        return
    raise ValidationError(f"non-JSON value at {path}: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Return sorted-key UTF-8 JSON without NaN, Infinity, or negative zero."""
    _check(value)
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def model_bytes(value: Any) -> bytes:
    """Canonicalize a Pydantic model or ordinary JSON-compatible value."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return canonical_bytes(value)


def digest_bytes(domain: str, payload: bytes) -> str:
    if not domain or "\x00" in domain:
        raise ValidationError("digest domain must be a nonempty string without NUL")
    return (
        "sha256:"
        + hashlib.sha256(
            b"ablagent\x00" + domain.encode("utf-8") + b"\x00v1\x00" + payload
        ).hexdigest()
    )


def digest_model(domain: str, value: Any) -> str:
    return digest_bytes(domain, model_bytes(value))


def read_json(path: Path, max_bytes: int = 4 * 1024 * 1024) -> Any:
    try:
        size = path.stat().st_size
        if size > max_bytes:
            raise ValidationError(f"JSON exceeds the {max_bytes}-byte limit: {path}")
        text = path.read_text(encoding="utf-8")
        return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except ValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON input {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> str:
    """Write canonical JSON without replacing an existing file."""
    if path.exists():
        raise ValidationError(f"refusing to overwrite an immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return digest_bytes("json-file", payload)
