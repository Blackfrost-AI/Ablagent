"""Strict offline benchmark IO; schemas never resolve external references."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("non-finite JSON number")


def loads(text):
    try:
        result = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
        canonical(result)  # Also rejects overflow (1e999) and invalid Unicode.
        return result
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid strict JSON") from exc


def canonical(obj):
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def digest(obj):
    """SHA-256 of the ENTIRE response object (no excluded fields)."""
    return hashlib.sha256(canonical(obj)).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(obj, name, version="v2"):
    path = (
        ROOT / "protocol" / (f"{name}.{version}.schema.json" if version else f"{name}.schema.json")
    )
    schema = loads(path.read_text(encoding="utf-8"))

    def reject_remote_refs(node):
        if isinstance(node, dict):
            ref = node.get("$ref") or node.get("$dynamicRef")
            if ref and not ref.startswith("#/"):
                raise ValueError("remote schema references forbidden")
            for value in node.values():
                reject_remote_refs(value)
        elif isinstance(node, list):
            for value in node:
                reject_remote_refs(value)

    reject_remote_refs(schema)
    Draft202012Validator.check_schema(schema)
    if next(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(obj), None):
        raise ValueError(f"invalid {name} {version or 'v1'} record")
    canonical(obj)


def jsonl(path, name=None, version="v2"):
    return jsonl_bytes(Path(path).read_bytes(), name, version)


def jsonl_bytes(data, name=None, version="v2"):
    rows = []
    # JSONL records are delimited by the byte LF only.  Unicode line
    # separators are valid JSON string content when emitted literally by our
    # ensure_ascii=False canonical encoder; treating them as delimiters would
    # corrupt otherwise valid records.  Drop exactly one delimiter newline,
    # while retaining interior/trailing blank records for rejection by loads.
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    for line in lines:
        line = line.decode("utf-8")
        obj = loads(line)
        if not isinstance(obj, dict):
            raise ValueError("JSONL record must be an object")
        if name:
            validate(obj, name, version)
        rows.append(obj)
    return rows


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
