"""Effective-policy compilation and digest verification."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..constants import EFFECTIVE_POLICY_SCHEMA, DEPENDENCIES, STAGES
from .codec import canonical_bytes, digest
from .errors import IntegrityError, InputError
from .files import atomic_write


def compile_policy(*, run: dict[str, Any], principal: dict[str, Any],
                   input_digests: dict[str, str], options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compile only resolved, content-addressed policy inputs."""
    if not isinstance(input_digests, dict) or any(not isinstance(v, str) or not v.startswith("sha256:") for v in input_digests.values()):
        raise InputError("every policy input must be a sha256 digest")
    policy = {
        "schema": EFFECTIVE_POLICY_SCHEMA,
        "version": 1,
        "run": run,
        "principal": principal,
        "inputs": dict(sorted(input_digests.items())),
        "options": options or {},
        "stages": list(STAGES),
        "dependencies": {stage: list(DEPENDENCIES[stage]) for stage in STAGES},
    }
    policy["digest"] = digest(policy)
    return policy


def write_policy(path: Path, policy: dict[str, Any]) -> str:
    if policy.get("schema") != EFFECTIVE_POLICY_SCHEMA:
        raise InputError("invalid effective-policy schema")
    content = dict(policy)
    content.pop("digest", None)
    actual = digest(content)
    if policy.get("digest") != actual:
        raise IntegrityError("effective policy digest mismatch")
    atomic_write(Path(path), canonical_bytes(policy))
    return actual


def verify_policy(path: Path) -> dict[str, Any]:
    from .files import read_stable
    from .codec import load_json
    value = load_json(read_stable(Path(path)), source=str(path))
    if not isinstance(value, dict) or value.get("schema") != EFFECTIVE_POLICY_SCHEMA:
        raise IntegrityError("invalid effective policy")
    claimed = value.get("digest")
    body = dict(value); body.pop("digest", None)
    if claimed != digest(body):
        raise IntegrityError("effective policy digest mismatch")
    return value
