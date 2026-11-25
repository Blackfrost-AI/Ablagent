"""Authority receipt validation. Direct greenlight is the built-in path.

Ed25519 is intentionally optional: requesting signed authority without an
installed provider fails closed; it never silently downgrades to direct mode.
"""

from __future__ import annotations

from typing import Any

from .codec import digest
from .errors import AuthorityError


def normalize_actions(actions: list[str]) -> list[str]:
    if (
        not isinstance(actions, list)
        or not actions
        or any(not isinstance(a, str) or not a for a in actions)
    ):
        raise AuthorityError("actions must be a non-empty list of strings")
    return sorted(set(actions))


def receipt_digest(receipt: dict[str, Any]) -> str:
    body = dict(receipt)
    body.pop("signature", None)
    return digest(body)


def verify_direct(
    *, principal: str, session_binding: str, stage: str, actions: list[str], expected_revision: int
) -> dict[str, Any]:
    if not principal or not session_binding or not stage:
        raise AuthorityError("principal, session, and stage are required")
    if type(expected_revision) is not int or expected_revision < 0:
        raise AuthorityError("invalid expected revision")
    return {
        "schema": "authority.v1",
        "mode": "direct",
        "principal": principal,
        "session_binding": session_binding,
        "stage": stage,
        "actions": normalize_actions(actions),
        "expected_revision": expected_revision,
    }


def verify_signed(receipt: dict[str, Any], *, provider: Any = None) -> dict[str, Any]:
    if provider is None:
        raise AuthorityError("signed authority requires an explicit Ed25519 provider")
    raise AuthorityError("Ed25519 provider integration is not enabled")
