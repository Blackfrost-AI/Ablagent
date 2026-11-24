"""Defensive verification hooks for evidence and release paths.

Red-line enforcement is intentionally not active in experimental V2 runs.
This package provides audit-only scanners and evidence integrity checks until
the operator promotes them into a proven quality gate.
"""

from .evidence import verify_evidence
from .secret_scan import scan_text

__all__ = ["scan_text", "verify_evidence"]
