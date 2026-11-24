"""Reconstruct the persisted baseline ArtifactReference from an intake receipt.

The receipt stores the exact reference emitted by ArtifactStore.save_checkpoint;
re-parsing it through the contract model keeps every digest verbatim (no
recomputation, no drift).
"""

from __future__ import annotations

from typing import Any

from .errors import ValidationError
from .interventions import ArtifactReference


def baseline_ref_from_receipt(intake_receipt: dict[str, Any]) -> ArtifactReference:
    raw = intake_receipt.get("baseline")
    if not isinstance(raw, dict):
        raise ValidationError("intake receipt carries no baseline artifact reference")
    reference = ArtifactReference.model_validate(raw)
    if reference.role != "baseline" or reference.artifact_id != "baseline":
        raise ValidationError("intake baseline reference is not the canonical baseline artifact")
    return reference
