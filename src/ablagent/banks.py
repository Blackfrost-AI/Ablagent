"""Versioned bank manifests and publication eligibility, not access control."""

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from ablagent.canonical import canonical_bytes, digest_model
from ablagent.contracts import Contract, Digest, Identifier
from ablagent.errors import PolicyBlocked, ValidationError

BankRole = Literal[
    "capture", "calibration", "evaluation", "release_redline", "mock_release_redline"
]
PUBLIC_LICENSES = frozenset({"CC0-1.0", "CC-BY-4.0"})


class LicenseTerms(Contract):
    spdx_id: str = Field(min_length=1, max_length=128)
    copyright: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=1000)
    license_text: str = Field(min_length=1, max_length=500)
    redistribution_allowed: bool | None = None
    transformation_lineage: str = Field(min_length=1, max_length=1000)
    generator_terms: str = Field(min_length=1, max_length=500)


class BankRecord(Contract):
    record_id: Identifier
    prompt: str = Field(min_length=1, max_length=32_000)
    group: Literal["benign", "hostile", "control"]
    domain: Identifier
    pair_id: Identifier | None = None


class BankManifest(Contract):
    schema_version: Literal["dwmip.bank.v0"] = "dwmip.bank.v0"
    bank_id: Identifier
    version: Identifier
    role: BankRole
    visibility: Literal["public", "private"]
    records: tuple[BankRecord, ...] = Field(min_length=1, max_length=100_000)
    license: LicenseTerms
    split_policy: str = Field(min_length=1, max_length=1000)
    notes: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def bank_integrity(self):
        ids = [record.record_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate bank record id")
        if self.role == "capture":
            counts = {
                group: sum(r.group == group for r in self.records)
                for group in ("benign", "hostile")
            }
            if counts["benign"] < 4 or counts["hostile"] != counts["benign"]:
                raise ValueError("capture bank needs at least four balanced benign/hostile pairs")
            if any(record.pair_id is None for record in self.records):
                raise ValueError("every capture record needs a pair_id")
            pairs: dict[str, set[str]] = {}
            for record in self.records:
                pairs.setdefault(record.pair_id, set()).add(record.group)
            if any(groups != {"benign", "hostile"} for groups in pairs.values()):
                raise ValueError(
                    "every capture pair must contain one benign and one hostile record"
                )
        if self.role == "release_redline" and self.visibility != "private":
            raise ValueError("a real release-redline bank must remain external and private")
        if self.visibility == "private" and self.license.redistribution_allowed:
            raise ValueError("a private bank cannot be marked redistributable")
        return self

    def digest(self) -> str:
        return digest_model("bank-manifest", self)

    def reference(self) -> "BankReference":
        return BankReference(
            bank_id=self.bank_id,
            version=self.version,
            role=self.role,
            visibility=self.visibility,
            manifest_digest=self.digest(),
            records_digest="sha256:"
            + hashlib.sha256(
                canonical_bytes([r.model_dump(mode="json") for r in self.records])
            ).hexdigest(),
            count=len(self.records),
        )


class BankReference(Contract):
    bank_id: Identifier
    version: Identifier
    role: BankRole
    visibility: Literal["public", "private"]
    manifest_digest: Digest
    records_digest: Digest
    count: int = Field(gt=0)


def require_publicly_bundlable(bank: BankManifest) -> None:
    """Refuse unknown licensing/private data. This does not publish anything."""
    if bank.visibility != "public":
        raise PolicyBlocked("private bank contents cannot be bundled publicly")
    if bank.license.redistribution_allowed is not True:
        raise PolicyBlocked("redistribution permission is missing or uncertain")
    if bank.license.spdx_id not in PUBLIC_LICENSES:
        raise PolicyBlocked(f"license {bank.license.spdx_id!r} is not approved for public bundling")
    if bank.role == "release_redline":
        raise PolicyBlocked("real release holdouts cannot be bundled")


def validate_bank_reference(bank: BankManifest, reference: BankReference) -> None:
    if bank.reference() != reference:
        raise ValidationError("bank reference does not match the complete bank manifest")
