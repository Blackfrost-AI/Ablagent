"""Deterministic, explicitly inert mock red-line evidence for policy wiring tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field
from pydantic import ValidationError as PydanticValidationError

from ablagent.artifacts import ArtifactStore, file_digest
from ablagent.banks import BankManifest, BankReference, validate_bank_reference
from ablagent.canonical import digest_bytes, digest_model, model_bytes, read_json, write_json
from ablagent.contracts import Contract, Digest, Identifier
from ablagent.errors import ValidationError

if TYPE_CHECKING:
    from ablagent.evidence import ArtifactBinding

MOCK_EVALUATOR = "synthetic-mock-v0"
MOCK_CONFIGURATION = "all_inert_controls_v0"
MOCK_CATEGORIES = (
    "biological",
    "chemical",
    "explosives_incendiary",
    "strategic_wmd",
    "cyber",
    "loss_of_control",
)


class MockRedlineResult(Contract):
    record_id: Identifier
    name: Literal[
        "biological",
        "chemical",
        "explosives_incendiary",
        "strategic_wmd",
        "cyber",
        "loss_of_control",
    ]
    outcome: Literal["PASS"]
    score: Literal[1.0] = 1.0
    observed: Literal[1] = 1
    eligible: Literal[1] = 1


class FileReference(Contract):
    path: str
    file_digest: Digest


class RawEvidenceReference(FileReference):
    json_digest: Digest


class MockRedlineRaw(Contract):
    schema_version: Literal["dwmip.mock-redline-raw.v0"] = "dwmip.mock-redline-raw.v0"
    bank: BankReference
    binding: ArtifactBinding
    evaluator: Literal["synthetic-mock-v0"] = MOCK_EVALUATOR
    configuration: Literal["all_inert_controls_v0"] = MOCK_CONFIGURATION
    results: tuple[MockRedlineResult, ...] = Field(min_length=6, max_length=6)
    measurement_scope: Literal["inert_policy_fixture_not_behavioral"] = (
        "inert_policy_fixture_not_behavioral"
    )


class MockRedlineAttestation(Contract):
    schema_version: Literal["dwmip.mock-redline-attestation.v0"] = (
        "dwmip.mock-redline-attestation.v0"
    )
    bank: BankReference
    binding: ArtifactBinding
    evaluator: Literal["synthetic-mock-v0"] = MOCK_EVALUATOR
    configuration: Literal["all_inert_controls_v0"] = MOCK_CONFIGURATION
    raw: RawEvidenceReference
    results: tuple[MockRedlineResult, ...] = Field(min_length=6, max_length=6)
    assurance_scope: Literal["inert_policy_fixture_not_behavioral"] = (
        "inert_policy_fixture_not_behavioral"
    )

    def digest(self) -> str:
        return digest_model("mock-redline-attestation", self)


def mock_redline_metrics(attestation: MockRedlineAttestation) -> tuple:
    """Derive policy metrics from the attestation, never from caller scores."""
    from ablagent.evidence import MetricResult

    return tuple(
        MetricResult(
            category="redline",
            name=result.name,
            outcome=result.outcome,
            score=result.score,
            observed=result.observed,
            eligible=result.eligible,
            evidence_digest=attestation.digest(),
            details=(
                f"Selected bank record {result.record_id}; inert {MOCK_EVALUATOR} fixture, "
                "not a behavioral or release-safety measurement."
            ),
        )
        for result in attestation.results
    )


def mock_redline_raw(bank: BankManifest, binding: ArtifactBinding) -> MockRedlineRaw:
    if bank.role != "mock_release_redline":
        raise ValidationError("only the inert mock-release bank supports synthetic redline replay")
    by_id = {record.record_id: record for record in bank.records}
    if len(by_id) != len(bank.records):
        raise ValidationError("mock bank record identifiers must be unique")
    results = []
    for name in MOCK_CATEGORIES:
        record_id = f"mock-{name}"
        record = by_id.get(record_id)
        if record is None or record.group != "control" or record.domain != "mock":
            raise ValidationError(f"mock bank lacks the exact inert control record: {record_id}")
        if record.prompt != f"inert schema fixture: {name}":
            raise ValidationError(f"mock bank record does not match its inert fixture: {record_id}")
        results.append(MockRedlineResult(record_id=record_id, name=name, outcome="PASS"))
    if set(by_id) != {result.record_id for result in results}:
        raise ValidationError("mock bank must contain exactly the six declared inert controls")
    return MockRedlineRaw(bank=bank.reference(), binding=binding, results=tuple(results))


def write_mock_redlines(
    store: ArtifactStore, bank: BankManifest, binding: ArtifactBinding, raw_path: str
) -> MockRedlineAttestation:
    raw = mock_redline_raw(bank, binding)
    path = store.path(raw_path)
    json_digest = write_json(path, raw)
    return MockRedlineAttestation(
        bank=raw.bank,
        binding=raw.binding,
        raw=RawEvidenceReference(
            path=raw_path, file_digest=file_digest(path), json_digest=json_digest
        ),
        results=raw.results,
    )


def verify_mock_redlines(
    store: ArtifactStore,
    bank: BankManifest,
    binding: ArtifactBinding,
    attestation: MockRedlineAttestation,
) -> None:
    expected = mock_redline_raw(bank, binding)
    validate_bank_reference(bank, attestation.bank)
    if attestation.binding != binding or attestation.results != expected.results:
        raise ValidationError("mock redline attestation does not match the selected bank/artifact")
    try:
        path = store.path(attestation.raw.path)
        actual = MockRedlineRaw.model_validate(read_json(path))
    except PydanticValidationError as exc:
        raise ValidationError(f"invalid mock redline raw evidence: {exc}") from exc
    if actual != expected:
        raise ValidationError("mock redline raw evidence does not reproduce from the selected bank")
    if file_digest(path) != attestation.raw.file_digest:
        raise ValidationError("mock redline raw file digest does not match its attestation")
    if digest_bytes("json-file", model_bytes(actual) + b"\n") != attestation.raw.json_digest:
        raise ValidationError("mock redline canonical JSON digest does not match its attestation")
