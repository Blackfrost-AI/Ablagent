import pytest
from pydantic import ValidationError as PydanticValidationError

from ablagent.banks import BankManifest, require_publicly_bundlable, validate_bank_reference
from ablagent.errors import PolicyBlocked, ValidationError


def test_public_bank_reference_covers_manifest_and_records(bank):
    validate_bank_reference(bank, bank.reference())
    changed = bank.reference().model_copy(update={"records_digest": "sha256:" + "0" * 64})
    with pytest.raises(ValidationError, match="reference"):
        validate_bank_reference(bank, changed)


@pytest.mark.parametrize("permission", [None, False])
def test_unknown_or_forbidden_redistribution_blocks(bank, permission):
    altered = bank.model_copy(
        update={"license": bank.license.model_copy(update={"redistribution_allowed": permission})}
    )
    with pytest.raises(PolicyBlocked, match="permission"):
        require_publicly_bundlable(altered)


def test_private_bank_cannot_be_publicly_bundled(bank):
    raw = bank.model_dump(mode="json")
    raw["visibility"] = "private"
    raw["license"]["redistribution_allowed"] = None
    private = BankManifest.model_validate(raw)
    with pytest.raises(PolicyBlocked, match="private"):
        require_publicly_bundlable(private)


def test_real_redline_bank_must_be_private(bank):
    raw = bank.model_dump(mode="json")
    raw["role"] = "release_redline"
    with pytest.raises(PydanticValidationError, match="private"):
        BankManifest.model_validate(raw)


def test_mock_bank_is_not_mistaken_for_real_holdout(mock_bank):
    require_publicly_bundlable(mock_bank)
    assert mock_bank.role == "mock_release_redline"


def test_capture_requires_balanced_complete_pairs(bank):
    raw = bank.model_dump(mode="json")
    raw["records"] = raw["records"][:-1]
    with pytest.raises(PydanticValidationError, match=r"balanced|pair"):
        BankManifest.model_validate(raw)
