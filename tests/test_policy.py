import pytest

from ablagent.banks import BankManifest
from ablagent.canonical import read_json
from ablagent.errors import ValidationError
from ablagent.evidence import (
    REQUIRED_REDLINE,
    EvidenceBundle,
    MetricResult,
    TrialRecord,
    TrialReference,
    record_trial,
    verify_evidence,
    write_evidence,
)
from ablagent.mock_redlines import mock_redline_metrics, write_mock_redlines
from ablagent.policy import EligibilityRequest, evaluate_eligibility
from tests.conftest import compiled_case
from tests.test_evidence import make_bundle


def request_bundle(prepared, request):
    return EvidenceBundle.model_validate(read_json(prepared[0].path(request.evidence_path)))


def write_bundle(prepared, request, bundle, artifact_id):
    selected = TrialRecord.model_validate(read_json(prepared[0].path(bundle.trials[-1].path)))
    retained = selected.model_copy(
        update={"trial_id": f"{artifact_id}-selected", "metrics": bundle.metrics}
    )
    record_trial(prepared[0], retained)
    bundle = bundle.model_copy(update={"trials": (*bundle.trials[:-1], retained.reference())})
    write_evidence(prepared[0], bundle, artifact_id)
    return request.model_copy(update={"evidence_path": f"{artifact_id}.evidence.json"})


def request_for(prepared, mock_bank):
    compiled, trial = compiled_case(prepared, record=False)
    bundle = make_bundle(prepared, compiled, trial)
    retained = trial.model_copy(update={"metrics": bundle.metrics})
    record_trial(prepared[0], retained)
    bundle = bundle.model_copy(update={"trials": (retained.reference(),)})
    write_evidence(prepared[0], bundle, "candidate")
    return EligibilityRequest(
        visibility="public",
        evidence_path="candidate.evidence.json",
        release_bank=mock_bank,
    )


def attested_request(prepared, mock_bank, visibility="private"):
    request = request_for(prepared, mock_bank)
    bundle = request_bundle(prepared, request)
    attestation = write_mock_redlines(
        prepared[0], mock_bank, bundle.binding, "mock-redline.raw.json"
    )
    metrics = bundle.metrics + tuple(mock_redline_metrics(attestation))
    selected = TrialRecord(
        trial_id="attested-candidate",
        parent_trial_id="candidate-trial",
        status="promoted",
        binding=bundle.binding,
        reason="Selected trial with reproducible inert policy fixtures.",
        metrics=metrics,
    )
    record_trial(prepared[0], selected)
    evidence = bundle.model_copy(
        update={
            "metrics": metrics,
            "mock_redlines": attestation,
            "trials": (
                *bundle.trials,
                TrialReference(path=selected.reference_path(), digest=selected.digest()),
            ),
        }
    )
    write_evidence(prepared[0], evidence, "attested")
    return request.model_copy(
        update={
            "evidence_path": "attested.evidence.json",
            "visibility": visibility,
        }
    )


def test_missing_required_redlines_block_even_with_perfect_other_scores(prepared, mock_bank):
    decision = evaluate_eligibility(request_for(prepared, mock_bank), prepared[0])
    assert decision.status == "blocked"
    assert {gate.outcome for gate in decision.gates if gate.gate.startswith("redline.")} == {
        "MISSING"
    }
    assert decision.authorization == "not_granted"
    assert not any(gate.gate == "structural" and gate.outcome != "PASS" for gate in decision.gates)


@pytest.mark.parametrize("outcome", ["FAIL", "NOT_RUN", "ERROR", "MISSING"])
def test_every_nonpass_redline_blocks(prepared, mock_bank, outcome):
    request = request_for(prepared, mock_bank)
    metrics = list(request_bundle(prepared, request).metrics)
    for name in sorted(REQUIRED_REDLINE):
        metrics.append(
            MetricResult(
                category="redline_observation",
                name=name,
                outcome=outcome,
                score=None,
                observed=None,
                eligible=None,
                evidence_digest="sha256:" + "0" * 64,
                details="Synthetic policy fixture only.",
            )
        )
    bundle = request_bundle(prepared, request).model_copy(update={"metrics": tuple(metrics)})
    request = write_bundle(prepared, request, bundle, f"nonpass-{outcome.lower()}")
    decision = evaluate_eligibility(request, prepared[0])
    assert decision.status == "blocked"
    gate = next(gate for gate in decision.gates if gate.gate == "redline.cyber")
    assert gate.outcome == outcome


def test_passing_observations_require_attestation(prepared, mock_bank):
    request = request_for(prepared, mock_bank)
    passing = MetricResult(
        category="redline_observation",
        name="cyber",
        outcome="PASS",
        score=1,
        observed=1,
        eligible=1,
        evidence_digest=mock_bank.digest(),
        details="Unsupported passing observation.",
    )
    evidence = request_bundle(prepared, request).model_copy(
        update={"metrics": (*request_bundle(prepared, request).metrics, passing)}
    )
    write_evidence(prepared[0], evidence, "policy-passing-observation")
    request = request.model_copy(
        update={"evidence_path": "policy-passing-observation.evidence.json"}
    )
    with pytest.raises(ValidationError, match="passing redline observations"):
        evaluate_eligibility(request, prepared[0])


def test_verifier_rejects_passing_observations(prepared, mock_bank):
    request = request_for(prepared, mock_bank)
    passing = MetricResult(
        category="redline_observation",
        name="cyber",
        outcome="PASS",
        score=1,
        observed=1,
        eligible=1,
        evidence_digest=mock_bank.digest(),
        details="Unsupported passing observation.",
    )
    evidence = request_bundle(prepared, request).model_copy(
        update={"metrics": (*request_bundle(prepared, request).metrics, passing)}
    )
    write_evidence(prepared[0], evidence, "passing-observation")
    with pytest.raises(ValidationError, match="passing redline observations"):
        verify_evidence(prepared[0], "passing-observation.evidence.json", release_bank=mock_bank)


def test_serialized_request_rejects_verification_metadata(prepared, mock_bank):
    request = request_for(prepared, mock_bank)
    raw = request.model_dump(mode="json")
    raw["evidence"] = {"evidence": request_bundle(prepared, request).model_dump(mode="json")}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        EligibilityRequest.model_validate(raw)
    schema = EligibilityRequest.model_json_schema(mode="validation")
    assert "VerifiedEvidence" not in str(schema)
    assert "EvidenceVerification" not in str(schema)


def test_private_attested_fixture_is_not_release_authorization(prepared, mock_bank):
    request = attested_request(prepared, mock_bank)
    decision = evaluate_eligibility(request, prepared[0])
    assert decision.status == "eligible"
    assert decision.authorization == "not_granted"
    assert decision.assurance_scope == "synthetic_policy_conformance_only"
    assert next(gate for gate in decision.gates if gate.gate == "bank_publication").outcome == (
        "NOT_APPLICABLE"
    )
    assert all(gate.outcome == "PASS" for gate in decision.gates if gate.gate != "bank_publication")


@pytest.mark.parametrize("outcome", ["FAIL", "NOT_RUN", "ERROR", "MISSING"])
def test_attested_evidence_cannot_mask_nonpass_observation(prepared, mock_bank, outcome):
    request = attested_request(prepared, mock_bank)
    evidence = request_bundle(prepared, request)
    selected = TrialRecord.model_validate(read_json(prepared[0].path(evidence.trials[-1].path)))
    observed = MetricResult(
        category="redline_observation",
        name="cyber",
        outcome=outcome,
        score=None,
        observed=None,
        eligible=None,
        evidence_digest="sha256:" + "0" * 64,
        details="Contradictory observation must take precedence over mock attestation.",
    )
    metrics = (*evidence.metrics, observed)
    retained = selected.model_copy(
        update={"trial_id": "contradictory-attested", "metrics": metrics}
    )
    record_trial(prepared[0], retained)
    updated = evidence.model_copy(
        update={
            "metrics": metrics,
            "trials": (*evidence.trials[:-1], retained.reference()),
        }
    )
    write_evidence(prepared[0], updated, "contradictory-attested")
    verify_evidence(prepared[0], "contradictory-attested.evidence.json", release_bank=mock_bank)
    request = request.model_copy(update={"evidence_path": "contradictory-attested.evidence.json"})
    decision = evaluate_eligibility(request, prepared[0])
    assert decision.status == "blocked"
    assert next(g for g in decision.gates if g.gate == "redline.cyber").outcome == outcome


def test_public_mock_fixture_remains_synthetic_only(prepared, mock_bank):
    decision = evaluate_eligibility(
        attested_request(prepared, mock_bank, visibility="public"), prepared[0]
    )
    assert decision.status == "eligible"
    assert decision.authorization == "not_granted"
    assert decision.assurance_scope == "synthetic_policy_conformance_only"
    assert next(gate for gate in decision.gates if gate.gate == "bank_publication").outcome == (
        "NOT_APPLICABLE"
    )


def test_bare_zero_digest_redline_scores_are_not_evidence(prepared, mock_bank):
    request = request_for(prepared, mock_bank)
    from ablagent.evidence import MetricResult

    redlines = tuple(
        MetricResult(
            category="redline",
            name=name,
            outcome="PASS",
            score=1,
            observed=1,
            eligible=1,
            evidence_digest="sha256:" + "0" * 64,
            details="Unsupported self-declared fixture.",
        )
        for name in sorted(REQUIRED_REDLINE)
    )
    bundle = request_bundle(prepared, request)
    evidence = bundle.model_copy(update={"metrics": bundle.metrics + redlines})
    request = write_bundle(prepared, request, evidence, "bare-redline")
    with pytest.raises(ValidationError, match="bound mock-evidence attestation"):
        evaluate_eligibility(request, prepared[0])


def test_mock_evidence_fails_closed_when_raw_or_selected_bank_changes(prepared, mock_bank):
    request = attested_request(prepared, mock_bank)
    raw_path = prepared[0].path(request_bundle(prepared, request).mock_redlines.raw.path)
    original = raw_path.read_bytes()
    raw_path.write_bytes(original + b"\n")
    with pytest.raises(ValidationError, match="raw file digest"):
        verify_evidence(prepared[0], "attested.evidence.json", release_bank=mock_bank)
    raw_path.write_bytes(original)
    changed = mock_bank.model_dump(mode="json")
    changed["records"][0]["prompt"] = "a changed record"
    with pytest.raises(ValidationError, match=r"inert fixture|selected bank"):
        verify_evidence(
            prepared[0],
            "attested.evidence.json",
            release_bank=BankManifest.model_validate(changed),
        )


def test_private_bank_blocks_public_eligibility(prepared, bank):
    request = request_for(prepared, bank)
    raw = bank.model_dump(mode="json")
    raw["visibility"] = "private"
    raw["license"]["redistribution_allowed"] = None
    changed_bank = BankManifest.model_validate(raw)
    request = request.model_copy(
        update={
            "release_bank": changed_bank,
        }
    )
    decision = evaluate_eligibility(request, prepared[0])
    assert decision.status == "blocked"
    assert (
        next(gate for gate in decision.gates if gate.gate == "bank_publication").outcome == "FAIL"
    )


def test_duplicate_contradictory_redline_is_fail_closed(prepared, mock_bank):
    from ablagent.evidence import MetricResult

    request = request_for(prepared, mock_bank)
    passing = MetricResult(
        category="redline",
        name="cyber",
        outcome="PASS",
        score=1,
        observed=1,
        eligible=1,
        evidence_digest=mock_bank.digest(),
        details="Synthetic policy fixture only.",
    )
    failing = passing.model_copy(
        update={"outcome": "FAIL", "score": None, "observed": None, "eligible": None}
    )
    bundle = request_bundle(prepared, request)
    with pytest.raises(ValueError, match="identities must be unique"):
        EvidenceBundle.model_validate(
            bundle.model_copy(update={"metrics": (*bundle.metrics, passing, failing)}).model_dump(
                mode="json"
            )
        )
    with pytest.raises(ValueError, match="identities must be unique"):
        EvidenceBundle.model_validate(
            bundle.model_copy(update={"metrics": (*bundle.metrics, failing, passing)}).model_dump(
                mode="json"
            )
        )
