import pytest

from ablagent.evidence import REQUIRED_REDLINE
from ablagent.policy import EligibilityRequest, evaluate_eligibility
from tests.conftest import compiled_case
from tests.test_evidence import make_bundle


def request_for(prepared, mock_bank):
    compiled, trial = compiled_case(prepared)
    bundle = make_bundle(prepared, compiled, trial)
    return EligibilityRequest(visibility="public", evidence=bundle, release_bank=mock_bank)


def test_missing_required_redlines_block_even_with_perfect_other_scores(prepared, mock_bank):
    decision = evaluate_eligibility(request_for(prepared, mock_bank))
    assert decision.status == "blocked"
    assert {gate.outcome for gate in decision.gates if gate.gate.startswith("redline.")} == {
        "MISSING"
    }
    assert decision.authorization == "not_granted"
    assert not any(gate.gate == "structural" and gate.outcome != "PASS" for gate in decision.gates)


@pytest.mark.parametrize("outcome", ["FAIL", "NOT_RUN", "ERROR", "MISSING"])
def test_every_nonpass_redline_blocks(prepared, mock_bank, outcome):
    request = request_for(prepared, mock_bank)
    metrics = list(request.evidence.metrics)
    for name in sorted(REQUIRED_REDLINE):
        from ablagent.evidence import MetricResult

        metrics.append(
            MetricResult(
                category="redline",
                name=name,
                outcome="PASS" if name != "cyber" else outcome,
                score=1 if name != "cyber" else None,
                observed=1 if name != "cyber" else None,
                eligible=1 if name != "cyber" else None,
                evidence_digest="sha256:" + "0" * 64,
                details="Synthetic policy fixture only.",
            )
        )
    request = request.model_copy(
        update={"evidence": request.evidence.model_copy(update={"metrics": tuple(metrics)})}
    )
    decision = evaluate_eligibility(request)
    assert decision.status == "blocked"
    gate = next(gate for gate in decision.gates if gate.gate == "redline.cyber")
    assert gate.outcome == outcome


def test_passing_policy_fixture_is_not_release_authorization(prepared, mock_bank):
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
            details="Inert synthetic policy fixture only.",
        )
        for name in sorted(REQUIRED_REDLINE)
    )
    request = request.model_copy(
        update={
            "evidence": request.evidence.model_copy(
                update={"metrics": request.evidence.metrics + redlines}
            )
        }
    )
    decision = evaluate_eligibility(request)
    assert decision.status == "eligible"
    assert decision.authorization == "not_granted"
    assert decision.assurance_scope == "synthetic_policy_conformance_only"


def test_private_bank_blocks_public_eligibility(prepared, bank):
    request = request_for(prepared, bank)
    raw = bank.model_dump(mode="json")
    raw["visibility"] = "private"
    raw["license"]["redistribution_allowed"] = None
    from ablagent.banks import BankManifest

    request = request.model_copy(update={"release_bank": BankManifest.model_validate(raw)})
    decision = evaluate_eligibility(request)
    assert decision.status == "blocked"
    assert (
        next(gate for gate in decision.gates if gate.gate == "bank_publication").outcome == "FAIL"
    )


def test_duplicate_contradictory_redline_is_fail_closed(prepared, mock_bank):
    from ablagent.errors import PolicyBlocked
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
    failing = passing.model_copy(update={"outcome": "FAIL", "score": None, "observed": None})
    request = request.model_copy(
        update={
            "evidence": request.evidence.model_copy(
                update={"metrics": (*request.evidence.metrics, passing, failing)}
            )
        }
    )
    with pytest.raises(PolicyBlocked, match="duplicate"):
        evaluate_eligibility(request)
    reversed_request = request.model_copy(
        update={
            "evidence": request.evidence.model_copy(
                update={"metrics": (*request.evidence.metrics[:-2], failing, passing)}
            )
        }
    )
    with pytest.raises(PolicyBlocked, match="duplicate"):
        evaluate_eligibility(reversed_request)
