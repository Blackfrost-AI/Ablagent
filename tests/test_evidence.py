import pytest
from pydantic import ValidationError as PydanticValidationError

from ablagent.artifacts import load_tensors
from ablagent.canonical import read_json
from ablagent.errors import ValidationError
from ablagent.evidence import (
    EvidenceBundle,
    MetricResult,
    TrialRecord,
    TrialReference,
    bind_compiled,
    promote_trial,
    record_trial,
    verify_evidence,
    write_evidence,
)
from ablagent.synthetic import observation_digest, observe_synthetic
from tests.conftest import compiled_case, dump_json


def make_bundle(prepared, compiled, trial):
    store, graph, tensors, _, _, _ = prepared
    observation = observe_synthetic(
        graph, tensors, load_tensors(store.path(compiled.candidate.path)), "diagnostic"
    )
    metrics = (
        MetricResult(
            category="structural",
            name="candidate_audit",
            outcome="PASS",
            score=1,
            observed=1,
            eligible=1,
            evidence_digest=compiled.structural_check.target_digest_after,
            details="Verified structural audit.",
        ),
        MetricResult(
            category="evaluation",
            name="synthetic_diagnostics",
            outcome="PASS",
            score=1,
            observed=1,
            eligible=1,
            evidence_digest=observation_digest(observation),
            details="Verified synthetic diagnostic.",
        ),
    )
    return EvidenceBundle(
        binding=bind_compiled(f"{compiled.candidate.artifact_id}.manifest.json", compiled),
        observations=(observation,),
        metrics=metrics,
        trials=(TrialReference(path=f"{trial.trial_id}.trial.json", digest=trial.digest()),),
        environment={"backend": "numpy-cpu"},
        decoding={"mode": "analytic"},
    )


def test_evidence_replays_without_database(prepared):
    store = prepared[0]
    compiled, trial = compiled_case(prepared, record=False)
    bundle = make_bundle(prepared, compiled, trial)
    retained_trial = trial.model_copy(update={"metrics": bundle.metrics})
    record_trial(store, retained_trial)
    bundle = bundle.model_copy(
        update={
            "trials": (
                TrialReference(path=f"{trial.trial_id}.trial.json", digest=retained_trial.digest()),
            )
        }
    )
    relative = write_evidence(store, bundle, "candidate")
    assert verify_evidence(store, relative) == bundle


@pytest.mark.parametrize(
    "target", ["baseline", "candidate", "manifest", "direction", "trial", "observation"]
)
def test_evidence_and_artifact_tampering_detected(prepared, target):
    store = prepared[0]
    compiled, trial = compiled_case(prepared)
    bundle = make_bundle(prepared, compiled, trial)
    relative = write_evidence(store, bundle, "candidate")
    if target in {"baseline", "candidate"}:
        reference = compiled.baseline if target == "baseline" else compiled.candidate
        path = store.path(reference.path)
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    elif target == "manifest":
        path = store.path("candidate.manifest.json")
        raw = read_json(path)
        raw["efficacy"] = "measured"
        dump_json(path, raw)
    elif target == "direction":
        path = store.path("candidate.manifest.json")
        raw = read_json(path)
        raw["direction_digest"] = "sha256:" + "0" * 64
        dump_json(path, raw)
    elif target == "trial":
        path = store.path("candidate-trial.trial.json")
        raw = read_json(path)
        raw["reason"] = "tampered retained trial"
        dump_json(path, raw)
    else:
        path = store.path(relative)
        raw = read_json(path)
        raw["observations"][0]["candidate_output"] = [9.0, 9.0, 9.0]
        dump_json(path, raw)
    with pytest.raises(ValidationError):
        if target in {"baseline", "candidate", "manifest", "direction"}:
            store.verify_compiled("candidate.manifest.json")
        else:
            verify_evidence(store, relative)


def test_negative_trial_is_retained_and_cannot_be_promoted(prepared):
    store = prepared[0]
    compiled, _ = compiled_case(prepared)
    binding = bind_compiled("candidate.manifest.json", compiled)
    rejected = TrialRecord(
        trial_id="rejected",
        status="rejected",
        binding=binding,
        reason="Analytic fixture rejected by structural test.",
        retained_outputs=("negative.json",),
    )
    relative = record_trial(store, rejected)
    assert store.path(relative).exists()
    with pytest.raises(ValidationError, match="cannot be promoted"):
        promote_trial(rejected, "promoted")


def test_promotion_preserves_parent_lineage(prepared):
    compiled, _ = compiled_case(prepared)
    metric = MetricResult(
        category="structural",
        name="candidate_audit",
        outcome="PASS",
        score=1,
        observed=1,
        eligible=1,
        evidence_digest=compiled.structural_check.target_digest_after,
        details="Structural fixture passed.",
    )
    original = TrialRecord(
        trial_id="original",
        status="completed",
        binding=bind_compiled("candidate.manifest.json", compiled),
        reason="Completed analytic test.",
        metrics=(metric,),
    )
    promoted = promote_trial(original, "promoted")
    assert original.status == "completed"
    assert promoted.parent_trial_id == "original"
    assert promoted.status == "promoted"


def test_metrics_cannot_pass_without_complete_denominator():
    with pytest.raises(PydanticValidationError, match="denominator"):
        MetricResult(
            category="redline",
            name="cyber",
            outcome="PASS",
            score=1,
            evidence_digest="sha256:" + "0" * 64,
            details="Incomplete fixture.",
        )


@pytest.mark.parametrize("observed,eligible", [(0, 1), (0, 0), (1, 0)])
def test_metrics_cannot_pass_with_incomplete_or_empty_denominator(observed, eligible):
    with pytest.raises(PydanticValidationError):
        MetricResult(
            category="redline",
            name="cyber",
            outcome="PASS",
            score=1,
            observed=observed,
            eligible=eligible,
            evidence_digest="sha256:" + "0" * 64,
            details="Incomplete or empty fixture.",
        )


def test_bundle_metrics_must_be_retained_in_its_trial(prepared):
    store = prepared[0]
    compiled, trial = compiled_case(prepared)
    bundle = make_bundle(prepared, compiled, trial)
    relative = write_evidence(store, bundle, "candidate")
    with pytest.raises(ValidationError, match="metrics"):
        verify_evidence(store, relative)
