import pytest
from pydantic import ValidationError as PydanticValidationError

from ablagent.artifacts import file_digest, load_tensors
from ablagent.canonical import digest_model, read_json, write_json
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
from ablagent.mock_redlines import FileReference
from ablagent.planner import ExperimentRequest, propose_experiment
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


def test_evidence_replays_without_database(prepared, mock_bank):
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
    verified = verify_evidence(store, relative)
    assert verified.evidence == bundle
    receipt = verified.verification
    assert receipt.store_root == str(store.root.resolve())
    assert receipt.evidence_path == str(store.path(relative).resolve())
    assert receipt.evidence_id == receipt.evidence_digest == digest_model("evidence", bundle)
    assert receipt.binding == bundle.binding
    assert receipt.bank_digest == "sha256:" + "0" * 64
    bank_bound = verify_evidence(store, relative, release_bank=mock_bank)
    assert bank_bound.verification.bank_digest == mock_bank.digest()


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
    write_json(store.path("negative.json"), {"rejected": True})
    rejected = TrialRecord(
        trial_id="rejected",
        status="rejected",
        binding=binding,
        reason="Analytic fixture rejected by structural test.",
        retained_outputs=(
            FileReference(
                path="negative.json", file_digest=file_digest(store.path("negative.json"))
            ),
        ),
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


def test_retained_trial_output_must_exist_and_match_its_digest(prepared):
    store = prepared[0]
    compiled, trial = compiled_case(prepared)
    bundle = make_bundle(prepared, compiled, trial)
    relative = write_evidence(store, bundle, "candidate")
    record = TrialRecord.model_validate(read_json(store.path("candidate-trial.trial.json")))
    retained = TrialRecord(
        trial_id="retained-output",
        parent_trial_id=record.trial_id,
        status="promoted",
        binding=record.binding,
        reason="Selected lineage output integrity fixture.",
        metrics=bundle.metrics,
        retained_outputs=(
            FileReference(
                path="candidate.manifest.json",
                file_digest=file_digest(store.path("candidate.manifest.json")),
            ),
        ),
    )
    record_trial(store, retained)
    bundle = bundle.model_copy(update={"trials": (*bundle.trials, retained.reference())})
    relative = write_evidence(store, bundle, "retained-output")
    output_path = store.path(retained.retained_outputs[0].path)
    output_path.write_bytes(output_path.read_bytes() + b"\n")
    with pytest.raises(ValidationError, match="retained trial output"):
        verify_evidence(store, relative)


def test_missing_parent_and_unused_trial_references_fail_closed(prepared):
    store = prepared[0]
    compiled, _ = compiled_case(prepared)
    parent = TrialRecord(
        trial_id="parent-with-audit",
        status="completed",
        binding=bind_compiled("candidate.manifest.json", compiled),
        reason="Complete parent lineage fixture.",
        metrics=(
            MetricResult(
                category="structural",
                name="candidate_audit",
                outcome="PASS",
                score=1,
                observed=1,
                eligible=1,
                evidence_digest=compiled.structural_check.target_digest_after,
                details="Parent structural audit.",
            ),
        ),
    )
    child = promote_trial(parent, "selected")
    record_trial(store, parent)
    record_trial(store, child)
    bundle = make_bundle(prepared, compiled, child)
    bundle = bundle.model_copy(update={"trials": (child.reference(),)})
    relative = write_evidence(store, bundle, "orphan")
    with pytest.raises(ValidationError, match="parent chain"):
        verify_evidence(store, relative)


def test_retained_plan_must_match_compiled_intervention(prepared):
    store = prepared[0]
    compiled, trial = compiled_case(prepared)
    bound = prepared[5]
    unrelated = propose_experiment(
        ExperimentRequest(request_id="unrelated-plan", direction=bound, alpha_values=(0.25,))
    )
    write_json(store.path("unrelated.plan.json"), unrelated)
    original = TrialRecord.model_validate(read_json(store.path("candidate-trial.trial.json")))
    altered = original.model_copy(
        update={
            "trial_id": "unrelated-plan-trial",
            "plan_digest": unrelated.digest(),
            "plan_reference": FileReference(
                path="unrelated.plan.json",
                file_digest=file_digest(store.path("unrelated.plan.json")),
            ),
            "metrics": make_bundle(prepared, compiled, trial).metrics,
        }
    )
    record_trial(store, altered)
    bundle = make_bundle(prepared, compiled, trial).model_copy(
        update={"trials": (altered.reference(),)}
    )
    relative = write_evidence(store, bundle, "unrelated-plan")
    with pytest.raises(ValidationError, match="compiled intervention"):
        verify_evidence(store, relative)


def test_promoted_lineage_cannot_switch_plan_identity(prepared):
    store = prepared[0]
    compiled, trial = compiled_case(prepared)
    bound = prepared[5]
    plan = propose_experiment(
        ExperimentRequest(request_id="selected-plan", direction=bound, alpha_values=(1.0,))
    )
    write_json(store.path("selected.plan.json"), plan)
    reference = FileReference(
        path="selected.plan.json", file_digest=file_digest(store.path("selected.plan.json"))
    )
    parent = trial.model_copy(
        update={
            "trial_id": "planned-parent",
            "plan_digest": plan.digest(),
            "plan_reference": reference,
            "metrics": make_bundle(prepared, compiled, trial).metrics,
        }
    )
    record_trial(store, parent)
    child = promote_trial(parent, "switched-plan")
    child = child.model_copy(
        update={
            "plan_digest": None,
            "plan_reference": None,
            "metrics": make_bundle(prepared, compiled, trial).metrics,
        }
    )
    record_trial(store, child)
    bundle = make_bundle(prepared, compiled, trial).model_copy(
        update={"trials": (parent.reference(), child.reference())}
    )
    relative = write_evidence(store, bundle, "switched-plan")
    with pytest.raises(ValidationError, match=r"cannot change.*plan identity"):
        verify_evidence(store, relative)


def test_typed_plan_reference_is_verified(prepared):
    store = prepared[0]
    compiled, trial = compiled_case(prepared)
    record = TrialRecord.model_validate(read_json(store.path("candidate-trial.trial.json")))
    write_json(store.path("other.json"), {"not_a_plan": True})
    altered = record.model_copy(
        update={
            "trial_id": "plan-check",
            "plan_digest": "sha256:" + "0" * 64,
            "plan_reference": FileReference(
                path="other.json", file_digest=file_digest(store.path("other.json"))
            ),
        }
    )
    record_trial(store, altered)
    bundle = make_bundle(prepared, compiled, trial)
    bundle = bundle.model_copy(update={"trials": (altered.reference(),)})
    relative = write_evidence(store, bundle, "bad-plan")
    with pytest.raises(ValidationError, match="retained plan"):
        verify_evidence(store, relative)
