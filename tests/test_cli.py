import json
import os
import subprocess
from pathlib import Path

from ablagent.artifacts import ArtifactStore
from ablagent.canonical import read_json
from ablagent.cli import main
from ablagent.evidence import TrialRecord, record_trial, verify_evidence, write_evidence
from ablagent.mock_redlines import mock_redline_metrics, write_mock_redlines

ROOT = Path(__file__).parents[1]


def test_offline_demo_verify_and_expected_blocked_gate(tmp_path, capsys):
    assert main(["demo", "--run-dir", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "synthetic_demo_complete"
    assert main(["verify", str(tmp_path / "candidate.manifest.json")]) == 0
    capsys.readouterr()
    assert main(["verify-evidence", str(tmp_path / "candidate.evidence.json")]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "gate",
                str(tmp_path / "candidate.evidence.json"),
                "--bank",
                "examples/banks/synthetic-release-mock.bank.json",
            ]
        )
        == 2
    )
    decision = json.loads(capsys.readouterr().out)
    assert decision["status"] == "blocked"
    assert decision["authorization"] == "not_granted"


def test_packaged_demo_works_outside_repository(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    command = [str(ROOT / ".venv/bin/ablagent"), "demo", "--run-dir", str(tmp_path / "run")]
    completed = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["status"] == "synthetic_demo_complete"


def test_plan_does_not_compile_or_authorize(tmp_path, capsys):
    assert main(["demo", "--run-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "plan",
                "bound.direction.json",
                "--run-dir",
                str(tmp_path),
                "--alpha",
                "0",
                "--alpha",
                "1",
                "--output",
                "two.plan.json",
            ]
        )
        == 0
    )
    proposal = json.loads(capsys.readouterr().out)
    assert proposal["execution_authority"] == "none"
    assert len(proposal["interventions"]) == 2
    assert not (tmp_path / "two.safetensors").exists()
    assert (
        main(
            [
                "compile",
                "bound.direction.json",
                "--run-dir",
                str(tmp_path),
                "--plan",
                "two.plan.json",
            ]
        )
        == 2
    )
    assert "exactly one" in capsys.readouterr().out


def test_documented_one_intervention_compile_and_relative_output(tmp_path, capsys):
    assert main(["demo", "--run-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "plan",
                "bound.direction.json",
                "--run-dir",
                str(tmp_path),
                "--alpha",
                "1",
                "--output",
                "one.plan.json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (tmp_path / "one.plan.json").exists()
    assert (
        main(
            [
                "compile",
                "bound.direction.json",
                "--run-dir",
                str(tmp_path),
                "--plan",
                "one.plan.json",
                "--artifact-id",
                "independent",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["verify", str(tmp_path / "independent.manifest.json")]) == 0
    capsys.readouterr()


def test_direction_cannot_escape_run_directory(tmp_path, capsys):
    assert main(["demo", "--run-dir", str(tmp_path / "run")]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "plan",
                "../outside.direction.json",
                "--run-dir",
                str(tmp_path / "run"),
                "--alpha",
                "1",
            ]
        )
        == 1
    )
    assert "within the run" in capsys.readouterr().out


def test_bank_input_is_size_bounded(tmp_path, capsys):
    oversized = tmp_path / "oversized.json"
    oversized.write_text('{"records":"' + "x" * (66 * 1024) + '"}', encoding="utf-8")
    assert main(["demo", "--run-dir", str(tmp_path / "run"), "--bank", str(oversized)]) == 1
    assert "limit" in capsys.readouterr().out


def test_invalid_input_is_nonzero_and_does_not_create_candidate(tmp_path, capsys):
    bad_bank = tmp_path / "bad-bank.json"
    bad_bank.write_text('{"schema_version":"dwmip.bank.v0"}', encoding="utf-8")
    assert main(["demo", "--run-dir", str(tmp_path / "run"), "--bank", str(bad_bank)]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "blocked_or_invalid"
    assert not (tmp_path / "run/candidate.safetensors").exists()


def test_cli_private_gate_accepts_verified_mock_fixture(tmp_path):
    run = tmp_path / "run"
    assert main(["demo", "--run-dir", str(run)]) == 0
    store = ArtifactStore(run)
    from ablagent.evidence import EvidenceBundle

    bundle = EvidenceBundle.model_validate(read_json(run / "candidate.evidence.json"))
    bank = ROOT / "examples/banks/synthetic-release-mock.bank.json"
    from ablagent.cli import _bank

    selected_bank = _bank(bank)
    attestation = write_mock_redlines(store, selected_bank, bundle.binding, "mock.raw.json")
    metrics = tuple(
        metric for metric in bundle.metrics if metric.category != "redline_observation"
    ) + tuple(mock_redline_metrics(attestation))
    parent = TrialRecord.model_validate(read_json(store.path(bundle.trials[-1].path)))
    selected = TrialRecord(
        trial_id="attested",
        parent_trial_id=parent.trial_id,
        status="promoted",
        binding=bundle.binding,
        plan_digest=parent.plan_digest,
        plan_reference=parent.plan_reference,
        reason="CLI inert policy wiring test; no behavioral claim.",
        metrics=metrics,
    )
    record_trial(store, selected)
    evidence = bundle.model_copy(
        update={
            "metrics": metrics,
            "mock_redlines": attestation,
            "trials": (*bundle.trials, selected.reference()),
        }
    )
    write_evidence(store, evidence, "attested")
    verify_evidence(store, "attested.evidence.json", release_bank=selected_bank)
    assert (
        main(
            [
                "verify-evidence",
                str(run / "attested.evidence.json"),
                "--bank",
                str(bank),
            ]
        )
        == 0
    )
    decision = run / "private.decision.json"
    assert (
        main(
            [
                "gate",
                str(run / "attested.evidence.json"),
                "--bank",
                str(bank),
                "--visibility",
                "private",
                "--output",
                str(decision),
            ]
        )
        == 0
    )
    assert decision.is_file()
