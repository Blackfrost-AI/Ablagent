"""Skip-redline (exploratory mode) tests: honest skipping, not silent passes."""
from __future__ import annotations

from pathlib import Path

import pytest

from ablagent.constants import SKIP_OPTIONAL_STAGES
from ablagent.control.store import ControlStore, InputError, StateError


def _store_with_s3(tmp_path: Path) -> ControlStore:
    store = ControlStore.initialize(tmp_path / "control",
                                     run={"run_id": "t", "pack_id": "toy"},
                                     principal={"kind": "test"})
    for stage in ("S0_INTAKE", "S2_DIRECTIONS", "S3_DERISK"):
        control = store.load()
        store.authorize_direct(stage, ["start"], expected_revision=control["revision"], principal="t")
        store.transition(stage, "start", expected_revision=control["revision"] + 1)
        control = store.load()
        store.transition(stage, "complete", expected_revision=control["revision"])
    return store


def test_sr_is_skip_optional():
    assert "SR_RELEASE_REDLINE" in SKIP_OPTIONAL_STAGES


def test_skip_stage_records_honest_dispositions(tmp_path):
    store = _store_with_s3(tmp_path)
    control = store.load()
    store.skip_stage("SR_RELEASE_REDLINE", expected_revision=control["revision"],
                     reason="operator: exploratory run")
    control = store.load()
    sr = control["stages"]["SR_RELEASE_REDLINE"]
    assert sr["execution"] == "skipped"
    assert sr["measurement"] == "not_applicable"
    assert sr["release"] == "not_eligible"
    assert sr["evidence"][-1]["gate_id"] == "stage-skip"
    assert "exploratory" in sr["evidence"][-1]["reason"]


def test_skip_requires_reason(tmp_path):
    store = _store_with_s3(tmp_path)
    control = store.load()
    with pytest.raises(InputError, match="reason"):
        store.skip_stage("SR_RELEASE_REDLINE", expected_revision=control["revision"], reason="  ")
    with pytest.raises(InputError, match="reason"):
        store.skip_stage("SR_RELEASE_REDLINE", expected_revision=control["revision"], reason="")


def test_non_optional_stage_cannot_be_skipped(tmp_path):
    store = _store_with_s3(tmp_path)
    control = store.load()
    with pytest.raises(InputError, match="SKIP_OPTIONAL_STAGES"):
        store.skip_stage("S6_DOMAIN_VERIFY", expected_revision=control["revision"],
                         reason="operator: exploratory run")


def test_cannot_skip_after_stage_started(tmp_path):
    store = _store_with_s3(tmp_path)
    control = store.load()
    # complete S5/S6/S7 first so SR can legitimately start
    for stage in ("S5_SERVE", "S6_DOMAIN_VERIFY", "S7_CAPABILITY_VERIFY"):
        store.authorize_direct(stage, ["start"], expected_revision=control["revision"], principal="t")
        store.transition(stage, "start", expected_revision=control["revision"] + 1)
        control = store.load()
        store.transition(stage, "complete", expected_revision=control["revision"])
        control = store.load()
    store.authorize_direct("SR_RELEASE_REDLINE", ["start"],
                           expected_revision=control["revision"], principal="t")
    control = store.load()
    store.transition("SR_RELEASE_REDLINE", "start", expected_revision=control["revision"])
    control = store.load()
    with pytest.raises(StateError, match="cannot be skipped"):
        store.skip_stage("SR_RELEASE_REDLINE", expected_revision=control["revision"],
                         reason="operator: exploratory run")


def test_skipped_stage_satisfies_dag_for_successors(tmp_path):
    store = _store_with_s3(tmp_path)
    control = store.load()
    store.skip_stage("SR_RELEASE_REDLINE", expected_revision=control["revision"],
                     reason="operator: exploratory run")
    # S8 depends on SR; a skipped SR must let S8 authorize (tolerant DAG)
    control = store.load()
    # S8 also needs S3/S6/S7 completed — complete the remaining chain
    for stage in ("S5_SERVE", "S6_DOMAIN_VERIFY", "S7_CAPABILITY_VERIFY"):
        store.authorize_direct(stage, ["start"], expected_revision=control["revision"], principal="t")
        store.transition(stage, "start", expected_revision=control["revision"] + 1)
        control = store.load()
        store.transition(stage, "complete", expected_revision=control["revision"])
        control = store.load()
    store.authorize_direct("S8_PRODUCTIZE", ["start"], expected_revision=control["revision"], principal="t")
    store.transition("S8_PRODUCTIZE", "start", expected_revision=control["revision"] + 1)
    control = store.load()
    assert control["stages"]["S8_PRODUCTIZE"]["execution"] == "running"


def test_release_check_tolerates_skipped_sr_but_keeps_not_eligible(tmp_path):
    store = _store_with_s3(tmp_path)
    control = store.load()
    store.skip_stage("SR_RELEASE_REDLINE", expected_revision=control["revision"],
                     reason="operator: exploratory run")
    for stage in ("S5_SERVE", "S6_DOMAIN_VERIFY", "S7_CAPABILITY_VERIFY"):
        control = store.load()
        store.authorize_direct(stage, ["start"], expected_revision=control["revision"], principal="t")
        store.transition(stage, "start", expected_revision=control["revision"] + 1)
        control = store.load()
        store.transition(stage, "complete", expected_revision=control["revision"])
        control = store.load()
        from ablagent.control.codec import digest
        store.record_measurement(stage, {"passed": True}, expected_revision=control["revision"],
                                 gate_id=f"{stage}-g", criteria_digest="sha256:" + "a" * 64)
        control = store.load()
    for stage in ("S8_PRODUCTIZE", "S9_CARD_DRAFT"):
        control = store.load()
        store.authorize_direct(stage, ["start"], expected_revision=control["revision"], principal="t")
        store.transition(stage, "start", expected_revision=control["revision"] + 1)
        control = store.load()
        store.transition(stage, "complete", expected_revision=control["revision"])
        control = store.load()
        store.record_measurement(stage, {"passed": True}, expected_revision=control["revision"],
                                 gate_id=f"{stage}-g", criteria_digest="sha256:" + "a" * 64)
        control = store.load()
    store.release_check(expected_revision=control["revision"])
    control = store.load()
    assert control["stages"]["SR_RELEASE_REDLINE"]["release"] == "not_eligible"
    assert control["stages"]["S8_PRODUCTIZE"]["release"] == "release_ready"
    # operator acceptance must refuse the skipped-SR stage
    with pytest.raises(StateError, match="not ready"):
        store.accept_release(expected_revision=control["revision"], principal="operator")
