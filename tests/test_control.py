from __future__ import annotations

import json
from pathlib import Path

import pytest

from ablagent.control import ConcurrentRevisionError, ControlStore, IntegrityError
from ablagent.control.codec import load_json
from ablagent.packs import load_pack
from ablagent.pipeline import run_mock
from ablagent.research.corpus import derive_jsonl
from ablagent.research.notes import Note, write_note
from ablagent.research.sources import Source, SourceRegistry


def test_control_init_authorize_transition_chain(tmp_path: Path):
    store = ControlStore.initialize(tmp_path / "run")
    control = store.load()
    assert control["revision"] == 0
    control = store.authorize_direct("S0", ["start"], expected_revision=0, principal="test")
    assert control["revision"] == 1
    control = store.transition("S0", "start", expected_revision=1)
    control = store.transition("S0", "complete", expected_revision=2)
    assert control["stages"]["S0_INTAKE"]["execution"] == "completed"
    report = store.verify()
    assert report.revision == 3
    assert report.events == 3


def test_control_cas_and_anchor_fail_closed(tmp_path: Path):
    store = ControlStore.initialize(tmp_path / "run")
    store.authorize_direct("S0", ["start"], expected_revision=0, principal="test")
    with pytest.raises(ConcurrentRevisionError):
        store.authorize_direct("S1", ["start"], expected_revision=0, principal="test")
    anchor = store.anchor_path
    value = json.loads(anchor.read_text())
    value["revision"] = 99
    anchor.write_text(json.dumps(value, sort_keys=True))
    anchor.chmod(0o600)
    with pytest.raises(IntegrityError, match="anchor mismatch"):
        store.load()


def test_control_precommit_transaction_is_discarded(tmp_path: Path):
    store = ControlStore.initialize(tmp_path / "run")
    current = store.load()
    from ablagent.control.codec import canonical_bytes, digest

    prepared = dict(current)
    prepared["revision"] = 1
    marker = {
        "schema": "ablagent.transaction.v1",
        "control_hex": canonical_bytes(prepared).hex(),
        "previous_control_hex": store.control_path.read_bytes().hex(),
        "event_hex": canonical_bytes({}).hex(),
        "control_digest": digest(prepared),
    }
    store.transaction_path.write_bytes(canonical_bytes(marker))
    store.transaction_path.chmod(0o600)
    assert store.load()["revision"] == 0
    assert not store.transaction_path.exists()


def test_strict_json_rejects_duplicates_nan_and_bools():
    with pytest.raises(IntegrityError):
        load_json(b'{"x": 1, "x": 2}')
    with pytest.raises(IntegrityError):
        load_json(b'{"x": NaN}')
    with pytest.raises(IntegrityError):
        load_json(b"\xff")


def test_mock_pipeline_completes_canonical_pack_driven_dag(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    pack = load_pack(root / "packs" / "toy-domain")
    result = run_mock(pack, runs_root=tmp_path / "runs")
    assert result["completed_stages"]
    assert result["completed_stages"][-1] == "S10_DELIVER"
    run_dir = Path(result["run_directory"])
    assert (run_dir / "RUN.yaml").is_file()
    # Evidence: one mock-output per stage + gate results (S6/S7/SR), the S8
    # secret-scan receipt, and the S9 card-draft completeness gate.
    assert len(list((run_dir / "evidence").glob("*.json"))) >= len(result["completed_stages"])
    gate_files = list((run_dir / "evidence").glob("*gate-result*.json"))
    scan_files = list((run_dir / "evidence").glob("*secret-scan*.json"))
    assert len(gate_files) >= 3  # S6, S7, SR gate results recorded
    assert len(scan_files) >= 1  # S8 secret scan receipt exists
    control = ControlStore(run_dir / "control").load()
    assert all(state["execution"] == "completed" for state in control["stages"].values())
    # Security-phase invariants: gate stages accepted, release ready.
    for stage in (
        "S6_DOMAIN_VERIFY",
        "S7_CAPABILITY_VERIFY",
        "SR_RELEASE_REDLINE",
        "S8_PRODUCTIZE",
        "S9_CARD_DRAFT",
    ):
        assert control["stages"][stage]["measurement"] == "accepted", stage
        assert control["stages"][stage]["release"] == "release_ready", stage


def test_research_registry_note_and_corpus_are_provenance_bound(tmp_path: Path):
    registry = SourceRegistry(tmp_path / "research")
    source = Source("nist-csf", "https://www.nist.gov/cyberframework", "CSF", "public", "research")
    registry.register(source)
    assert registry.get("nist-csf") == source
    note = Note(
        "note-1",
        "A defensible research claim",
        ({"claim": "x"},),
        (source.source_id,),
        ("sha256:" + "a" * 64,),
    )
    note_path = write_note(tmp_path / "notes", note)
    import json

    note_value = json.loads(note_path.read_text())
    receipt = derive_jsonl(
        [note_value], tmp_path / "derived.jsonl", dataset="research", label="reference"
    )
    assert receipt["rows"] == 1
