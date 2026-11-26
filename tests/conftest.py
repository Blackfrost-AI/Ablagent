import json
import os
import socket
from pathlib import Path

import pytest

from ablagent.artifacts import ArtifactStore
from ablagent.banks import BankManifest
from ablagent.canonical import read_json
from ablagent.evidence import TrialRecord, bind_compiled, record_trial
from ablagent.interventions import InterventionSpec
from ablagent.synthetic import derive_contrast, reference_graph, reference_tensors


@pytest.fixture
def bank() -> BankManifest:
    path = Path(__file__).parents[1] / "examples/banks/synthetic-capture.bank.json"
    return BankManifest.model_validate(read_json(path))


@pytest.fixture
def mock_bank() -> BankManifest:
    path = Path(__file__).parents[1] / "examples/banks/synthetic-release-mock.bank.json"
    return BankManifest.model_validate(read_json(path))


@pytest.fixture
def prepared(tmp_path: Path, bank: BankManifest):
    store = ArtifactStore(tmp_path)
    graph = reference_graph()
    tensors = reference_tensors()
    store.write_graph(graph, "baseline")
    baseline = store.save_checkpoint(tensors, graph, "baseline")
    candidate = derive_contrast(graph, tensors, bank, "candidate")
    bound = store.derive_candidate(candidate, graph, tensors, "reference-writer-0", "bound")
    return store, graph, tensors, baseline, candidate, bound


def compiled_case(prepared, alpha=1.0, artifact_id="candidate", *, record=True):
    store, graph, _, baseline, _, bound = prepared
    intervention = InterventionSpec(direction=bound, alpha=alpha)
    compiled = store.compile(graph, baseline, intervention, artifact_id)
    trial = TrialRecord(
        trial_id=f"{artifact_id}-trial",
        status="completed",
        binding=bind_compiled(f"{artifact_id}.manifest.json", compiled),
        reason="Test fixture.",
    )
    if record:
        record_trial(store, trial)
    return compiled, trial


def dump_json(path: Path, raw: dict) -> None:
    path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")), encoding="utf-8")


@pytest.fixture(autouse=True)
def offline_test_transport(monkeypatch):
    """Tests must use transport stubs, never actual sockets."""

    def denied(*args, **kwargs):
        raise AssertionError("network is forbidden in offline tests")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("ABLAGENT_REQUIRE_REAL_TESTS") == "1":
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter and reporter.stats.get("skipped"):
            session.exitstatus = 1
