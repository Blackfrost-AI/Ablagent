import json
from pathlib import Path

from ablagent.canonical import read_json
from ablagent.cli import main
from ablagent.conformance import run_conformance
from scripts.export_schemas import CONTRACTS, check_schemas, export_schemas


def test_conformance_cli_passes_real_fixtures(tmp_path, capsys):
    fixtures = Path(__file__).parents[1] / "spec/dwmip/v0/conformance"
    assert (
        main(
            [
                "conformance",
                "--fixtures",
                str(fixtures),
                "--run-dir",
                str(tmp_path / "conformance"),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["count"] == 13
    assert report["claim_scope"] == "synthetic_protocol_only"
    assert all(case["passed"] for case in report["results"])
    assert any(
        case["observed"] == "reject" and case["expected"] == "reject" for case in report["results"]
    )


def test_conformance_unexpected_acceptance_fails(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    operations = {
        "synthetic": "matrix",
        "manifests": "reference_graph",
        "identity": "matching_direction",
        "verification": "valid_compile",
    }
    for suite, operation in operations.items():
        (fixtures / f"{suite}.json").write_text(
            json.dumps(
                {
                    "schema_version": f"dwmip.conformance.{suite}.v0",
                    "cases": [
                        {
                            "case_id": f"unexpected-{suite}",
                            "operation": operation,
                            "outcome": "reject",
                            "baseline": [[1]],
                            "expected_unit_vector": [1],
                            "alpha": 1,
                            "expected_writer": [[0]],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    report = run_conformance(fixtures, tmp_path / "run")
    assert report["passed"] is False
    assert report["count"] == 4
    assert all(not case["passed"] for case in report["results"])
    assert all(case["observed"] == "accept" for case in report["results"])


def test_conformance_missing_fixture_is_a_cli_error(tmp_path, capsys):
    assert (
        main(
            [
                "conformance",
                "--fixtures",
                str(tmp_path / "missing"),
                "--run-dir",
                str(tmp_path / "run"),
            ]
        )
        == 1
    )
    assert "invalid JSON input" in capsys.readouterr().out


def test_exported_schemas_are_current():
    schema_dir = Path(__file__).parents[1] / "spec/dwmip/v0/schema"
    assert set(CONTRACTS) == {
        path.name.removesuffix(".schema.json") for path in schema_dir.glob("*.json")
    }
    for name, model in CONTRACTS.items():
        assert read_json(schema_dir / f"{name}.schema.json") == model.model_json_schema(
            mode="validation"
        )
    before = {path: path.read_bytes() for path in schema_dir.glob("*.json")}
    assert check_schemas(schema_dir) == []
    assert {path: path.read_bytes() for path in schema_dir.glob("*.json")} == before


def test_schema_check_detects_drift_without_writing(tmp_path):
    export_schemas(tmp_path)
    path = tmp_path / "graph.schema.json"
    drifted = b'{"stale":true}\n'
    path.write_bytes(drifted)
    assert any("stale schema" in problem for problem in check_schemas(tmp_path))
    assert path.read_bytes() == drifted
