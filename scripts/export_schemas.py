"""Export the runtime contracts as JSON Schema without a second schema source."""

import argparse
import sys
from pathlib import Path

from ablagent.banks import BankManifest, BankRecord, BankReference, LicenseTerms
from ablagent.canonical import canonical_bytes, read_json
from ablagent.contracts import Contract
from ablagent.evidence import (
    ArtifactBinding,
    EvidenceBundle,
    MetricResult,
    TrialRecord,
    TrialReference,
)
from ablagent.identity import (
    AdapterCapabilities,
    GraphSpec,
    SiteCoordinate,
    SiteSpec,
    SpaceSpec,
    WriterSpec,
)
from ablagent.interventions import (
    ArtifactReference,
    BoundDirection,
    CompiledArtifact,
    DirectionCandidate,
    InterventionSpec,
    StructuralCheck,
)
from ablagent.mock_redlines import (
    FileReference,
    MockRedlineAttestation,
    MockRedlineRaw,
    RawEvidenceReference,
)
from ablagent.planner import ExperimentRequest, ProposedPlan
from ablagent.policy import EligibilityDecision, EligibilityRequest, GateResult
from ablagent.synthetic import SyntheticObservation

CONTRACTS = {
    "contract": Contract,
    "site-coordinate": SiteCoordinate,
    "graph": GraphSpec,
    "site": SiteSpec,
    "space": SpaceSpec,
    "writer": WriterSpec,
    "artifact-reference": ArtifactReference,
    "bank": BankManifest,
    "bank-record": BankRecord,
    "bank-reference": BankReference,
    "license-terms": LicenseTerms,
    "direction-candidate": DirectionCandidate,
    "bound-direction": BoundDirection,
    "intervention": InterventionSpec,
    "experiment-request": ExperimentRequest,
    "proposed-plan": ProposedPlan,
    "structural-check": StructuralCheck,
    "compiled-artifact": CompiledArtifact,
    "artifact-binding": ArtifactBinding,
    "observation": SyntheticObservation,
    "adapter-capabilities": AdapterCapabilities,
    "metric-result": MetricResult,
    "file-reference": FileReference,
    "raw-evidence-reference": RawEvidenceReference,
    "mock-redline-raw": MockRedlineRaw,
    "mock-redline-attestation": MockRedlineAttestation,
    "trial": TrialRecord,
    "trial-reference": TrialReference,
    "evidence": EvidenceBundle,
    "gate-result": GateResult,
    "eligibility-request": EligibilityRequest,
    "eligibility-decision": EligibilityDecision,
}


def check_schemas(destination: Path) -> list[str]:
    """Read-only drift check; never writes to tracked manifests."""
    errors = []
    existing = {path.name for path in destination.glob("*.schema.json")}
    expected = {f"{name}.schema.json" for name in CONTRACTS}
    if existing != expected:
        errors.append("schema file set differs from the runtime contract registry")
    for name, contract in CONTRACTS.items():
        path = destination / f"{name}.schema.json"
        if not path.is_file():
            errors.append(f"missing schema: {path.name}")
            continue
        if read_json(path) != contract.model_json_schema(mode="validation"):
            errors.append(f"stale schema: {path.name}")
        if (
            path.read_bytes()
            != canonical_bytes(contract.model_json_schema(mode="validation")) + b"\n"
        ):
            errors.append(f"noncanonical schema bytes: {path.name}")
    return errors


def export_schemas(destination: Path) -> list[str]:
    import os

    written = []
    destination.mkdir(parents=True, exist_ok=True)
    for name, contract in CONTRACTS.items():
        schema = contract.model_json_schema(mode="validation")
        path = destination / f"{name}.schema.json"
        expected = canonical_bytes(schema) + b"\n"
        if path.exists() and path.read_bytes() == expected:
            continue
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(expected)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        written.append(path.name)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check schemas without writing")
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    schema_root = project_root / "spec" / "dwmip" / "v0" / "schema"
    if arguments.check:
        problems = check_schemas(schema_root)
        for problem in problems:
            print(problem, file=sys.stderr)
        raise SystemExit(bool(problems))
    for schema_name in export_schemas(schema_root):
        print(schema_name)
