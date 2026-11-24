"""Command-line boundary. The runtime makes no network requests."""

import argparse
import json
import sys
from importlib.resources import as_file, files
from pathlib import Path, PurePosixPath

from pydantic import ValidationError as PydanticValidationError

from ablagent import __version__
from ablagent.artifacts import ArtifactStore, load_tensors
from ablagent.banks import BankManifest
from ablagent.canonical import canonical_bytes, read_json, write_json
from ablagent.conformance import run_conformance
from ablagent.errors import AblagentError, PolicyBlocked, ValidationError
from ablagent.evidence import (
    EvidenceBundle,
    MetricResult,
    TrialRecord,
    bind_compiled,
    record_trial,
    verify_evidence,
    write_evidence,
)
from ablagent.planner import ExperimentRequest, ProposedPlan, propose_experiment
from ablagent.policy import EligibilityRequest, evaluate_eligibility
from ablagent.synthetic import (
    capabilities,
    derive_contrast,
    observation_digest,
    observe_synthetic,
    reference_graph,
    reference_tensors,
)


def _print(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(canonical_bytes(value).decode("utf-8"))


def _model(model_type, path: Path, max_bytes: int = 4 * 1024 * 1024):
    try:
        return model_type.model_validate(read_json(path, max_bytes=max_bytes))
    except PydanticValidationError as exc:
        raise ValidationError(f"invalid {model_type.__name__}: {exc}") from exc


def _bank(path: Path) -> BankManifest:
    if path.is_absolute() or path.exists():
        return _model(BankManifest, path, max_bytes=64 * 1024)
    resource = files("ablagent").joinpath("resources", "banks", path.name)
    if path.as_posix() not in {
        "examples/banks/synthetic-capture.bank.json",
        "examples/banks/synthetic-release-mock.bank.json",
    }:
        raise ValidationError(f"bank file does not exist: {path}")
    with as_file(resource) as bundled:
        return _model(BankManifest, bundled, max_bytes=64 * 1024)


def _direction(store: ArtifactStore, relative: str):
    # Directions are run-scoped inputs, unlike explicitly external bank files.
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or Path(relative).is_absolute():
        raise ValidationError("direction must be a relative path within the run directory")
    return store.load_direction(pure.as_posix())


def _store(args: argparse.Namespace) -> ArtifactStore:
    return ArtifactStore(Path(args.run_dir))


def _graph(store: ArtifactStore, baseline_id: str):
    return store.load_graph(f"{baseline_id}.graph.json")


def demo(args: argparse.Namespace) -> int:
    store = _store(args)
    graph = reference_graph(args.dtype)
    tensors = reference_tensors(args.dtype)
    store.write_graph(graph, "baseline")
    baseline = store.save_checkpoint(tensors, graph, "baseline")
    bank = _bank(Path(args.bank))
    candidate = derive_contrast(graph, tensors, bank, "candidate")
    store.write_direction(candidate, "candidate")
    bound = store.derive_candidate(candidate, graph, tensors, "reference-writer-0", "bound")
    request = ExperimentRequest(
        request_id="reference-request", direction=bound, alpha_values=(0.0, 1.0)
    )
    plan = propose_experiment(request)
    write_json(store.path("proposed.plan.json"), plan)
    compiled = store.compile(
        graph, baseline, plan.interventions[-1], "candidate", direction_path="bound.direction.json"
    )
    binding = bind_compiled("candidate.manifest.json", compiled)
    baseline_tensors = load_tensors(store.path(baseline.path))
    candidate_tensors = load_tensors(store.path(compiled.candidate.path))
    observation = observe_synthetic(
        graph, baseline_tensors, candidate_tensors, "reference-output-diagnostic"
    )
    metrics = (
        MetricResult(
            category="structural",
            name="candidate_audit",
            outcome="PASS",
            score=1.0,
            observed=1,
            eligible=1,
            evidence_digest=compiled.structural_check.target_digest_after,
            details="Exact synthetic tensor update and unchanged non-target tensors verified.",
        ),
        MetricResult(
            category="evaluation",
            name="synthetic_diagnostics",
            outcome="PASS",
            score=1.0,
            observed=1,
            eligible=1,
            evidence_digest=observation_digest(observation),
            details="Analytic output observation only; not a language or capability evaluation.",
        ),
        MetricResult(
            category="redline",
            name="cyber",
            outcome="NOT_RUN",
            evidence_digest=observation_digest(observation),
            details=(
                "Required private behavioral evaluation was not run. Other required "
                "redlines are absent; all block eligibility."
            ),
        ),
    )
    trial = TrialRecord(
        trial_id="trial-reference",
        status="completed",
        binding=binding,
        plan_digest=plan.digest(),
        reason=(
            "Synthetic analytic vertical slice; structural evidence is not behavioral "
            "or release proof."
        ),
        metrics=metrics,
        retained_outputs=("candidate.manifest.json",),
    )
    record_trial(store, trial)
    bundle = EvidenceBundle(
        binding=binding,
        observations=(observation,),
        metrics=metrics,
        trials=({"path": "trial-reference.trial.json", "digest": trial.digest()},),
        environment={"backend": "numpy-cpu", "adapter": graph.adapter_id},
        decoding={"mode": "not_applicable_analytic_matrix"},
    )
    evidence_path = write_evidence(store, bundle, "candidate")
    verify_evidence(store, evidence_path)
    _print(
        {
            "status": "synthetic_demo_complete",
            "manifest": "candidate.manifest.json",
            "evidence": evidence_path,
            "release": "not_evaluated",
            "note": "The gate is expected to block incomplete redline evidence.",
        }
    )
    return 0


def plan(args: argparse.Namespace) -> int:
    store = _store(args)
    direction = _direction(store, args.direction)
    from ablagent.interventions import BoundDirection

    if not isinstance(direction, BoundDirection):
        raise ValidationError("planning requires a bound direction, not a raw candidate")
    request = ExperimentRequest(
        request_id=args.request_id, direction=direction, alpha_values=args.alpha
    )
    proposal = propose_experiment(request)
    destination = store.path(args.output or "proposed.plan.json")
    write_json(destination, proposal)
    _print(proposal)
    return 0


def compile_plan(args: argparse.Namespace) -> int:
    store = _store(args)
    proposal = _model(ProposedPlan, store.path(args.plan))
    if proposal.execution_authority != "none" or len(proposal.interventions) != 1:
        raise PolicyBlocked(
            "compile accepts exactly one proposal-only intervention; "
            "no planner execution API exists"
        )
    from ablagent.interventions import BoundDirection

    direction = _direction(store, args.direction)
    if not isinstance(direction, BoundDirection):
        raise ValidationError("compile requires a bound direction")
    intervention = proposal.interventions[0]
    if (
        intervention.direction != direction
        or proposal.direction_digest != direction.direction_digest()
    ):
        raise ValidationError("proposal is not bound to the explicitly supplied direction")
    baseline_id = args.baseline_id
    graph = _graph(store, baseline_id)
    baseline = store.reference(f"{baseline_id}.safetensors", "baseline", graph)
    if baseline.tensor_digest != proposal.baseline_digest:
        raise ValidationError("proposal baseline does not match the selected baseline")
    compiled = store.compile(
        graph, baseline, intervention, args.artifact_id, direction_path=args.direction
    )
    binding = bind_compiled(f"{args.artifact_id}.manifest.json", compiled)
    trial = TrialRecord(
        trial_id=f"{args.artifact_id}-trial",
        status="completed",
        binding=binding,
        plan_digest=proposal.digest(),
        reason="Explicit CLI compile request for one bounded intervention.",
        retained_outputs=(f"{args.artifact_id}.manifest.json",),
    )
    record_trial(store, trial)
    _print(compiled)
    return 0


def gate(args: argparse.Namespace) -> int:
    store = ArtifactStore(Path(args.evidence).parent)
    bundle = verify_evidence(store, Path(args.evidence).name)
    bank = _bank(Path(args.bank))
    decision = evaluate_eligibility(
        EligibilityRequest(
            visibility=args.visibility,
            evidence=bundle,
            release_bank=bank,
        )
    )
    if args.output:
        write_json(Path(args.output), decision)
    _print(decision)
    return 0 if decision.status == "eligible" else 2


def conformance_command(args: argparse.Namespace) -> int:
    report = run_conformance(Path(args.fixtures), Path(args.run_dir))
    _print(report)
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ablagent", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities", help="print exact synthetic-adapter capabilities")
    demo_parser = commands.add_parser(
        "demo", help="run the offline analytic synthetic vertical slice"
    )
    demo_parser.add_argument("--run-dir", required=True)
    demo_parser.add_argument("--bank", default="examples/banks/synthetic-capture.bank.json")
    demo_parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    demo_parser.set_defaults(handler=demo)
    for command, handler in (("verify", "verify"), ("verify-evidence", "evidence")):
        verify_parser = commands.add_parser(
            command, help="independently verify a manifest or evidence bundle"
        )
        verify_parser.add_argument("path")
        verify_parser.set_defaults(handler=handler)
    plan_parser = commands.add_parser(
        "plan", help="produce a bounded proposal without executing it"
    )
    plan_parser.add_argument("direction")
    plan_parser.add_argument("--run-dir", required=True)
    plan_parser.add_argument("--request-id", default="reference-request")
    plan_parser.add_argument("--alpha", type=float, action="append", required=True)
    plan_parser.add_argument("--output")
    plan_parser.set_defaults(handler=plan)
    compile_parser = commands.add_parser(
        "compile", help="validate and execute exactly one selected intervention"
    )
    compile_parser.add_argument("direction")
    compile_parser.add_argument("--run-dir", required=True)
    compile_parser.add_argument("--plan", required=True)
    compile_parser.add_argument("--baseline-id", default="baseline")
    compile_parser.add_argument("--artifact-id", default="candidate")
    compile_parser.set_defaults(handler=compile_plan)
    gate_parser = commands.add_parser(
        "gate", help="evaluate eligibility; never publish or authorize"
    )
    gate_parser.add_argument("evidence")
    gate_parser.add_argument("--bank", required=True)
    gate_parser.add_argument("--visibility", choices=("public", "private"), default="public")
    gate_parser.add_argument("--output")
    gate_parser.set_defaults(handler=gate)
    conformance_parser = commands.add_parser(
        "conformance", help="Run the explicit synthetic DWMIP conformance vectors"
    )
    conformance_parser.add_argument("--fixtures", type=Path, required=True)
    conformance_parser.add_argument("--run-dir", type=Path, required=True)
    conformance_parser.set_defaults(handler=conformance_command)
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            _print(capabilities())
            return 0
        if args.handler == "verify":
            store = ArtifactStore(Path(args.path).parent)
            _print(store.verify_compiled(Path(args.path).name))
            return 0
        if args.handler == "evidence":
            store = ArtifactStore(Path(args.path).parent)
            _print(verify_evidence(store, Path(args.path).name))
            return 0
        return args.handler(args)
    except (AblagentError, OSError, json.JSONDecodeError) as exc:
        _print(
            {"status": "blocked_or_invalid", "error_type": type(exc).__name__, "message": str(exc)}
        )
        return 2 if isinstance(exc, PolicyBlocked) else 1


if __name__ == "__main__":
    sys.exit(main())
