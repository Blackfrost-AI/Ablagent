"""Command-line boundary.

The default (synthetic/mock) paths make no network requests. Only the explicit
`run --executor real` path downloads a pinned HuggingFace snapshot.
"""

import argparse
import json
import sys
from importlib.resources import as_file, files
from pathlib import Path, PurePosixPath

from pydantic import ValidationError as PydanticValidationError

from ablagent import __version__
from ablagent.artifacts import ArtifactStore, file_digest, load_tensors
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
from ablagent.mock_redlines import FileReference
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
            category="redline_observation",
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
        plan_reference=FileReference(
            path="proposed.plan.json", file_digest=file_digest(store.path("proposed.plan.json"))
        ),
        reason=(
            "Synthetic analytic vertical slice; structural evidence is not behavioral "
            "or release proof."
        ),
        metrics=metrics,
        retained_outputs=(
            FileReference(
                path="candidate.manifest.json",
                file_digest=file_digest(store.path("candidate.manifest.json")),
            ),
        ),
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
        plan_reference=FileReference(
            path=args.plan, file_digest=file_digest(store.path(args.plan))
        ),
        reason="Explicit CLI compile request for one bounded intervention.",
        retained_outputs=(
            FileReference(
                path=f"{args.artifact_id}.manifest.json",
                file_digest=file_digest(store.path(f"{args.artifact_id}.manifest.json")),
            ),
        ),
    )
    record_trial(store, trial)
    _print(compiled)
    return 0


def gate(args: argparse.Namespace) -> int:
    store = ArtifactStore(Path(args.evidence).parent)
    bank = _bank(Path(args.bank))
    decision = evaluate_eligibility(
        EligibilityRequest(
            visibility=args.visibility,
            evidence_path=Path(args.evidence).name,
            release_bank=bank,
        ),
        store,
    )
    if args.output:
        write_json(Path(args.output), decision)
    _print(decision)
    return 0 if decision.status == "eligible" else 2


def _packs_dir_default() -> Path:
    return Path(__file__).resolve().parents[2] / "packs"


def doctor(args: argparse.Namespace) -> int:
    import platform

    from ablagent.constants import DEPENDENCIES, STAGES
    from ablagent.packs import PackError, discover_packs

    ok = True
    print(f"python    {platform.python_version()} ({platform.machine()})")
    try:
        import yaml  # noqa: F401

        print("pyyaml    ok")
    except ImportError:
        print("pyyaml    MISSING (pip install PyYAML>=6.0)")
        ok = False
    dag_ok = True
    for stage, deps in DEPENDENCIES.items():
        for dep in deps:
            if dep not in STAGES:
                print(f"dag       BROKEN: {stage} depends on unknown {dep}")
                dag_ok = False
    ok = ok and dag_ok
    print("dag       ok" if dag_ok else "dag       see errors above")
    root = Path(args.packs_dir) if args.packs_dir else _packs_dir_default()
    if root.is_dir():
        try:
            packs = discover_packs(root)
            if not packs and args.packs_dir is not None:
                print(f"packs     BROKEN: no packs found under explicit directory {root}")
                ok = False
            else:
                print(f"packs     ok ({len(packs)} discovered under {root})")
        except (PackError, OSError, ValueError) as exc:
            print(f"packs     BROKEN: {exc}")
            ok = False
    else:
        print(f"packs     none at {root} (ok for source-only installs)")
    return 0 if ok else 1


def packs_command(args: argparse.Namespace) -> int:
    from ablagent.packs import discover_packs, load_pack

    root = Path(args.packs_dir) if args.packs_dir else _packs_dir_default()
    if args.pack_id:
        pack = load_pack(root / args.pack_id)
        print(f"{pack.pack_id}  v{pack.version}  [{pack.domain}/{pack.purpose}]")
        print(f"  title:    {pack.title}")
        print(f"  summary:  {pack.summary}")
        print(f"  manifest: {pack.manifest_path}")
        print(f"  sha256:   {pack.manifest_sha256}")
        for stage in pack.stage_names():
            print(f"  stage {stage}: {pack.stage_config(stage).get('summary', '')}")
        return 0
    packs = discover_packs(root)
    if not packs:
        print(f"no packs found under {root}")
        return 1
    for pack in packs.values():
        print(f"{pack.pack_id}  v{pack.version}  [{pack.domain}/{pack.purpose}]  {pack.title}")
    return 0


def stages_command(_args: argparse.Namespace) -> int:
    from ablagent.constants import DEPENDENCIES, STAGES

    print("Ablagent pipeline DAG (domain-pack driven):")
    for stage in STAGES:
        deps = DEPENDENCIES.get(stage, ())
        print(f"  {stage}  <- {', '.join(deps) if deps else '(root)'}")
    return 0


def init_command(args: argparse.Namespace) -> int:
    from ablagent.control import ControlStore

    store = ControlStore.initialize(Path(args.run_dir))
    print(f"initialized control: {store.control_path}")
    print(f"anchor: {store.anchor_path}")
    return 0


def status_command(args: argparse.Namespace) -> int:
    from ablagent.control import ControlStore

    control = ControlStore(Path(args.run_dir)).load()
    print(f"revision: {control['revision']}")
    print(f"event_head: {control['event_head']}")
    for stage, state in control["stages"].items():
        print(
            f"{stage}: {state['execution']} / {state['authorization']}"
            f" / {state['measurement']} / {state['release']}"
        )
    return 0


def run_command(args: argparse.Namespace) -> int:
    from ablagent.packs import load_pack

    pack = load_pack(Path(args.packs_dir) / args.pack)
    if args.executor == "real":
        from ablagent.adapters.hf_intake import parse_model_ref, require_real_deps
        from ablagent.pipeline_real import run_real

        # Preflight BEFORE any run state is created (fail closed, no litter).
        require_real_deps()
        parse_model_ref(args.model)
        if not 4 <= args.max_pairs <= 16:
            raise ValidationError("--max-pairs must be in 4..16")
        if not 1 <= args.baseline_rows <= 64:
            raise ValidationError("--baseline-rows must be in 1..64")
        if args.skip_redline:
            raise ValidationError(
                "--skip-redline is a mock-path flag; the real executor always "
                "records SR_RELEASE_REDLINE as an operator-supplied skip"
            )
        result = run_real(
            pack,
            model=args.model,
            runs_root=Path(args.runs_dir),
            alpha=args.alpha,
            writer=args.writer,
            max_pairs=args.max_pairs,
            baseline_rows=args.baseline_rows,
        )
        print(f"run: {result['run_directory']}")
        print("completed: " + ", ".join(result["completed_stages"]))
        print("skipped (untested, recorded): " + ", ".join(result["skipped_stages"]))
        print(f"selected writer: {result['selected_writer']} (alpha {result['alpha']})")
        print(f"manifest: {result['manifest']}")
        print(f"export: {result['export_dir']}")
        print("release: not_eligible — verification gates were not run; do not publish")
        return 0
    from ablagent.pipeline import run_mock

    result = run_mock(
        pack,
        runs_root=Path(args.runs_dir),
        model=args.model,
        skip_redline=args.skip_redline,
    )
    print(f"run: {result['run_directory']}")
    print("completed: " + ", ".join(result["completed_stages"]))
    return 0


def conformance_command(args: argparse.Namespace) -> int:
    report = run_conformance(Path(args.fixtures), Path(args.run_dir))
    _print(report)
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ablagent", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    capabilities_parser = commands.add_parser(
        "capabilities", help="print exact adapter capabilities"
    )
    capabilities_parser.add_argument("--adapter", choices=("synthetic", "hf"), default="synthetic")
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
        if command == "verify-evidence":
            verify_parser.add_argument(
                "--bank", help="selected bank; required for mock redline evidence"
            )
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
    doctor_parser = commands.add_parser("doctor", help="verify local environment and packs")
    doctor_parser.add_argument("--packs-dir", default=None)
    doctor_parser.set_defaults(handler=doctor)
    packs_parser = commands.add_parser("packs", help="list or inspect domain packs")
    packs_parser.add_argument("--packs-dir", default=None, help="packs root (default: repo packs/)")
    packs_parser.add_argument("pack_id", nargs="?", default=None)
    packs_parser.set_defaults(handler=packs_command)
    stages_parser = commands.add_parser("stages", help="print the pipeline DAG")
    stages_parser.set_defaults(handler=stages_command)
    init_parser = commands.add_parser("init", help="initialize a control-plane run")
    init_parser.add_argument("run_dir")
    init_parser.set_defaults(handler=init_command)
    status_parser = commands.add_parser("status", help="verify and show a control-plane run")
    status_parser.add_argument("run_dir")
    status_parser.set_defaults(handler=status_command)
    run_parser = commands.add_parser("run", help="run a pack with an executor")
    run_parser.add_argument("--pack", required=True)
    run_parser.add_argument("--model", default="mock://tiny")
    run_parser.add_argument("--executor", choices=("mock", "real"), default="mock")
    run_parser.add_argument(
        "--alpha", type=float, default=1.0, help="real executor: rank-one edit strength in (0, 1]"
    )
    run_parser.add_argument(
        "--writer",
        default=None,
        help="real executor: writer_id or tensor_name to edit "
        "(default: strongest captured contrast)",
    )
    run_parser.add_argument(
        "--max-pairs", type=int, default=16, help="real executor: capture pairs (4..16)"
    )
    run_parser.add_argument(
        "--baseline-rows", type=int, default=8, help="real executor: S1 baseline generation rows"
    )
    run_parser.add_argument("--packs-dir", default="packs")
    run_parser.add_argument("--runs-dir", default="runs")
    run_parser.add_argument(
        "--skip-redline",
        action="store_true",
        help="exploratory mode: skip SR red-line gate (recorded as skipped, not passed)",
    )
    run_parser.set_defaults(handler=run_command)
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            if getattr(args, "adapter", "synthetic") == "hf":
                from ablagent.adapters.hf_intake import require_real_deps
                from ablagent.constants import HF_ADAPTER_ID
                from ablagent.identity import AdapterCapabilities

                versions = require_real_deps()
                _print(
                    {
                        "capabilities": AdapterCapabilities(
                            adapter_id=HF_ADAPTER_ID,
                            executable_topologies=("dense_linear",),
                            supported_operations=("rank_one_output",),
                            supported_codecs=("safetensors",),
                            real_checkpoint_support=True,
                        ).model_dump(mode="json"),
                        "dependency_versions": versions,
                    }
                )
            else:
                _print(capabilities())
            return 0
        if args.handler == "verify":
            store = ArtifactStore(Path(args.path).parent)
            _print(store.verify_compiled(Path(args.path).name))
            return 0
        if args.handler == "evidence":
            store = ArtifactStore(Path(args.path).parent)
            bank = _bank(Path(args.bank)) if args.bank else None
            _print(verify_evidence(store, Path(args.path).name, release_bank=bank))
            return 0
        return args.handler(args)
    except (AblagentError, OSError, json.JSONDecodeError) as exc:
        _print(
            {"status": "blocked_or_invalid", "error_type": type(exc).__name__, "message": str(exc)}
        )
        return 2 if isinstance(exc, PolicyBlocked) else 1


if __name__ == "__main__":
    sys.exit(main())
