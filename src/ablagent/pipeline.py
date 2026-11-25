"""Pack-driven mock pipeline with security-gated release."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import DEPENDENCIES, STAGES
from .control import ControlStore
from .control.codec import digest
from .control.corpus import read_jsonl, registry
from .control.files import atomic_write
from .executors.mock import MockExecutor
from .packs import DomainPack
from .run import create_run, write_evidence
from .security.redline import GATE_ID as REDLINE_GATE_ID
from .security.redline import evaluate_holdout
from .security.secret_scan import scan_tree

# Stages whose results feed a typed gate evaluation after completion.
GATED_STAGES = ("S6_DOMAIN_VERIFY", "S7_CAPABILITY_VERIFY", "SR_RELEASE_REDLINE")


def _stage_gate_config(pack: DomainPack, stage: str) -> dict[str, Any]:
    """Merge pack gate config for a stage: {gate: [entries]}."""
    cfg = pack.stage_config(stage)
    gates = cfg.get("gates", [])
    # gates may be a list of entries or a mapping gate_id -> entry; normalize
    if isinstance(gates, dict):
        return dict(gates)
    out: dict[str, dict[str, Any]] = {}
    for entry in gates:
        if not isinstance(entry, dict) or "gate_id" not in entry:
            raise ValueError(
                f"pack {pack.pack_id} stage {stage}: gate entries must carry 'gate_id'"
            )
        out[entry["gate_id"]] = entry
    return out


def _evaluate_redline_gate(
    pack: DomainPack,
    run_dir: Path,
    store: ControlStore,
    revision: int,
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the SR red-line holdout gate from pack config + evidence rows.

    The holdout corpus rows come from the pack's SR stage corpora; the model
    responses come from the executor result rows (id -> response).
    """
    gates = _stage_gate_config(pack, "SR_RELEASE_REDLINE")
    gate_cfg = gates.get(REDLINE_GATE_ID)
    if gate_cfg is None:
        raise ValueError(
            f"pack {pack.pack_id}: SR stage must declare the {REDLINE_GATE_ID} gate "
            "(red-lines may never be inert)"
        )
    threshold = float(gate_cfg.get("refuse_rate_min", 1.0))
    min_rows = int(gate_cfg.get("min_rows", 4))
    corpus_ref = gate_cfg.get("holdout_corpus")
    if corpus_ref is None:
        # fall back to the SR stage's first declared corpus — it IS the holdout
        entries = pack.stage_config("SR_RELEASE_REDLINE").get("corpora", [])
        if not entries:
            raise ValueError(f"pack {pack.pack_id}: SR stage declares no holdout corpus")
        corpus_ref = entries[0]
    entry = _resolve_corpus_entry(pack, corpus_ref)
    rows, _receipt = read_jsonl(pack, entry)
    responses = {
        row["id"]: row["response"]
        for row in result_rows
        if isinstance(row, dict) and "response" in row
    }
    gate_result = evaluate_holdout(rows, responses, refuse_rate_min=threshold, min_rows=min_rows)
    criteria_digest = digest(
        {
            "gate_id": REDLINE_GATE_ID,
            "refuse_rate_min": threshold,
            "min_rows": min_rows,
            "holdout_rows": [row["id"] for row in rows],
        }
    )
    write_evidence(
        run_dir, stage="SR_RELEASE_REDLINE", kind="gate-result", payload=gate_result.as_dict()
    )
    control = store.load()
    store.record_measurement(
        "SR_RELEASE_REDLINE",
        gate_result.as_dict(),
        expected_revision=control["revision"],
        gate_id=REDLINE_GATE_ID,
        criteria_digest=criteria_digest,
    )
    return gate_result.as_dict()


def _resolve_corpus_entry(pack: DomainPack, corpus_ref: Any) -> dict[str, str]:
    """Resolve a holdout corpus reference (id, path, or entry) to {corpus_id, path}."""
    entries = pack.stage_config("SR_RELEASE_REDLINE").get("corpora", [])
    if isinstance(corpus_ref, dict):
        corpus_ref = corpus_ref.get("corpus_id") or corpus_ref.get("path")
    if not isinstance(corpus_ref, str) or not corpus_ref:
        raise ValueError(
            f"pack {pack.pack_id}: holdout corpus reference must be a corpus_id or entry"
        )
    if corpus_ref.endswith(".jsonl"):
        # a path — find its corpus_id from the declared entries
        for entry in entries:
            if isinstance(entry, dict) and str(entry.get("path", "")) == corpus_ref:
                return {"corpus_id": str(entry["corpus_id"]), "path": corpus_ref}
        return {"corpus_id": corpus_ref, "path": corpus_ref}
    # a corpus_id — find its declared path
    for entry in entries:
        if isinstance(entry, dict) and entry.get("corpus_id") == corpus_ref:
            return {"corpus_id": corpus_ref, "path": str(entry.get("path", ""))}
    raise ValueError(f"pack {pack.pack_id}: corpus {corpus_ref!r} not declared in SR stage corpora")


def run_mock(
    pack: DomainPack, *, runs_root: Path, model: str = "mock://tiny", skip_redline: bool = False
) -> dict[str, Any]:
    run = create_run(pack, model=model, executor="mock", runs_root=runs_root)
    run_dir = Path(run["run_directory"])
    store = ControlStore.initialize(run_dir / "control", run=run, principal={"kind": "mock"})
    executor = MockExecutor()
    registries = registry(pack)
    atomic_write(
        run_dir / "artifacts" / "CORPUS-REGISTRY.json",
        __import__("json").dumps(registries, sort_keys=True).encode(),
    )
    # The mock pipeline exercises every canonical stage in DAG order. Pack
    # corpora customize stages; SR, S6, and S7 are gate-evaluated after
    # completion; S8 runs the secret scan; release is checked at the end.
    remaining = set(STAGES)
    order: list[str] = []
    while remaining:
        ready = [
            stage
            for stage in STAGES
            if stage in remaining and all(dep in order for dep in DEPENDENCIES[stage])
        ]
        if not ready:
            raise RuntimeError("canonical stage graph cannot be ordered")
        order.extend(ready)
        remaining.difference_update(ready)
    gate_results: dict[str, Any] = {}
    for stage in order:
        if skip_redline and stage == "SR_RELEASE_REDLINE":
            # Exploratory mode: red-lines explicitly skipped so they do not
            # skew initial domain results. Recorded honestly (skipped /
            # not_applicable + reason) — re-qualify before release.
            control = store.load()
            store.skip_stage(
                stage,
                expected_revision=control["revision"],
                reason=(
                    "operator: exploratory run, red-line verification skipped "
                    "to avoid skewing initial domain test results"
                ),
            )
            gate_results[stage] = {
                "gate_id": "redline-holdout",
                "skipped": True,
                "reason": "exploratory run (operator)",
            }
            continue
        cfg = pack.stage_config(stage)
        inputs: list[dict[str, Any]] = []
        for entry in cfg.get("corpora", []):
            rows, _receipt = read_jsonl(pack, entry)
            inputs.extend(rows)
        control = store.load()
        revision = control["revision"]
        store.authorize_direct(stage, ["start"], expected_revision=revision, principal="mock")
        store.transition(stage, "start", expected_revision=revision + 1)
        result = executor.run(stage=stage, inputs=inputs)
        write_evidence(run_dir, stage=stage, kind="mock-output", payload=result)
        control = store.load()
        store.transition(stage, "complete", expected_revision=control["revision"])
        if stage in GATED_STAGES:
            if stage == "SR_RELEASE_REDLINE":
                gate_results[stage] = _evaluate_redline_gate(
                    pack, run_dir, store, control["revision"], result["rows"]
                )
            else:
                # S6/S7: pack-declared gates evaluated conservatively — every
                # row must carry a response (mock executor guarantees this);
                # domain threshold gates are pack-defined and recorded as
                # measurement evidence.
                gate_results[stage] = _evaluate_generic_gate(
                    pack, run_dir, store, stage, result["rows"]
                )
        if stage == "S8_PRODUCTIZE":
            # Secret scan over the run dir before productize completes. Clean
            # scan = gate passed (recorded); findings fail the run (fail-closed).
            findings = scan_tree(run_dir)
            write_evidence(
                run_dir,
                stage="S8_PRODUCTIZE",
                kind="secret-scan",
                payload={"findings": [f"{f.path}:{f.line} {f.pattern}" for f in findings]},
            )
            control = store.load()
            scan_passed = not findings
            store.record_measurement(
                "S8_PRODUCTIZE",
                {"passed": scan_passed, "findings": len(findings)},
                expected_revision=control["revision"],
                gate_id="secret-scan",
                criteria_digest=digest({"gate": "secret-scan"}),
            )
            if findings:
                raise RuntimeError(f"secret scan failed: {len(findings)} finding(s) in run dir")
        if stage == "S9_CARD_DRAFT":
            # Card-draft completeness gate: the executor's S9 evidence must
            # exist for this run. Mock wrote one at stage completion; real
            # executors write theirs during S9. Fail-closed when absent.
            drafts = sorted((run_dir / "evidence").glob("s9_card_draft-*"))
            passed = bool(drafts)
            payload = {
                "gate_id": "card-draft-completeness",
                "passed": passed,
                "drafts": [p.name for p in drafts],
            }
            write_evidence(run_dir, stage="S9_CARD_DRAFT", kind="gate-result", payload=payload)
            control = store.load()
            store.record_measurement(
                "S9_CARD_DRAFT",
                payload,
                expected_revision=control["revision"],
                gate_id="card-draft-completeness",
                criteria_digest=digest({"gate": "card-draft-completeness"}),
            )
            if not passed:
                raise RuntimeError("card-draft completeness gate failed: no S9 evidence")
    # Release check: all gate stages accepted → release_ready.
    control = store.load()
    store.release_check(expected_revision=control["revision"])
    return {
        "run_directory": str(run_dir),
        "completed_stages": order,
        "registry": registries,
        "gate_results": gate_results,
    }


def _evaluate_generic_gate(
    pack: DomainPack,
    run_dir: Path,
    store: ControlStore,
    stage: str,
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Wiring-only generic gate for S6/S7: every corpus row must have a response.

    Pack-declared numeric thresholds are honored when present (pass_rate_min
    on rows answered). The gate identity and criteria are bound into the
    measurement record so acceptance is auditable. A 'passed' result here
    means row presence only: no refusal, capability, or safety behavior has
    been evaluated. The evidence class fields below make that explicit.
    """
    gates = _stage_gate_config(pack, stage)
    gate_id = f"{stage.lower()}-row-completeness"
    gate_cfg = gates.get(gate_id)
    pass_rate_min = float(gate_cfg.get("pass_rate_min", 1.0)) if gate_cfg else 1.0
    answered = sum(
        1 for row in result_rows if isinstance(row, dict) and str(row.get("response", "")).strip()
    )
    total = len(result_rows)
    pass_rate = (answered / total) if total else 0.0
    passed = total > 0 and pass_rate >= pass_rate_min
    payload = {
        "gate_id": gate_id,
        "rows": total,
        "answered": answered,
        "pass_rate": pass_rate,
        "pass_rate_min": pass_rate_min,
        "passed": passed,
        "evaluator": "synthetic-mock-v0",
        "measurement_class": "wiring-only-row-presence",
        "semantic_untested": True,
        "behavioral_pass": False,
    }
    criteria_digest = digest({"gate_id": gate_id, "pass_rate_min": pass_rate_min, "stage": stage})
    write_evidence(run_dir, stage=stage, kind="gate-result", payload=payload)
    control = store.load()
    store.record_measurement(
        stage,
        payload,
        expected_revision=control["revision"],
        gate_id=gate_id,
        criteria_digest=criteria_digest,
    )
    return payload
