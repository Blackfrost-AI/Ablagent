"""Pack-driven REAL pipeline: HF checkpoint → S0/S1/S2/S3 → honest skips → S8/S9.

The DWMIP artifact chain (intake → capture → bind → compile → verify) is the
evidence spine; the v2 control plane records every stage transition. Stages the
v0 real adapter cannot honestly execute (S4 quant, S5 serve, S6/S7 verify,
S10 deliver) are recorded as skipped with reasons — never faked. SR red-line
stays operator-supplied and skipped by default. Release stays not_eligible:
this pipeline produces an evidence-bound experimental export, not a release.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapters import hf_bake, hf_capture, hf_intake
from .artifacts import ArtifactStore
from .constants import DEPENDENCIES, REAL_UNMET_STAGES, STAGES
from .control import ControlStore
from .control.codec import digest
from .control.corpus import read_jsonl, registry
from .control.files import atomic_write
from .errors import ValidationError
from .packs import DomainPack
from .run import create_run, write_evidence
from .security.secret_scan import scan_tree_report

_SKIP_REASONS: dict[str, str] = {
    "S4_QUANT": "v0 real adapter has no quantizer; quantization not executed this run",
    "S5_SERVE": "v0 real adapter has no serving stack; candidate not served this run",
    "S6_DOMAIN_VERIFY": (
        "requires a served candidate (S5 skipped); domain behavior untested this run"
    ),
    "S7_CAPABILITY_VERIFY": (
        "requires a served candidate (S5 skipped); capability floor untested this run"
    ),
    "S10_DELIVER": "no delivery channel in the v0 real adapter; nothing delivered",
}


def _dag_order() -> list[str]:
    remaining = set(STAGES)
    order: list[str] = []
    while remaining:
        ready = [s for s in STAGES if s in remaining and all(d in order for d in DEPENDENCIES[s])]
        if not ready:
            raise RuntimeError("canonical stage graph cannot be ordered")
        order.extend(ready)
        remaining.difference_update(ready)
    return order


def run_real(
    pack: DomainPack,
    *,
    model: str,
    runs_root: Path,
    alpha: float = 1.0,
    writer: str | None = None,
    max_pairs: int = 16,
    baseline_rows: int = 8,
) -> dict[str, Any]:
    """Execute the real vertical slice. Preflight deps BEFORE creating run state."""
    hf_intake.require_real_deps()
    hf_intake.parse_model_ref(model)
    if not 0.0 < alpha <= 1.0:
        raise ValidationError("alpha must be in (0, 1]")

    run = create_run(
        pack,
        model=model,
        executor="real",
        runs_root=runs_root,
        options={
            "alpha": alpha,
            "writer": writer,
            "max_pairs": max_pairs,
            "baseline_rows": baseline_rows,
        },
    )
    run_dir = Path(run["run_directory"])
    store = ControlStore.initialize(
        run_dir / "control", run=run, principal={"kind": "operator-cli"}
    )
    artifacts_root = run_dir / "artifacts"
    artifact_store = ArtifactStore(artifacts_root)
    snapshot_dir = run_dir / "hf-snapshot"
    registries = registry(pack)
    atomic_write(
        artifacts_root / "CORPUS-REGISTRY.json", json.dumps(registries, sort_keys=True).encode()
    )

    def authorize_start(stage: str) -> None:
        control = store.load()
        store.authorize_direct(
            stage, ["start"], expected_revision=control["revision"], principal="operator-cli"
        )
        store.transition(stage, "start", expected_revision=control["revision"] + 1)

    def complete(stage: str) -> None:
        control = store.load()
        store.transition(stage, "complete", expected_revision=control["revision"])

    def skip(stage: str, reason: str) -> None:
        control = store.load()
        store.skip_stage(
            stage,
            expected_revision=control["revision"],
            reason=reason,
            allowed_stages=REAL_UNMET_STAGES | frozenset({"SR_RELEASE_REDLINE"}),
        )
        write_evidence(run_dir, stage=stage, kind="stage-skip", payload={"reason": reason})

    def measure(stage: str, payload: dict[str, Any], gate_id: str) -> None:
        write_evidence(run_dir, stage=stage, kind="gate-result", payload=payload)
        control = store.load()
        store.record_measurement(
            stage,
            payload,
            expected_revision=control["revision"],
            gate_id=gate_id,
            criteria_digest=digest({"gate": gate_id, "stage": stage}),
        )

    order = _dag_order()
    completed: list[str] = []
    skipped: list[str] = []
    intake_receipt: dict[str, Any] = {}
    capture_receipt: dict[str, Any] = {}
    bake_receipt: dict[str, Any] = {}
    export_receipt: dict[str, Any] = {}
    baseline_result: dict[str, Any] = {}

    for stage in order:
        if stage in _SKIP_REASONS:
            skip(stage, _SKIP_REASONS[stage])
            skipped.append(stage)
            continue
        if stage == "SR_RELEASE_REDLINE":
            skip(
                stage,
                "release red-line holdout is operator-supplied (private); "
                "not executed by the open-source runtime — release stays not_eligible",
            )
            skipped.append(stage)
            continue

        authorize_start(stage)
        try:
            if stage == "S0_INTAKE":
                intake_receipt = hf_intake.intake(
                    model, snapshot_dir=snapshot_dir, store_root=artifacts_root
                )
                write_evidence(run_dir, stage=stage, kind="intake", payload=intake_receipt)
                findings, unscanned = scan_tree_report(snapshot_dir)
                write_evidence(
                    run_dir,
                    stage=stage,
                    kind="snapshot-secret-scan",
                    payload={
                        "findings": [f"{f.path}:{f.line} {f.pattern}" for f in findings],
                        "unscanned": list(unscanned),
                    },
                )
                if findings:
                    raise ValidationError(
                        f"pinned HF snapshot failed secret scan: {len(findings)} finding(s)"
                    )
                complete(stage)

            elif stage == "S1_BASELINE":
                rows: list[dict[str, Any]] = []
                for entry in pack.stage_config("S0_INTAKE").get("corpora", []):
                    corpus_rows, _receipt = read_jsonl(pack, entry)
                    rows.extend(corpus_rows)
                rows = rows[:baseline_rows]
                if not rows:
                    raise ValidationError("pack declares no corpus rows for baseline generation")
                baseline_result = hf_capture.baseline_generate(snapshot_dir, rows)
                write_evidence(
                    run_dir, stage=stage, kind="baseline-generation", payload=baseline_result
                )
                complete(stage)

            elif stage == "S2_DIRECTIONS":
                entries = pack.stage_config(stage).get("corpora", [])
                if not entries:
                    raise ValidationError(f"pack {pack.pack_id}: S2 declares no capture corpus")
                capture_rows, corpus_receipt = read_jsonl(pack, entries[0])
                bank = hf_capture.derive_capture_bank(
                    capture_rows,
                    bank_id=f"{pack.pack_id}-capture-derived",
                    pack_id=pack.pack_id,
                    corpus_sha256=corpus_receipt["sha256"],
                    max_pairs=max_pairs,
                )
                artifact_store.write_bank(bank, "capture-bank")
                graph = artifact_store.load_graph("baseline.graph.json")
                from .intake_support import baseline_ref_from_receipt

                baseline = baseline_ref_from_receipt(intake_receipt)
                capture_receipt = hf_capture.capture_directions(
                    snapshot_dir,
                    graph,
                    baseline,
                    bank,
                    store=artifact_store,
                    max_pairs=max_pairs,
                )
                write_evidence(run_dir, stage=stage, kind="capture", payload=capture_receipt)
                complete(stage)

            elif stage == "S3_DERISK":
                graph = artifact_store.load_graph("baseline.graph.json")
                baseline = baseline_ref_from_receipt(intake_receipt)
                rankings = capture_receipt["rankings"]
                if writer is not None:
                    selected = next(
                        (
                            r
                            for r in rankings
                            if r["writer_id"] == writer or r["tensor_name"] == writer
                        ),
                        None,
                    )
                    if selected is None:
                        raise ValidationError(
                            f"writer {writer!r} has no captured candidate; "
                            f"captured writers: {[r['writer_id'] for r in rankings]}"
                        )
                else:
                    selected = rankings[0]  # strongest contrast norm
                entry = next(
                    c
                    for c in capture_receipt["candidates"]
                    if c["writer_id"] == selected["writer_id"]
                )
                bake_receipt = hf_bake.bake(
                    artifact_store,
                    graph,
                    baseline,
                    candidate_path=entry["path"],
                    writer_id=selected["writer_id"],
                    alpha=alpha,
                )
                write_evidence(run_dir, stage=stage, kind="bake", payload=bake_receipt)
                export_receipt = hf_bake.export_hf(
                    artifact_store,
                    graph,
                    candidate_path=bake_receipt["candidate"]["path"],
                    snapshot_dir=snapshot_dir,
                    export_dir=run_dir / "export",
                    intake_receipt=intake_receipt,
                    tensor_name=bake_receipt["tensor_name"],
                )
                write_evidence(run_dir, stage=stage, kind="hf-export", payload=export_receipt)
                complete(stage)
                measure(
                    stage,
                    {
                        "passed": bool(
                            bake_receipt["structural_check_passed"]
                            and bake_receipt["sign_test"]["passed"]
                        ),
                        "structural_check_passed": bake_receipt["structural_check_passed"],
                        "sign_test_passed": bake_receipt["sign_test"]["passed"],
                        "manifest": bake_receipt["manifest"],
                    },
                    "dwmip-structural-audit",
                )

            elif stage == "S8_PRODUCTIZE":
                # Export must exist and re-verify before productize evidence.
                verified = artifact_store.verify_compiled(bake_receipt["manifest"])
                findings, unscanned = scan_tree_report(run_dir)
                write_evidence(
                    run_dir,
                    stage=stage,
                    kind="secret-scan",
                    payload={
                        "findings": [f"{f.path}:{f.line} {f.pattern}" for f in findings],
                        "unscanned": list(unscanned),
                    },
                )
                if findings:
                    raise ValidationError(
                        f"secret scan failed: {len(findings)} finding(s) in run dir"
                    )
                complete(stage)
                measure(
                    stage,
                    {
                        "passed": True,
                        "findings": 0,
                        "verified_manifest": bake_receipt["manifest"],
                        "candidate_tensor_digest": verified.candidate.tensor_digest,
                        "release": "not_eligible",
                    },
                    "secret-scan",
                )

            elif stage == "S9_CARD_DRAFT":
                # S10 is skipped AFTER S9 in DAG order; the card must declare
                # the full untested set, not the skips seen so far.
                card = _card_draft(
                    run,
                    intake_receipt,
                    bake_receipt,
                    export_receipt,
                    baseline_result,
                    sorted(REAL_UNMET_STAGES | {"SR_RELEASE_REDLINE"}),
                )
                atomic_write(
                    run_dir / "evidence" / "s9_card_draft-0000.json",
                    json.dumps(card, sort_keys=True, indent=2).encode(),
                )
                complete(stage)
                drafts = sorted((run_dir / "evidence").glob("s9_card_draft-*"))
                measure(
                    stage,
                    {
                        "gate_id": "card-draft-completeness",
                        "passed": bool(drafts),
                        "drafts": [p.name for p in drafts],
                    },
                    "card-draft-completeness",
                )

            else:
                raise RuntimeError(f"unhandled real stage: {stage}")
        except Exception:
            control = store.load()
            if control["stages"][stage]["execution"] == "running":
                store.transition(stage, "fail", expected_revision=control["revision"])
            raise
        completed.append(stage)

    control = store.load()
    # Honest release evaluation: skipped gate stages keep release not_eligible.
    # release_check() would mark completed gates release_ready but skipped ones
    # stay untested — we do NOT call it here because SR/S6/S7 are skipped and
    # the run must not present any release-ready state.
    release_states = {s: control["stages"][s]["release"] for s in control["stages"]}
    return {
        "run_directory": str(run_dir),
        "completed_stages": completed,
        "skipped_stages": skipped,
        "release": "not_eligible",
        "release_states": release_states,
        "registry": registries,
        "revision": control["revision"],
        "manifest": bake_receipt.get("manifest"),
        "export_dir": export_receipt.get("export_dir"),
        "selected_writer": bake_receipt.get("writer_id"),
        "alpha": alpha,
    }


def _card_draft(
    run: dict[str, Any],
    intake: dict[str, Any],
    bake_receipt: dict[str, Any],
    export: dict[str, Any],
    baseline: dict[str, Any],
    skipped: list[str],
) -> dict[str, Any]:
    """Evidence-bound card DRAFT. Never claims release or measured safety."""
    return {
        "schema": "ablagent.card-draft.v1",
        "status": "draft_experimental_not_a_release",
        "run_id": run["run_id"],
        "base_model": {
            "repo_id": intake.get("repo_id"),
            "revision": intake.get("revision"),
            "checkpoint_id": intake.get("checkpoint_id"),
        },
        "intervention": {
            "operation": "weight_rank_one",
            "algorithm_id": "dense.rank_one_output.v0",
            "alpha": bake_receipt.get("alpha"),
            "writer_id": bake_receipt.get("writer_id"),
            "tensor_name": bake_receipt.get("tensor_name"),
            "direction_digest": bake_receipt.get("direction_digest"),
            "evidence_class": "correlational",
        },
        "verification": {
            "structural_check_passed": bake_receipt.get("structural_check_passed"),
            "sign_test": bake_receipt.get("sign_test"),
            "manifest": bake_receipt.get("manifest"),
        },
        "export": {
            "export_dir": export.get("export_dir"),
            "original_dtype_restored": export.get("original_dtype"),
            "non_edited_shards_byte_identical": export.get("non_edited_shards_byte_identical"),
        },
        "baseline_generation": {
            "row_count": baseline.get("row_count"),
            "claim": baseline.get("claim"),
        },
        "untested_stages": skipped,
        "release_eligibility": "not_eligible",
        "limitations": [
            "Direction capture is correlational, not causal.",
            "Domain behavior, capability floor, and red-lines were NOT tested this run.",
            "No serving, quantization, or delivery was performed.",
            "Do not publish or deploy without operator-run verification gates.",
        ],
    }
