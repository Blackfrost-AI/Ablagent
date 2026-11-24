"""CAS-backed control state with an append-only hash chain and external anchor."""
from __future__ import annotations

import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .codec import canonical_bytes, digest, load_json
from .errors import AuthorityError, ConcurrentRevisionError, IntegrityError, InputError, StateError
from .files import atomic_write, ensure_private_dir, ensure_private_file, read_stable
from .schemas import dependencies_satisfied, empty_stage_state, normalize_stage, validate_control
from ..constants import CONTROL_SCHEMA, LIFECYCLE_ACTIONS, RELEASE_GATE_STAGES, STAGES

CONTROL_FILE = "CONTROL.json"
OPERATIONS_FILE = "OPERATIONS.jsonl"
LOCK_FILE = ".CONTROL.lock"
TRANSACTION_FILE = ".CONTROL.txn"
ANCHOR_SCHEMA = "ablagent.anchor.v2"
EVENT_SCHEMA = "ablagent.event.v2"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _event_hash(event: dict[str, Any]) -> str:
    return digest(event)


def _state_digest(control: dict[str, Any]) -> str:
    body = {k: v for k, v in control.items() if k not in {"state_digest", "event_head"}}
    return digest(body)


@dataclass(frozen=True)
class VerificationReport:
    revision: int
    event_head: str
    state_digest: str
    events: int


class ControlStore:
    def __init__(self, run_directory: Path, *, anchor_path: Path | None = None):
        self.run_directory = ensure_private_dir(Path(run_directory))
        self.control_path = self.run_directory / CONTROL_FILE
        self.operations_path = self.run_directory / OPERATIONS_FILE
        self.lock_path = self.run_directory / LOCK_FILE
        self.transaction_path = self.run_directory / TRANSACTION_FILE
        self.anchor_path = Path(anchor_path) if anchor_path else self.run_directory.parent / "anchors" / f"{self.run_directory.name}.json"

    @classmethod
    def initialize(cls, run_directory: Path, *, run: dict[str, Any] | None = None,
                   principal: dict[str, Any] | None = None, session_binding: str | None = None,
                   anchor_path: Path | None = None) -> "ControlStore":
        run_directory = Path(run_directory)
        if run_directory.exists():
            if run_directory.is_symlink() or not run_directory.is_dir():
                raise IntegrityError("run directory is not a real directory")
            if any(run_directory.iterdir()):
                raise IntegrityError(f"run directory is not empty: {run_directory}")
            run_directory.chmod(0o700)
        else:
            run_directory.mkdir(mode=0o700, parents=True)
        store = cls(run_directory, anchor_path=anchor_path)
        anchor_parent = store.anchor_path.parent
        anchor_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if store.anchor_path.exists() or store.anchor_path.is_symlink():
            raise IntegrityError(f"anchor already exists: {store.anchor_path}")
        fd = os.open(store.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        control = {
            "schema": CONTROL_SCHEMA, "version": 1,
            "control_instance_id": secrets.token_hex(16),
            "run_directory": str(run_directory.resolve()), "revision": 0,
            "event_head": "sha256:" + "0" * 64,
            "state_digest": "",
            "session_binding": session_binding or secrets.token_hex(16),
            "run": run or {}, "principal": principal or {},
            "stages": {stage: empty_stage_state(stage) for stage in STAGES},
            "used_authority_receipts": [],
        }
        control["state_digest"] = _state_digest(control)
        validate_control(control)
        atomic_write(store.control_path, canonical_bytes(control))
        atomic_write(store.operations_path, b"")
        anchor = {"schema": ANCHOR_SCHEMA, "control_instance_id": control["control_instance_id"],
                  "run_directory": control["run_directory"], "revision": 0,
                  "event_head": control["event_head"], "state_digest": control["state_digest"]}
        atomic_write(store.anchor_path, canonical_bytes(anchor))
        return store

    def _lock(self):
        class Guard:
            def __enter__(inner):
                self.lock_fd = os.open(self.lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
                return inner
            def __exit__(inner, *args):
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
        return Guard()

    def _read_control(self) -> dict[str, Any]:
        control = load_json(read_stable(self.control_path), source=str(self.control_path))
        if not isinstance(control, dict):
            raise IntegrityError("CONTROL.json must be an object")
        validate_control(control)
        return control

    def _recover_transaction(self) -> None:
        """Complete a prepared mutation, or fail closed if its files drifted."""
        if not self.transaction_path.exists():
            return
        txn = load_json(read_stable(self.transaction_path), source=str(self.transaction_path))
        if not isinstance(txn, dict) or txn.get("schema") != "ablagent.transaction.v1":
            raise IntegrityError("invalid control transaction marker")
        # A marker is only recoverable when it contains the exact bytes that
        # were prepared. No ordinary command may invent or repair a marker.
        try:
            control_bytes = bytes.fromhex(txn["control_hex"])
            previous_control_bytes = bytes.fromhex(txn["previous_control_hex"])
            event_bytes = bytes.fromhex(txn["event_hex"])
        except (KeyError, ValueError, TypeError) as exc:
            raise IntegrityError("invalid control transaction bytes") from exc
        prepared_control = load_json(control_bytes, source="control transaction")
        previous_control = load_json(previous_control_bytes, source="control transaction")
        if digest(prepared_control) != txn["control_digest"]:
            raise IntegrityError("control transaction digest mismatch")
        current = self.control_path.read_bytes()
        if current == previous_control_bytes:
            # The marker was durable but CONTROL.json was not replaced. The
            # mutation is not committed; discard only this exact prepared txn.
            self.transaction_path.unlink()
            return
        if current != control_bytes:
            raise IntegrityError("control transaction has unexpected current state")
        event = load_json(event_bytes, source="control transaction")
        if not isinstance(event, dict) or event.get("schema") != EVENT_SCHEMA:
            raise IntegrityError("invalid prepared control event")
        claimed = event.get("event_hash")
        body = dict(event); body.pop("event_hash", None)
        if claimed != _event_hash(body):
            raise IntegrityError("prepared control event hash mismatch")
        if event.get("revision") != prepared_control["revision"] or event.get("previous_event_hash") != previous_control["event_head"]:
            raise IntegrityError("prepared control event metadata mismatch")
        if event.get("state_digest") != prepared_control["state_digest"] or prepared_control.get("event_head") != claimed:
            raise IntegrityError("prepared control event state mismatch")
        line = event_bytes + b"\n"
        existing = self.operations_path.read_bytes() if self.operations_path.exists() else b""
        if existing.endswith(line):
            pass  # crash occurred after the append and before marker removal
        elif line in existing:
            raise IntegrityError("control transaction event appears at an unexpected log position")
        else:
            with self.operations_path.open("ab") as fh:
                fh.write(line)
                fh.flush(); os.fsync(fh.fileno())
        anchor = {"schema": ANCHOR_SCHEMA, "control_instance_id": txn["control_instance_id"],
                  "run_directory": txn["run_directory"], "revision": txn["revision"],
                  "event_head": txn["event_head"], "state_digest": txn["state_digest"]}
        atomic_write(self.anchor_path, canonical_bytes(anchor))
        self.transaction_path.unlink()

    def _read_anchor(self) -> dict[str, Any]:
        anchor = load_json(read_stable(self.anchor_path), source=str(self.anchor_path))
        if not isinstance(anchor, dict) or anchor.get("schema") != ANCHOR_SCHEMA:
            raise IntegrityError("invalid anchor")
        return anchor

    def _assert_anchor(self, control: dict[str, Any]) -> None:
        anchor = self._read_anchor()
        for key in ("control_instance_id", "run_directory", "revision", "event_head", "state_digest"):
            if anchor.get(key) != control.get(key):
                raise IntegrityError(f"anchor mismatch at {key}; refusing recovery or repair")

    def load(self) -> dict[str, Any]:
        with self._lock():
            self._recover_transaction()
            control = self._read_control()
            self._assert_anchor(control)
            self.verify()
            return control

    def verify(self) -> VerificationReport:
        control = self._read_control()
        self._assert_anchor(control)
        previous = "sha256:" + "0" * 64
        revision = 0
        events = 0
        if self.operations_path.exists():
            for line in read_stable(self.operations_path).splitlines():
                event = load_json(line, source=str(self.operations_path))
                if not isinstance(event, dict) or event.get("schema") != EVENT_SCHEMA:
                    raise IntegrityError("invalid operation event")
                if event.get("revision") != revision + 1:
                    raise IntegrityError("operation revisions are not contiguous")
                if event.get("previous_event_hash") != previous:
                    raise IntegrityError("operation hash chain is broken")
                claimed = event.get("event_hash")
                body = dict(event); body.pop("event_hash", None)
                if claimed != _event_hash(body):
                    raise IntegrityError("operation event hash mismatch")
                previous = claimed; revision += 1; events += 1
        if revision != control["revision"] or previous != control["event_head"]:
            raise IntegrityError("control does not match operation head")
        if _state_digest(control) != control["state_digest"]:
            raise IntegrityError("control state digest mismatch")
        return VerificationReport(control["revision"], control["event_head"], control["state_digest"], events)

    def mutate(self, expected_revision: int, operation: Callable[[dict[str, Any]], tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        if type(expected_revision) is not int or expected_revision < 0:
            raise InputError("expected_revision must be a non-negative integer")
        with self._lock():
            self._recover_transaction()
            control = self._read_control()
            self._assert_anchor(control)
            self.verify()
            if control["revision"] != expected_revision:
                raise ConcurrentRevisionError(f"expected revision {expected_revision}, got {control['revision']}")
            candidate = copy.deepcopy(control)
            event_type, details = operation(candidate)
            if not isinstance(event_type, str) or not event_type:
                raise InputError("event type must be non-empty")
            candidate["revision"] = control["revision"] + 1
            candidate["state_digest"] = _state_digest(candidate)
            event = {"schema": EVENT_SCHEMA, "revision": candidate["revision"], "timestamp": _now(),
                     "type": event_type, "session_binding": candidate["session_binding"],
                     "previous_event_hash": control["event_head"], "state_digest": candidate["state_digest"],
                     "details": details}
            event["event_hash"] = _event_hash(event)
            candidate["event_head"] = event["event_hash"]
            validate_control(candidate)
            # Prepare a durable marker before changing CONTROL.json. On a
            # crash, the next load completes the exact prepared bytes.
            control_bytes = canonical_bytes(candidate)
            event_bytes = canonical_bytes(event)
            txn = {"schema": "ablagent.transaction.v1", "control_hex": control_bytes.hex(),
                   "previous_control_hex": canonical_bytes(control).hex(),
                   "event_hex": event_bytes.hex(), "control_digest": digest(candidate),
                   "control_instance_id": candidate["control_instance_id"],
                   "run_directory": candidate["run_directory"], "revision": candidate["revision"],
                   "event_head": candidate["event_head"], "state_digest": candidate["state_digest"]}
            atomic_write(self.transaction_path, canonical_bytes(txn))
            atomic_write(self.control_path, control_bytes)
            self._recover_transaction()
            return candidate

    def authorize_direct(self, stage: str, actions: list[str], *, expected_revision: int,
                         principal: str) -> dict[str, Any]:
        stage = normalize_stage(stage)
        if not isinstance(principal, str) or not principal:
            raise AuthorityError("principal is required")
        actions = sorted(set(actions))
        if not actions or any(action not in LIFECYCLE_ACTIONS for action in actions):
            raise AuthorityError("unknown or empty lifecycle actions")
        def op(control: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            if not dependencies_satisfied(control, stage):
                raise StateError(f"dependencies are incomplete for {stage}")
            state = control["stages"][stage]
            if state["authorization"] not in {"ungranted", "expired", "revoked"}:
                raise AuthorityError(f"stage already authorized: {stage}")
            state["authorization"] = "active"
            state["execution"] = "awaiting_greenlight"
            return "authority_granted", {"stage": stage, "actions": actions, "principal": principal, "mode": "direct"}
        return self.mutate(expected_revision, op)

    def transition(self, stage: str, action: str, *, expected_revision: int) -> dict[str, Any]:
        stage = normalize_stage(stage)
        if action not in LIFECYCLE_ACTIONS:
            raise InputError(f"unknown lifecycle action: {action}")
        def op(control: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            state = control["stages"][stage]
            if action == "start":
                if state["authorization"] != "active" or not dependencies_satisfied(control, stage):
                    raise StateError(f"stage cannot start: {stage}")
                state["execution"] = "running"
            elif action == "complete":
                if state["execution"] != "running":
                    raise StateError(f"stage is not running: {stage}")
                state["execution"] = "completed"; state["authorization"] = "consumed"
            elif action in {"fail", "cancel", "terminate"}:
                if state["execution"] not in {"running", "awaiting_greenlight"}:
                    raise StateError(f"stage cannot {action}: {stage}")
                state["execution"] = {"fail": "failed", "cancel": "cancelled", "terminate": "terminated"}[action]
                state["authorization"] = "consumed"
            elif action == "pause":
                if state["execution"] != "running": raise StateError("stage is not running")
                state["execution"] = "paused"
            elif action == "freeze":
                if state["execution"] not in {"running", "paused", "completed"}: raise StateError("stage cannot freeze")
                state["execution"] = "frozen"
            else:
                raise StateError(f"action not implemented for stage state: {action}")
            return "stage_transition", {"stage": stage, "action": action}
        return self.mutate(expected_revision, op)

    def skip_stage(self, stage: str, *, expected_revision: int, reason: str,
                   allowed_stages: frozenset[str] | None = None) -> dict[str, Any]:
        """Explicitly skip an optional stage for this run (exploratory mode).

        Records execution="skipped" + measurement="not_applicable" with the
        operator's reason — never a silent pass. By default only stages in
        SKIP_OPTIONAL_STAGES may be skipped (e.g. SR_RELEASE_REDLINE for
        exploratory runs). A caller may pass an explicit ``allowed_stages``
        set (e.g. REAL_UNMET_STAGES for the v0 real executor); the default
        operator authority surface is unchanged. Skipped gate stages always
        leave release not_eligible.
        """
        from ..constants import SKIP_OPTIONAL_STAGES
        stage = normalize_stage(stage)
        allowed = SKIP_OPTIONAL_STAGES if allowed_stages is None else allowed_stages
        if stage not in allowed:
            raise InputError(f"stage {stage} cannot be skipped (not in SKIP_OPTIONAL_STAGES)")
        if not isinstance(reason, str) or not reason.strip():
            raise InputError("a non-empty skip reason is required (honest evidence)")
        def op(control: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            state = control["stages"][stage]
            if state["execution"] not in {"planned", "awaiting_greenlight"}:
                raise StateError(f"stage cannot be skipped from execution state {state['execution']!r}: {stage}")
            state["execution"] = "skipped"
            state["measurement"] = "not_applicable"
            state["evidence"].append({
                "gate_id": "stage-skip",
                "reason": reason,
                "recorded_kind": "stage-skip",
                "stage": stage,
            })
            return "stage_skipped", {"stage": stage, "reason": reason}
        return self.mutate(expected_revision, op)

    # -- measurement and release ------------------------------------------------
    #
    # Gate results (red-line holdout, domain verify, capability floor) are
    # recorded as typed measurement evidence bound to the gate identity and
    # criteria digest, then drive release transitions. Fail-closed: a stage
    # cannot be marked accepted without recorded gate evidence; release cannot
    # be accepted unless every gate stage is accepted.

    def record_measurement(self, stage: str, result: dict[str, Any], *, expected_revision: int,
                           gate_id: str, criteria_digest: str) -> dict[str, Any]:
        """Record a gate result and set the stage measurement state.

        The gate result payload must carry ``passed: bool``. A passed gate
        sets measurement to ``accepted``; anything else is ``automatic_failed``
        (the model's behavior failed a gate) or raises (the gate result itself
        was malformed — corpus or harness bug, never silently swallowed).
        """
        stage = normalize_stage(stage)
        if not isinstance(gate_id, str) or not gate_id:
            raise InputError("gate_id is required")
        if not isinstance(criteria_digest, str) or not criteria_digest.startswith("sha256:"):
            raise InputError("criteria_digest must be a sha256:… digest")
        if not isinstance(result, dict) or type(result.get("passed")) is not bool:
            raise InputError("gate result must be a mapping with boolean 'passed'")
        def op(control: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            state = control["stages"][stage]
            if state["execution"] not in {"completed", "running"}:
                raise StateError(f"cannot record measurement for stage in execution state {state['execution']!r}: {stage}")
            passed = result["passed"]
            state["measurement"] = "accepted" if passed else "automatic_failed"
            state["evidence"].append({
                "gate_id": gate_id,
                "criteria_digest": criteria_digest,
                "passed": passed,
                "recorded_kind": "gate-result",
                "stage": stage,
            })
            return ("measurement_recorded" if passed else "measurement_failed",
                    {"stage": stage, "gate_id": gate_id, "criteria_digest": criteria_digest, "passed": passed})
        return self.mutate(expected_revision, op)

    def release_check(self, *, expected_revision: int,
                     release_gate_stages: tuple[str, ...] = RELEASE_GATE_STAGES) -> dict[str, Any]:
        """Evaluate release readiness. Gates accepted → release_ready.

        Returns the post-check control state. Fails closed (raises) when a
        gate stage has not been accepted or its execution is not complete.
        Calling this before all gate stages are completed is an operator
        error, not a run-state corruption, so StateError is raised.
        """
        def op(control: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            unaccepted: list[str] = []
            incomplete: list[str] = []
            for stage in release_gate_stages:
                state = control["stages"][stage]
                if state["execution"] == "skipped":
                    continue  # not_applicable: red-lines untested this run (exploratory)
                if state["execution"] != "completed":
                    incomplete.append(stage)
                if state["measurement"] not in ("accepted", "not_applicable"):
                    unaccepted.append(stage)
            problems = []
            if incomplete:
                problems.append(f"stages not completed: {incomplete}")
            if unaccepted:
                problems.append(f"stages not accepted: {unaccepted}")
            if problems:
                raise StateError("release check failed: " + "; ".join(problems))
            for stage in release_gate_stages:
                if control["stages"][stage]["execution"] == "skipped":
                    continue  # skipped stage: release stays not_eligible (untested, fail-closed)
                control["stages"][stage]["release"] = "release_ready"
            return "release_check_passed", {"stages": list(release_gate_stages)}
        return self.mutate(expected_revision, op)

    def accept_release(self, *, expected_revision: int, principal: str,
                      release_gate_stages: tuple[str, ...] = RELEASE_GATE_STAGES) -> dict[str, Any]:
        """Operator transition: release_ready → accepted for the gate stages."""
        if not isinstance(principal, str) or not principal:
            raise AuthorityError("principal is required")
        def op(control: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            not_ready = [stage for stage in release_gate_stages
                         if control["stages"][stage]["release"] != "release_ready"]
            if not_ready:
                raise StateError(f"release not ready for stages: {not_ready}")
            for stage in release_gate_stages:
                if control["stages"][stage]["release"] == "release_ready":
                    control["stages"][stage]["release"] = "accepted"
            return "release_accepted", {"stages": list(release_gate_stages), "principal": principal}
        return self.mutate(expected_revision, op)
