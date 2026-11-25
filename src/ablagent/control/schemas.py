"""Control-state schema and stage normalization."""

from __future__ import annotations

from typing import Any

from ..constants import (
    AUTHORIZATION_STATES,
    DEPENDENCIES,
    EXECUTION_STATES,
    MEASUREMENT_STATES,
    RELEASE_STATES,
    STAGE_ALIASES,
    STAGES,
)
from .errors import InputError, StateError


def normalize_stage(value: str) -> str:
    if not isinstance(value, str):
        raise InputError("stage must be a string")
    try:
        return STAGE_ALIASES[value.upper()]
    except KeyError as exc:
        raise InputError(f"unknown stage: {value!r}") from exc


def empty_stage_state(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "execution": "planned",
        "authorization": "ungranted",
        "measurement": "not_started",
        "release": "not_eligible",
        "evidence": [],
    }


def validate_control(control: dict[str, Any]) -> None:
    required = {
        "schema",
        "version",
        "control_instance_id",
        "run_directory",
        "revision",
        "event_head",
        "state_digest",
        "session_binding",
        "stages",
        "used_authority_receipts",
    }
    missing = required - set(control)
    if missing:
        raise StateError(f"control missing fields: {sorted(missing)}")
    if control["schema"] != "ablagent.control.v2" or type(control["version"]) is not int:
        raise StateError("invalid control schema/version")
    if type(control["revision"]) is not int or control["revision"] < 0:
        raise StateError("revision must be a non-negative integer")
    if not isinstance(control["stages"], dict) or set(control["stages"]) != set(STAGES):
        raise StateError("control stages must contain exactly the canonical stage set")
    for stage in STAGES:
        state = control["stages"][stage]
        if not isinstance(state, dict) or state.get("stage") != stage:
            raise StateError(f"invalid state for {stage}")
        if state.get("execution") not in EXECUTION_STATES:
            raise StateError(f"invalid execution state for {stage}")
        if state.get("authorization") not in AUTHORIZATION_STATES:
            raise StateError(f"invalid authorization state for {stage}")
        if state.get("measurement") not in MEASUREMENT_STATES:
            raise StateError(f"invalid measurement state for {stage}")
        if state.get("release") not in RELEASE_STATES:
            raise StateError(f"invalid release state for {stage}")
        if not isinstance(state.get("evidence"), list):
            raise StateError(f"evidence must be a list for {stage}")


def dependencies_satisfied(control: dict[str, Any], stage: str) -> bool:
    stage = normalize_stage(stage)
    return all(
        control["stages"][dep]["execution"] in ("completed", "skipped")
        for dep in DEPENDENCIES[stage]
    )
