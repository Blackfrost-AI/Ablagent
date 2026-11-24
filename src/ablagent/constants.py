"""Canonical pipeline constants for Ablagent V2.

Ported from V1 ``src/ablagentctl.py`` (workspace Ablagent, 2026-10) so the
V2 pipeline graph, stage names, and schema identifiers stay identical in spirit.
The V2 pipeline generalizes V1's refusal-removal focus into domain packs:
each stage's semantics are defined by the active domain pack, while the DAG,
gating model, and fail-closed control plane stay domain-independent.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema identifiers (wire/format contracts)
# ---------------------------------------------------------------------------

CONTROL_SCHEMA = "ablagent.control.v2"
CONTROL_REQUEST_SCHEMA = "ablagent.control-request.v2"
AUTHORITY_SCHEMA = "authority.v1"  # unchanged from V1; operator receipts
EFFECTIVE_POLICY_SCHEMA = "ablagent.effective-policy.v2"
BENCHMARK_REGISTRY_SCHEMA = "ablagent.benchmarks.v2"
PACK_MANIFEST_SCHEMA = "ablagent.pack.v1"
RUN_STATE_SCHEMA = "ablagent.run-state.v2"
EVIDENCE_RECEIPT_SCHEMA = "ablagent.evidence.v2"

# ---------------------------------------------------------------------------
# Pipeline stages (V1-compatible names, V2 pack-driven semantics)
# ---------------------------------------------------------------------------

STAGES: tuple[str, ...] = (
    "S0_INTAKE",
    "S1_BASELINE",
    "S2_DIRECTIONS",
    "S3_DERISK",
    "S4_QUANT",
    "S5_SERVE",
    "S6_DOMAIN_VERIFY",
    "S7_CAPABILITY_VERIFY",
    "SR_RELEASE_REDLINE",
    "S8_PRODUCTIZE",
    "S9_CARD_DRAFT",
    "S10_DELIVER",
)

STAGE_ALIASES: dict[str, str] = {stage: stage for stage in STAGES} | {
    "S0": "S0_INTAKE",
    "S1": "S1_BASELINE",
    "S2": "S2_DIRECTIONS",
    "S3": "S3_DERISK",
    "S4": "S4_QUANT",
    "S5": "S5_SERVE",
    "S6": "S6_DOMAIN_VERIFY",  # V1: S6_REFUSAL_VERIFY
    "S7": "S7_CAPABILITY_VERIFY",
    "SR": "SR_RELEASE_REDLINE",
    "S8": "S8_PRODUCTIZE",
    "S9": "S9_CARD_DRAFT",
    "S10": "S10_DELIVER",
}

# Exact DAG. S4 is added to S8 only when the run selects quantization.
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "S0_INTAKE": (),
    "S1_BASELINE": ("S0_INTAKE",),
    "S2_DIRECTIONS": ("S0_INTAKE",),
    "S3_DERISK": ("S2_DIRECTIONS",),
    "S4_QUANT": ("S3_DERISK",),
    "S5_SERVE": ("S3_DERISK",),
    "S6_DOMAIN_VERIFY": ("S5_SERVE",),
    "S7_CAPABILITY_VERIFY": ("S5_SERVE",),
    "SR_RELEASE_REDLINE": ("S6_DOMAIN_VERIFY", "S7_CAPABILITY_VERIFY"),
    "S8_PRODUCTIZE": (
        "S3_DERISK",
        "S6_DOMAIN_VERIFY",
        "S7_CAPABILITY_VERIFY",
        "SR_RELEASE_REDLINE",
    ),
    "S9_CARD_DRAFT": ("S8_PRODUCTIZE",),
    "S10_DELIVER": (
        "S6_DOMAIN_VERIFY",
        "S7_CAPABILITY_VERIFY",
        "SR_RELEASE_REDLINE",
        "S8_PRODUCTIZE",
        "S9_CARD_DRAFT",
    ),
}

RELEASE_GATE_STAGES: tuple[str, ...] = (
    "S6_DOMAIN_VERIFY",
    "S7_CAPABILITY_VERIFY",
    "SR_RELEASE_REDLINE",
    "S8_PRODUCTIZE",
    "S9_CARD_DRAFT",
)

# Stages that emit typed measurement results into the run state.
TYPED_STAGES: frozenset[str] = frozenset({
    "S3_DERISK", "S4_QUANT", "S5_SERVE", "S6_DOMAIN_VERIFY",
    "S7_CAPABILITY_VERIFY", "SR_RELEASE_REDLINE", "S8_PRODUCTIZE",
    "S9_CARD_DRAFT",
})

# ---------------------------------------------------------------------------
# Run-state vocabularies (fail-closed: unknown values are rejected)
# ---------------------------------------------------------------------------

AUTHORIZATION_STATES: tuple[str, ...] = (
    "ungranted", "active", "consumed", "expired", "revoked",
)
EXECUTION_STATES: tuple[str, ...] = (
    "planned", "awaiting_greenlight", "running", "frozen", "paused",
    "verify_pending", "completed", "failed", "cancelled", "terminated",
    "skipped",
)
MEASUREMENT_STATES: tuple[str, ...] = (
    "not_started", "measurement_incomplete", "automatic_failed",
    "hitl_pending", "accepted", "rejected",
    # Skipped-for-this-run stages (exploratory mode): explicitly untested,
    # never silently passed. See store.skip_stage().
    "not_applicable",
)
RELEASE_STATES: tuple[str, ...] = (
    "not_eligible", "evidence_pending", "release_ready", "delivered",
    "accepted",
)
TERMINAL_EXECUTION_STATES: frozenset[str] = frozenset({
    "completed", "failed", "cancelled", "terminated",
})
LIFECYCLE_ACTIONS: frozenset[str] = frozenset({
    "start", "complete", "fail", "freeze", "pause", "cancel", "terminate",
    "release-check", "accept-release", "adopt-policy", "classify-experimental",
    "private-archive",
})

# V1 state dimensions kept for porting parity (V1: S6 was refusal-specific;
# V2 renames the dimension to domain-verify while retaining the vocabulary).
DISPOSITION_NOT_APPLICABLE = "not_applicable"

# Run-level mode: stages that may be explicitly skipped for exploratory runs.
# A skipped stage is recorded as execution="skipped" + measurement="not_applicable"
# (via store.skip_stage) — never silently passed. The evidence trail must show
# what was untested. Skipped stages satisfy DAG dependencies (their results are
# not required by downstream gates that also tolerate not_applicable).
SKIP_OPTIONAL_STAGES: frozenset[str] = frozenset({"SR_RELEASE_REDLINE"})

# Real-executor adapter identity (dense HuggingFace checkpoints, CPU torch).
HF_ADAPTER_ID = "hf-dense-linear-v0"

# Stages the v0 real adapter cannot honestly execute (no quantizer, no serving
# stack, no delivery channel). A real run may skip exactly these — explicitly,
# with a recorded reason — via store.skip_stage(allowed_stages=...). They are
# NOT added to SKIP_OPTIONAL_STAGES: the default operator authority surface is
# unchanged, and release stays not_eligible while any gate stage is skipped.
REAL_UNMET_STAGES: frozenset[str] = frozenset({
    "S4_QUANT", "S5_SERVE", "S6_DOMAIN_VERIFY", "S7_CAPABILITY_VERIFY",
    "S10_DELIVER",
})

# ---------------------------------------------------------------------------
# Supported domains (first-class packs shipped with the platform)
# ---------------------------------------------------------------------------

BUILTIN_DOMAINS: tuple[str, ...] = (
    "cybersecurity",
    "healthcare",
    "finance",
    "logistics",
    "derisk",  # general false-refusal removal — the V1 flagship use case
)
