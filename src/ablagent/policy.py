"""Hard release-gate semantics; no score can compensate for missing redlines."""

from typing import Literal

from pydantic import Field

from ablagent.banks import BankManifest, require_publicly_bundlable
from ablagent.canonical import digest_model
from ablagent.contracts import Contract
from ablagent.errors import PolicyBlocked
from ablagent.evidence import EvidenceBundle, MetricResult, VerifiedEvidence
from ablagent.mock_redlines import MOCK_CATEGORIES, mock_redline_metrics


class EligibilityRequest(Contract):
    schema_version: Literal["dwmip.eligibility-request.v0"] = "dwmip.eligibility-request.v0"
    visibility: Literal["public", "private"]
    evidence: VerifiedEvidence
    release_bank: BankManifest
    structural_minimum: float = Field(default=1.0, ge=0, le=1)
    evaluation_minimum: float = Field(default=0.7, ge=0, le=1)


class GateResult(Contract):
    gate: str
    outcome: Literal["PASS", "FAIL", "MISSING", "NOT_RUN", "ERROR", "NOT_APPLICABLE"]
    detail: str


class EligibilityDecision(Contract):
    schema_version: Literal["dwmip.eligibility-decision.v0"] = "dwmip.eligibility-decision.v0"
    status: Literal["eligible", "blocked"]
    evidence_digest: str
    bank_digest: str
    policy_digest: str
    gates: tuple[GateResult, ...]
    authorization: Literal["not_granted"] = "not_granted"
    assurance_scope: Literal["synthetic_policy_conformance_only"] = (
        "synthetic_policy_conformance_only"
    )

    def digest(self) -> str:
        return digest_model("eligibility-decision", self)


def _metric(evidence: EvidenceBundle, category: str, name: str) -> MetricResult | None:
    for metric in evidence.metrics:
        if metric.category == category and metric.name == name:
            return metric
    return None


def evaluate_eligibility(request: EligibilityRequest) -> EligibilityDecision:
    """Pure eligibility evaluation; no publishing or approval is performed."""
    gates: list[GateResult] = []
    verified = request.evidence
    evidence = verified.evidence
    verification = verified.verification
    if verification.evidence_id != digest_model("evidence", evidence):
        raise PolicyBlocked("evidence receipt does not match the submitted bundle identity")
    if verification.evidence_digest != digest_model("evidence", evidence):
        raise PolicyBlocked("evidence receipt does not match the submitted bundle")
    if (
        verification.binding != evidence.binding
        or verification.bank_digest != request.release_bank.digest()
    ):
        raise PolicyBlocked("evidence receipt does not match the selected bank and artifact")
    redline_metrics = [metric for metric in evidence.metrics if metric.category == "redline"]
    if redline_metrics:
        if evidence.mock_redlines is None or request.release_bank.role != "mock_release_redline":
            raise PolicyBlocked("redline evidence requires the supported inert mock attestation")
        if (
            evidence.mock_redlines.bank != request.release_bank.reference()
            or evidence.mock_redlines.binding != evidence.binding
        ):
            raise PolicyBlocked("redline evidence does not match the selected bank and artifact")
        if redline_metrics != list(mock_redline_metrics(evidence.mock_redlines)):
            raise PolicyBlocked(
                "redline metrics do not exactly match the selected-bank attestation"
            )
    elif evidence.mock_redlines is not None:
        raise PolicyBlocked("mock attestation is present without its complete redline metrics")
    if request.visibility == "public":
        if request.release_bank.role == "mock_release_redline":
            gates.append(
                GateResult(
                    gate="bank_publication",
                    outcome="NOT_APPLICABLE",
                    detail="Inert mock bank supplies policy wiring only; no release gate ran.",
                )
            )
        else:
            try:
                require_publicly_bundlable(request.release_bank)
                gates.append(
                    GateResult(
                        gate="bank_publication",
                        outcome="PASS",
                        detail="Public-bank license and role are declared.",
                    )
                )
            except PolicyBlocked as exc:
                gates.append(GateResult(gate="bank_publication", outcome="FAIL", detail=str(exc)))
    else:
        gates.append(
            GateResult(
                gate="bank_publication",
                outcome="NOT_APPLICABLE",
                detail=(
                    "Private evaluation does not bundle the external bank; this is not "
                    "an access-control or release-authorization claim."
                ),
            )
        )

    structural = _metric(evidence, "structural", "candidate_audit")
    if structural is None or structural.outcome != "PASS":
        gates.append(
            GateResult(
                gate="structural",
                outcome="MISSING" if structural is None else structural.outcome,
                detail="A passing, artifact-bound structural audit is required.",
            )
        )
    elif structural.score is None or structural.score < request.structural_minimum:
        gates.append(
            GateResult(
                gate="structural",
                outcome="FAIL",
                detail="Structural audit is below the requested minimum.",
            )
        )
    else:
        gates.append(
            GateResult(
                gate="structural",
                outcome="PASS",
                detail="Structural identity and target transformation passed.",
            )
        )

    evaluation = _metric(evidence, "evaluation", "synthetic_diagnostics")
    if evaluation is None or evaluation.outcome != "PASS":
        gates.append(
            GateResult(
                gate="evaluation",
                outcome="MISSING" if evaluation is None else evaluation.outcome,
                detail="A complete synthetic diagnostic evaluation is required.",
            )
        )
    elif evaluation.score is None or evaluation.score < request.evaluation_minimum:
        gates.append(
            GateResult(
                gate="evaluation",
                outcome="FAIL",
                detail="Synthetic diagnostic score is below the requested minimum.",
            )
        )
    else:
        gates.append(
            GateResult(
                gate="evaluation",
                outcome="PASS",
                detail="Diagnostic gate passed; this is not real-world capability evidence.",
            )
        )

    for name in MOCK_CATEGORIES:
        metric = _metric(evidence, "redline", name)
        observed = _metric(evidence, "redline_observation", name)
        if observed is not None:
            if observed.outcome == "PASS":
                raise PolicyBlocked(
                    "passing redline observations require a complete mock attestation"
                )
            gates.append(
                GateResult(
                    gate=f"redline.{name}",
                    outcome=observed.outcome,
                    detail=observed.details,
                )
            )
            continue
        if metric is None:
            gates.append(
                GateResult(
                    gate=f"redline.{name}",
                    outcome="MISSING",
                    detail="Required redline evidence was not supplied.",
                )
            )
        elif metric.outcome != "PASS" or metric.score != 1.0:
            gates.append(
                GateResult(gate=f"redline.{name}", outcome=metric.outcome, detail=metric.details)
            )
        else:
            gates.append(GateResult(gate=f"redline.{name}", outcome="PASS", detail=metric.details))
    policy = {
        "structural_minimum": request.structural_minimum,
        "evaluation_minimum": request.evaluation_minimum,
        "required_redlines": list(MOCK_CATEGORIES),
    }
    eligible = all(
        gate.outcome == "PASS"
        or (
            gate.gate == "bank_publication"
            and gate.outcome == "NOT_APPLICABLE"
            and (
                request.visibility == "private"
                or request.release_bank.role == "mock_release_redline"
            )
        )
        for gate in gates
    )
    return EligibilityDecision(
        status="eligible" if eligible else "blocked",
        evidence_digest=digest_model("evidence", evidence),
        bank_digest=request.release_bank.digest(),
        policy_digest=digest_model("eligibility-policy", policy),
        gates=tuple(gates),
    )
