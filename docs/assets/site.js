/* Ablagent site — content grounded in src/ablagent/policy.py, docs/methodology.md,
   docs/measurement.md, spec/dwmip/v0/README.md. The simulator below is a faithful
   client-side mirror of evaluate_eligibility() for a VERIFIED synthetic bundle. */
"use strict";

/* ---------------- process stages ---------------- */
const STAGES = [
  {
    title: "Declare the computation",
    purpose: "Start from a checkpoint-specific graph and adapter declaration — not a remembered architecture name.",
    points: [
      "Identify logical sites, storage axes, routing, recurrent states, shared weights, normalization, and the exact capture point.",
      "Unknown semantics stay UNSUPPORTED — no suffix or shape fallback ever activates a writer.",
      "Structural coverage and decode coverage are different evidence; a state dictionary alone does not encode the forward computation."
    ],
    artifacts: ["GraphSpec", "SiteCoordinate", "WriterSpec", "BankManifest", "AdapterCapabilities"],
    boundary: "The v0 reference runtime executes only the analytic dense_linear synthetic adapter. Dense/MoE/recurrent/looped coordinates can be described, but description is not execution."
  },
  {
    title: "Derive a candidate, not a conclusion",
    purpose: "Contrast statistics produce DirectionCandidate records. They are correlational — never treated as bound directions.",
    points: [
      "Mean differences and probes may carry topic, wording, length, style, or source confounds; a probe can recover information the model does not use.",
      "Separate capture and calibration collections, balanced controls, full counts, and bank digests.",
      "Held-out evaluation sources stay held out: a private release holdout never becomes a tuning signal."
    ],
    artifacts: ["DirectionCandidate", "capture bank digest", "calibration bank digest", "full counts"],
    boundary: "A candidate vector cannot be treated as a bound direction. Binding requires one writer, one declared output space, unit normalization, and the full identity chain."
  },
  {
    title: "Compile one bounded change",
    purpose: "Planning proposes; compilation is the only step that writes a candidate artifact — from immutable, digest-bound inputs.",
    points: [
      "The v0 operation: W′ = W − α·d(dᵀW), unit d in the declared output space, 0 ≤ α ≤ 1, one coordinate per intervention.",
      "α = 0 is an analytic control; α = 1 removes the d component. Arbitrary α is a rank-one update, not generally a projection.",
      "Multiple non-orthogonal updates are order-dependent — v0 does not silently sum or compose directions.",
      "The artifact binds baseline and candidate tensor/file digests, codec readback, and transformation metadata."
    ],
    artifacts: ["InterventionSpec", "CompiledArtifact", "baseline digest", "candidate digest"],
    boundary: "A compiled artifact is not behaviorally evaluated. Loading it proves decodability, nothing about behavior."
  },
  {
    title: "Verify independently",
    purpose: "Verification reloads the graph and tensors and recomputes the update and observation — it never trusts the producer's numbers.",
    points: [
      "verify: schema, identities, exact algebra, non-target preservation, codec readback.",
      "verify-evidence: selected-trial parent chain, metric references, environment/decoding configuration, offline — no database, credentials, or network.",
      "Mock red-line evidence must bind the selected mock_release_redline bank and the exact artifact pair, and match all six inert fixtures exactly; missing, extra, or altered records fail closed."
    ],
    artifacts: ["EvidenceBundle", "trial lineage", "metric evidence digests", "VerifiedEvidence"],
    boundary: "Hashes provide tamper evidence, not author authenticity. Signed approvals and a protected release system are separate responsibilities."
  },
  {
    title: "Decide eligibility — policy only",
    purpose: "evaluate_eligibility() computes gate outcomes from verified evidence. Every gate must pass; scores never compensate for missing red-lines.",
    points: [
      "Nine gate outcomes, conjunctive: structural, evaluation, bank_publication, and six redline.<category> checks. Artifact verification is a precondition, not a gate.",
      "bank_publication is the only gate with NOT_APPLICABLE — private evaluation, or the inert mock bank that supplies policy wiring only.",
      "Passing red-line metrics are accepted only with a complete digest-bound mock attestation for the selected bank and artifact pair; red-line PASS requires score exactly 1.0.",
      "authorization is always not_granted; assurance_scope is always synthetic_policy_conformance_only."
    ],
    artifacts: ["EligibilityDecision", "gate outcomes", "hard-block exceptions"],
    boundary: "Eligible means policy-conformant on synthetic evidence — not safe, not behaviorally validated, not authorized for release. Release is a separate, human-authorized decision."
  }
];

/* ---------------- gates (mirrors policy.py) ---------------- */
const GATES = [
  {
    name: "verify_evidence (precondition)", tag: "precondition", summary: "Re-hash, reload, replay — before any gate runs",
    purpose: "Not a gate in the decision: evaluate_eligibility() first calls verify_evidence(), which re-hashes and replays the artifact, the trial chain, and metric references. Unverified evidence never reaches the gates.",
    detail: "Identities, codec readback, exact recomputation of the rank-one update, non-target preservation, selected-trial parent chain, environment/decoding configuration — all checked offline, with no credentials or network.",
    fail: "Any digest mismatch, algebra deviation, or non-target change means the evidence is not what it claims; evaluation is refused before a decision exists, not scored."
  },
  {
    name: "structural", tag: "hard", summary: "PASS with score ≥ declared minimum",
    purpose: "Structural verification of the transformation: the edit is exactly the declared operation on exactly the declared tensor, and everything else is byte-preserved.",
    detail: "Requires state PASS and score ≥ minimum. A structural PASS is necessary but never sufficient — it says nothing about behavior, capability, or serving.",
    fail: "FAIL, MISSING, NOT_RUN, or ERROR all block. A high score in another gate cannot offset a structural failure."
  },
  {
    name: "evaluation", tag: "hard", summary: "PASS with score ≥ declared minimum",
    purpose: "The scoped evaluation metric (in v0: the analytic synthetic observation) must be complete and meet its declared threshold.",
    detail: "Requires state PASS and score ≥ minimum. The bundle is explicitly synthetic_reference_only: this gate proves fixture wiring and measurement discipline, not real behavioral change.",
    fail: "Any non-PASS state blocks. 'Behavior failed' (FAIL) and 'the gate could not evaluate behavior' (MISSING/NOT_RUN/ERROR) remain distinct facts."
  },
  {
    name: "bank_publication", tag: "conditional", summary: "Bundlable bank — or N/A (private / inert mock)",
    purpose: "A public release must not bundle a restricted evaluation bank. Public bundling requires declared public visibility, redistribution permission, an approved SPDX license, and a role other than release_redline.",
    detail: "The only gate with NOT_APPLICABLE, by explicit predicate: private evaluation does not bundle the external bank, and the inert mock_release_redline bank supplies policy wiring only — no release gate ran. N/A counts toward eligibility only under those two predicates. Absence alone never implies N/A.",
    fail: "Public visibility with a private, unlicensed, or release-holdout bank FAILs with the blocking reason recorded in the gate detail."
  },
  {
    name: "redline.<category> (×6)", tag: "hard", summary: "All six categories: attested PASS at score 1.0",
    purpose: "Six mandatory red-line categories — biological, chemical, explosives_incendiary, strategic_wmd, cyber, loss_of_control — each require state PASS with score exactly 1.0, backed by a complete digest-bound mock attestation. There is no adjustable red-line threshold.",
    detail: "Passing metrics are only accepted from the attested mock evaluator: the attestation must bind the selected mock bank, the exact baseline/candidate artifact pair, evaluator/configuration literals, and all six results — and the bundle's metrics must equal the attestation-derived metrics exactly. redline_observation records explicit non-passing diagnostics; it can never supply a passing metric.",
    fail: "Any category in any state other than attested PASS at 1.0 blocks. Scores in other gates cannot compensate. Tampering raises PolicyBlocked before any decision exists."
  },
  {
    name: "authorization", tag: "invariant", summary: "Always not_granted",
    purpose: "The protocol never grants release authorization. This field is a constant, not a gate you can pass.",
    detail: "Every EligibilityDecision carries authorization = not_granted, even when all gates pass. Release requires a separate human-authorized process with its own evidence.",
    fail: "There is no input that changes this. A decision claiming authorization would be a protocol violation, not a policy outcome."
  },
  {
    name: "assurance_scope", tag: "invariant", summary: "Always synthetic_policy_conformance_only",
    purpose: "The decision's scope is pinned: it certifies policy conformance on synthetic evidence, nothing more.",
    detail: "Real-model behavioral claims, capability preservation, serving readiness, and safety certification each require separately qualified evidence that v0 does not produce.",
    fail: "Scope cannot be widened by any score. This is the protocol's anti-overclaim invariant."
  }
];

/* ---------------- evidence states ---------------- */
const STATES = [
  { name: "PASS", cls: "outcome-PASS", text: "A complete measurement meeting the declared threshold. Score and full denominator present." },
  { name: "FAIL", cls: "outcome-FAIL", text: "Measured and below threshold. Behavior failed — a real, recorded negative result." },
  { name: "MISSING", cls: "outcome-MISSING", text: "Required evidence was not supplied. Not the same as failing." },
  { name: "NOT_RUN", cls: "outcome-NOT_RUN", text: "Evaluation was not executed. The gate could not evaluate behavior." },
  { name: "ERROR", cls: "outcome-ERROR", text: "Execution, parsing, or validation failed. Stays in the denominator." },
  { name: "NOT_APPLICABLE", cls: "outcome-NOT_APPLICABLE", text: "Only by an explicit policy predicate with a recorded reason. Absence alone never implies it." }
];

const CHAIN = [
  ["GraphSpec + CaptureContract + BankManifest", "declared identities"],
  ["DirectionCandidate", "correlational statistic"],
  ["BoundDirection", "one writer · one space · unit d"],
  ["InterventionSpec", "one coordinate · one operation"],
  ["CompiledArtifact", "baseline + candidate digests"],
  ["EvidenceBundle", "trials · observations · results"],
  ["EligibilityDecision", "policy conformance only"]
];

const REDLINES = ["biological", "chemical", "explosives_incendiary", "strategic_wmd", "cyber", "loss_of_control"];
const METRIC_STATES = ["PASS", "FAIL", "MISSING", "NOT_RUN", "ERROR"];

/* ---------------- rendering ---------------- */
function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}

function renderStages() {
  const panel = document.getElementById("stage-panel");
  const buttons = document.querySelectorAll("#stepper .step");
  function show(i) {
    const s = STAGES[i];
    panel.innerHTML = "";
    panel.appendChild(el("h3", null, s.title));
    panel.appendChild(el("p", "stage-purpose", s.purpose));
    const ul = el("ul");
    s.points.forEach(p => ul.appendChild(el("li", null, p)));
    panel.appendChild(ul);
    const arts = el("div", "stage-artifacts");
    s.artifacts.forEach(a => arts.appendChild(el("code", null, a)));
    panel.appendChild(arts);
    panel.appendChild(el("div", "stage-boundary", s.boundary));
    buttons.forEach((b, j) => {
      b.classList.toggle("active", j === i);
      b.setAttribute("aria-selected", j === i ? "true" : "false");
    });
  }
  buttons.forEach(b => b.addEventListener("click", () => show(Number(b.dataset.stage))));
  show(0);
}

function renderGates() {
  const list = document.getElementById("gate-list");
  GATES.forEach(g => {
    const gate = el("div", "gate");
    const head = el("button", "gate-head");
    head.setAttribute("aria-expanded", "false");
    head.appendChild(el("span", "gate-name", g.name));
    head.appendChild(el("span", "gate-tag" + (g.tag === "hard" ? " hard" : ""), g.tag));
    head.appendChild(el("span", "gate-summary", g.summary));
    head.appendChild(el("span", "gate-caret", "▸"));
    const body = el("div", "gate-body");
    body.appendChild(el("p", "purpose", g.purpose));
    body.appendChild(el("p", null, g.detail));
    body.appendChild(el("p", "fail-mode", "<strong>Failure semantics:</strong> " + g.fail));
    head.addEventListener("click", () => {
      const open = gate.classList.toggle("open");
      head.setAttribute("aria-expanded", open ? "true" : "false");
    });
    gate.appendChild(head);
    gate.appendChild(body);
    list.appendChild(gate);
  });
}

function renderStates() {
  const grid = document.getElementById("states-grid");
  STATES.forEach(s => {
    const c = el("div", "state-card");
    c.appendChild(el("span", "state-name " + s.cls, s.name));
    c.appendChild(el("p", null, s.text));
    grid.appendChild(c);
  });
  const chain = document.getElementById("chain");
  CHAIN.forEach((node, i) => {
    if (i > 0) chain.appendChild(el("span", "chain-arrow", "→"));
    const n = el("div", "chain-node", node[0] + "<small>" + node[1] + "</small>");
    chain.appendChild(n);
  });
}

/* ---------------- simulator: mirror of evaluate_eligibility ---------------- */
function simRead() {
  const v = id => document.getElementById(id).value;
  const f = id => parseFloat(document.getElementById(id).value);
  const redlines = {};
  REDLINES.forEach(c => { redlines[c] = { state: v("rl-" + c + "-state") }; });
  return {
    visibility: v("sim-visibility"),
    bank: v("sim-bank"),
    tamper: v("sim-tamper"),
    structural: { state: v("sim-structural-state"), score: f("sim-structural-score"), min: f("sim-structural-min") },
    evaluation: { state: v("sim-evaluation-state"), score: f("sim-evaluation-score"), min: f("sim-evaluation-min") },
    redlines
  };
}

/* Mirrors policy.py evaluate_eligibility() for a verified bundle.
   Red-line metrics are attestation-derived (PASS ⇒ score 1.0, fixed — no sliders).
   Hard blocks raise before any decision. */
function simEvaluate(inp) {
  const mockBank = inp.bank === "mock";
  const attestationComplete = inp.tamper === "none";
  const anyMetric = REDLINES.some(c => inp.redlines[c].state !== "MISSING");
  if (anyMetric && (!attestationComplete || !mockBank)) {
    const why = inp.tamper === "bind"
      ? "redline evidence does not match the selected bank and artifact"
      : inp.tamper === "partial"
        ? "redline metrics do not exactly match the selected-bank attestation"
        : "redline evidence requires the supported inert mock attestation";
    return { hardBlock: "PolicyBlocked: " + why };
  }
  // 'bind' and 'partial' both mean an attestation is present; without the full
  // metric set the real policy raises before any decision (policy.py elif branch).
  if (!anyMetric && inp.tamper !== "none" && inp.tamper !== "obs") {
    return { hardBlock: "PolicyBlocked: mock attestation is present without its complete redline metrics" };
  }
  const gates = [];
  let bankOutcome, bankNote;
  if (inp.visibility === "public") {
    if (mockBank) {
      bankOutcome = "NOT_APPLICABLE"; bankNote = "inert mock bank supplies policy wiring only; no release gate ran";
    } else if (inp.bank === "public") {
      bankOutcome = "PASS"; bankNote = "public visibility, redistribution allowed, approved license";
    } else {
      bankOutcome = "FAIL"; bankNote = "real release holdouts cannot be bundled";
    }
  } else {
    bankOutcome = "NOT_APPLICABLE"; bankNote = "private evaluation does not bundle the external bank";
  }
  gates.push({ name: "bank_publication", outcome: bankOutcome, note: bankNote });
  const simple = m => m.state !== "PASS" ? m.state : (m.score >= m.min ? "PASS" : "FAIL");
  gates.push({ name: "structural", outcome: simple(inp.structural), note: "candidate_audit · score " + inp.structural.score.toFixed(2) + " vs min " + inp.structural.min.toFixed(2) });
  gates.push({ name: "evaluation", outcome: simple(inp.evaluation), note: "synthetic_diagnostics · score " + inp.evaluation.score.toFixed(2) + " vs min " + inp.evaluation.min.toFixed(2) });
  REDLINES.forEach(c => {
    const m = inp.redlines[c];
    gates.push({
      name: "redline." + c,
      outcome: m.state,
      note: m.state === "PASS" ? "attested · score 1.00 (fixed)" : "no attested pass"
    });
  });
  const bankOk = bankOutcome === "PASS" ||
    (bankOutcome === "NOT_APPLICABLE" && (inp.visibility === "private" || mockBank));
  const eligible = gates.every(g => g.name === "bank_publication" ? bankOk : g.outcome === "PASS");
  return { gates, eligible };
}

function simRender() {
  const inp = simRead();
  ["structural", "evaluation"].forEach(k => {
    document.getElementById("sim-" + k + "-score-val").textContent = inp[k].score.toFixed(2);
    document.getElementById("sim-" + k + "-min-val").textContent = inp[k].min.toFixed(2);
  });
  const verdict = document.getElementById("sim-verdict");
  const results = document.getElementById("sim-gate-results");
  results.innerHTML = "";
  const res = simEvaluate(inp);
  if (res.hardBlock) {
    verdict.className = "verdict hardblock";
    verdict.innerHTML = '<div class="verdict-status">HARD BLOCK</div><div class="verdict-detail">' + res.hardBlock +
      " — evaluation aborted before any decision exists. Tampering is an exception, not a score.</div>";
    return;
  }
  verdict.className = "verdict " + (res.eligible ? "eligible" : "blocked");
  verdict.innerHTML = '<div class="verdict-status">' + (res.eligible ? "ELIGIBLE" : "NOT ELIGIBLE") + "</div>" +
    '<div class="verdict-detail">' + (res.eligible
      ? "All gates pass on synthetic evidence. authorization remains not_granted; this is not release authorization."
      : "One or more mandatory gates did not pass. Scores cannot compensate for missing or failed red-line evidence.") + "</div>";
  res.gates.forEach(g => {
    const row = el("div", "sim-gate");
    row.appendChild(el("span", "g-name", g.name + ' <span style="opacity:0.6">· ' + g.note + "</span>"));
    row.appendChild(el("span", "g-outcome outcome-" + g.outcome, g.outcome));
    results.appendChild(row);
  });
}

function simBuild() {
  ["structural", "evaluation"].forEach(k => {
    const sel = document.getElementById("sim-" + k + "-state");
    METRIC_STATES.forEach(s => sel.appendChild(el("option", null, s)));
    sel.value = "PASS";
  });
  const rl = document.getElementById("sim-redlines");
  rl.appendChild(el("h4", null, "redline metrics — all six mandatory"));
  rl.appendChild(el("p", "redline-note", "Red-line scores are attestation-derived: a PASS is always 1.00 with a fixed policy minimum of 1.00 — there is no adjustable red-line threshold."));
  REDLINES.forEach(c => {
    const row = el("div", "redline-row");
    row.appendChild(el("span", null, c));
    const sel = el("select", "state-select");
    sel.id = "rl-" + c + "-state";
    METRIC_STATES.forEach(s => sel.appendChild(el("option", null, s)));
    sel.value = "PASS";
    row.appendChild(sel);
    rl.appendChild(row);
  });
  document.querySelectorAll(".sim select, .sim input").forEach(n => n.addEventListener("input", simRender));
  document.getElementById("sim-reset").addEventListener("click", () => {
    document.getElementById("sim-visibility").value = "private";
    document.getElementById("sim-bank").value = "mock";
    document.getElementById("sim-tamper").value = "none";
    ["structural", "evaluation"].forEach(k => {
      document.getElementById("sim-" + k + "-state").value = "PASS";
    });
    document.getElementById("sim-structural-score").value = "1";
    document.getElementById("sim-structural-min").value = "1";
    document.getElementById("sim-evaluation-score").value = "0.9";
    document.getElementById("sim-evaluation-min").value = "0.7";
    REDLINES.forEach(c => {
      document.getElementById("rl-" + c + "-state").value = "PASS";
    });
    simRender();
  });
  simRender();
}

/* ---------------- copy buttons ---------------- */
function initCopy() {
  document.querySelectorAll(".copy").forEach(btn => {
    btn.addEventListener("click", async () => {
      const text = document.getElementById(btn.dataset.copy).textContent;
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = "copied ✓";
      } catch {
        btn.textContent = "select + copy";
      }
      setTimeout(() => { btn.textContent = "copy"; }, 1600);
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  renderStages();
  renderGates();
  renderStates();
  simBuild();
  initCopy();
});