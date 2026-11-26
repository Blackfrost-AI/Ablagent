# Support matrix

| Capability | Status | Evidence/limit |
|---|---|---|
| Canonical manifest identities | Implemented | Local canonical JSON v1; not RFC 8785 |
| Safe tensor reference codec | Implemented | float32/float64 `safetensors`; bounded local parsing |
| Dense rank-one update | Implemented | Analytic CPU reference; one site, one direction |
| Complex graph representation | Schema only | MoE/recurrent/looped coordinates; no execution |
| Proposal-only rule planner | Implemented | Maximum four alpha values and eight plan steps |
| Independent evidence replay | Implemented | Local graph, tensor, observation, and trial verification |
| Fail-closed eligibility rules | Implemented | Policy conformance only; not authorization |
| Original synthetic banks | Implemented | Numeric capture and inert synthetic development fixtures |
| Public benchmark corpora | Implemented (prompts + protocol) | 443 eval / smoke-50 / cyber-10 in `benchmarks/`; prompt-only, hash-pinned, offline scorer |
| Real checkpoint support | Implemented (qualified envelope) | CPU-only `dense_out_in` safetensors intake/capture/bake/export via `--executor real`; ≤ ~900 MiB f32 surface |
| Neural activation capture | Implemented (dense CPU only) | HF forward-pass contrast directions, labeled **correlational** |
| Cross-space/model transfer | **Not supported** | Requires explicit map and fresh qualification |
| Multi-direction composition | **Not supported** | Ordering/orthogonality not implicit |
| Recurrent/MoE/looped/SSM execution | **Unqualified** | Fails closed or not implemented in v0; recipes are references, not executors |
| Conv1D (gpt2/gpt_neo/gptj) | **Fails closed** | No transpose heuristic; rejected at intake |
| Quantized/fused codecs | **Not supported** | No codec inference from extensions; S4 skipped |
| Serving readiness | **Not implemented** | No serving engine or readiness claim; S5 skipped |
| Behavioral efficacy/safety | **Not measured** | S6/S7 skipped; mock gates are row-presence wiring checks; benchmark corpora are tools, not results |
| Public release/publishing | **Not implemented** | No publication or approval API; every real run `not_eligible` |
| LLM provider integration | **Not implemented** | Planner is deterministic and offline |
| Confidentiality/access control | **Not provided** | Bank visibility is a policy label only |

An adapter may be added without weakening these defaults. Its support must be stated independently, with passing conformance and real checkpoint/serving/behavior evidence appropriate to its claim. Structural validation alone never justifies a real-model support claim. See [AGENTS.md](../AGENTS.md) for the current executor envelope and [benchmarks/README.md](../benchmarks/README.md) for the measurement layer.


## Standalone benchmark v2 boundary

`benchmarks/score.py` validates local v2 schemas and exact response/judgment binding.
It separates harmful engagement from benign false refusal, reports unresolved
coverage, and provides no smoke/cyber pass rates. Collection completion verifies
file integrity, not checkpoint identity or model quality. Requested settings are
not observed settings. V1 remains explicitly legacy. See
[the operator procedure](../benchmarks/protocol/RUNS.md).

No benchmark receipt is imported as a pipeline gate: real S6/S7 remain skipped,
release remains `not_eligible`. Live qualification still requires authorized
checkpoint/serving identity and readiness, behavior/capability evaluation, the
operator release-only holdout, and an authenticated release decision. Offline
unit tests—including tiny synthetic local HF fixtures—prove tooling only.
