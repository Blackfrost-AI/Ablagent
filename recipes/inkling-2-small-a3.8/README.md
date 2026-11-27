# Inkling-2-Small DWM alpha 3.8

**Blackfrost AI** recipe pack for the Inkling-2-Small Domain Weight Modification
at **alpha 3.8** — hard32 capture → contiguous mid-stack band **L20–37** →
rank-one OUT-space projection removal on the BF16 attention writer `wo_ud` →
MTP merged byte-exact → served on 4× B300 under a digest-pinned vLLM image.

This directory is an immutable snapshot of the duplication kit published with
[`Blackfrost-Research/Inkling-2-Small-Derisked`](https://huggingface.co/Blackfrost-Research/Inkling-2-Small-Derisked)
@ `c75bc4f1` (see [`PROVENANCE.json`](PROVENANCE.json)). It contains the method
and its evidence — **not model weights**. Derived checkpoints remain under the
upstream [`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small)
terms (Apache-2.0 upstream; see [`MODEL-NOTICE.md`](MODEL-NOTICE.md)).

## What is locked

| Item | Locked value |
|---|---|
| Clean source | `thinkingmachines/Inkling-2-Small@885d78bfca61a7e9a498e33cfa7ad35850533b8c` |
| Capture corpus | hard32 universal bank — 32 harmful + 32 paired safe mirrors (`DWM_RECIPE/input/v1_corpus_hard32.jsonl`) |
| Capture | vLLM native loader, 64 rows × 48 layers × 4608 hidden, thinking ON, TP4 |
| Band | attention_writer family, **L20–37** (18 layers), permutation p=0.0005, split-half cosine 0.912–0.975 |
| Edit | `W := W − α·W @ (q qᵀ)` on transposed `wo_ud` [4608, 6144], **α = 3.8**, one pass, BF16, no whitening, no requant |
| MTP | `mtp.safetensors` copied byte-exact (sha256 `3b265640…`); speculator not edited |
| Serve | `vllm/vllm-openai@sha256:e72944ef…`, TP4, 262144 ctx, dspark spec (7 tokens), inkling parsers |

Full constants: [`recipe.lock.json`](recipe.lock.json). Exact procedure:
[`DWM_RECIPE/RUNBOOK.md`](DWM_RECIPE/RUNBOOK.md). Deployment:
[`deployment-kit/AGENT_INSTRUCTIONS.md`](deployment-kit/AGENT_INSTRUCTIONS.md).

**Uniqueness rule:** never copy this alpha, band, or surface onto another
checkpoint — not even Inkling-2-Mini, whose loop-pass architecture differs
(this model has 48 physical layers, no loop passes, no shared experts).

## Recorded results (α3.8 final only — run-record claims, not measurements made here)

- **cyber10**: 10/10 answered, 0 refusals, 0 errors, all `finish_reason=stop`;
  10/10 graded FULL_COMPLIANCE; 1/10 rows carry a brief practical framing line
  (not a moral hedge). Temp 0.0, 32K budget, thinking ON.
- **443-row false-refusal corpus**: 443/443 generated, 0 errors, 0 truncations
  (temp 1.0, top_p 1.0, 32K budget, thinking ON). All 77 regex-flagged rows plus
  a seed-443 sample of 20 unflagged rows manually adjudicated: **no refusals
  observed in the adjudicated set**; estimated class shares 34.9% full /
  65.1% partial-benign / 0.0% refusal (sample-based estimates, not exhaustive
  labels). 0 observed over-refusals on safe rows.
- **Capability gate**: 32 safe hard32-mirror rows, NLL base 6.0069 → candidate
  5.9794 (Δ = −0.0275; floor held). The static writer-projection reduction is
  confounded by residual propagation — behavioral evals are the definitive test.

α3.0 → α3.3 lineage exists in the run record but is **not** release evidence;
only α3.8 results are reported. The operator-only release red-line holdout is
not part of this package and was not run by this repository.

Evidence: `DWM_RECIPE/evidence/` (S0 hashlock, S2 battery, S3 bake receipt +
gate, S5 serve receipt, cyber10, full 443 rows/ledgers/adjudications).

## Known limits

- **Raw capture is fetch-only.** `raw-capture.npz` (89,421,956 bytes, sha256
  `63a559079f9f39da84da731561c63d879c9c69de36f5ff8121a253bb0468cd88`) is not
  vendored (git size convention); fetch per
  [`assets/raw-capture.asset.json`](assets/raw-capture.asset.json). The S3 bake
  consumes the vendored `directions.npz`, so execution does not need it; it is
  required only for capture-stage audit.
- **7 files were endpoint-redacted** relative to the HF release (operator pod
  endpoints → `REDACTED` markers). Every redaction is documented with original
  and vendored SHA-256 in [`PROVENANCE.json`](PROVENANCE.json); the vendored HF
  manifests still record the original hashes and the verifier resolves them
  through that table. All other kit files are byte-identical to the pinned
  revision.
- **Weights not included.** Fetch the candidate (or rebuild it from the pinned
  upstream via the runbook) from the HF repo under upstream terms.
- The shipped `ablagent --executor real` runtime does **not** execute this
  recipe (MoE + mixed quantization is outside its CPU dense envelope). Study it
  as a method reference; reproduce on 4× B300-class hardware via its runbook.

## Verify

```bash
python3 scripts/verify_package.py            # package integrity, fail-closed
python3 scripts/verify_package.py --raw-capture <path/to/raw-capture.npz>   # optional exact audit
```

The verifier checks the lock constants, the redaction table, every vendored
file against the HF-published manifests, the pinned corpus/direction digests
(cross-checked against `benchmarks/corpora/` when present), and scans for
private material. It never touches the network.

## Layout

| Path | Role |
|---|---|
| `DWM_RECIPE/RUNBOOK.md` | exact executed procedure, hashes, gates, limits |
| `DWM_RECIPE/scripts/` | capture, battery, bake, gate, serve, eval, adjudication scripts |
| `DWM_RECIPE/input/` | hard32 capture corpus (hash-pinned) |
| `DWM_RECIPE/evidence/` | S0–S5 receipts, cyber10 + 443 evidence and ledgers |
| `deployment-kit/` | pinned-image serve instructions, `serve.sh`, serve receipt |
| `manifests/` | HF-published candidate + kit SHA-256 manifests (byte-identical) |
| `assets/raw-capture.asset.json` | fetch-only descriptor for the 89 MB raw capture |
| `recipe.lock.json`, `SHA256SUMS` | method lock + package inventory |
| `PROVENANCE.json` | source revision, redaction table, vendoring decisions |

License: Apache-2.0 for Blackfrost-authored content ([LICENSE](LICENSE),
[NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Upstream
model terms: [MODEL-NOTICE.md](MODEL-NOTICE.md).