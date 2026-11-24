# Initial repository verification

Verified locally on 2026-11-24 with Python 3.14.3 on macOS arm64.

| Check | Result |
|---|---|
| Tests | 90 passed |
| Ruff lint | Passed |
| Ruff format check | Passed |
| Exported schemas | Read-only freshness check passed |
| Dependency audit | No known vulnerabilities; local `ablagent` package skipped because it is not on PyPI |
| Secret scan | 91 files scanned, zero findings |
| DWMIP fixtures | 13 of 13 passed from a clean wheel installation |
| Wheel workflow | Demo, one-proposal plan, compile, verification, and conformance passed |
| Mock policy paths | Incomplete evidence blocked; bound inert attestation verified; private mock decision eligible with authorization not granted |
| Independent reviews | Multiple reviews completed; path-based verification, red-line observation precedence, and plan-binding corrections added with passing regressions |

Eligibility requests identify an evidence file within an explicit `ArtifactStore`; `evaluate_eligibility` verifies that file and its selected bank before evaluating gates. Verification receipts are returned for inspection but are not accepted in serialized requests as proof that verification ran.

The clean wheel was installed into an isolated environment and executed against the checked-in conformance fixtures. Generated artifacts remained outside the source package. These results do not validate a real checkpoint, a particular GPU/serving stack, a capability claim, or a publishing decision.

The release-only capability gate is intentionally unimplemented pending a separately selected external benchmark. No private holdout or release policy is part of this repository.

## GitHub Actions limitation

The initial remote workflow run `45344338868` was blocked before runner allocation by the account's failed-payment or spending-limit status. No CI test step ran. The local checks above passed; rerun GitHub Actions after the repository owner's billing/spending issue is resolved.

Reproduce the principal checks:

```sh
python -m pip install -e '.[dev]'
ruff check src tests scripts
ruff format --check src tests scripts
python scripts/export_schemas.py --check
pytest --cov=ablagent --cov-report=term-missing
python -m build
pip-audit
ablagent demo --run-dir .artifacts/verification
ablagent plan bound.direction.json --run-dir .artifacts/verification \
  --alpha 1 --output .artifacts/verification/one.plan.json
ablagent compile bound.direction.json --run-dir .artifacts/verification \
  --plan .artifacts/verification/one.plan.json --artifact-id candidate
ablagent verify .artifacts/verification/candidate.manifest.json
ablagent conformance --fixtures spec/dwmip/v0/conformance \
  --run-dir .artifacts/conformance
```

## Real executor verification (2026-11-24)

Verified locally with Python 3.14.3 on macOS arm64, torch 2.15 / transformers
5.24 (the `[real]` extra), CPU only.

| Check | Result |
|---|---|
| Full test suite | 163 passed (includes 15 real-adapter tests building tiny local bf16 checkpoints; no network) |
| Ruff lint + format (new real-path files) | Passed |
| Exported schemas freshness | Passed after additive re-export (`hf-dense-linear-v0` adapter id; `weight_rank_one` operation) |
| Vendored example provenance | Re-verified, hashes intact |
| Tiny local llama end-to-end | S0–S3 + S8/S9 completed; S4/S5/S6/S7/SR/S10 recorded skipped with reasons; all release states `not_eligible` |
| Live HF smoke: HuggingFaceTB/SmolLM2-135M | Pinned commit downloaded + hashed; 211-writer surface; capture → bake → export completed |
| Live export independence | `AutoModelForCausalLM.from_pretrained(export)` loads and generates coherent text; exactly one tensor differs from the pinned snapshot (`model.layers.29.mlp.down_proj.weight`), bf16 restored |
| Manifest re-verification | `verify_compiled` passed independently; float32 candidate deviation 1.2e-07; exported bf16 recast deviation 0.0114 recorded in the export receipt (within bf16 rounding) |
| Tied weights | Alias excluded when present in shards; ties preserved through export reload |
| Envelope + failure paths | Oversize surface (pre-download shard-size preflight + local cap), non-safetensors dir, non-finite tensors, unknown writer, alpha out of range, Conv1D families (gpt2/gpt_neo/gptj) all fail closed (tested) |
| Perturbation regression | A deliberately perturbed persisted f32 candidate fails `verify_compiled` (tested) |
| Export numeric receipt | `ABLAGENT-EXPORT-RECEIPT.json` records `edited_tensor_recast_max_deviation` (measured, e.g. 0.000122 on the tiny bf16 llama) |
| Capture evidence completeness | Receipt records chat-template usage, prompt token cap, and pairs dropped for prefill truncation; secret-scan evidence lists unscanned files with reasons |

Audit-semantics correction made during this work: `compile` and
`verify_compiled` now share one persisted audit contract — expected values are
the exact float64 edit, digests compare the deterministic dtype recast, and
`maximum_deviation` records the real dtype rounding error. Previously the two
paths disagreed for non-float32 checkpoints.

These results validate the v0 real-adapter envelope only: CPU, dense
safetensors, one rank-one site. They do not validate behavior change, safety,
capability preservation, quantization, serving, or any release decision —
those stages are recorded as skipped (untested) in every real run.
