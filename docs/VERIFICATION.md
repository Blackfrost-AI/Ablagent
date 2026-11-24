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
