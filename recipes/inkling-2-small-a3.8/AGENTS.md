# Agent contract: Inkling-2-Small DWM alpha 3.8

Read these files completely and in this order before proposing a command:

1. `recipe.lock.json`
2. `PROVENANCE.json`
3. `DWM_RECIPE/RUNBOOK.md`
4. `assets/raw-capture.asset.json`
5. `deployment-kit/AGENT_INSTRUCTIONS.md`

The method parameters are immutable. Alpha is `3.8`, rank is one, the operation
is one pass, layers 20–37 are edited on the BF16 attention writer `wo_ud` in
OUT space, and `mtp.safetensors` is copied byte-exact from the clean upstream
source. Never substitute a source revision, direction set, capture corpus,
capture loader, target surface, layer band, arithmetic rule, or pass count and
call the result this method. Never carry this alpha or band onto a different
checkpoint — including Inkling-2-Mini, whose loop-pass architecture differs.

Run `python3 scripts/verify_package.py` before execution. The vendored kit
matches the pinned HF revision except for 7 documented endpoint redactions
(`PROVENANCE.json`); the verifier resolves them and fails closed on any other
drift. The 89 MB raw capture is fetch-only and needed for capture-stage audit,
not for the bake — the bake consumes the vendored, hash-pinned `directions.npz`.
Stop on any mismatch.

Capture uses the vLLM native loader inside the digest-pinned image
(`vllm/vllm-openai@sha256:e72944ef…`). The host transformers path cannot load
the packed modelopt experts; the superseded HF-path scripts under
`DWM_RECIPE/scripts/superseded-hf-path/` are kept as labeled negative evidence,
not as an alternative execution route.

Evaluation discipline: only α3.8 results are release results. The α3.0/α3.3
lineage is process provenance and must never be reported as α3.8 evidence.
S2 directions are correlational contrasts, not causal claims. The 443 class
shares are sample-based estimates (77 flagged + 20 sampled rows adjudicated),
not exhaustive labels. The operator-only release red-line holdout is absent
from this package; a reproduction is unqualified until the operator's current
release gates pass.

This package is a method reference. The shipped Ablagent `--executor real`
runtime does not execute it (MoE + mixed NVFP4/MXFP8 quantization is outside its
CPU dense envelope). Reproduction requires 4× B300-class GPUs and the pinned
container; no command here authorizes a serve, a publish, or a release.