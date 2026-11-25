# Agent runbook: Qwen3.6-35B-A3B stock BF16 DWM

This runbook reproduces the alpha-1.0 ×3 method. Read `AGENTS.md`,
`recipe.lock.json`, `environment.lock.json`, and all files under `evidence/`
before executing it. Stop on every mismatch.

## 1. Claims boundary

The runbook can reproduce an offline candidate, verify the weight surface,
repeat the recorded small distribution comparison, and start the pinned vLLM
profile. One historical candidate passed those checks and received operator
HITL approval.

A fresh output remains a candidate. The recorded run did not execute S1, S6,
S7, or the release-only red-line gate. Complete those gates under the current
Blackfrost pipeline before a release claim.

## 2. Resources

- Linux with Docker and NVIDIA Container Toolkit.
- One CUDA GPU able to hold the ~35B MoE BF16 checkpoint. The recorded GPU was
  an NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Roughly 160 GB of free storage for the 71.9 GB source, output, and receipts.
- Enough host memory to inventory and rewrite individual safetensor shards.
- Hugging Face access to the pinned public source revision.

The captured environment is locked by the image digest in
`environment.lock.json`. Method equivalence requires the same source and
semantic locks; byte-identical output is not promised across hardware.

## 3. Set isolated paths

Run from the package root. Choose an empty output parent and a new run folder.

```bash
export PACKAGE_ROOT="$PWD"
export SOURCE_ROOT="/data/Qwen3.6-35B-A3B-995ad96"
export OUTPUT_PARENT="/data/dwm-output"
export OUTPUT_NAME="Qwen3.6-35B-A3B-DWM-a10x3"
export OUTPUT_ROOT="$OUTPUT_PARENT/$OUTPUT_NAME"
export RUN_ROOT="/data/dwm-runs/qwen36-a10x3-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTPUT_PARENT" "$RUN_ROOT"
test ! -e "$OUTPUT_ROOT"
```

Never reuse a partial output directory. Keep `SOURCE_ROOT` read-only during
capture and editing.

## 4. Verify the package

```bash
python3 scripts/verify_package.py
```

Expected high-level values:

| Check | Required value |
|---|---:|
| Package version | `1.0.0` |
| Pair rows | 64, split 32 harmful and 32 safe |
| Initial direction | 40 tensors of width 2048 |
| Alpha | 1.0 |
| Passes | 3 |
| Layers | 2–39 |
| Targets | 38 |
| Protected MTP tensors | 19 |
| Ablagent stages | 8, read-only |

## 5. Build the pinned runtime

The Dockerfile has no package installation step. Its immutable base image is
the exact vLLM image used for capture, edit, measurement, and serving.

```bash
docker build \
  --tag local/qwen36-35b-a3b-dwm:1.0.0 \
  --file runtime/Dockerfile \
  runtime
docker image inspect local/qwen36-35b-a3b-dwm:1.0.0 \
  --format '{{.RepoDigests}} {{.Id}}' | tee "$RUN_ROOT/runtime-image.txt"
```

## 6. Download and verify the clean source

Fetch the commit, never a moving branch:

```bash
hf download Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --local-dir "$SOURCE_ROOT"
```

Verify all 40 source files against `assets/source-lock.json`:

```bash
python3 scripts/verify_package.py --source "$SOURCE_ROOT" \
  | tee "$RUN_ROOT/source-verification.json"
chmod -R a-w "$SOURCE_ROOT"
```

Required source facts include 26 weight shards, 1,045 indexed tensors, 19 MTP
tensors, and canonical checkpoint digest
`sha256:88760187ccb2560500b1d51f0e75610658c956be010b4477bc11bb8265c7413b`.

## 7. Optional diagnostic recapture

The exact initial direction and baseline activations are already bundled. A
fresh diagnostic recapture can test the environment before a build:

```bash
mkdir -p "$RUN_ROOT/recapture"
docker run --rm \
  --gpus all \
  --network none \
  --ipc host \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$SOURCE_ROOT:/source:ro" \
  -v "$PACKAGE_ROOT:/package:ro" \
  -v "$RUN_ROOT/recapture:/recapture" \
  local/qwen36-35b-a3b-dwm:1.0.0 \
  python3 /package/scripts/capture_directions.py \
    --model /source \
    --out /recapture
```

The recaptured tensor may differ at the byte level across kernels. Inspect its
semantic report. The locked executor performs its own fresh pass-1 capture and
requires L39 cosine ≥0.99 to the exact bundled S2 direction before accepting
the build.

## 8. Build the three-pass candidate

```bash
mkdir -p "$RUN_ROOT/build"
docker run --rm \
  --gpus all \
  --network none \
  --ipc host \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$SOURCE_ROOT:/source:ro" \
  -v "$OUTPUT_PARENT:/outputs" \
  -v "$PACKAGE_ROOT:/package:ro" \
  -v "$RUN_ROOT:/run" \
  local/qwen36-35b-a3b-dwm:1.0.0 \
  python3 /package/scripts/run_locked.py \
    --source /source \
    --output "/outputs/$OUTPUT_NAME" \
    --evidence /run/build \
  2>&1 | tee "$RUN_ROOT/build.log"
```

There is no `--alpha`, `--passes`, `--skip-early`, `--bank`, `--directions`,
`--revision`, or target-surface option. The executor captures the current
model immediately before each of three passes and applies the layer-matched
direction to all 38 attention output weights. It also rehashes all 40 source
files itself before loading the model, so skipping the earlier source-check
command does not weaken the executable lock.

Required terminal result: `BUILD_DONE`. The structural report must show:

- 38 changed targets and no missed target;
- zero changed non-targets;
- zero non-finite target tensors;
- all 19 MTP tensors exact; and
- pass 1 L39 cosine at least 0.99 to the bundled initial direction.

## 9. Verify the output independently

The independent verifier rereads both checkpoints. It derives the target set
from the pinned name pattern and does not trust the executor's changed-tensor
list.

```bash
docker run --rm \
  --network none \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$SOURCE_ROOT:/source:ro" \
  -v "$OUTPUT_ROOT:/output:ro" \
  -v "$PACKAGE_ROOT:/package:ro" \
  -v "$RUN_ROOT:/run" \
  local/qwen36-35b-a3b-dwm:1.0.0 \
  python3 /package/scripts/verify_output.py \
    --source /source \
    --output /output \
    --build-record /run/build/build-record.json \
  | tee "$RUN_ROOT/output-verification.json"
```

Expected: `passed: true`, 38 changed targets, no changed non-targets, and 19
exact MTP tensors.

## 10. Repeat the recorded distribution and geometry gate

This is the original eight-prompt comparison gate. It is a sensitive smoke
check and does not replace a full capability suite.

```bash
mkdir -p "$RUN_ROOT/measurement"
docker run --rm \
  --gpus all \
  --network none \
  --ipc host \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$SOURCE_ROOT:/source:ro" \
  -v "$OUTPUT_ROOT:/output:ro" \
  -v "$PACKAGE_ROOT:/package:ro" \
  -v "$RUN_ROOT:/run" \
  local/qwen36-35b-a3b-dwm:1.0.0 \
  python3 /package/scripts/measure_compare.py \
    --base /source \
    --edited /output \
    --structural /run/build/structural-metrics.json \
    --out /run/measurement \
  2>&1 | tee "$RUN_ROOT/measurement.log"
```

The automatic gate requires relative NLL delta ≤0.10, mean JS ≤0.02, and
top-1 agreement ≥0.85. The recorded candidate measured −0.02444808,
0.00520880, and 0.95569620, respectively.

## 11. Inventory the new candidate

```bash
python3 scripts/inventory_checkpoint.py \
  --checkpoint "$OUTPUT_ROOT" \
  --artifact-id qwen36-35b-a3b-dwm-a10x3-reproduction \
  --model-id Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output "$RUN_ROOT/candidate-inventory.json"
```

Do not expect a reproduction to inherit the recorded candidate digest. Bind
all later measurements to the new inventory digest.

## 12. Serve with the recorded vLLM profile

The helper binds loopback by default:

```bash
scripts/serve_vllm.sh "$OUTPUT_ROOT"
docker logs -f qwen36-35b-a3b-dwm-a10x3
```

For a trusted private interface, pass its bind address and port explicitly:

```bash
scripts/serve_vllm.sh "$OUTPUT_ROOT" 10.0.0.24 8800
```

Readiness and chat smoke:

```bash
curl --fail --silent http://127.0.0.1:8800/v1/models | python3 -m json.tool
curl --fail --silent http://127.0.0.1:8800/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.6-35B-A3B","messages":[{"role":"user","content":"Reply with exactly READY."}],"temperature":0,"max_tokens":1024}' \
  | tee "$RUN_ROOT/chat-smoke.json" | python3 -m json.tool
```

Verify the model ID, HTTP success, a separate non-empty reasoning field,
`finish_reason=stop`, and final answer `READY`.

## 13. Complete qualification

Before promotion or delivery, bind every result to the new candidate inventory
and complete the current pipeline's remaining gates, including:

1. the registered capability profile;
2. the current refusal measurement corpus;
3. the separate release-only red-line holdout;
4. service stability and complete receipts; and
5. final operator acceptance under the current control-plane policy.

The historical HITL receipt applies only to the recorded candidate identity.
It does not automatically approve a reproduction.

