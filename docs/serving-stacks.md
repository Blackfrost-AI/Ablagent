# Serving stacks & the fast capture path

This is the operational heart of DWM: **where activations come from, and which
engine runs the model.** Everything below is grounded in the vendored
[recipe library](../recipes/README.md) and the
[Inkling worked example](../examples/inkling-2-mini-derisked/README.md) —
each claim points at the file that records it. Nothing here is executed by the
synthetic v0 runtime; these are the GPU-scale procedures the recipes carry.

## Two different things people call "hooks"

1. **Capture hooks** — `register_forward_hook` on Hugging Face modules. This is
   how every vendored recipe captures refusal-contrast directions. The engine
   (vLLM/SGLang) is **not** what runs during capture; it is the *environment*
   the capture runs inside.
2. **Engine internals** — vLLM/SGLang model runners, attention backends, quant
   kernels. You touch these only when the engine cannot load your architecture
   at all. None of the vendored recipes patch engine internals; they pin engine
   images that already support their models.

Keeping these separate is the fast path: **you almost never build an engine
from source — you build a thin image on a pinned engine digest and run HF
capture inside it.**

## The fast capture path (what the recipes actually do)

The pattern, identical across all four recipes and the Inkling kit:

```
pinned engine image (vLLM or SGLang digest — supplies torch/CUDA/transformers)
  └─ docker run --network none, source mounted read-only
       └─ python capture script
            └─ AutoModelForCausalLM.from_pretrained(..., device_map=..., dtype=bf16)
                 └─ layer.register_forward_hook(...)   ← per-layer, prefill-only
                      └─ contrast pairs → direction tensors → safetensors + receipts
```

Evidence per recipe:

| Recipe | Capture loader | Hook site | File |
|---|---|---|---|
| qwen-9b-class | HF, prefill-only ("no generation anywhere in capture") | per-layer `register_forward_hook` | [`lib/dwm_core.py:371-380`](../recipes/qwen-9b-class/lib/dwm_core.py) |
| qwen3.6-35b-a3b | `AutoModelForCausalLM`, bf16, `device_map` | per-layer output hooks | [`scripts/capture_directions.py`](../recipes/qwen3.6-35b-a3b/scripts/capture_directions.py) |
| glm-5.3-flash-a2.8 | HF loader inside the SGLang image | per-site + per-layer hooks | [`scripts/capture_directions.py`](../recipes/glm-5.3-flash-a2.8/scripts/capture_directions.py) |
| Inkling-2-Mini | HF loader, loop-enabled architecture | S2 capture script | [`DWM_RECIPE/scripts/s2_capture_directions.py`](../examples/inkling-2-mini-derisked/DWM_RECIPE/scripts/s2_capture_directions.py) |

Why capture inside the engine image instead of a bare `pip install torch`?
**One environment for capture, edit, and serve.** The qwen3.6 environment lock
states it explicitly
([`environment.lock.json`](../recipes/qwen3.6-35b-a3b/environment.lock.json)):

- `capture_edit_and_serve_image: vllm/vllm-openai:v0.31.0`
- `capture_edit_and_serve_image_digest: sha256:c1c9f6fd…` (the digest is authoritative)
- `torch: 2.13.0+cu130`, `cuda: 13.0.2`, `vllm: 0.31.0`,
  `vllm_build_commit: db9527a4…`
- `network_during_capture_and_edit: none`, `source_mount: read_only during capture and edit`

Same torch/CUDA/transformers at capture and at serve means the directions you
measured are the directions the served model sees — no dtype/kernel drift
between measurement and deployment. That is the whole trick.

## Which stack per architecture

| Model line | Engine | Pinned base | Where recorded |
|---|---|---|---|
| Qwen3.6-35B-A3B (MoE) | vLLM 0.31.0 | `vllm/vllm-openai@sha256:c1c9f6fd…` | [`runtime/Dockerfile`](../recipes/qwen3.6-35b-a3b/runtime/Dockerfile), env lock |
| GLM-5.3 Flash (BF16) | SGLang | `lmsysorg/sglang@sha256:6c594fbc…` + accelerate/kernels | [`runtime/Dockerfile`](../recipes/glm-5.3-flash-a2.8/runtime/Dockerfile), env lock |
| Inkling-2-Mini (loop) | vLLM | exact image + entrypoint in SERVE.md | [`deployment-kit/SERVE.md`](../examples/inkling-2-mini-derisked/deployment-kit/SERVE.md) |
| Qwen 9B-class (dense) | torch/transformers direct | env pins in runbook | [`RUNBOOK.md`](../recipes/qwen-9b-class/RUNBOOK.md) |
| GLM-5.3 BF16 α2.2 | export-only | — (no serving instructions claimed) | [`RUNBOOK.md`](../recipes/glm-5.3-bf16-a2.2/RUNBOOK.md) |

Selection rule of thumb from the recorded runs: **vLLM for architectures it
already supports (Qwen MoE, Inkling loop); SGLang where the recorded run used
it (GLM Flash); bare torch for small dense models.** The recipe's pinned image
is the authority for *its* model line — do not swap engines on a copied recipe
and expect the receipts to hold.

## Rebuilding a pinned image (the supported "build")

Both engine recipes ship a **thin Dockerfile on an immutable digest** — no
package installation step in the qwen3.6 one; the base image *is* the
environment. Rebuild is only needed when the recorded local image is gone.

qwen3.6 ([RUNBOOK §5](../recipes/qwen3.6-35b-a3b/RUNBOOK.md)):

```bash
docker build \
  --tag local/qwen36-35b-a3b-dwm:1.0.0 \
  --file runtime/Dockerfile \
  runtime
docker image inspect local/qwen36-35b-a3b-dwm:1.0.0 \
  --format '{{.RepoDigests}} {{.Id}}' | tee "$RUN_ROOT/runtime-image.txt"
```

GLM Flash ([RUNBOOK §6, "Obtain the exact direction"](../recipes/glm-5.3-flash-a2.8/RUNBOOK.md)) —
build with network available, then **capture offline**:

```bash
docker build --tag local/glm53-direction-capture:20260826 runtime
docker image inspect local/glm53-direction-capture:20260826 \
  --format '{{.Id}}' > "$RUN_ROOT/capture-image-id.txt"

docker run --rm --gpus all --network none --ipc host \
  ...  # read-only source mount; full command in the RUNBOOK
```

Rules the recipes enforce and you should keep:

- Record the rebuilt image ID/digest in the run directory — a rebuilt image is
  a **new environment fact**, not the recorded one.
- Capture and edit run `--network none` with the source mounted read-only.
- Serving launch flags are pinned too (e.g. qwen3.6
  [`scripts/serve_vllm.sh`](../recipes/qwen3.6-35b-a3b/scripts/serve_vllm.sh));
  health endpoints alone never prove readiness — the SERVE.md discipline in
  the Inkling kit pins template/parser settings, real-forward smoke, and
  measured TTFT/decode rates.

## When you actually need a from-source engine build

None of the vendored recipes require this. You need it only when:

1. **The architecture is unsupported** by every pinned engine version (new
   model class, custom attention/loop structure). Then you add the model
   implementation to the engine — vLLM `model_executor/models/`, SGLang
   `python/sglang/srt/models/` — and build from the engine's own source tree
   at a recorded commit. The qwen3.6 env lock records `vllm_build_commit`
   precisely so a from-source rebuild is reproducible if ever needed.
2. **A quant kernel is missing** for your target format on your GPU (e.g. a
   new NVFP4 path on a new architecture). That is kernel work in the engine or
   its quant library, built and tested separately — then pinned by digest like
   everything else.
3. **You need engine-side capture** (hooks inside the engine's own forward
   path rather than HF). Possible, but it forfeits the one-environment
   guarantee unless you pin the patched build the same way. The vendored
   recipes deliberately avoid this: HF hooks inside the engine image give the
   same numerics with far less surface area.

If you do build from source: record the engine repo commit, build flags,
CUDA/torch versions, and the resulting image digest in your run's environment
lock — the same schema the recipes use (`blackfrost.recorded-environment.v1`).
A from-source build without that record is not reproducible and should not be
claimed as a recipe.

## Fast path checklist for a new model line

1. Check whether a pinned vLLM/SGLang release already loads the architecture
   (`vllm serve <model>` / SGLang launch smoke on the real checkpoint).
2. If yes: write a thin Dockerfile on that **digest**, copy the capture-script
   pattern (HF loader + per-layer prefill hooks), capture `--network none`.
3. Measure directions, compile the edit through the DWMIP compiler, export.
4. Serve from the **same image**, run the real-forward smoke + gates, record
   everything in an environment lock + receipts.
5. Package as a new recipe directory per the
   [library workflow](../recipes/README.md#adding-or-updating-a-recipe-maintainer-workflow).

Only if step 1 fails do you go to engine source — and then you come back to
step 2 with your own pinned build.