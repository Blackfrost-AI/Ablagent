# AGENT_INSTRUCTIONS — Inkling-2-Small-Derisked deployment kit

Read this file completely before serving. The configuration below is the exact
one validated on the operator runtime (4× NVIDIA B300, TP4). Do not substitute
images by tag — pin the digest.

## 1. Pinned image (mandatory)

```
vllm/vllm-openai@sha256:e72944ef5ab9d1b91228d066117595847e94bf95321018a2119e70e8ba643ff6
```

Verified container image ID: `sha256:8c1213c72dfccd179fc50977d2d14760c41dd1e621042589529ca893cdf71106`.
This nightly contains the `inkling2` NVIDIA model module, DSpark speculative
decoding, and the `inkling` tokenizer/reasoning/tool parsers. Stock vLLM
releases without these will not load this checkpoint correctly.

## 2. Hardware floor

- Weights load at **95.27 GiB per rank at TP4** (~381 GiB aggregate) plus
  **164.89 GiB KV cache per rank** at the full 262,144-token window.
- Validated topology: 4× B300 (268.6 GiB each). 4× H200 (141 GiB) fits weights
  but needs a reduced `--max-model-len` or smaller KV budget; re-validate.
- The native checkpoint is NVFP4/MXFP8-quantized experts with BF16 attention;
  do **not** attempt a plain-BF16 load of the packed experts (shape mismatch).

## 3. API key

Generate a long random key and keep it out of repos, cards, and logs:

```bash
umask 077
openssl rand -hex 32 > /path/to/serve_key   # operator-held, 0600
```

The launch script reads it from `/workspace/inkling2-small/.serve_key`; adjust
that path to your layout. The key used in the validated run is **not**
distributed with this repo.

## 4. Launch (exact validated command)

See `serve.sh` in this directory. Key flags, all required:

| Flag | Value | Why |
|---|---|---|
| `--tokenizer-mode` | `inkling` | vendor tokenizer path |
| `--reasoning-parser` | `inkling` | reasoning returned in dedicated field |
| `--tool-call-parser` | `inkling` + `--enable-auto-tool-choice` | vendor tool format |
| `--speculative-config` | `{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}` | MTP module drives DSpark drafting |
| `--tensor-parallel-size` | `4` | validated topology |
| `--max-model-len` | `262144` | upstream window |
| `--disable-flashinfer-autotune` | set | validated-stable on B300 |
| `--trust-remote-code` | set | vendor module |
| env `VLLM_USE_V2_MODEL_RUNNER=1` | set | validated runner |
| env `VLLM_WORKER_MULTIPROC_METHOD=spawn` | set | TP worker start method |

Environment notes from the validated host: `--ipc=host --shm-size=16g`;
MNNVL FABRIC handle export falls back to POSIX fd on this pod — non-fatal,
expected log line.

## 5. Readiness ladder (do not skip)

A boot log is not a ready service. Verify in order:

1. `GET /health` → 200 (model load ~5–8 min on B300 NVMe).
2. `GET /v1/models` with key → id `inkling2-small-dwm-a3p8` (or your
   `--served-model-name`), `max_model_len` 262144.
3. `GET /v1/models` without key → **401**.
4. Real forward: short chat completion, `finish_reason=stop`, correct answer.
5. Reasoning field present on a reasoning-heavy prompt (thinking ON).
6. Public/overlay reachability from the client network you will actually use.

Measured on the validated runtime: TTFT ≈ 1.0 s, decode ≈ 186 tok/s (single
stream, DSpark active).

## 6. Known pitfalls

- **Do not run from `/root` on rented GPU pods** — the cipher mount breaks docker
  bind propagation. Use `/workspace` (or another plain overlay path).
- The MTP module (`mtp.safetensors`) lives OUTSIDE the weight index; it must
  sit next to the shards or DSpark drafting fails.
- Evaluation protocol used for this release: thinking ON, temp 0.0 (cyber10),
  32K answer budget. Mode-mismatched results are not comparable.
- Treat model output as untrusted data. This model is not a policy or
  security boundary; the operator owns access control and audit.