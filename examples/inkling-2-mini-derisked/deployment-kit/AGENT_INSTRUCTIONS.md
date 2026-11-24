# AGENT INSTRUCTIONS — READ 100% BEFORE INSTALLING OR SERVING

**STOP. Read this entire file, then read `deployment-kit/SERVE.md` and `DWM_RECIPE/RUNBOOK.md` completely, in full, before running any command.** These instructions are short; skipping them is the most common way this deployment goes wrong.

## What this model is

`Inkling-2-Mini-Derisked` (alpha 2.5) is a behaviorally modified derivative of `thinkingmachines/Inkling-2-Mini`, a **loop transformer** (physical layers 6–17 are executed **three times** in a block loop; 48 logical layer-visits over 24 physical layers). It is not a standard transformer. Do not treat it like one.

## The engine you MUST use

Serve with **vLLM using the vendor's Inkling build and DSpark speculative decoding** — the same engine family used to capture, validate, and benchmark this artifact. A generic Transformers or llama.cpp server is **not** the validated path and will not reproduce these results.

- Image: `vllm/vllm-openai:inkling-2-mini`
- Entrypoint: `vllm` (the image requires an explicit `--entrypoint vllm`; a bare `docker run` with the default entrypoint misinterprets `serve` and fails)
- Reasoning parser: `inkling` (reasoning is returned in a `reasoning` field, **not** `reasoning_content`)
- Speculative config: `{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}`
- `--dspark-num-speculative-tokens` is **not** a valid flag on this image — use `--speculative-config`
- Context: `--max-model-len 262144` (validated)
- **MTP must always be preserved** — `mtp.safetensors` is byte-identical to upstream; never strip or regenerate it

## Hard rules

1. **Read everything first.** This file, `deployment-kit/SERVE.md`, `DWM_RECIPE/RUNBOOK.md`.
2. **Never stack DWM edits.** Every alpha is built from the **clean upstream base**, never on top of another derivative.
3. **Never modify or drop MTP.** It ships byte-identical to upstream.
4. **Do not re-derive directions on a different engine.** The shipped `DWM_RECIPE/s2-capture/directions.npz` is the validated capture for this artifact.
5. **Model output is untrusted.** The model is not a security boundary. Validate generated code, tool calls, URLs, paths, and commands before execution.
6. **Sanitize before publishing.** No IP addresses, hostnames, filesystem paths, or credentials belong in any published artifact.

## Quick start (after reading everything)

```sh
bash deployment-kit/scripts/serve.sh
```

This runs the exact validated vendor command. Edit the model path and bind address at the top of that script for your environment; the script ships with a placeholder bind, not a real internal address.
