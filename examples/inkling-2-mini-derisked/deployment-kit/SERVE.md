# SERVE.md — validated serving configuration

## Engine (validated path)

| Item | Value |
|---|---|
| Container image | `vllm/vllm-openai:inkling-2-mini` |
| Entrypoint | `vllm` (explicit `--entrypoint vllm` required) |
| Command | `vllm serve <model-path>` |
| Reasoning parser | `inkling` |
| Speculative config | `{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}` |
| Max model len | `262144` |
| Tensor parallel | 1 |
| MTP | Preserved (byte-identical upstream `mtp.safetensors`) |

## Requirements

- GPU: single card with ≥ 21 GiB free for weights + KV at 262144 ctx (fp8 KV)
- Docker with NVIDIA runtime (`--gpus all --ipc=host`)
- Weights: this repo's model files (`model-00001-of-00001.safetensors`, `mtp.safetensors`, configs, tokenizer) in one directory

## Launch (validated command, sanitized)

```sh
docker run -d --name inkling-derisked \
  --gpus all --ipc=host --restart unless-stopped --network host \
  -v /path/to/model/dir:/models/inkling-derisked:ro \
  --entrypoint vllm vllm/vllm-openai:inkling-2-mini \
  serve /models/inkling-derisked \
  --served-model-name Inkling-2-Mini-Derisked \
  --host <bind-address> --port 8899 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --reasoning-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

Edit `/path/to/model/dir` and `<bind-address>` for your environment. Bind to a trusted interface only; the model has no built-in authentication — whoever reaches the port can use it.

## Readiness

`/v1/models` returning 200 is **not** sufficient. Readiness requires:

1. `Application startup complete` in the server log, and
2. a real chat completion returning coherent text in both `reasoning` and `content` fields.

Model load + CUDA graph capture takes several minutes on first boot.

## Template / parser notes

- Chat template: vendor template shipped in tokenizer config. Reasoning effort is rendered by the template (`Thinking effort level:` prefix).
- Reasoning is returned in a **`reasoning`** field (not `reasoning_content`).
- Normalize curly apostrophes (’) before applying refusal or content regexes.

## Verbatim serving receipt

The validated launch on the development host used host networking with the bind address restricted to a private wireguard interface; the bind address is environment-specific and is intentionally not shipped. Command shape is identical to the sanitized launch above.
