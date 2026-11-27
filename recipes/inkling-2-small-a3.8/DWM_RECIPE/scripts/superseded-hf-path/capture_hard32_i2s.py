#!/usr/bin/env python3
"""Inkling-2-Small (thinkingmachines/Inkling-2-Small, rev 885d78bf) S2:
real stateful full-prefill hard32 capture. Adapted from Inkling-Small v1
capture_hard32_v1.py (run 20261127-inkling-small-s0-s2-crystal300-v1, proven).

Inkling-2-Small differences (verified on-disk 2026-11-27):
- 48 physical layers, hidden 4608, NO loop passes (every layer once)
- MoE every layer: 384 routed experts top-8, NO shared experts (n_shared_experts=0)
  -> families: layer_output, attention_writer, mlp_output, moe_experts
  (shared_experts family dropped: absent by architecture, not skipped silently)
- attention_writer = self_attn.o_proj = Inkling2GatedOutputProjection;
  its forward returns linear(gated attn) == wo_ud matmul output, OUT space 4608.
  wo_ud stored [4608,6144] BF16 (48 full heads x 128 = 6144 in-dim).
- block attention residuals (attn_res queries, 6-layer blocks), rel-bias attn
  (d_rel 16, extent 1024), sconv-k4 x4/layer, SWA-1024 on 40 local layers,
  8 full-attention layers (5,11,17,23,29,35,41,47).
- DSpark speculator (mtp.safetensors, 84 tensors) excluded from capture:
  trunk prefill only; speculator heads not exercised.
- Experts NVFP4 (L0-39) / MXFP8 (L40-47) on disk; HF inkling2 dequantizes to
  BF16 compute at load (parameter_groups compute mapping). Capture is stock
  BF16-compute inference; no weight changes anywhere.

Read-only: no weights/runtime/tokenizer/checkpoint changed. Full forwards only;
no decode/cache reuse; prefill-only, last-nonpadding-token readout.
Thinking ON per operator S1 standard (reasoning_effort 0.9, explicit thinking open).
GPUs 4-7 only (CUDA_VISIBLE_DEVICES set by launcher); GPUs 0-3 hold another run.
"""
import argparse, datetime, hashlib, json, pathlib, socket, time
import numpy as np

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return "sha256:" + h.hexdigest()

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--corpus", required=True)
p.add_argument("--output", required=True)
p.add_argument("--limit", type=int, default=64)
p.add_argument("--allow-noncanonical-limit", action="store_true")
p.add_argument("--probe", action="store_true", help="single-row structural probe; writes probe report, no full capture")
a = p.parse_args()
out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
if (a.limit != 64) and not a.allow_noncanonical_limit and not a.probe:
    raise SystemExit("Canonical hard32 requires exactly 64 rows")
mem = float([x for x in pathlib.Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")][0].split()[1]) / 1024**2
s = socket.socket(); s.settimeout(2); port = s.connect_ex(("127.0.0.1", 8901)); s.close()
if port == 0 or mem < 80:
    raise SystemExit(f"GPU GUARD HALT port8901={port} MemAvailableGiB={mem:.1f}")
import torch, transformers

# HF device_map="auto" = naive pipeline sharding across the 4 visible GPUs:
# each layer lives WHOLE on one device, so every captured tensor is a complete
# post-allreduce residual-stream value — no TP partial sums. (Not vLLM TP.)

CANONICAL_HARD32 = "sha256:902ac2472d572ff5724e0b70012efe687c87d19c655f09cd528d8a9bca60f7a8"
assert sha(a.corpus) == CANONICAL_HARD32, "corpus does not match canonical hard32 hash"
rows = [json.loads(x) for x in pathlib.Path(a.corpus).read_text().splitlines() if x.strip()]
if a.probe:
    rows = rows[:1]
else:
    rows = rows[:a.limit]
    assert len(rows) == a.limit
    assert [x["label"] for x in rows] == ["harmful" if i % 2 == 0 else "safe" for i in range(len(rows))], "locked alternating hard32 pair order"
script_sha = sha(__file__)
started = utc()

from transformers import AutoConfig, AutoTokenizer, AutoModelForImageTextToText
cfg_full = AutoConfig.from_pretrained(a.checkpoint)
tok = AutoTokenizer.from_pretrained(a.checkpoint)
# Pad token ships as <|endoftext|> in this checkpoint; keep in-memory assignment
# fallback only. Rows sliced to true length before forward; pad never enters a forward.
if tok.pad_token is None:
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    assert isinstance(eot, int) and eot >= 0, "no <|endoftext|> in vocab"
    tok.pad_token = "<|endoftext|>"
    print(f"S2_PAD_TOKEN_ASSIGNED_IN_MEMORY <|endoftext|> id={tok.pad_token_id}", flush=True)
rendered = []
for r in rows:
    text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                   add_generation_prompt=True, reasoning_effort=0.9)
    text = text + "<|message_model|><|content_thinking|>"
    assert "Thinking effort level: 0.9" in text, "thinking-ON template marker missing"
    rendered.append(text)
enc = tok(rendered, add_special_tokens=False, padding=True, return_tensors="pt")

print(f"S2_CAPTURE_BEGIN rows={len(rows)} probe={a.probe} checkpoint={a.checkpoint}", flush=True)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(a.checkpoint, dtype=torch.bfloat16,
                                                    device_map="auto", attn_implementation="eager")
model.eval()
print(f"S2_LOADED seconds={time.time()-t0:.2f} class={type(model).__name__}", flush=True)

tc = cfg_full.text_config if hasattr(cfg_full, "text_config") else cfg_full
hidden = tc.hidden_size; physical = tc.num_hidden_layers
assert (physical, hidden) == (48, 4608), f"unexpected geometry {physical}x{hidden}"
trunk = model.model.language_model if hasattr(model.model, "language_model") else model.model
layers = trunk.layers
assert len(layers) == physical, f"decoder len {len(layers)} != {physical}"

inventory = {}
for i, layer in enumerate(layers):
    inventory[i] = {name: type(mod).__name__ for name, mod in layer.named_children()}
(out / "module-inventory.json").write_text(json.dumps(inventory, indent=1) + "\n")
if a.probe:
    print("PROBE_INVENTORY_WRITTEN", flush=True)

expected = {i: 1 for i in range(physical)}

def find_attr(layer, *names):
    for n in names:
        cur = layer; ok = True
        for part in n.split("."):
            if hasattr(cur, part): cur = getattr(cur, part)
            else: ok = False; break
        if ok and isinstance(cur, torch.nn.Module):
            return n, cur
    return None, None

family_paths = {}
resolved = {}
for i, layer in enumerate(layers):
    resolved[i] = {}
    attn_name, attn = find_attr(layer, "self_attn.o_proj", "self_attn.o_proj.linear", "attn.o_proj")
    if attn is not None: resolved[i]["attention_writer"] = attn; family_paths.setdefault("attention_writer", attn_name)
    mlp_name, mlp = find_attr(layer, "mlp", "feed_forward", "ffn")
    if mlp is not None: resolved[i]["mlp_output"] = mlp; family_paths.setdefault("mlp_output", mlp_name)
    exp_name, exp = find_attr(layer, "mlp.experts", "mlp.routed_experts")
    if exp is not None: resolved[i]["moe_experts"] = exp; family_paths.setdefault("moe_experts", exp_name)
assert family_paths.get("attention_writer") == "self_attn.o_proj", f"attention writer path {family_paths}"

FAMILIES = ["layer_output", "attention_writer", "mlp_output", "moe_experts"]
sites = physical
raw = {name: np.zeros((len(rows), sites, hidden), dtype=np.float32) for name in FAMILIES}
present = {name: np.zeros((len(rows), sites), dtype=bool) for name in FAMILIES}
visits = {}; captures = []; handles = []; current = [-1]

def last_token(value):
    if isinstance(value, tuple): value = value[0]
    if value.ndim == 3: return value[0, -1, :]
    if value.ndim == 2: return value[-1, :]
    raise RuntimeError(f"unexpected shape {tuple(value.shape)}")

for i, layer in enumerate(layers):
    for family, module in resolved[i].items():
        def hook(mod, args, kwargs, output, layer_idx=i, fam=family):
            assert visits.get(layer_idx, 0) == 0, f"writer before layer_output L{layer_idx}"
            raw[fam][current[0], layer_idx] = last_token(output).detach().float().cpu().numpy()
            present[fam][current[0], layer_idx] = True
        handles.append(module.register_forward_hook(hook, with_kwargs=True))
    def layer_output_hook(mod, args, kwargs, output, layer_idx=i):
        raw["layer_output"][current[0], layer_idx] = last_token(output).detach().float().cpu().numpy()
        present["layer_output"][current[0], layer_idx] = True
    def layer_count_hook(mod, args, kwargs, output, layer_idx=i):
        v = visits.get(layer_idx, 0)
        assert v < expected[layer_idx], f"extra visit L{layer_idx}"
        captures.append((layer_idx, v)); visits[layer_idx] = v + 1
    handles.append(layer.register_forward_hook(layer_output_hook, with_kwargs=True))
    handles.append(layer.register_forward_hook(layer_count_hook, with_kwargs=True))

with torch.inference_mode():
    for i, r in enumerate(rows):
        visits.clear(); captures.clear(); current[0] = i
        n = int(enc["attention_mask"][i].sum())
        ids = enc["input_ids"][i:i+1, :n].to(model.get_input_embeddings().weight.device)
        mask = enc["attention_mask"][i:i+1, :n].to(model.get_input_embeddings().weight.device)
        assert n > 0
        result = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
        logits = result.logits[0, -1].float()
        assert torch.isfinite(logits).all(), f"nonfinite logits row {i}"
        assert dict(visits) == expected, f"wrong visit counts row {i}: {visits}"
        assert captures == [(l, 0) for l in range(physical)], f"noncanonical visit order row {i}: {captures}"
        assert np.isfinite(raw["layer_output"][i]).all()
        print(f"S2_ROW {i:02d}/{len(rows)-1} id={r.get('id', i)} label={r['label']} tokens={n} visits={sum(visits.values())} logit_absmax={float(logits.abs().max()):.3f}", flush=True)
        del result, logits, ids, mask
for h in handles: h.remove()

if a.probe:
    report = {"probe": True, "loaded_class": type(model).__name__, "family_paths": family_paths,
              "presence": {k: v[0].tolist() for k, v in present.items()},
              "layer_output_norms": [float(np.linalg.norm(raw["layer_output"][0, l])) for l in range(physical)],
              "seconds": time.time() - t0}
    (out / "probe-report.json").write_text(json.dumps(report, indent=1) + "\n")
    print("S2_PROBE_PASS", json.dumps({k: report[k] for k in ("family_paths", "seconds")}), flush=True)
    raise SystemExit(0)

metadata = {
    "schema": "inkling2-small.s2-raw-capture.v1", "started_at": started, "finished_at": utc(),
    "checkpoint": str(a.checkpoint), "revision": "885d78bfca61a7e9a498e33cfa7ad35850533b8c",
    "transformers": transformers.__version__, "torch": torch.__version__,
    "capture_script_sha256": script_sha, "corpus_sha256": sha(a.corpus),
    "row_count": len(rows), "pair_count": len(rows)//2, "hidden_size": hidden, "physical_layers": physical,
    "loop_passes": None, "visit_semantics": "Inkling-2-Small trunk: every physical layer once per forward; DSpark speculator excluded",
    "reasoning_effort": 0.9, "thinking_mode": "template_mechanically_on",
    "template_add_generation_prompt": True, "explicit_content_thinking": True, "use_cache": False,
    "prefill_only": True, "generation_performed": False, "last_token_position": "last_nonpadding_token",
    "family_paths": family_paths, "family_presence": {k: v.all(axis=0).tolist() for k, v in present.items()},
    "family_semantics": {
        "layer_output": "decoder output after block-attn-residual add; not a single writer",
        "attention_writer": "gated attention output projection (o_proj module output = wo_ud matmul) before residual",
        "mlp_output": "full SparseMoeBlock output (router-weighted routed experts; no shared experts in this arch)",
        "moe_experts": "Inkling2Experts module output (fp32-accumulated router-weighted sum cast to bf16)",
    },
    "prompt_sha256": [hashlib.sha256(x.encode()).hexdigest() for x in rendered],
    "token_counts": [int(x) for x in enc["attention_mask"].sum(1)],
    "row_ids": [str(x.get("id", i)) for i, x in enumerate(rows)], "labels": [x["label"] for x in rows],
    "pairing": "corpus_order_adjacent_harmful_safe",
    "dtype_storage": "float32 raw after stock BF16-compute inference (experts dequantized by HF at load)",
}
np.savez_compressed(out / "raw-capture.npz", **raw)
(out / "capture-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps({"status": "S2_CAPTURE_PASS", "raw_npz_sha256": sha(out / "raw-capture.npz"),
                  "metadata_sha256": sha(out / "capture-metadata.json"), "sites": sites,
                  "seconds": time.time() - t0}, indent=2), flush=True)