#!/usr/bin/env python3
"""S3 gate metrics: candidate vs base on SAFE hard32 rows (prefill only).
Produces clean/candidate activations + logits for parity gates.
Base writer projections come from S2 raw-capture; candidate from live hooks.
"""
import argparse, datetime, hashlib, json, pathlib, socket, gc

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
p.add_argument("--tag", required=True)
a = p.parse_args()
out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)

mem = float([x for x in pathlib.Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")][0].split()[1]) / 1024**2
s = socket.socket(); s.settimeout(2); port = s.connect_ex(("127.0.0.1", 8899)); s.close()
if port == 0 or mem < 80:
    raise SystemExit(f"GPU GUARD HALT port8899={port} MemAvailableGiB={mem:.1f}")

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

rows = [json.loads(x) for x in pathlib.Path(a.corpus).read_text().splitlines() if x.strip()]
safe = [r for r in rows if r["label"] == "safe"]
assert len(safe) == 32

proc = AutoProcessor.from_pretrained(a.checkpoint)
model = AutoModelForMultimodalLM.from_pretrained(a.checkpoint, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
model.eval()
lm = model.model.language_model; cfg = lm.config
hidden = cfg.hidden_size
loop_start, loop_end, passes = cfg.loop_layer_index_start, cfg.loop_layer_index_end, cfg.loop_num_passes
vocab = None  # derived from first logits row at runtime (config text_config disagrees)

# hooks: writer output at L9-18 (per visit) + final norm (last hidden state)
writer = {}
handles = []
LSEL = list(range(9, 19))
def mk_writer(layer_idx):
    def hook(mod, args, kwargs, output):
        visit = writer.setdefault(layer_idx, 0)
        value = output[0] if isinstance(output, tuple) else output
        writer[f"{layer_idx}:{visit}"] = value[0, -1, :].detach().float().cpu().numpy()
        writer[layer_idx] = visit + 1
    return hook
for L in LSEL:
    handles.append(lm.layers[L].self_attn.o_proj.linear.register_forward_hook(mk_writer(L), with_kwargs=True))
final_state = {}
def fs_hook(mod, args, kwargs, output):
    value = output[0] if isinstance(output, tuple) else output
    final_state["last"] = value[0, -1, :].detach().float().cpu().numpy()
handles.append(lm.norm.register_forward_hook(fs_hook, with_kwargs=True))

W = np.zeros((32, 10 * passes, hidden), dtype=np.float32)   # writer outputs per (layer,pass); L18 row uses only p0
FS = np.zeros((32, hidden), dtype=np.float32)               # final hidden state
LOGITS = None  # allocated after first forward (vocab derived from logits shape)
NLL = np.zeros(32, dtype=np.float64)                       # mean per-token NLL of prompt

with torch.inference_mode():
    for i, r in enumerate(safe):
        text = proc.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, reasoning_effort=0.9)
        text = text + "<|message_model|><|content_thinking|>"
        ids = proc.tokenizer(text, add_special_tokens=False, return_tensors="pt").to(model.device)
        res = model(**ids, use_cache=False)
        logits = res.logits[0].float()           # (T, V) all positions
        if LOGITS is None:
            vocab = logits.shape[-1]
            LOGITS = np.zeros((32, vocab), dtype=np.float32)
        LOGITS[i] = logits[-1].cpu().numpy()
        # per-token NLL of the prompt (predict token t from <t)
        logp = torch.log_softmax(logits[:-1], dim=-1)
        tgt = ids["input_ids"][0, 1:]
        NLL[i] = float(torch.nn.functional.cross_entropy(logits[:-1], tgt).item())
        FS[i] = final_state["last"]
        k = 0
        expected_visits = {L: (passes if 6 <= L < 18 else 1) for L in LSEL}
        for L in LSEL:
            for v in range(expected_visits[L]):
                W[i, k] = writer[f"{L}:{v}"]; k += 1
        # remaining slots for L18 (p1,p2) stay zero — single-pass layer
        writer.clear()
        assert torch.isfinite(logits).all()
        del res, logits, ids

for h in handles: h.remove()
del model; gc.collect(); torch.cuda.empty_cache()

np.savez_compressed(out / f"{a.tag}-activations.npz", writer=W, final_state=FS, logits=LOGITS)
meta = {"tag": a.tag, "rows": 32, "layers": LSEL, "passes_per_layer": passes,
        "hidden": hidden, "vocab": vocab, "nll": NLL.tolist(),
        "checkpoint": str(a.checkpoint), "timestamp_utc": utc(),
        "activations_sha256": sha(out / f"{a.tag}-activations.npz")}
(out / f"{a.tag}-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps({k: meta[k] for k in ("tag", "rows", "activations_sha256")}))
print(f"{a.tag.upper()}_GATE_CAPTURE_PASS nll_mean={NLL.mean():.6f}")
