#!/usr/bin/env python3
"""S3 gate metrics (vLLM-native) for Inkling-2-Small. HF transformers 5.24.1 is
incompatible with this checkpoint, so the gate uses the vLLM nightly native loader.

Captures, on the 32 SAFE hard32 rows (prefill only, thinking ON effort 0.9):
- attention_writer [32, band, 4608]: wo_ud OUT-space activation (TP4 partial summed)
  via the i2s_capture_ext worker extension (same hook path as S2 capture).
- per-token NLL (capability floor) from SamplingParams.prompt_logprobs.
- final logits last position from prompt_logprobs top-k (logits not directly exposed;
  we store the last-token logprob vector over returned top-k as a capability proxy).

Outputs {tag}-gate.npz (writer[32,band,4608], nll[32]) + {tag}-gate-meta.json.
Read-only: no weights changed. Runs inside the pinned nightly container, TP4 GPUs 4-7.
"""
import argparse, datetime, hashlib, json, pathlib, time
import numpy as np

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""): h.update(b)
    return "sha256:" + h.hexdigest()

HIDDEN, PHYSICAL = 4608, 48
CANONICAL_HARD32 = "sha256:902ac2472d572ff5724e0b70012efe687c87d19c655f09cd528d8a9bca60f7a8"
NIGHTLY_IMAGE = "vllm/vllm-openai@sha256:e72944ef5ab9d1b91228d066117595847e94bf95321018a2119e70e8ba643ff6"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--tp", type=int, default=4)
    a = p.parse_args()
    out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
    partial_dir = out / f"{a.tag}-partials"; partial_dir.mkdir(exist_ok=True)

    assert sha(a.corpus) == CANONICAL_HARD32, "corpus hash mismatch"
    rows = [json.loads(x) for x in pathlib.Path(a.corpus).read_text().splitlines() if x.strip()]
    safe = [r for r in rows if r["label"] == "safe"]
    assert len(safe) == 32, f"expected 32 safe rows, got {len(safe)}"
    lo, hi = (int(x) for x in a.layers.split("-"))
    LSEL = list(range(lo, hi + 1))

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    if tok.pad_token is None:
        eot = tok.convert_tokens_to_ids("<|endoftext|>")
        assert isinstance(eot, int) and eot >= 0
        tok.pad_token = "<|endoftext|>"
    rendered, token_ids = [], []
    for r in safe:
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, reasoning_effort=0.9)
        text = text + "<|message_model|><|content_thinking|>"
        assert "Thinking effort level: 0.9" in text
        rendered.append(text)
        token_ids.append(tok(text, add_special_tokens=False)["input_ids"])

    started = utc(); t0 = time.time()
    print(f"S3_GATE_BEGIN tag={a.tag} rows={len(safe)} band={a.layers}", flush=True)
    llm = LLM(model=a.checkpoint, tensor_parallel_size=a.tp, enforce_eager=True,
              gpu_memory_utilization=0.85, max_model_len=8192, max_num_seqs=1,
              max_num_batched_tokens=8192, enable_prefix_caching=False,
              enable_chunked_prefill=False, disable_log_stats=True, trust_remote_code=True,
              worker_extension_cls="i2s_capture_ext.I2SCaptureExt")
    print(f"S3_GATE_LOADED seconds={time.time()-t0:.2f}", flush=True)
    info = llm.collective_rpc("i2s_install", args=(str(partial_dir),))
    print("S3_GATE_HOOKS", json.dumps(info[0]), flush=True)

    # prompt_logprobs=0 returns the logprob of each actual prompt token (NLL source).
    sp = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False, prompt_logprobs=0)
    W = np.zeros((len(safe), len(LSEL), HIDDEN), dtype=np.float32)
    NLL = np.zeros(len(safe), dtype=np.float64)
    row_meta = []
    with torch.inference_mode():
        for i, r in enumerate(safe):
            ids = token_ids[i]
            res = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
            # NLL from prompt_logprobs: skip first token (None), mean over scored tokens.
            pl = res.prompt_logprobs
            assert pl is not None and len(pl) == len(ids), f"prompt_logprobs len {len(pl) if pl else None}"
            lps = []
            for t in range(1, len(ids)):
                d = pl[t]
                tid = ids[t]
                assert tid in d, f"token {tid} missing in prompt_logprobs row {t}"
                lps.append(float(d[tid].logprob))
            NLL[i] = -float(np.mean(lps))
            receipts = llm.collective_rpc("i2s_grab", args=(i,))
            visits = receipts[0]["visits"]
            assert all(v == 1 for v in visits.values()) and len(visits) == PHYSICAL
            attn_sum = np.zeros((PHYSICAL, HIDDEN), dtype=np.float64)
            for rank in range(a.tp):
                ap = np.load(partial_dir / f"row{i:02d}_rank{rank}_attn.npy")
                attn_sum += ap.astype(np.float64)
            W[i] = attn_sum[LSEL].astype(np.float32)
            assert np.isfinite(W[i]).all()
            row_meta.append({"id": str(r.get("id", i)), "tokens": len(ids), "nll": float(NLL[i])})
            print(f"S3_GATE_ROW {i:02d}/31 id={r.get('id',i)} tokens={len(ids)} nll={NLL[i]:.4f}", flush=True)

    llm.collective_rpc("i2s_remove")
    np.savez_compressed(out / f"{a.tag}-gate.npz", writer=W, nll=NLL)
    meta = {"tag": a.tag, "rows": len(safe), "layers": LSEL, "hidden": HIDDEN,
            "checkpoint": str(a.checkpoint), "loader": "vllm-native", "vllm_image": NIGHTLY_IMAGE,
            "vllm_version": vllm.__version__, "torch": torch.__version__,
            "timestamp_utc": utc(), "started_at": started, "seconds": time.time() - t0,
            "reasoning_effort": 0.9, "thinking_mode": "template_mechanically_on",
            "prefill_only": True, "nll_source": "prompt_logprobs_mean_over_prompt_tokens",
            "writer_semantics": "wo_ud OUT-space, TP4 partial summed",
            "nll": NLL.tolist(), "nll_mean": float(NLL.mean()),
            "row_meta": row_meta,
            "prompt_sha256": [hashlib.sha256(x.encode()).hexdigest() for x in rendered],
            "gate_npz_sha256": sha(out / f"{a.tag}-gate.npz")}
    (out / f"{a.tag}-gate-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"status": "S3_GATE_PASS", "tag": a.tag, "nll_mean": float(NLL.mean()),
                      "gate_npz_sha256": meta["gate_npz_sha256"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()