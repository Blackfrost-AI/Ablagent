#!/usr/bin/env python3
"""Inkling-2-Small S2 hard32 capture via vLLM NATIVE loader (nightly image).

HF transformers 5.24.1 is incompatible with this checkpoint (0 key match, no conversion).
Authoritative loader = vllm/models/inkling2/nvidia. Requires vLLM nightly 0.35.0+ per the
vLLM cookbook (recipes.vllm.ai/thinkingmachines/Inkling-2-Small): the 0.34.x pinned image's
Triton lacks gluon tensor_memory_descriptor._reinterpret needed by the tcgen05 _verify_p kernel.

Pinned nightly image: vllm/vllm-openai@sha256:e72944ef5ab9d1b91228d066117595847e94bf95321018a2119e70e8ba643ff6
(vllm 0.34.1rc1.dev254+g..., Triton 3.8.0 WITH _reinterpret). TP4 on GPUs 4-7.

Hooks installed via worker_extension_cls (i2s_capture_ext.I2SCaptureExt) + collective_rpc
with STRING method names (no function pickling across the spawn boundary).
- attention_writer: layer.attn partial [T,4608], summed across TP ranks == full wo_ud OUT.
- mlp_output/moe_experts: layer.mlp full all-reduced [T,4608] (ep_size=1). Identical.
- layer_output: fused residual -> absent (honest). Bake only consumes attention_writer.

enforce_eager=True, batch=1 prefill, max_tokens=1, no prefix cache/chunked prefill.
Thinking ON per operator S1 (reasoning_effort 0.9 + explicit thinking open). Read-only.
"""
import argparse, datetime, hashlib, json, os, pathlib, time
import numpy as np

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""): h.update(b)
    return "sha256:" + h.hexdigest()

HIDDEN, PHYSICAL = 4608, 48
CANONICAL_HARD32 = "sha256:902ac2472d572ff5724e0b70012efe687c87d19c655f09cd528d8a9bca60f7a8"
FAMILIES = ["layer_output", "attention_writer", "mlp_output", "moe_experts"]
NIGHTLY_IMAGE = "vllm/vllm-openai@sha256:e72944ef5ab9d1b91228d066117595847e94bf95321018a2119e70e8ba643ff6"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--probe", action="store_true")
    a = p.parse_args()
    out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
    partial_dir = out / "partials"; partial_dir.mkdir(exist_ok=True)

    assert sha(a.corpus) == CANONICAL_HARD32, "corpus does not match canonical hard32 hash"
    rows = [json.loads(x) for x in pathlib.Path(a.corpus).read_text().splitlines() if x.strip()]
    if a.probe:
        rows = rows[:1]
    else:
        rows = rows[:a.limit]
        assert len(rows) == a.limit == 64, "canonical hard32 requires 64 rows"
        assert [x["label"] for x in rows] == ["harmful" if i % 2 == 0 else "safe" for i in range(64)]

    import torch
    import vllm  # registers inkling2 config with transformers
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    if tok.pad_token is None:
        eot = tok.convert_tokens_to_ids("<|endoftext|>")
        assert isinstance(eot, int) and eot >= 0
        tok.pad_token = "<|endoftext|>"
    rendered, token_ids = [], []
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, reasoning_effort=0.9)
        text = text + "<|message_model|><|content_thinking|>"
        assert "Thinking effort level: 0.9" in text, "thinking-ON marker missing"
        rendered.append(text)
        token_ids.append(tok(text, add_special_tokens=False)["input_ids"])

    script_sha = sha(__file__)
    ext_sha = sha(pathlib.Path(__file__).parent / "i2s_capture_ext.py")
    started = utc()
    print(f"S2_VLLM_CAPTURE_BEGIN rows={len(rows)} probe={a.probe} tp={a.tp}", flush=True)
    t0 = time.time()

    llm = LLM(model=a.checkpoint, tensor_parallel_size=a.tp, enforce_eager=True,
              gpu_memory_utilization=0.85, max_model_len=8192, max_num_seqs=1,
              max_num_batched_tokens=8192, enable_prefix_caching=False,
              enable_chunked_prefill=False, disable_log_stats=True, trust_remote_code=True,
              worker_extension_cls="i2s_capture_ext.I2SCaptureExt")
    print(f"S2_VLLM_LOADED seconds={time.time()-t0:.2f}", flush=True)

    info = llm.collective_rpc("i2s_install", args=(str(partial_dir),))
    print("S2_HOOKS_INSTALLED", json.dumps(info), flush=True)

    sp = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)
    row_meta = []
    with torch.inference_mode():
        for i, r in enumerate(rows):
            ids = token_ids[i]
            assert len(ids) > 0
            llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
            receipts = llm.collective_rpc("i2s_grab", args=(i,))
            visits = receipts[0]["visits"]
            assert all(v == 1 for v in visits.values()) and len(visits) == PHYSICAL, f"visits {visits}"
            row_meta.append({"id": str(r.get("id", i)), "label": r["label"], "tokens": len(ids)})
            print(f"S2_ROW {i:02d}/{len(rows)-1} label={r['label']} tokens={len(ids)} ranks={len(receipts)}", flush=True)

    raw = {f: np.zeros((len(rows), PHYSICAL, HIDDEN), dtype=np.float32) for f in FAMILIES}
    present = {f: np.zeros((len(rows), PHYSICAL), dtype=bool) for f in FAMILIES}
    for i in range(len(rows)):
        attn_sum = np.zeros((PHYSICAL, HIDDEN), dtype=np.float64)
        mlp_ref = None
        for rank in range(a.tp):
            ap = np.load(partial_dir / f"row{i:02d}_rank{rank}_attn.npy")
            mp = np.load(partial_dir / f"row{i:02d}_rank{rank}_mlp.npy")
            attn_sum += ap.astype(np.float64)
            if mlp_ref is None: mlp_ref = mp.astype(np.float64)
            else: assert np.allclose(mp, mlp_ref, atol=1e-3), f"mlp differs across ranks row {i}"
        raw["attention_writer"][i] = attn_sum.astype(np.float32)
        raw["mlp_output"][i] = mlp_ref.astype(np.float32)
        raw["moe_experts"][i] = mlp_ref.astype(np.float32)
        present["attention_writer"][i] = True
        present["mlp_output"][i] = True
        present["moe_experts"][i] = True
        assert np.isfinite(raw["attention_writer"][i]).all()

    if a.probe:
        report = {"probe": True, "hook_info": info, "tp": a.tp,
                  "attention_writer_norms": [float(np.linalg.norm(raw["attention_writer"][0, l])) for l in range(PHYSICAL)],
                  "mlp_output_norms": [float(np.linalg.norm(raw["mlp_output"][0, l])) for l in range(PHYSICAL)],
                  "seconds": time.time() - t0}
        (out / "probe-report.json").write_text(json.dumps(report, indent=1) + "\n")
        print("S2_VLLM_PROBE_PASS", json.dumps({"seconds": report["seconds"]}), flush=True)
        llm.collective_rpc("i2s_remove")
        return

    metadata = {
        "schema": "inkling2-small.s2-raw-capture.vllm.v1", "started_at": started, "finished_at": utc(),
        "checkpoint": str(a.checkpoint), "revision": "885d78bfca61a7e9a498e33cfa7ad35850533b8c",
        "loader": "vllm-native", "vllm_image": NIGHTLY_IMAGE,
        "vllm_version": vllm.__version__, "torch": torch.__version__,
        "tensor_parallel_size": a.tp, "enforce_eager": True,
        "capture_script_sha256": script_sha, "extension_sha256": ext_sha, "corpus_sha256": sha(a.corpus),
        "row_count": len(rows), "pair_count": len(rows)//2, "hidden_size": HIDDEN, "physical_layers": PHYSICAL,
        "loop_passes": None, "visit_semantics": "every physical layer once per forward; DSpark speculator excluded",
        "reasoning_effort": 0.9, "thinking_mode": "template_mechanically_on",
        "template_add_generation_prompt": True, "explicit_content_thinking": True,
        "prefill_only": True, "generation_performed": False, "max_tokens": 1,
        "last_token_position": "last_prompt_token",
        "family_paths": {"attention_writer": "model.layers[i].attn (wo_ud OUT, TP partial summed)",
                         "mlp_output": "model.layers[i].mlp (Inkling2MoE full reduced)",
                         "moe_experts": "model.layers[i].mlp (routed-only == mlp_output)",
                         "layer_output": "ABSENT (fused block-attn-residual; no clean module output)"},
        "family_presence": {k: v.all(axis=0).tolist() for k, v in present.items()},
        "family_semantics": {
            "attention_writer": "wo_ud OUT-space activation = sum of TP-rank partials (RowParallelLinear reduce_results=False)",
            "mlp_output": "full router-weighted routed-expert output (all-reduced, ep_size=1)",
            "moe_experts": "identical to mlp_output (no shared experts in this arch)",
            "layer_output": "absent by capture design (fused residual); not used by bake",
        },
        "prompt_sha256": [hashlib.sha256(x.encode()).hexdigest() for x in rendered],
        "token_counts": [m["tokens"] for m in row_meta],
        "row_ids": [m["id"] for m in row_meta], "labels": [m["label"] for m in row_meta],
        "pairing": "corpus_order_adjacent_harmful_safe",
        "dtype_storage": "float32 raw after vLLM BF16-compute inference (NVFP4/MXFP8 experts native)",
    }
    np.savez_compressed(out / "raw-capture.npz", **raw)
    (out / "capture-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    llm.collective_rpc("i2s_remove")
    print(json.dumps({"status": "S2_VLLM_CAPTURE_PASS", "raw_npz_sha256": sha(out / "raw-capture.npz"),
                      "metadata_sha256": sha(out / "capture-metadata.json"),
                      "seconds": time.time() - t0}, indent=2), flush=True)

if __name__ == "__main__":
    main()