#!/usr/bin/env python3
"""Inkling-2-Mini S2: real stateful full-prefill hard32 capture.

Physical module instances are visited repeatedly by the stock Inkling loop model.
Capture physical layer x visit index, never fold visits. Read-only: no weights,
runtime, tokenizer, or checkpoint are changed. Full forwards only; no decode/cache
reuse, synthetic loop, or generation passed off as thinking-state capture.
"""
import argparse, datetime, hashlib, json, os, pathlib, socket, time
import numpy as np

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return "sha256:"+h.hexdigest()

p=argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--corpus", required=True)
p.add_argument("--output", required=True)
p.add_argument("--limit", type=int, default=64)
p.add_argument("--allow-noncanonical-limit", action="store_true")
a=p.parse_args()
out=pathlib.Path(a.output); out.mkdir(parents=True,exist_ok=True)
lab=out.parent
if (a.limit != 64) and not a.allow_noncanonical_limit:
    raise SystemExit("Canonical hard32 requires exactly 64 rows")
mem=float([x for x in pathlib.Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")][0].split()[1])/1024**2
s=socket.socket(); s.settimeout(2); port=s.connect_ex(("127.0.0.1",8899)); s.close()
if port==0 or mem<80:
    raise SystemExit(f"GPU GUARD HALT port8899={port} MemAvailableGiB={mem:.1f}")
import torch, transformers
from transformers import AutoProcessor, AutoModelForMultimodalLM

CANONICAL_HARD32 = "sha256:902ac2472d572ff5724e0b70012efe687c87d19c655f09cd528d8a9bca60f7a8"
assert sha(a.corpus) == CANONICAL_HARD32, "corpus does not match canonical hard32 hash (state.yaml pinned)"
rows=[json.loads(x) for x in pathlib.Path(a.corpus).read_text().splitlines() if x.strip()]
if a.limit: rows=rows[:a.limit]
assert len(rows)==a.limit
assert [x["label"] for x in rows]==["harmful" if i%2==0 else "safe" for i in range(len(rows))], "Expected locked alternating hard32 pair order"
script_sha=sha(__file__)
started=utc()
proc=AutoProcessor.from_pretrained(a.checkpoint)
rendered=[]
for r in rows:
    text=proc.apply_chat_template([{"role":"user","content":r["prompt"]}],tokenize=False,add_generation_prompt=True,reasoning_effort=0.9)
    text=text+"<|message_model|><|content_thinking|>"
    assert "Thinking effort level: 0.9" in text
    rendered.append(text)
enc=proc.tokenizer(rendered,add_special_tokens=False,padding=True,return_tensors="pt")
# All prompts are prefill only. Forward individually: no padded-token choice can
# contaminate last-token hidden states and no state is reused across rows.
print(f"S2_CAPTURE_BEGIN rows={len(rows)} checkpoint={a.checkpoint}",flush=True)
t0=time.time()
model=AutoModelForMultimodalLM.from_pretrained(a.checkpoint,dtype=torch.bfloat16,device_map="cuda",attn_implementation="eager")
model.eval()
lm=model.model.language_model; cfg=lm.config
hidden=cfg.hidden_size; physical=cfg.num_hidden_layers
loop_start,loop_end,passes=cfg.loop_layer_index_start,cfg.loop_layer_index_end,cfg.loop_num_passes
assert (physical,loop_start,loop_end,passes)==(24,6,18,3)
expected={i:(passes if loop_start<=i<loop_end else 1) for i in range(physical)}
print(f"S2_LOADED seconds={time.time()-t0:.2f} logical_sites={sum(expected.values())}",flush=True)

families={
 "layer_output":lambda l:l,
 "attention_writer":lambda l:l.self_attn.o_proj.linear,
 "moe_down":lambda l:l.mlp.experts.down_proj,
 "mlp_output":lambda l:l.mlp,
}
# Down_proj is an nn.Parameter in the fused expert bank, not an nn.Module.
# The experts module's output is already router-weighted and summed, so capture
# that exact writer-family output as moe_down and label it accordingly.
families["moe_down"]=lambda l:l.mlp.experts
raw={name:np.zeros((len(rows),sum(expected.values()),hidden),dtype=np.float32) for name in families}
controllers={name:np.zeros((len(rows),sum(expected.values()),hidden),dtype=np.float32) for name in ["forget","message"]}
visits={}; captures=[]; handles=[]; current=[-1]
positions=[]
for layer_idx,layer in enumerate(lm.layers):
    for family,getter in families.items():
        module=getter(layer)
        def hook(mod,args,kwargs,output,layer_idx=layer_idx,family=family):
            visit=visits.get(layer_idx,0)
            expected_visit=0 if not (loop_start<=layer_idx<loop_end) else visit
            if visit>=expected[layer_idx]: raise RuntimeError(f"extra visit L{layer_idx}")
            value=output[0] if isinstance(output,tuple) else output
            if value.ndim==3: value=value[0,-1,:]
            elif value.ndim==2: value=value[-1,:]
            else: raise RuntimeError(f"unexpected {family} shape {value.shape} L{layer_idx}")
            raw[family][current[0],positions[layer_idx]+expected_visit]=value.detach().float().cpu().numpy()
        handles.append(module.register_forward_hook(hook,with_kwargs=True))
    # layer output hooks must increment after all child writer/gate hooks.
    def layer_hook(mod,args,kwargs,output,layer_idx=layer_idx):
        visit=visits.get(layer_idx,0)
        assert visit<expected[layer_idx]
        captures.append((layer_idx,visit))
        visits[layer_idx]=visit+1
    # layer_output hook is separate so writer hook uses the pre-increment index.
    handles[-len(families)].remove()
    def layer_output_hook(mod,args,kwargs,output,layer_idx=layer_idx):
        visit=visits.get(layer_idx,0)
        value=output[0] if isinstance(output,tuple) else output
        raw["layer_output"][current[0],positions[layer_idx]+visit]=value[0,-1,:].detach().float().cpu().numpy()
    handles.append(layer.register_forward_hook(layer_output_hook,with_kwargs=True))
    handles.append(layer.register_forward_hook(layer_hook,with_kwargs=True))
    if hasattr(layer,"q_message_gate"):
        for gate_name in ["forget_gate_proj","message_gate_proj"]:
            def gate_hook(mod,args,kwargs,output,layer_idx=layer_idx,gate_name=gate_name):
                visit=visits.get(layer_idx,0)
                controllers["forget" if gate_name.startswith("forget") else "message"][current[0],positions[layer_idx]+visit]=output[0,-1,:].detach().float().cpu().numpy()
            handles.append(getattr(layer.q_message_gate,gate_name).register_forward_hook(gate_hook,with_kwargs=True))
positions=[]; offset=0
for i in range(physical): positions.append(offset); offset+=expected[i]

prefill_sha=None
with torch.inference_mode():
    for i,r in enumerate(rows):
        visits.clear(); captures.clear(); current[0]=i
        ids=enc["input_ids"][i:i+1,:int(enc["attention_mask"][i].sum())].to(model.device)
        mask=enc["attention_mask"][i:i+1,:ids.shape[1]].to(model.device)
        assert ids.shape[1]>0 and mask.sum()==ids.shape[1]
        result=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1)
        logits=result.logits[0,-1].float()
        assert torch.isfinite(logits).all() and float(logits.abs().max())<=30.001
        assert dict(visits)==expected, f"wrong visit counts row {i}: {visits}"
        expected_order=[(layer,v) for v in range(passes) for layer in range(loop_start,loop_end)]
        assert captures==[(layer,0) for layer in range(loop_start)]+expected_order+[(layer,0) for layer in range(loop_end,physical)], f"noncanonical block-loop visit order row {i}: {captures}"
        assert np.isfinite(raw["layer_output"][i]).all()
        print(f"S2_ROW {i:02d}/{len(rows)-1} id={r.get('id', i)} label={r['label']} tokens={ids.shape[1]} visits={sum(visits.values())}",flush=True)
        del result,logits,ids,mask
for h in handles: h.remove()

metadata={
 "schema":"inkling2-mini.s2-raw-capture.v1","started_at":started,"finished_at":utc(),
 "checkpoint":str(a.checkpoint),"transformers":transformers.__version__,"torch":torch.__version__,
 "capture_script_sha256":script_sha,"corpus_sha256":sha(a.corpus),"hard32_corpus_sha256":CANONICAL_HARD32,
 "row_count":len(rows),"pair_count":len(rows)//2,"hidden_size":hidden,"physical_layers":physical,
 "loop_layer_index_start":loop_start,"loop_layer_index_end":loop_end,"loop_num_passes":passes,
 "positions":positions,"expected_visits":expected,"reasoning_effort":0.9,"thinking_mode":"template_mechanically_on",
 "template_add_generation_prompt":True,"explicit_content_thinking":True,"use_cache":False,
 "prefill_only":True,"generation_performed":False,"last_token_position":"last_nonpadding_token",
 "prompt_sha256":[hashlib.sha256(x.encode()).hexdigest() for x in rendered],
 "token_counts":[int(x) for x in enc["attention_mask"].sum(1)],
 "row_ids":[str(x.get("id",i)) for i,x in enumerate(rows)],"labels":[x["label"] for x in rows],
 "pairing":"corpus_order_adjacent_harmful_safe",
 "family_semantics":{
   "layer_output":"decoder output after residual add; not a single writer",
   "attention_writer":"self_attn.o_proj.linear output before gated output residual path",
   "moe_down":"router-weighted fused experts module output; down_proj parameter has no module hook; ROUTED-EXPERTS-ONLY (no shared expert)",
   "mlp_output":"SparseMoeBlock output == moe_down reshaped (routed-only MoE; no shared-expert addition)",
 },
 "gate_semantics":"q_message_gate forget/message projection outputs before tanh; measure only; not message/mlp gate or block-attention controller",
 "visit_semantics":"physical layer x zero-based pass; looped L6-17 passes 0,1,2; other layers pass 0; no flattening",
 "dtype_storage":"float32 raw after stock BF16 inference",
}
np.savez_compressed(out/"raw-capture.npz",**raw,**{f"gate_{k}":v for k,v in controllers.items()})
(out/"capture-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
print(json.dumps({"status":"S2_CAPTURE_PASS","raw_npz_sha256":sha(out/"raw-capture.npz"),"metadata_sha256":sha(out/"capture-metadata.json"),"logical_sites":offset,"seconds":time.time()-t0},indent=2),flush=True)
