#!/usr/bin/env python3
"""S2 battery for loop-transformer raw activations. No weight changes."""
import argparse, hashlib, json, pathlib, datetime
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('--capture-dir',required=True); p.add_argument('--output',required=True); p.add_argument('--permutations',type=int,default=2000); a=p.parse_args()
out=pathlib.Path(a.output); out.mkdir(parents=True,exist_ok=True); cp=pathlib.Path(a.capture_dir)
meta=json.loads((cp/'capture-metadata.json').read_text()); z=np.load(cp/'raw-capture.npz')
assert meta['row_count']==64 and meta['pair_count']==32
positions=meta['positions']; visits=meta['expected_visits']; physical=meta['physical_layers']; hidden=meta['hidden_size']; passes=meta['loop_num_passes']; ls=meta['loop_layer_index_start']; le=meta['loop_layer_index_end']
rows=[]
for layer in range(physical):
    for visit in range(visits[str(layer)]):
        rows.append({'layer':layer,'visit':visit,'logical_site':positions[layer]+visit,'looped':ls<=layer<le,'writer_family':'attention_o_proj','attention_writer_projection':'self_attn.o_proj.linear','downstream_expert_branch':'fused_experts_router_weighted_output','candidate_tensor':'self_attn.o_proj.linear.output','capture_hook':'self_attn.o_proj.linear','semantics_note':'SparseMoeBlock is routed-experts-ONLY (no shared expert); mlp_output is moe_down reshaped, kept as duplicate consistency check'})
assert len(rows)==48
sites=[r['logical_site'] for r in rows]
labels=np.array([1 if x=='harmful' else 0 for x in meta['labels']],dtype=np.int8)
assert np.array_equal(labels,np.tile([1,0],32))
rng=np.random.default_rng(20261120)
permutations=np.array([np.array([labels[i] for j in range(32) for i in rng.permutation([2*j,2*j+1])],dtype=np.int8) for _ in range(a.permutations)])
# Each permutation independently flips each adjacent harmful/safe pair.
assert np.all((permutations[:,::2]+permutations[:,1::2])==1)
families=['layer_output','attention_writer','moe_down','mlp_output']
# Architecture fact (source-verified): routed experts only, so these two hooks
# must agree exactly. Assert rather than treat them as independent evidence.
assert np.allclose(z['moe_down'],z['mlp_output'],atol=1e-6), 'moe_down/mlp_output diverge: hook semantics wrong'
directions=np.zeros((48,hidden),dtype=np.float32)
results={}
def unit(v):
    n=np.linalg.norm(v.astype(np.float64)); return v/n if n>0 and np.isfinite(n) else np.zeros_like(v)
def cos(x,y):
    return float(np.dot(unit(x),unit(y)))
for family in families:
    x=z[family].astype(np.float64)
    assert x.shape==(64,48,hidden) and np.isfinite(x).all()
    x-=x.mean(axis=0,keepdims=True) # per-SITE centring across the 64 rows (axis0=rows); not across sites
    harmful=x[0::2]; safe=x[1::2]
    delta=(harmful-safe).mean(axis=0)
    norms=np.linalg.norm(delta,axis=1)
    unit_delta=delta/np.maximum(norms[:,None],1e-30)
    if family=='attention_writer': directions=unit_delta.astype(np.float32)
    halves=[]
    for h in range(2):
        idx=np.arange(h,32,2)
        d=(x[2*idx]-x[2*idx+1]).mean(axis=0)
        halves.append(d)
    split=np.array([cos(halves[0][j],halves[1][j]) for j in range(48)])
    # Same-index permuted labels produce null difference vectors per pair.
    signed=(x[0::2]-x[1::2])
    null=np.empty((a.permutations,48),dtype=np.float64)
    for n in range(a.permutations):
        signs=(permutations[n,0::2]*2-1).astype(np.float64)
        null[n]=np.linalg.norm((signed*signs[:,None,None]).mean(axis=0),axis=1)
    pvals=(1+(null>=norms[None,:]).sum(axis=0))/(a.permutations+1)
    pass_cos={}
    for l in range(ls,le):
        ix=[positions[l]+v for v in range(passes)]
        pass_cos[str(l)]=[[cos(delta[i],delta[j]) for j in ix] for i in ix]
    famres={'rows':[],'pass_cosines':pass_cos,'direction_norms':norms.tolist(),'split_half_cosine':split.tolist(),'permutation_p':pvals.tolist(),'null_norm_quantiles':{str(q):np.quantile(null,q,axis=0).tolist() for q in [0.05,0.5,0.95]},'null_permutations':a.permutations}
    for j,r in enumerate(rows):
        e=float(np.mean((x[:,j]@unit_delta[j])**2)) # activation-projection energy; not a writer claim
        q=x[:,j]@unit_delta[j]
        famres['rows'].append({**r,'family':family,'direction_norm':float(norms[j]),'split_half_cosine':float(split[j]),'permutation_p':float(pvals[j]),'energy_mean_projection_squared':e,'energy_harmful_mean':float(np.mean(q[labels==1]**2)),'energy_safe_mean':float(np.mean(q[labels==0]**2))})
    results[family]=famres
# Controller energy only; direction cosines measured before selection above.
gates={}
for name in ['forget','message']:
    x=z['gate_'+name].astype(np.float64)
    x-=x.mean(axis=0,keepdims=True)
    d=(x[0::2]-x[1::2]).mean(axis=0); norms=np.linalg.norm(d,axis=1)
    du=d/np.maximum(norms[:,None],1e-30)
    active=np.array([any(r['logical_site']==j and r['looped'] for r in rows) for j in range(48)])
    gate_rows=[]
    for r in rows:
        if not r['looped']: continue
        j=r['logical_site']; projection=x[:,j]@du[j]
        gate_rows.append({**r,'direction_norm':float(norms[j]),'energy_mean_projection_squared':float(np.mean(projection**2))})
    gates[name]={'rows':gate_rows,'direction_norms':norms.tolist(),'semantics':'pre-tanh q_message_gate projection output; no editing recommendation; other controllers not measured'}
# Cross-pass cosine is a first-class artifact, separately easy to audit.
cross={'family':'attention_writer','layers':{str(l):results['attention_writer']['pass_cosines'][str(l)] for l in range(ls,le)},'interpretation':'Same physical layer, different actual loop passes; not independent layers and not proof of intervention transfer.'}
(out/'cross-pass-cosines.json').write_text(json.dumps(cross,indent=2)+'\n')
(out/'direction-battery.json').write_text(json.dumps({'schema':'inkling2-mini.s2-direction-battery.v1','metadata':meta,'permutation_count':a.permutations,'split_half':'even pairs vs odd pairs; cosine of paired difference vectors','normalization':'paired per-site L2 centroid differences; per-site harmful/safe centring','families':results,'gates':gates},indent=2)+'\n')
np.savez_compressed(out/'directions.npz',directions=directions,logical_sites=np.array(sites,dtype=np.int32),layers=np.array([r['layer'] for r in rows],dtype=np.int16),visits=np.array([r['visit'] for r in rows],dtype=np.int16),labels=labels)
summary={'status':'S2_BATTERY_MEASURED_NOT_GO','measurement_complete':False,'reason':'S2 capture/battery only; no S3 derivative, no target family/gate, no writer intervention evidence','capture_rows':64,'logical_sites':48,'cross_pass_cosines_first':True,'families':families,'hard32_corpus_sha256':meta['hard32_corpus_sha256'],'directions_sha256':'sha256:'+hashlib.sha256((out/'directions.npz').read_bytes()).hexdigest(),'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out/'s2-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
