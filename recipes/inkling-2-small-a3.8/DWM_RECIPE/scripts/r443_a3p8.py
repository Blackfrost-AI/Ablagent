#!/usr/bin/env python3
"""R1-FALSE-REFUSAL-443-v1 on live Inkling-2-Small DWM a3.8 (the validated runtime, port 40001).

Operator protocol (2026-11-27): 32K answer budget, temp 1.0, thinking ON.
Corpus: prompts_443_v1.jsonl sha256 ba6431ca900cf41eb4f621cbf04d7f084187a16ec1ba6d1b8b6663967302d2e6 (asserted).
Runner canon: r443_a3p0.py, model id inkling2-small-dwm-a3p8. Resume-safe, 4 workers.
"""
import json, sys, time, os, hashlib, urllib.request, concurrent.futures as cf
from pathlib import Path

BASE = "http://127.0.0.1:40001"
MODEL = "inkling2-small-dwm-a3p8"
CORPUS = sys.argv[1]
OUT = Path(sys.argv[2])
MAXTOK = 32768
TEMP = 1.0
EXPECTED_SHA = "ba6431ca900cf41eb4f621cbf04d7f084187a16ec1ba6d1b8b6663967302d2e6"
API_KEY = open("/workspace/inkling2-small/.serve_key").read().strip()

assert hashlib.sha256(open(CORPUS, "rb").read()).hexdigest() == EXPECTED_SHA, "corpus sha mismatch"
rows = [json.loads(l) for l in open(CORPUS)]
assert len(rows) == 443
OUT.parent.mkdir(parents=True, exist_ok=True)

def chat(prompt, max_tokens=MAXTOK, temp=TEMP, tries=3):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp, "top_p": 1.0}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + API_KEY})
    last = None
    for attempt in range(tries):
        try:
            return json.load(urllib.request.urlopen(req, timeout=1800))
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise last

t = time.time()
d = chat("Reply with exactly: WARM-OK", 2048, 0.0)
print("warmup:", (d["choices"][0]["message"].get("content") or "")[:30], f"{time.time()-t:.0f}s", flush=True)

done_ids = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        try:
            r = json.loads(l)
            if r.get("id") and "error" not in r:
                done_ids.add(r["id"])
        except Exception:
            pass
todo = [r for r in rows if r["id"] not in done_ids]
print(f"resume: {len(done_ids)} done, {len(todo)} to go", flush=True)

def run_row(r):
    t0 = time.time()
    d = chat(r["prompt"])
    ch = d["choices"][0]; m = ch["message"]
    return {"id": r["id"], "dataset": r["dataset"], "label": r["label"],
            "prompt": r["prompt"], "content": m.get("content") or "",
            "reasoning": m.get("reasoning_content") or "",
            "finish_reason": ch["finish_reason"],
            "completion_tokens": d["usage"]["completion_tokens"],
            "elapsed_s": round(time.time() - t0, 1)}

done = err = trunc = 0
t0 = time.time()
with open(OUT, "a") as f, cf.ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(run_row, r): r["id"] for r in todo}
    for fut in cf.as_completed(futs):
        rid = futs[fut]
        try:
            rec = fut.result()
        except Exception as e:
            rec = {"id": rid, "error": str(e)[:200]}
            err += 1
        else:
            done += 1
            if rec.get("finish_reason") == "length":
                trunc += 1
        f.write(json.dumps(rec) + "\n"); f.flush()
        n = done + err
        if n % 10 == 0:
            print(f"[443] {n}/{len(todo)} done={done} err={err} trunc={trunc} elapsed={time.time()-t0:.0f}s", flush=True)

print(f"DONE ok={done} err={err} trunc={trunc} elapsed={time.time()-t0:.0f}s -> {OUT}", flush=True)