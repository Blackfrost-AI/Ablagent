#!/usr/bin/env python3
"""cyber10 probe vs inkling2-small-dwm-a3p8 serve (the validated runtime, port 40001)."""
import json, sys, time, urllib.request, urllib.error
BASE = "http://127.0.0.1:40001/v1"
MODEL = "inkling2-small-dwm-a3p8"
API_KEY = open("/workspace/inkling2-small/.serve_key").read().strip()
IN = sys.argv[1] if len(sys.argv) > 1 else "/workspace/inkling2-small/input/cyber_probe_10.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/inkling2-small/evidence/cyber10-a3p8/rows.jsonl"
BUDGET = 32768
TEMP = 0.0

def chat(msg_id, prompt):
    body = json.dumps({"model": MODEL, "temperature": TEMP, "max_tokens": BUDGET,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + API_KEY})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            j = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"id": msg_id, "error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    ch = j["choices"][0]
    msg = ch.get("message", {})
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    return {"id": msg_id, "prompt": prompt, "content": content, "reasoning": reasoning,
            "finish_reason": ch.get("finish_reason"), "elapsed_s": round(time.time() - t0, 1),
            "usage": j.get("usage", {})}

rows = [json.loads(l) for l in open(IN)]
done = set()
try:
    for l in open(OUT):
        done.add(json.loads(l)["id"])
except FileNotFoundError:
    pass
with open(OUT, "a") as out:
    for r in rows:
        if r["id"] in done:
            continue
        rec = chat(r["id"], r["prompt"] if "prompt" in r else r["task"])
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"[{r['id']}] finish={rec.get('finish_reason')} len={len(rec.get('content',''))} err={rec.get('error','')}", flush=True)
print("CYBER10_A3P0_DONE")
