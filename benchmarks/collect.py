#!/usr/bin/env python3
"""Explicit, network-isolated collector for frozen benchmark responses.

This is the ONLY tool in benchmarks/ that may touch the network. It requires
an operator-supplied OpenAI-compatible chat-completions endpoint, refuses to
overwrite an existing output, and never modifies the prompt corpora. All other
benchmark tooling is strictly offline.

No endpoint default, no telemetry, no background behavior. The endpoint is
used only for the exact rows of the selected corpus. Set credentials through
your own environment; this tool reads no API keys from files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPORA = ROOT / "corpora"
CORPUS_IDS = ("R1-FALSE-REFUSAL-443-v1", "R1-SMOKE-50-v1", "CYBER-REASONING-10-v1")
COLLECTOR_VERSION = "blackfrost.benchmark-collector.v1"


def corpus_sha(corpus_id: str) -> str:
    lines = (ROOT / "REGISTRY.yaml").read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"- id: {corpus_id}":
            for nxt in lines[i + 1 : i + 8]:
                if nxt.strip().startswith("sha256:"):
                    return nxt.strip().split("sha256:", 1)[1].strip()
    raise ValueError(f"no sha256 for {corpus_id!r}")


def row_id(corpus_id: str, row: dict, position: int) -> str:
    return row.get("id") or f"smoke-{position}" if corpus_id == "R1-SMOKE-50-v1" else row["id"]


def chat_completion(endpoint: str, payload: dict, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    # The endpoint is operator-supplied and required explicitly; no defaults.
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", required=True, choices=CORPUS_IDS)
    ap.add_argument(
        "--endpoint", required=True, help="operator's OpenAI-compatible chat/completions URL"
    )
    ap.add_argument("--model", required=True, help="model id served at the endpoint")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--chat-template", default="model-default")
    ap.add_argument(
        "--thinking-mode", choices=["on", "off", "intrinsic", "unknown"], default="unknown"
    )
    ap.add_argument(
        "--model-digest", default="", help="operator-recorded checkpoint digest, when known"
    )
    ap.add_argument(
        "--header", action="append", default=[], help="extra request header 'Name: value'"
    )
    args = ap.parse_args()

    corpus_path = CORPORA / f"{args.corpus}.jsonl"
    if hashlib.sha256(corpus_path.read_bytes()).hexdigest() != corpus_sha(args.corpus):
        print("COLLECT: FAIL — corpus digest drift; run validate.py first", file=sys.stderr)
        return 1
    if args.out.exists():
        print(f"COLLECT: FAIL — refusing to overwrite {args.out}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    headers: dict[str, str] = {}
    for h in args.header:
        if ":" not in h:
            ap.error(f"--header needs 'Name: value', got {h!r}")
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()
    token = os.environ.get("ABLAGENT_BENCHMARK_TOKEN")
    if token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {token}"

    decoding = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "chat_template": args.chat_template,
        "thinking_mode": args.thinking_mode,
    }
    rows = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    out_lines: list[str] = []
    for position, row in enumerate(rows, 1):
        rid = row_id(args.corpus, row, position)
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": row["prompt"]}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
        if args.seed is not None:
            payload["seed"] = args.seed
        try:
            raw = chat_completion(args.endpoint, payload, headers)
            choice = (raw.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            finish = choice.get("finish_reason") or "unknown"
            rec = {
                "schema": "blackfrost.benchmark-response.v1",
                "corpus_id": args.corpus,
                "corpus_sha256": corpus_sha(args.corpus),
                "row_id": rid,
                "response_text": json.dumps(raw, ensure_ascii=False),
                "reasoning_text": reasoning,
                "answer_text": text,
                "finish_reason": finish
                if finish in {"stop", "length", "error", "unknown", "timeout"}
                else "unknown",
                "status": "ok"
                if finish == "stop"
                else ("truncated" if finish == "length" else "error"),
                "decoding": {k: v for k, v in decoding.items() if v is not None},
                "model": args.model,
                "model_digest": args.model_digest,
                "endpoint_local_reference": "operator-supplied",
                "collector_version": COLLECTOR_VERSION,
                "captured_at": datetime.now(UTC).isoformat(),
                "sequence": position - 1,
            }
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            rec = {
                "schema": "blackfrost.benchmark-response.v1",
                "corpus_id": args.corpus,
                "corpus_sha256": corpus_sha(args.corpus),
                "row_id": rid,
                "response_text": "",
                "reasoning_text": "",
                "answer_text": "",
                "finish_reason": "error",
                "status": "error",
                "decoding": {k: v for k, v in decoding.items() if v is not None},
                "model": args.model,
                "model_digest": args.model_digest,
                "endpoint_local_reference": "operator-supplied",
                "collector_version": COLLECTOR_VERSION,
                "captured_at": datetime.now(UTC).isoformat(),
                "sequence": position - 1,
            }
            print(
                f"COLLECT: row {rid} failed: {type(exc).__name__} — recorded as ERROR",
                file=sys.stderr,
            )
        out_lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))
        print(f"COLLECT: {position}/{len(rows)}", file=sys.stderr)

    args.out.write_text("\n".join(out_lines) + "\n")
    out_digest = hashlib.sha256(args.out.read_bytes()).hexdigest()[:16]
    print(f"COLLECT: wrote {args.out} ({len(out_lines)} rows, sha256 {out_digest}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
