#!/usr/bin/env python3
"""Explicit OpenAI-compatible collector. No redirect, resume, overwrite or telemetry."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from . import contracts as c
    from . import registry
    from .scoring_v2 import row_map, usable
except ImportError:
    import contracts as c
    import registry
    from scoring_v2 import row_map, usable

ROOT = Path(__file__).resolve().parent
COLLECTOR_VERSION = "blackfrost.benchmark-collector.v2"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect refused")


def endpoint_check(endpoint):
    try:
        url = urllib.parse.urlsplit(endpoint)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or any(ord(ch) <= 32 or ord(ch) == 127 for ch in endpoint)
        ):
            raise ValueError("invalid endpoint")
        _ = url.port
    except (ValueError, AttributeError) as exc:
        raise ValueError("invalid endpoint") from exc


def parse_headers(values):
    headers = {}
    for value in values:
        if ":" not in value:
            raise ValueError("invalid header")
        key, val = value.split(":", 1)
        key, val = key.strip().lower(), val.strip()
        if (
            not re.fullmatch(r"[a-z0-9!#$%&\'*+.^_`|~-]+", key)
            or key in headers
            or key in {"host", "content-length", "content-type", "transfer-encoding"}
            or any(ord(ch) < 32 or ord(ch) == 127 or ord(ch) > 255 for ch in val)
        ):
            raise ValueError("invalid or duplicate header")
        headers[key] = val
    token = os.environ.get("ABLAGENT_BENCHMARK_TOKEN")
    if token and "authorization" not in headers:
        if any(ord(ch) < 32 or ord(ch) > 126 for ch in token):
            raise ValueError("invalid token encoding")
        headers["authorization"] = f"Bearer {token}"
    return headers


def chat_completion(endpoint, payload, headers):
    endpoint_check(endpoint)
    req = urllib.request.Request(
        endpoint,
        data=c.canonical(payload),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    # Disable environment proxies: credentials go only to the explicit endpoint.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(req, timeout=600) as resp:
        if not 200 <= resp.status < 300:
            raise ValueError("non-success HTTP status")
        data = resp.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise ValueError("response exceeds byte limit")
        return c.loads(data.decode("utf-8"))


def extract(raw, model):
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("choices"), list)
        or len(raw["choices"]) != 1
    ):
        raise ValueError("invalid response shape")
    observed = raw.get("model")
    if observed is not None and (not isinstance(observed, str) or observed != model):
        raise ValueError("returned model mismatch")
    choice = raw["choices"][0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("invalid response choice")
    msg = choice["message"]
    answer = msg.get("content")
    reasoning = msg.get("reasoning_content", msg.get("reasoning", ""))
    if not isinstance(answer, str) or not isinstance(reasoning, str):
        raise ValueError("non-text response")
    finish = choice.get("finish_reason")
    if finish not in ("stop", "length"):
        raise ValueError("unsupported finish reason")
    return answer, reasoning, finish, observed


def collect(args):
    endpoint_check(args.endpoint)
    headers = parse_headers(args.header)
    if (
        not c.finite(args.temperature)
        or not 0 <= args.temperature <= 2
        or not c.finite(args.top_p)
        or not 0 < args.top_p <= 1
        or type(args.max_tokens) is not int
        or args.max_tokens < 1
        or not re.fullmatch("[0-9a-f]{64}", args.model_digest)
        or not args.run_id.strip()
        or not args.model.strip()
        or not args.chat_template.strip()
    ):
        raise ValueError("invalid collection settings/identity")
    rows = row_map(registry.rows(ROOT, args.corpus))
    corpus_sha = registry.load(ROOT)[args.corpus]["sha256"]
    settings = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "chat_template": args.chat_template,
        "thinking_mode": args.thinking_mode,
    }
    identity = {
        "run_id": args.run_id,
        "model_id": args.model,
        "model_digest": args.model_digest,
        "corpus_id": args.corpus,
        "corpus_sha256": corpus_sha,
    }
    completion = Path(str(args.out) + ".complete.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if completion.exists() or completion.is_symlink():
        raise ValueError("completion path already exists")
    errors = 0
    written_hash = hashlib.sha256()
    # Exclusive creation also refuses dangling symlinks. Interruptions leave incomplete output.
    with args.out.open("xb") as output:
        for position, (rid, row) in enumerate(rows.items()):
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": row["prompt"]}],
                **{k: settings[k] for k in ("temperature", "top_p", "max_tokens")},
            }
            if args.seed is not None:
                payload["seed"] = args.seed
            rec = {
                "schema": "blackfrost.benchmark-response.v2",
                **identity,
                "row_id": rid,
                "sequence": position,
                "requested_settings": settings,
                "observed_settings": {
                    "model_id": None,
                    "chat_template": None,
                    "thinking_mode": "unknown",
                },
                "collector_version": COLLECTOR_VERSION,
                "response_text": "",
                "answer_text": "",
                "reasoning_text": "",
                "finish_reason": "error",
                "status": "error",
            }
            try:
                raw = chat_completion(args.endpoint, payload, headers)
                answer, reasoning, finish, observed = extract(raw, args.model)
                rec.update(
                    response_text=c.canonical(raw).decode("utf-8"),
                    answer_text=answer,
                    reasoning_text=reasoning,
                    finish_reason=finish,
                    status="ok" if finish == "stop" else "truncated",
                )
                rec["observed_settings"]["model_id"] = observed
            except (OSError, ValueError, UnicodeError, RecursionError):
                # Never persist exception text, URL, headers or a malformed server body.
                print(f"COLLECT: row {position + 1} ERROR", file=sys.stderr)
            c.validate(rec, "responses")
            errors += not usable(rec)
            encoded = c.canonical(rec) + b"\n"
            output.write(encoded)
            written_hash.update(encoded)
            output.flush()
            os.fsync(output.fileno())
    if c.sha256(args.out) != written_hash.hexdigest():
        raise ValueError("response output changed during collection")
    receipt = {
        "schema": "blackfrost.benchmark-completion.v2",
        **identity,
        "responses_sha256": written_hash.hexdigest(),
        "rows": len(rows),
        "errors": errors,
        "complete": True,
        "release_eligible": False,
    }
    c.validate(receipt, "completion")
    with completion.open("xb") as output:
        output.write(c.canonical(receipt) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=sorted(registry.IDS))
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-digest", required=True, help="operator-asserted checkpoint SHA-256")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--chat-template", default="model-default", help="intent only; not enforced")
    ap.add_argument(
        "--thinking-mode",
        choices=["on", "off", "intrinsic", "unknown"],
        default="unknown",
        help="intent only; not enforced",
    )
    ap.add_argument("--header", action="append", default=[])
    args = ap.parse_args()
    try:
        receipt = collect(args)
    except (ValueError, OSError, UnicodeError):
        print(
            "COLLECT: FAIL — invalid input or IO failure; output may be incomplete", file=sys.stderr
        )
        return 1
    print(f"COLLECT: complete; sha256={receipt['responses_sha256']}; errors={receipt['errors']}")
    return 0 if receipt["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
