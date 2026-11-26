#!/usr/bin/env python3
"""Strict offline verifier for the public benchmark corpora.

No network, no writes. Enforces byte digests, row counts, schemas, the
smoke-50 prompt-sequence relation, license-file presence, and a namespace
allowlist. It intentionally contains no private identifiers; protected
operator-controlled material is absent by construction, and any attempt to
add it fails the allowlist below.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

try:
    from . import registry
    from .contracts import jsonl
except ImportError:
    import registry
    from contracts import jsonl

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "REGISTRY.yaml"

# Only these corpora may be published under benchmarks/corpora/. A new corpus
# requires an explicit registry entry (and a versioned id), never a silent add.
ALLOWED_FILES = frozenset(
    {
        "R1-FALSE-REFUSAL-443-v1.jsonl",
        "R1-SMOKE-50-v1.jsonl",
        "CYBER-REASONING-10-v1.jsonl",
    }
)
ALLOWED_IDS = frozenset({"R1-FALSE-REFUSAL-443-v1", "R1-SMOKE-50-v1", "CYBER-REASONING-10-v1"})
ALLOWED_FIELDS = {
    "R1-FALSE-REFUSAL-443-v1": frozenset({"id", "dataset", "prompt", "label"}),
    "R1-SMOKE-50-v1": frozenset({"dataset", "prompt", "label"}),
    "CYBER-REASONING-10-v1": frozenset({"id", "topic", "prompt"}),
}
ANSWER_LIKE = frozenset(
    {
        "response",
        "responses",
        "answer",
        "answers",
        "completion",
        "output",
        "score",
        "grade",
        "judgment",
        "rubric",
    }
)
# Generic structural guards only — no protected or private literals.
FORBIDDEN_FIELDS = frozenset(
    {"release_redline", "redline_holdout", "operator_only", "private", "secret"}
)
# Credential and private-infrastructure shapes only. Bare RFC1918 IPs are
# legitimate inside research prompts and are deliberately NOT banned.
SUSPICIOUS_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*\S+"),
    re.compile(r"(?i)(?:password|token|secret)\s*[:=]\s*\S+"),
    re.compile(r"/(?:home|opt|Volumes|Users)/[A-Za-z0-9_.-]+/"),
    re.compile(r"(?i)blackfrost_ai@\S+"),
)

EXPECTED_ROWS = {"R1-FALSE-REFUSAL-443-v1": 443, "R1-SMOKE-50-v1": 50, "CYBER-REASONING-10-v1": 10}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_registry() -> dict:
    return {"corpora": registry.load(ROOT)}


def load_jsonl(path: Path) -> list[dict]:
    return jsonl(path)


def check() -> list[str]:
    errors: list[str] = []
    try:
        reg = parse_registry()["corpora"]
    except (ValueError, OSError) as exc:
        return [str(exc)]
    if set(reg) != ALLOWED_IDS:
        errors.append(f"registry corpus ids drift: {sorted(reg)}")

    actual_files = {p.name for p in (ROOT / "corpora").glob("*.jsonl")}
    if actual_files != ALLOWED_FILES:
        errors.append(f"corpora file allowlist violated: {sorted(actual_files)}")

    for cid, entry in reg.items():
        path = registry.confined(ROOT, entry["file"])
        if not path.exists():
            errors.append(f"{cid}: missing {entry['file']}")
            continue
        digest = sha256(path)
        if digest != entry["sha256"]:
            errors.append(f"{cid}: sha256 {digest} != registry {entry['sha256']}")
        try:
            rows = load_jsonl(path)
            registry.validate_rows(cid, rows, entry)
        except (ValueError, OSError, UnicodeError) as exc:
            errors.append(str(exc))
            continue
        if len(rows) != EXPECTED_ROWS[cid] or len(rows) != entry["rows"]:
            errors.append(f"{cid}: {len(rows)} rows, expected {EXPECTED_ROWS[cid]}")
        for i, row in enumerate(rows, 1):
            keys = frozenset(row)
            if keys != ALLOWED_FIELDS[cid]:
                errors.append(f"{cid}:{i}: fields {sorted(keys)} != {sorted(ALLOWED_FIELDS[cid])}")
                break
            bad = keys & ANSWER_LIKE | keys & FORBIDDEN_FIELDS
            if bad:
                errors.append(f"{cid}:{i}: disallowed fields {sorted(bad)}")
                break
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                errors.append(f"{cid}:{i}: empty/non-string prompt")
                break
        # no protected namespace text or suspicious infra patterns anywhere in bytes
        raw = path.read_text(errors="replace")
        for pat in SUSPICIOUS_PATTERNS:
            if pat.search(raw):
                errors.append(f"{cid}: suspicious pattern {pat.pattern!r} present")
        for token in FORBIDDEN_FIELDS:
            if f'"{token}"' in raw:
                errors.append(f"{cid}: forbidden namespace token {token!r} present in rows")

    # smoke-50 is the first 50 prompt strings of the 443, in order.
    p443 = ROOT / "corpora" / "R1-FALSE-REFUSAL-443-v1.jsonl"
    p50 = ROOT / "corpora" / "R1-SMOKE-50-v1.jsonl"
    if p443.exists() and p50.exists():
        try:
            rows443, rows50 = load_jsonl(p443), load_jsonl(p50)
            a = [r.get("prompt") for r in rows443[:50]]
            b = [r.get("prompt") for r in rows50]
            if a != b:
                errors.append("smoke-50 prompt sequence != first 50 prompts of 443")
            # id fields must NOT be present in smoke rows (it is a prompt sequence)
            if any("id" in r for r in rows50):
                errors.append("smoke-50 rows unexpectedly carry id fields")
        except (ValueError, OSError, UnicodeError) as exc:
            errors.append(str(exc))

    for name in ("AdvBench-MIT.txt", "StrongREJECT-MIT.txt", "CC-BY-4.0-legalcode.txt"):
        f = ROOT / "LICENSES" / name
        if not f.exists() or not f.read_text().strip():
            errors.append(f"missing/empty license file: LICENSES/{name}")
        elif f.stat().st_size < 900:
            errors.append(f"license text suspiciously short: LICENSES/{name}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any finding")
    args = ap.parse_args()
    errors = check()
    if errors:
        print("BENCHMARK VALIDATION: FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1 if args.strict else 0
    print("BENCHMARK VALIDATION: PASS — corpora, registry, licenses, namespace")
    for name in sorted(ALLOWED_FILES):
        print(f"  {name}: {sha256(ROOT / 'corpora' / name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
