"""Conservative local secret-pattern scanner."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    # Keyword may stand alone (token=...) or be embedded in a longer env-var
    # style identifier (AWS_SECRET_ACCESS_KEY=...). Fail loud, not narrow.
    re.compile(
        r"(?i)\b[a-z0-9_-]*(?:api[_-]?key|password|secret|token)[a-z0-9_-]*\s*[:=]\s*['\"]?[^\s'\"]{12,}"
    ),
    # AWS secret access keys have a distinctive 40-char base64-ish shape.
    # Pure lowercase-hex 40-char runs are SHA-1/git commit ids (pinned HF
    # revisions, cache metadata), not credentials — excluded to avoid
    # false-positiving every real intake receipt. Real AWS keys are mixed-case
    # base64; an all-lowercase-hex key has probability ~(16/64)^40 = 4^-40.
    # Keyword-labeled assignments (api_key=<40 hex>) are still caught by the
    # keyword pattern above; only bare unlabeled 40-hex tokens are excluded.
    # The token must be a whole run of exactly 40 characters. Matching any
    # 40-char window of a longer run flags ordinary filesystem paths ("/" is in
    # the class and also a word boundary), e.g. the macOS per-user temp root
    # /private/var/folders/<2>/<30>/T/... recorded in run evidence. "=" may
    # precede the token so an unlabeled NAME=<key> assignment is still caught.
    # Boundaries use \w (Unicode-aware, like the \b they replace) plus the
    # non-word class members: an identifier tail such as cache_<40 alnum> or
    # café<40 alnum> is one longer run, not a key. The hex exclusion's inner
    # boundary mirrors the trailing boundary so <40 hex>_tail stays unflagged.
    re.compile(
        r"(?<!\w)(?<![/+])"
        r"(?![0-9a-f]{40}(?!\w)(?![/+=]))"
        r"[A-Za-z0-9/+=]{40}"
        r"(?!\w)(?![/+=])"
    ),
)


@dataclass(frozen=True)
class SecretFinding:
    path: str
    line: int
    pattern: str


def scan_text(text: str, *, path: str = "<text>") -> tuple[SecretFinding, ...]:
    findings: list[SecretFinding] = []
    for number, line in enumerate(text.splitlines(), 1):
        for pattern in PATTERNS:
            if pattern.search(line):
                findings.append(SecretFinding(path, number, pattern.pattern))
    return tuple(findings)


# Binary weight/media artifacts are not text secret carriers; skip by suffix
# and by size so a real run's HF snapshot doesn't get read as UTF-8 line noise.
BINARY_SUFFIXES = frozenset(
    {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
        ".onnx",
        ".npz",
        ".npy",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".gz",
        ".tar",
        ".whl",
        ".so",
        ".dylib",
        ".pyc",
    }
)
MAX_TEXT_BYTES = 2 * 1024 * 1024


def scan_tree(root: Path) -> tuple[SecretFinding, ...]:
    return scan_tree_report(root)[0]


def scan_tree_report(
    root: Path,
) -> tuple[tuple[SecretFinding, ...], tuple[dict[str, str], ...]]:
    """Scan a tree; also report UNscanned files so 'clean' never means 'unscanned'.

    Binary-suffix files, oversize files, and undecodable files are skipped for
    text scanning; each skip is returned with its reason for the evidence log.
    """
    findings: list[SecretFinding] = []
    unscanned: list[dict[str, str]] = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or ".git" in path.parts or ".venv" in path.parts:
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            unscanned.append({"path": str(path), "reason": "binary-suffix"})
            continue
        try:
            if path.stat().st_size > MAX_TEXT_BYTES:
                unscanned.append({"path": str(path), "reason": "over-size-cap"})
                continue
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            unscanned.append({"path": str(path), "reason": "undecodable-or-unreadable"})
            continue
        findings.extend(scan_text(text, path=str(path)))
    return tuple(findings), tuple(unscanned)
