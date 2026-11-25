"""Domain-pack manifest loading, validation, and pin verification.

A domain pack is the V2 unit of specialization: it declares how the generic
DWM pipeline applies to one commercial domain (cybersecurity, healthcare,
finance, logistics, ...). The platform ships a runnable toy pack plus domain
packs; each pack's guide (``packs/<name>/GUIDE.md``) documents how to fill in
the domain-specific corpora, probes, and release gates.

Security posture (fail closed):
  * YAML aliases/anchors-that-are-aliased are rejected at parse time, so a
    small manifest cannot expand into an unbounded object graph.
  * Every required scalar is type-checked *before* any coercion.
  * Corpus paths must be relative and resolve inside the pack directory;
    symlinks that escape the pack are refused.
  * Every referenced corpus must carry a 64-hex SHA-256 pin, and the pin is
    verified against the bytes on disk.
  * The manifest is read once; the parsed bytes and the recorded hash are the
    same bytes (no re-read window).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - hard dep of the platform
    raise RuntimeError(
        "PyYAML is required: pip install ablagent  (or pip install PyYAML>=6.0)"
    ) from exc

from .constants import PACK_MANIFEST_SCHEMA, STAGES

# Manifest fields required in every pack.
REQUIRED_FIELDS: tuple[str, ...] = (
    "schema",
    "pack_id",
    "domain",
    "version",
    "title",
    "summary",
    "vendor",
    "license",
    "stages",
)

# Fields that must be strings when present.
STRING_FIELDS: tuple[str, ...] = (
    *(f for f in REQUIRED_FIELDS if f != "stages"),
    "contact",
    "purpose",
)

# Accepted top-level keys. Unknown keys are rejected so a typo cannot silently
# disable a security-relevant block (e.g. corpora_sha256 / red_lines).
ALLOWED_FIELDS: frozenset[str] = frozenset(
    set(STRING_FIELDS) | {"stages", "corpora_sha256", "red_lines", "metadata", "research"}
)

# Vocabularies validated during load.
VALID_PURPOSES: tuple[str, ...] = (
    "specialize",  # domain specialization (cyber, health, finance, ...)
    "derisk",  # false-refusal removal (the V1 flagship)
)
VALID_STAGE_KEYS: tuple[str, ...] = (
    "purpose",
    "summary",
    "corpora",
    "probes",
    "gates",
    "scripts",
)

MAX_MANIFEST_BYTES = 4 * 1024 * 1024  # 4 MiB

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_PACK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
# Red-line enforcement is intentionally deferred until the experimental
# pipeline and quality gates are proven. Manifests may carry inert metadata,
# but the loader must not validate, merge, or apply red-lines in V2 experiments.


class PackError(ValueError):
    """A pack failed validation. Always a data error (CLI exit 1)."""


class _NoAliasLoader(yaml.SafeLoader):
    """SafeLoader that refuses alias nodes (bounded object graphs only)."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.events.AliasEvent):
            mark = self.peek_event().start_mark
            raise PackError(
                "YAML aliases are not allowed in pack manifests "
                f"(line {mark.line + 1}, column {mark.column + 1})"
            )
        return super().compose_node(parent, index)


@dataclass(frozen=True)
class DomainPack:
    """A validated view of one domain pack.

    ``raw`` is the parsed manifest; treat it as read-only. ``manifest_sha256``
    is the hash of the exact bytes that were parsed.
    """

    pack_id: str
    domain: str
    version: str
    title: str
    summary: str
    vendor: str
    license: str
    purpose: str
    manifest_path: Path
    manifest_sha256: str
    raw: dict[str, Any] = field(repr=False)
    pins_verified: bool = False

    @property
    def pack_dir(self) -> Path:
        return self.manifest_path.parent

    def stage_names(self) -> tuple[str, ...]:
        """Names of pipeline stages this pack customizes."""
        return tuple(self.raw.get("stages", {}).keys())

    def stage_config(self, stage: str) -> dict[str, Any]:
        """Domain-pack configuration for one pipeline stage (may be empty)."""
        cfg = self.raw.get("stages", {}).get(stage, {})
        if not isinstance(cfg, dict):
            raise PackError(f"pack {self.pack_id}: stage {stage} config must be a mapping")
        return cfg

    def corpora_ids(self) -> tuple[str, ...]:
        """IDs of all corpora declared anywhere in this pack's stages."""
        ids: list[str] = []
        for stage in self.stage_names():
            for entry in self.stage_config(stage).get("corpora", []):
                if isinstance(entry, dict):
                    cid = entry.get("corpus_id")
                    if cid:
                        ids.append(str(cid))
        return tuple(dict.fromkeys(ids))

    def corpus_entries(self) -> tuple[dict[str, Any], ...]:
        """Every declared corpus entry, deduplicated by (corpus_id, path)."""
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for stage in self.stage_names():
            for entry in self.stage_config(stage).get("corpora", []):
                key = (str(entry["corpus_id"]), str(entry["path"]))
                seen.setdefault(key, entry)
        return tuple(seen.values())

    def pins(self) -> dict[str, str]:
        return dict(self.raw.get("corpora_sha256", {}) or {})


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_manifest_bytes(path: Path) -> bytes:
    """Single read of the manifest, size-capped."""
    size = path.stat().st_size
    if size > MAX_MANIFEST_BYTES:
        raise PackError(f"pack manifest too large: {path} ({size} bytes)")
    return path.read_bytes()


def _parse_manifest(raw: bytes, source: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PackError(f"{source}: manifest is not valid UTF-8") from exc
    try:
        data = yaml.load(text, Loader=_NoAliasLoader)
    except PackError as exc:
        raise PackError(f"{source}: {exc}") from exc
    except (yaml.YAMLError, RecursionError) as exc:
        raise PackError(f"{source}: manifest is not parseable YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise PackError(f"{source}: pack manifest must be a YAML mapping")
    return data


def _require_str(data: dict[str, Any], key: str, source: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise PackError(f"{source}: '{key}' must be a non-empty string")
    return value


def _check_relative_path(value: Any, source: str, label: str) -> str:
    """A corpus/script path must be relative, clean, and inside the pack."""
    if not isinstance(value, str) or not value.strip():
        raise PackError(f"{source}: {label} must be a non-empty string path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or value.startswith("\\"):
        raise PackError(f"{source}: {label} must be relative to the pack dir: {value!r}")
    if ".." in pure.parts:
        raise PackError(f"{source}: {label} may not contain '..': {value!r}")
    if not pure.parts:
        raise PackError(f"{source}: {label} is empty")
    return value


def _resolve_inside(pack_dir: Path, rel: str, source: str, label: str) -> Path:
    """Resolve a pack-relative path, refusing symlink escapes."""
    root = pack_dir.resolve()
    target = (pack_dir / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise PackError(f"{source}: {label} resolves outside the pack directory: {rel!r}") from exc
    return target


def _typed_list(cfg: dict[str, Any], key: str, source: str, stage: str) -> list[Any]:
    value = cfg.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise PackError(
            f"{source}: stage '{stage}' field '{key}' must be a list, got {type(value).__name__}"
        )
    return value


def validate_manifest(
    data: dict[str, Any],
    source: str,
    pack_dir: Path | None = None,
) -> None:
    """Fail-closed manifest validation. Raises PackError on any problem.

    When ``pack_dir`` is given, corpus/script paths are also resolved and
    checked for containment (symlink escapes refused).
    """
    if not isinstance(data, dict):
        raise PackError(f"{source}: pack manifest must be a mapping")

    for fld in REQUIRED_FIELDS:
        if fld not in data:
            raise PackError(f"{source}: missing required field '{fld}'")

    unknown_fields = set(data.keys()) - ALLOWED_FIELDS
    if unknown_fields:
        raise PackError(f"{source}: unknown top-level field(s): {sorted(unknown_fields)}")

    for fld in STRING_FIELDS:
        if fld in data:
            _require_str(data, fld, source)

    if data["schema"] != PACK_MANIFEST_SCHEMA:
        raise PackError(
            f"{source}: schema must be {PACK_MANIFEST_SCHEMA!r}, got {data['schema']!r}"
        )
    if not _PACK_ID_RE.match(data["pack_id"]):
        raise PackError(
            f"{source}: pack_id must be lowercase alphanumeric with . _ - (got {data['pack_id']!r})"
        )

    purpose = data.get("purpose", "specialize")
    if purpose not in VALID_PURPOSES:
        raise PackError(f"{source}: purpose must be one of {list(VALID_PURPOSES)}")

    # Red-lines are inert metadata in this version. Do not validate or apply
    # them: doing so would alter experimental responses before quality gates
    # are proven. Enforcement belongs to the later security milestone.

    # --- corpus pins -----------------------------------------------------
    pins = data.get("corpora_sha256", {})
    if pins is None:
        pins = {}
    if not isinstance(pins, dict):
        raise PackError(f"{source}: 'corpora_sha256' must be a mapping of corpus_id -> sha256")
    for cid, digest in pins.items():
        if not isinstance(cid, str) or not cid.strip():
            raise PackError(f"{source}: corpus pin keys must be non-empty strings")
        if not isinstance(digest, str) or not _HEX64_RE.match(digest):
            raise PackError(
                f"{source}: corpus '{cid}' pin must be 64 lowercase hex chars (got {digest!r})"
            )

    # --- stages ----------------------------------------------------------
    if not isinstance(data["stages"], dict) or not data["stages"]:
        raise PackError(f"{source}: 'stages' must be a non-empty mapping")

    unknown_stages = set(data["stages"].keys()) - set(STAGES)
    if unknown_stages:
        raise PackError(f"{source}: unknown stage(s) in 'stages': {sorted(unknown_stages)}")

    referenced: set[str] = set()
    for stage, cfg in data["stages"].items():
        if not isinstance(cfg, dict):
            raise PackError(f"{source}: stage '{stage}' config must be a mapping")
        unknown_keys = set(cfg.keys()) - set(VALID_STAGE_KEYS)
        if unknown_keys:
            raise PackError(f"{source}: stage '{stage}' has unknown key(s): {sorted(unknown_keys)}")
        if "summary" in cfg:
            _require_str(cfg, "summary", f"{source}: stage '{stage}'")
        if "purpose" in cfg and cfg["purpose"] not in VALID_PURPOSES:
            raise PackError(
                f"{source}: stage '{stage}' purpose must be one of {list(VALID_PURPOSES)}"
            )

        for corpus in _typed_list(cfg, "corpora", source, stage):
            if not isinstance(corpus, dict):
                raise PackError(f"{source}: stage '{stage}' corpus entries must be mappings")
            if "corpus_id" not in corpus or "path" not in corpus:
                raise PackError(
                    f"{source}: stage '{stage}' corpus entries need 'corpus_id' and 'path'"
                )
            cid = _require_str(corpus, "corpus_id", f"{source}: stage '{stage}' corpus")
            rel = _check_relative_path(corpus["path"], source, f"corpus '{cid}' path")
            referenced.add(cid)
            if pack_dir is not None:
                _resolve_inside(pack_dir, rel, source, f"corpus '{cid}' path")

        for probe in _typed_list(cfg, "probes", source, stage):
            if not isinstance(probe, dict) or "probe_id" not in probe:
                raise PackError(
                    f"{source}: stage '{stage}' probe entries must be mappings with 'probe_id'"
                )
            _require_str(probe, "probe_id", f"{source}: stage '{stage}' probe")
            if "path" in probe:
                rel = _check_relative_path(probe["path"], source, "probe path")
                if pack_dir is not None:
                    _resolve_inside(pack_dir, rel, source, "probe path")

        for gate in _typed_list(cfg, "gates", source, stage):
            if not isinstance(gate, dict) or "gate_id" not in gate:
                raise PackError(
                    f"{source}: stage '{stage}' gate entries must be mappings with 'gate_id'"
                )
            _require_str(gate, "gate_id", f"{source}: stage '{stage}' gate")
            if "threshold" in gate and not isinstance(gate["threshold"], (int, float)):
                raise PackError(f"{source}: gate '{gate['gate_id']}' threshold must be numeric")

        for script in _typed_list(cfg, "scripts", source, stage):
            if not isinstance(script, dict) or "path" not in script:
                raise PackError(
                    f"{source}: stage '{stage}' script entries must be mappings with 'path'"
                )
            rel = _check_relative_path(script["path"], source, "script path")
            if "sha256" in script:
                digest = script["sha256"]
                if not isinstance(digest, str) or not _HEX64_RE.match(digest):
                    raise PackError(
                        f"{source}: script '{rel}' sha256 must be 64 lowercase hex chars"
                    )
            if "args" in script:
                args = script["args"]
                if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
                    raise PackError(f"{source}: script '{rel}' args must be a list of strings")
            if pack_dir is not None:
                _resolve_inside(pack_dir, rel, source, "script path")

    missing_pins = referenced - set(pins.keys())
    if missing_pins:
        raise PackError(
            f"{source}: corpora referenced without a corpora_sha256 pin: {sorted(missing_pins)}"
        )


def verify_corpus_pins(pack: DomainPack) -> tuple[dict[str, str], ...]:
    """Verify every declared corpus against its pin. Raises on any mismatch.

    Returns one receipt per corpus: corpus_id, path, sha256, rows.
    """
    receipts: list[dict[str, str]] = []
    pins = pack.pins()
    for entry in pack.corpus_entries():
        cid = str(entry["corpus_id"])
        rel = str(entry["path"])
        target = _resolve_inside(pack.pack_dir, rel, str(pack.manifest_path), f"corpus '{cid}'")
        if not target.is_file():
            raise PackError(f"pack {pack.pack_id}: corpus '{cid}' file missing: {rel}")
        actual = sha256_file(target)
        expected = pins.get(cid)
        if expected is None:
            raise PackError(f"pack {pack.pack_id}: corpus '{cid}' has no pin")
        if actual != expected:
            raise PackError(
                f"pack {pack.pack_id}: corpus '{cid}' SHA-256 mismatch "
                f"(expected {expected}, got {actual}) - refusing to load"
            )
        receipts.append(
            {
                "corpus_id": cid,
                "path": rel,
                "sha256": actual,
                "bytes": str(target.stat().st_size),
            }
        )
    return tuple(receipts)


def load_pack(pack_dir: Path, *, verify_pins: bool = True) -> DomainPack:
    """Load and validate the pack rooted at ``pack_dir`` (needs PACK.yaml).

    Corpus pins are verified by default; pass ``verify_pins=False`` only for
    read-only inspection of a pack whose corpora are intentionally absent.
    """
    pack_dir = Path(pack_dir)
    manifest = pack_dir / "PACK.yaml"
    if manifest.is_symlink():
        raise PackError(f"pack manifest may not be a symlink: {manifest}")
    if not manifest.is_file():
        raise FileNotFoundError(f"no PACK.yaml in {pack_dir}")
    root = pack_dir.resolve()
    if manifest.resolve().parent != root:
        raise PackError(f"pack manifest resolves outside the pack directory: {manifest}")

    raw = _read_manifest_bytes(manifest)
    data = _parse_manifest(raw, source=str(manifest))
    validate_manifest(data, source=str(manifest), pack_dir=pack_dir)

    pack = DomainPack(
        pack_id=data["pack_id"],
        domain=data["domain"],
        version=data["version"],
        title=data["title"],
        summary=data["summary"],
        vendor=data["vendor"],
        license=data["license"],
        purpose=data.get("purpose", "specialize"),
        manifest_path=manifest,
        manifest_sha256=sha256_bytes(raw),
        pins_verified=False,
        raw=data,
    )
    if verify_pins:
        verify_corpus_pins(pack)
        object.__setattr__(pack, "pins_verified", True)
    return pack


def discover_packs(packs_root: Path, *, verify_pins: bool = True) -> dict[str, DomainPack]:
    """Discover every valid pack under ``packs_root`` (one level deep).

    Returns a mapping of pack_id -> DomainPack. Directories without a
    PACK.yaml are ignored (not an error) so operator working dirs can live
    beside packs. A manifest that fails validation raises.
    """
    packs_root = Path(packs_root)
    packs: dict[str, DomainPack] = {}
    if not packs_root.is_dir():
        return packs
    root = packs_root.resolve()
    for entry in sorted(packs_root.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink():
            raise PackError(f"pack directory may not be a symlink: {entry}")
        if not entry.is_dir():
            continue
        if entry.resolve().parent != root:
            raise PackError(f"pack directory resolves outside the packs root: {entry}")
        if not (entry / "PACK.yaml").is_file():
            continue
        pack = load_pack(entry, verify_pins=verify_pins)
        if pack.pack_id in packs:
            raise PackError(f"duplicate pack_id: {pack.pack_id}")
        packs[pack.pack_id] = pack
    return packs
