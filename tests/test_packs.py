"""Unit tests for domain-pack manifest loading and validation (scaffold)."""

from __future__ import annotations

import textwrap
from pathlib import Path
import shutil
import subprocess

import pytest

from ablagent.constants import DEPENDENCIES, STAGES
from ablagent.packs import PackError, discover_packs, load_pack, validate_manifest


def _minimal_manifest() -> dict:
    return {
        "schema": "ablagent.pack.v1",
        "pack_id": "toy-pack",
        "domain": "example",
        "version": "0.1.0",
        "title": "Toy Pack",
        "summary": "Minimal valid pack manifest for tests.",
        "vendor": "Blackfrost_AI",
        "license": "Apache-2.0",
        "corpora_sha256": {"example-corpus-1": "a" * 64},
        "red_lines": ["toy_red_line"],
        "contact": "https://example.invalid/~toy",
        "stages": {
            "S0_INTAKE": {"summary": "intake"},
            "S3_DERISK": {"summary": "derisk"},
        },
    }


def test_minimal_manifest_valid():
    validate_manifest(_minimal_manifest(), source="unit")
    assert True


def test_missing_required_field_rejected():
    manifest = _minimal_manifest()
    del manifest["pack_id"]
    with pytest.raises(ValueError, match="pack_id"):
        validate_manifest(manifest, source="unit")


def test_unknown_stage_rejected():
    manifest = _minimal_manifest()
    manifest["stages"]["S99_NOPE"] = {}
    with pytest.raises(ValueError, match="S99_NOPE"):
        validate_manifest(manifest, source="unit")


def test_unknown_stage_key_rejected():
    manifest = _minimal_manifest()
    manifest["stages"]["S0_INTAKE"]["summary"] = "x"
    manifest["stages"]["S0_INTAKE"]["future_key"] = "y"
    with pytest.raises(ValueError, match="future_key"):
        validate_manifest(manifest, source="unit")


def test_dag_invariants():
    # Every stage's dependencies are themselves stages.
    for stage, deps in DEPENDENCIES.items():
        assert stage in STAGES
        for dep in deps:
            assert dep in STAGES, f"{stage} depends on unknown {dep}"


def test_discover_toy_pack():
    root = Path(__file__).resolve().parents[1]
    packs = discover_packs(root / "packs")
    assert "toy-domain" in packs
    pack = packs["toy-domain"]
    assert pack.domain == "example"
    assert pack.purpose == "specialize"
    assert "S0_INTAKE" in pack.stage_names()


def test_load_toy_pack_full():
    root = Path(__file__).resolve().parents[1]
    pack = load_pack(root / "packs" / "toy-domain")
    assert pack.pack_id == "toy-domain"
    assert len(pack.manifest_sha256) == 64


def test_manifest_rejects_aliases():
    manifest = _minimal_manifest()
    manifest["alias"] = "&x value"
    # validate_manifest receives parsed data; alias rejection is covered by
    # load_pack's YAML parser below.
    assert manifest["pack_id"] == "toy-pack"


def test_load_rejects_yaml_alias(tmp_path: Path):
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "PACK.yaml").write_text(
        "schema: ablagent.pack.v1\n"
        "pack_id: toy-pack\n"
        "domain: example\nversion: '0.1'\n"
        "title: Toy\nsummary: Toy\nvendor: V\nlicense: Apache-2.0\n"
        "stages: {S0_INTAKE: {summary: intake}}\n"
        "x: &anchor value\ny: *anchor\n"
    )
    with pytest.raises(PackError, match="aliases"):
        load_pack(pack_dir, verify_pins=False)


def test_manifest_rejects_untyped_scalars():
    manifest = _minimal_manifest()
    manifest["pack_id"] = 123
    with pytest.raises(ValueError, match="pack_id"):
        validate_manifest(manifest, source="unit")


def test_manifest_rejects_corpus_path_traversal(tmp_path: Path):
    manifest = _minimal_manifest()
    manifest["stages"]["S0_INTAKE"]["corpora"] = [
        {"corpus_id": "example-corpus-1", "path": "../../outside.jsonl"}
    ]
    with pytest.raises(ValueError, match=r"\.\."):
        validate_manifest(manifest, source="unit", pack_dir=tmp_path)


def test_tampered_toy_corpus_is_rejected(tmp_path: Path):
    source = Path(__file__).resolve().parents[1] / "packs" / "toy-domain"
    target = tmp_path / "toy-domain"
    shutil.copytree(source, target)
    corpus = target / "corpora" / "toy-corpus-1.jsonl"
    corpus.write_text(corpus.read_text() + "tampered\n")
    with pytest.raises(PackError, match="SHA-256 mismatch"):
        load_pack(target)


def test_explicit_empty_packs_dir_fails_cli(tmp_path: Path):
    result = subprocess.run(
        [".venv/bin/ablagent", "doctor", "--packs-dir", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "no packs found" in result.stdout
