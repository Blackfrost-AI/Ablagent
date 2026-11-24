"""Offline integrity checks for the vendored Inkling-2-Mini-Derisked example.

These tests never download anything. They recompute SHA-256 of every vendored
file against examples/inkling-2-mini-derisked/PROVENANCE.json (pinned to HF
revision c352150aacc7d0ca4f6e4f4994cfe39b1077daa8) and check the kit's internal
hash chain. The fetch-only raw-capture.npz is verified only when present.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "examples" / "inkling-2-mini-derisked"
MANIFEST_PATH = EXAMPLE_DIR / "PROVENANCE.json"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_example_provenance.py"

DIRECTIONS_SHA = "6674ddc8f6e8762c10f14aa37fdcef914186fda1276804f37d7bd30365fb1b74"
HARD32_SHA = "902ac2472d572ff5724e0b70012efe687c87d19c655f09cd528d8a9bca60f7a8"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_manifest_exists_and_pins_revision(manifest: dict) -> None:
    assert manifest["schema"] == "ablagent.example-provenance.v1"
    assert manifest["source"]["revision"] == "c352150aacc7d0ca4f6e4f4994cfe39b1077daa8"
    assert manifest["source"]["repository"].endswith("Blackfrost-AI/Inkling-2-Mini-Derisked")


def test_every_vendored_file_matches_manifest(manifest: dict) -> None:
    assert len(manifest["files"]) == 15
    for entry in manifest["files"]:
        path = EXAMPLE_DIR / entry["path"]
        assert path.is_file(), f"missing vendored file: {entry['path']}"
        assert path.stat().st_size == entry["size"], entry["path"]
        assert _sha256(path) == entry["sha256"], entry["path"]


def test_directions_hash_chain(manifest: dict) -> None:
    directions = EXAMPLE_DIR / "DWM_RECIPE" / "s2-capture" / "directions.npz"
    assert _sha256(directions) == DIRECTIONS_SHA
    summary = json.loads(
        (EXAMPLE_DIR / "DWM_RECIPE" / "s2-capture" / "s2-summary.json").read_text()
    )
    receipt = json.loads((EXAMPLE_DIR / "DWM_RECIPE" / "s3-edit-receipt-a25.json").read_text())
    assert summary["directions_sha256"] == f"sha256:{DIRECTIONS_SHA}"
    assert receipt["directions_sha256"] == f"sha256:{DIRECTIONS_SHA}"
    # The receipt binds the PRE-sanitization battery digest; the shipped
    # battery differs only by one sanitized path field (documented erratum).
    assert receipt["battery_sha256"] == (
        "sha256:1ec4212bde7f6b990f6c96407cf389ae6dd20bced098b13cd350129bedf49af8"
    )


def test_hard32_corpus_hash_chain() -> None:
    corpus = EXAMPLE_DIR / "DWM_RECIPE" / "scripts" / "v1_corpus_hard32.jsonl"
    assert _sha256(corpus) == HARD32_SHA
    meta = json.loads(
        (EXAMPLE_DIR / "DWM_RECIPE" / "s2-capture" / "capture-metadata.json").read_text()
    )
    assert meta["corpus_sha256"] == f"sha256:{HARD32_SHA}"
    assert meta["hard32_corpus_sha256"] == f"sha256:{HARD32_SHA}"
    rows = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    assert len(rows) == 64  # 32 adjacent harmful/safe pairs
    assert {r["label"] for r in rows} == {"harmful", "safe"}


def test_receipt_binds_shipped_weight_shard() -> None:
    receipt = json.loads((EXAMPLE_DIR / "DWM_RECIPE" / "s3-edit-receipt-a25.json").read_text())
    assert receipt["alpha"] == 2.5
    assert receipt["layers"] == list(range(9, 19))
    digest = receipt["candidate_file_digests"]["model-00001-of-00001.safetensors"]
    assert digest == ("sha256:b1156b5d53ddabf927a8e3f209eebc5da610d2b90ed8c14a6a945b78080bef21")


def test_directions_geometry() -> None:
    np = pytest.importorskip("numpy")
    z = np.load(EXAMPLE_DIR / "DWM_RECIPE" / "s2-capture" / "directions.npz")
    assert z["directions"].shape == (48, 2304)
    norms = np.linalg.norm(z["directions"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert z["layers"].shape == (48,) and z["visits"].shape == (48,)


def test_verifier_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fetch_only_asset_verified_when_present(manifest: dict) -> None:
    entry = manifest["fetch_only"][0]
    path = EXAMPLE_DIR / entry["path"]
    if not path.is_file():
        pytest.skip("raw-capture.npz not downloaded (fetch-only asset)")
    assert path.stat().st_size == entry["size"]
    assert _sha256(path) == entry["sha256"]
