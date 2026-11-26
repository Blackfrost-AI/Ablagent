"""Structured registry loader shared by all benchmark tools."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

try:
    from .contracts import jsonl, sha256
except ImportError:
    from contracts import jsonl, sha256

IDS = {"R1-FALSE-REFUSAL-443-v1": 443, "R1-SMOKE-50-v1": 50, "CYBER-REASONING-10-v1": 10}


class UniqueLoader(yaml.SafeLoader):
    pass


def mapping(loader, node):
    result = {}
    for key_node, val_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ValueError("invalid or duplicate registry key")
        result[key] = loader.construct_object(val_node, deep=True)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)


def confined(root, relative):
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("invalid corpus path")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("corpus path escapes root")
    return resolved


def load(root):
    import re

    try:
        raw = yaml.load((root / "REGISTRY.yaml").read_text(encoding="utf-8"), Loader=UniqueLoader)
        expected = {
            "schema",
            "version",
            "description",
            "boundary",
            "operator_standard",
            "corpora",
            "licenses",
            "protocols",
            "tooling",
            "exclusions",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("invalid registry fields")
        if raw["schema"] != "blackfrost.benchmark-registry.v1":
            raise ValueError("invalid registry schema")
        if not isinstance(raw["version"], (str, date)):
            raise ValueError("invalid registry version")
        for field in ("description", "boundary"):
            if not isinstance(raw[field], str) or not raw[field].strip():
                raise ValueError("invalid registry metadata")
        for field in ("operator_standard", "protocols", "tooling"):
            if not isinstance(raw[field], dict):
                raise ValueError("invalid registry mapping")
        for field in ("licenses", "exclusions", "corpora"):
            if not isinstance(raw[field], list) or not raw[field]:
                raise ValueError("invalid registry list")

        def text_map(value, keys):
            if set(value) != set(keys) or any(
                not isinstance(v, str) or not v.strip() for v in value.values()
            ):
                raise ValueError("invalid registry metadata fields")

        text_map(raw["operator_standard"], ("capture_thinking_mode", "grading_region", "denial"))
        text_map(raw["tooling"], ("validate", "score", "collect", "ci"))
        protocols = raw["protocols"]
        if set(protocols) != {"scoring", "runs", "schemas"}:
            raise ValueError("invalid registry protocols")
        if any(
            not isinstance(protocols[k], str) or not protocols[k].strip()
            for k in ("scoring", "runs")
        ):
            raise ValueError("invalid protocol path")
        if (
            not isinstance(protocols["schemas"], list)
            or not protocols["schemas"]
            or any(not isinstance(v, str) or not v.strip() for v in protocols["schemas"])
        ):
            raise ValueError("invalid protocol schemas")
        for license_entry in raw["licenses"]:
            if not isinstance(license_entry, dict):
                raise ValueError("invalid registry license")
            text_map(license_entry, ("name", "license", "file"))
        if any(not isinstance(v, str) or not v.strip() for v in raw["exclusions"]):
            raise ValueError("invalid exclusions")
        # Metadata paths are descriptive only and are never dereferenced here.
        entries = {}
        for entry in raw["corpora"]:
            required = {
                "id",
                "file",
                "sha256",
                "rows",
                "fields",
                "composition",
                "stage",
                "split",
                "provenance",
            }
            if (
                not isinstance(entry, dict)
                or not required <= entry.keys()
                or entry.keys() - required - {"usage_rule", "relation"}
            ):
                raise ValueError("invalid corpus registry fields")
            cid = entry["id"]
            if not isinstance(cid, str) or cid not in IDS or cid in entries:
                raise ValueError("unknown or duplicate corpus id")
            if type(entry["rows"]) is not int or entry["rows"] != IDS[cid]:
                raise ValueError("invalid corpus count")
            if not isinstance(entry["sha256"], str) or not re.fullmatch(
                "[0-9a-f]{64}", entry["sha256"]
            ):
                raise ValueError("invalid corpus digest")
            confined(root, entry["file"])
            if entry["file"] != f"corpora/{cid}.jsonl":
                raise ValueError("noncanonical corpus path")
            expected_fields = (
                ["id", "topic", "prompt"]
                if cid.startswith("CYBER")
                else ["id", "dataset", "prompt", "label"]
            )
            if cid.startswith("R1-SMOKE"):
                expected_fields.remove("id")
            if entry["fields"] != expected_fields or not isinstance(entry["composition"], dict):
                raise ValueError("invalid corpus fields/composition")
            for field in ("stage", "split", "provenance", "usage_rule", "relation"):
                if field in entry and (
                    not isinstance(entry[field], str) or not entry[field].strip()
                ):
                    raise ValueError("invalid corpus metadata")
            entries[cid] = entry
        if set(entries) != set(IDS):
            raise ValueError("registry corpus ids drift")
        return entries
    except (yaml.YAMLError, TypeError, RecursionError, OSError, UnicodeError) as exc:
        raise ValueError("invalid registry") from exc


def rows(root, cid):
    entries = load(root)
    if cid not in entries:
        raise ValueError("unknown corpus id")
    entry = entries[cid]
    path = confined(root, entry["file"])
    if sha256(path) != entry["sha256"]:
        raise ValueError("corpus sha256 mismatch")
    result = jsonl(path)
    validate_rows(cid, result, entry)
    return result


def validate_rows(cid, records, entry):
    from collections import Counter

    if len(records) != IDS[cid]:
        raise ValueError("corpus row count mismatch")
    seen = set()
    composition = Counter()
    labels = Counter()
    for n, row in enumerate(records, 1):
        if set(row) != set(entry["fields"]) or any(
            not isinstance(v, str) or not v.strip() for v in row.values()
        ):
            raise ValueError("invalid corpus record")
        if "id" in row:
            expected = f"cyb-{n:02d}" if cid.startswith("CYBER") else f"r1fr443-{n:04d}"
            if row["id"] in seen or row["id"] != expected:
                raise ValueError("invalid or duplicate row id")
            seen.add(row["id"])
        if "dataset" in row:
            ds = row["dataset"]
            allowed = {
                "advbench": {"harmful"},
                "strongreject": {"harmful"},
                "xstest": {"safe", "unsafe"},
            }
            if ds not in allowed or row["label"] not in allowed[ds]:
                raise ValueError("invalid dataset/label")
            composition[ds] += 1
            labels[(ds, row["label"])] += 1
    expected = (
        {"security_reasoning_probes": 10}
        if cid.startswith("CYBER")
        else {
            ds: {
                "rows": count,
                **(
                    {"labels": {label: labels[(ds, label)] for label in ("safe", "unsafe")}}
                    if ds == "xstest"
                    else {"label": "harmful"}
                ),
                **(
                    {"license": "CC-BY-4.0" if ds == "xstest" else "MIT"}
                    if cid == "R1-FALSE-REFUSAL-443-v1"
                    else {}
                ),
            }
            for ds, count in composition.items()
        }
    )
    if expected != entry["composition"]:
        raise ValueError("corpus composition mismatch")
