#!/usr/bin/env python3
"""Capture a hash-locked thinking-on Qwen3.6 refusal direction bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from locked_source import verify_source


REPO_ID = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
CHECKPOINT_DIGEST = "sha256:88760187ccb2560500b1d51f0e75610658c956be010b4477bc11bb8265c7413b"
BANK_SHA256 = "902ac2472d572ff5724e0b70012efe687c87d19c655f09cd528d8a9bca60f7a8"
EXPECTED_LAYERS = 40
EXPECTED_HIDDEN_SIZE = 2048


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_bank(path: Path, expected_sha256: str) -> tuple[list[str], list[str]]:
    observed = sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"bank sha256 mismatch: expected {expected_sha256}, got {observed}")
    harmful, safe = [], []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["label"] == "harmful":
            harmful.append(row["prompt"])
        elif row["label"] == "safe":
            safe.append(row["prompt"])
        else:
            raise ValueError(f"unknown label: {row.get('label')}")
    if len(harmful) != 32 or len(safe) != 32:
        raise ValueError(f"expected 32/32 bank, got {len(harmful)}/{len(safe)}")
    return harmful, safe


def layers_of(model):
    return model.model.language_model.layers


def templated_ids(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=True,
        return_tensors="pt",
    )
    if not torch.is_tensor(encoded):
        encoded = encoded["input_ids"]
    return encoded.to(device)


@torch.inference_mode()
def capture(model, tokenizer, prompts: list[str], device: torch.device, label: str) -> torch.Tensor:
    layers = layers_of(model)
    hits: list[torch.Tensor | None] = [None] * len(layers)

    def hook(index: int):
        def inner(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hits[index] = hidden[:, -1, :].detach().float().cpu()
        return inner

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    rows = []
    try:
        for index, prompt in enumerate(prompts, 1):
            ids = templated_ids(tokenizer, prompt, device)
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            if any(item is None for item in hits):
                raise RuntimeError("one or more language-layer hooks did not fire")
            rows.append(torch.cat([item for item in hits if item is not None], dim=0))
            print(f"CAPTURE {label} {index}/{len(prompts)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(rows, dim=0)


def direction_stats(harmful: torch.Tensor, safe: torch.Tensor) -> tuple[torch.Tensor, dict]:
    delta = harmful.mean(0) - safe.mean(0)
    separation = delta.norm(dim=1)
    if not torch.isfinite(separation).all() or (separation <= 0).any():
        raise RuntimeError("non-finite or zero direction separation")
    directions = delta / separation[:, None]
    split_a = harmful[::2].mean(0) - safe[::2].mean(0)
    split_b = harmful[1::2].mean(0) - safe[1::2].mean(0)
    split_cos = torch.nn.functional.cosine_similarity(split_a, split_b, dim=1)
    variance = harmful.var(0, unbiased=True) / harmful.shape[0]
    variance += safe.var(0, unbiased=True) / safe.shape[0]
    snr = separation / variance.sum(dim=1).sqrt().clamp_min(1.0e-12)
    selected = int(separation.argmax())
    centered = torch.cat([harmful[:, selected], safe[:, selected]], dim=0)
    centered -= centered.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    cumulative = energy.cumsum(0) / energy.sum().clamp_min(1.0e-12)
    rank90 = int((cumulative < 0.9).sum().item() + 1)
    report = {
        "selected_layer": selected,
        "separation_at_selected": separation[selected].item(),
        "direction_norm_before_normalization": separation[selected].item(),
        "separation_by_layer": separation.tolist(),
        "split_half_cosine_by_layer": split_cos.tolist(),
        "split_half_cosine_selected": split_cos[selected].item(),
        "snr_by_layer": snr.tolist(),
        "snr_selected": snr[selected].item(),
        "rank90_selected": rank90,
        "principal_energy_first5": (energy[:5] / energy.sum()).tolist(),
    }
    return directions, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-id", default="qwen36-35b-a10x3-recapture")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    bank = root / "assets" / "v1_corpus_hard32.jsonl"
    source_lock = root / "assets" / "source-lock.json"
    verified_source = verify_source(args.model, source_lock)
    if verified_source["checkpoint_digest"] != CHECKPOINT_DIGEST:
        raise RuntimeError("verified source identity changed")
    args.out.mkdir(parents=True, exist_ok=True)
    harmful, safe = load_bank(bank, BANK_SHA256)
    device = torch.device("cuda:0")
    started = datetime.now(timezone.utc).isoformat()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    template = tokenizer.chat_template
    template_bytes = (template if isinstance(template, str) else json.dumps(template, sort_keys=True)).encode()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device).eval()
    layers = layers_of(model)
    hidden_size = int(model.config.text_config.hidden_size)
    if len(layers) != EXPECTED_LAYERS or hidden_size != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(
            f"architecture mismatch: layers={len(layers)}, hidden={hidden_size}, "
            f"expected {EXPECTED_LAYERS}/{EXPECTED_HIDDEN_SIZE}"
        )
    print(f"MODEL {type(model).__name__} layers={len(layers)} hidden={hidden_size}", flush=True)

    harmful_acts = capture(model, tokenizer, harmful, device, "harmful")
    safe_acts = capture(model, tokenizer, safe, device, "safe")
    directions, stats = direction_stats(harmful_acts, safe_acts)

    tensors = {f"layer_{i}": directions[i].to(torch.bfloat16) for i in range(directions.shape[0])}
    direction_path = args.out / "directions_on.safetensors"
    activation_path = args.out / "capture-activations.safetensors"
    save_file(tensors, str(direction_path), metadata={
        "run_id": args.run_id,
        "model_revision": REVISION,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "bank_sha256": sha256(bank),
        "thinking": "on",
        "activation_site": "language_decoder_layer_output_last_prefill_token",
    })
    save_file({
        "harmful": harmful_acts.to(torch.bfloat16),
        "safe": safe_acts.to(torch.bfloat16),
    }, str(activation_path), metadata={"run_id": args.run_id, "thinking": "on"})

    provenance = {
        "schema": "ablagent.prompt-provenance.v1",
        "run_id": args.run_id,
        "bank_sha256": sha256(bank),
        "thinking": "on",
        "system_prompt": None,
        "injection_point": "request-supplied user message",
        "tokenization_boundary": "apply_chat_template(add_generation_prompt=True, enable_thinking=True)",
        "pooling": "last prefill token",
        "activation_site": "language decoder layer output",
        "truncation": "none",
        "chat_template_sha256": sha256_bytes(template_bytes),
    }
    provenance_path = args.out / "prompt-provenance.json"
    provenance_path.write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")

    report = {
        "schema": "ablagent.direction-report.v1",
        "run_id": args.run_id,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "model_path": str(args.model.resolve()),
        "model_revision": REVISION,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "bank_path": str(bank.resolve()),
        "bank_sha256": sha256(bank),
        "harmful_count": len(harmful),
        "safe_count": len(safe),
        "activation_shape": list(harmful_acts.shape),
        "prompt_condition": "bare system / thinking on",
        "chat_template_sha256": provenance["chat_template_sha256"],
        **stats,
    }
    report_path = args.out / "direction-report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")

    manifest = {
        "schema": "ablagent.s2-capture-manifest.v1",
        "run_id": args.run_id,
        "model_revision": REVISION,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "bank_sha256": sha256(bank),
        "thinking": "on",
        "files": [],
    }
    for role, path in (
        ("directions", direction_path),
        ("raw_activations", activation_path),
        ("prompt_provenance", provenance_path),
        ("direction_report", report_path),
    ):
        manifest["files"].append({"role": role, "path": path.name, "size": path.stat().st_size, "sha256": sha256(path)})
    (args.out / "capture-manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    print(json.dumps({k: report[k] for k in (
        "selected_layer", "separation_at_selected", "split_half_cosine_selected",
        "snr_selected", "rank90_selected")}, indent=2), flush=True)
    print("CAPTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
