#!/usr/bin/env python3
"""Sequential persisted-source versus Qwen3.6 DWM measurement.

Only one 35B model is resident at a time.  Baseline logits are retained on CPU,
the source model is released, and the persisted candidate is loaded and
measured against the exact same frozen token sequences.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from capture_directions import capture, direction_stats, load_bank
from locked_source import CHECKPOINT_DIGEST, verify_source


BANK_SHA256 = "902ac2472d572ff5724e0b70012efe687c87d19c655f09cd528d8a9bca60f7a8"
HELDOUT_SHA256 = "8b9c4695422caaac2854c250c3e874f263b2ac820c08c5c3ac8ef9ed2d237309"
BASELINE_ACTIVATIONS_SHA256 = "2ef775b19e30a90b73116baba69942549ef176a232d7b38062b6acde3fa95d00"
BASELINE_DIRECTIONS_SHA256 = "fc2e01feace4215a2c27bc84bc7a7c6d576013ab358671870db7357ccc07d9b2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ids_for(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    value = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=True,
        return_tensors="pt",
    )
    if not torch.is_tensor(value):
        value = value["input_ids"]
    return value.to(device)


def load_model(path: Path, device: torch.device):
    return AutoModelForImageTextToText.from_pretrained(
        path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device).eval()


@torch.inference_mode()
def baseline_logits(model, tokenizer, prompts: list[dict], device: torch.device) -> list[dict]:
    rows = []
    for index, row in enumerate(prompts, 1):
        ids = ids_for(tokenizer, row["prompt"], device)
        logits = model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).logits[:, :-1].float().cpu()
        rows.append({"id": row["id"], "targets": ids[:, 1:].cpu(), "logits": logits})
        print(f"BASELINE {index}/{len(prompts)}", flush=True)
    return rows


@torch.inference_mode()
def compare_candidate(model, tokenizer, prompts: list[dict], baseline: list[dict],
                      device: torch.device) -> dict:
    prompt_rows, token_rows = [], []
    accum = {name: [] for name in (
        "base_nll", "edited_nll", "forward_kl", "reverse_kl", "js", "tv",
        "base_entropy", "edited_entropy",
    )}
    top1_equal = token_count = 0
    for index, (row, base_row) in enumerate(zip(prompts, baseline, strict=True), 1):
        if row["id"] != base_row["id"]:
            raise RuntimeError("baseline prompt order mismatch")
        ids = ids_for(tokenizer, row["prompt"], device)
        targets = ids[:, 1:].cpu()
        if not torch.equal(targets, base_row["targets"]):
            raise RuntimeError(f"tokenization changed for {row['id']}")
        edited_logits = model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).logits[:, :-1].float().cpu()
        base_logits = base_row["logits"]
        bp = torch.softmax(base_logits, dim=-1)
        ep = torch.softmax(edited_logits, dim=-1)
        bl = torch.log_softmax(base_logits, dim=-1)
        el = torch.log_softmax(edited_logits, dim=-1)
        mix = 0.5 * (bp + ep)
        ml = mix.clamp_min(1e-30).log()
        bnll = -bl.gather(-1, targets[..., None]).squeeze(-1)
        enll = -el.gather(-1, targets[..., None]).squeeze(-1)
        fkl = (bp * (bl - el)).sum(-1)
        rkl = (ep * (el - bl)).sum(-1)
        js = 0.5 * ((bp * (bl - ml)).sum(-1) + (ep * (el - ml)).sum(-1))
        tv = 0.5 * (bp - ep).abs().sum(-1)
        be = -(bp * bl).sum(-1)
        ee = -(ep * el).sum(-1)
        agree = base_logits.argmax(-1).eq(edited_logits.argmax(-1))
        values = {
            "base_nll": bnll, "edited_nll": enll, "forward_kl": fkl,
            "reverse_kl": rkl, "js": js, "tv": tv,
            "base_entropy": be, "edited_entropy": ee,
        }
        count = targets.numel()
        for name, value in values.items():
            accum[name].extend(value.flatten().tolist())
        top1_equal += int(agree.sum())
        token_count += count
        prompt_rows.append({
            "id": row["id"], "scored_tokens": count,
            **{name: value.mean().item() for name, value in values.items()},
            "top1_agreement": agree.float().mean().item(),
        })
        for token_index in range(count):
            token_rows.append({
                "prompt_id": row["id"], "token_index": token_index,
                "target_token_id": int(targets[0, token_index]),
                **{name: value[0, token_index].item() for name, value in values.items()},
                "top1_agree": bool(agree[0, token_index]),
            })
        print(f"CANDIDATE {index}/{len(prompts)}", flush=True)
        del edited_logits, bp, ep, bl, el, mix, ml

    def mean(name: str) -> float:
        return sum(accum[name]) / len(accum[name])

    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        position = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return ordered[position]

    base_nll = mean("base_nll")
    edited_nll = mean("edited_nll")
    base_ppl = math.exp(base_nll)
    edited_ppl = math.exp(edited_nll)
    summary = {
        "prompt_count": len(prompts),
        "scored_token_count": token_count,
        "base_nll": base_nll,
        "edited_nll": edited_nll,
        "nll_delta": edited_nll - base_nll,
        "nll_relative_delta": (edited_nll - base_nll) / base_nll,
        "base_perplexity": base_ppl,
        "edited_perplexity": edited_ppl,
        "perplexity_relative_delta": (edited_ppl - base_ppl) / base_ppl,
        "forward_kl": mean("forward_kl"),
        "reverse_kl": mean("reverse_kl"),
        "jensen_shannon": mean("js"),
        "total_variation": mean("tv"),
        "base_entropy": mean("base_entropy"),
        "edited_entropy": mean("edited_entropy"),
        "entropy_delta": mean("edited_entropy") - mean("base_entropy"),
        "top1_agreement": top1_equal / token_count,
        "next_token_probe_agreement": top1_equal / token_count,
        "forward_kl_percentiles": {
            label: percentile(accum["forward_kl"], fraction)
            for label, fraction in (("p50", .50), ("p90", .90), ("p95", .95),
                                    ("p99", .99), ("max", 1.0))
        },
    }
    return {
        "summary": summary,
        "per_prompt": prompt_rows,
        "per_token": token_rows,
        "worst_tokens_by_kl": sorted(
            token_rows, key=lambda item: item["forward_kl"], reverse=True
        )[:20],
        "worst_prompts_by_nll_delta": sorted(
            prompt_rows, key=lambda item: item["edited_nll"] - item["base_nll"], reverse=True
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--edited", required=True, type=Path)
    parser.add_argument("--structural", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    heldout = root / "assets" / "heldout_distribution.jsonl"
    bank = root / "assets" / "v1_corpus_hard32.jsonl"
    baseline_activations = root / "assets" / "capture-activations.safetensors"
    baseline_directions = root / "assets" / "directions_on.safetensors"
    source_lock = root / "assets" / "source-lock.json"
    expected = {
        bank: BANK_SHA256,
        heldout: HELDOUT_SHA256,
        baseline_activations: BASELINE_ACTIVATIONS_SHA256,
        baseline_directions: BASELINE_DIRECTIONS_SHA256,
    }
    for path, wanted in expected.items():
        if sha256(path) != wanted:
            raise ValueError(f"locked measurement input mismatch: {path.name}")
    verified_source = verify_source(args.base, source_lock)
    if verified_source["checkpoint_digest"] != CHECKPOINT_DIGEST:
        raise RuntimeError("verified source identity changed")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    started = datetime.now(timezone.utc).isoformat()
    prompts = load_jsonl(heldout)
    harmful, safe = load_bank(bank, BANK_SHA256)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)

    base_model = load_model(args.base, device)
    baseline = baseline_logits(base_model, tokenizer, prompts, device)
    del base_model
    gc.collect(); torch.cuda.empty_cache()

    edited_model = load_model(args.edited, device)
    distribution = compare_candidate(edited_model, tokenizer, prompts, baseline, device)
    final_harmful = capture(edited_model, tokenizer, harmful, device, "final-harmful")
    final_safe = capture(edited_model, tokenizer, safe, device, "final-safe")
    final_directions, final_report = direction_stats(final_harmful, final_safe)
    save_file(
        {"harmful": final_harmful.to(torch.bfloat16), "safe": final_safe.to(torch.bfloat16)},
        args.out / "capture-final-activations.safetensors",
        metadata={"thinking": "on", "checkpoint": "persisted-candidate"},
    )
    save_file(
        {f"layer_{i}": final_directions[i].to(torch.bfloat16) for i in range(40)},
        args.out / "directions-final.safetensors",
        metadata={"thinking": "on", "checkpoint": "persisted-candidate"},
    )
    (args.out / "direction-final-report.json").write_text(
        json.dumps(final_report, sort_keys=True, indent=2) + "\n"
    )
    del edited_model, baseline
    gc.collect(); torch.cuda.empty_cache()

    baseline_acts = load_file(baseline_activations)
    baseline_dirs = load_file(baseline_directions)
    layer = 39
    direction = baseline_dirs[f"layer_{layer}"].float()
    base_h = (baseline_acts["harmful"][:, layer].float() @ direction).mean().item()
    base_s = (baseline_acts["safe"][:, layer].float() @ direction).mean().item()
    edited_h = (final_harmful[:, layer].float() @ direction).mean().item()
    edited_s = (final_safe[:, layer].float() @ direction).mean().item()
    baseline_sep = abs(base_h - base_s)
    edited_sep = abs(edited_h - edited_s)
    structural = json.loads(args.structural.read_text())
    geometry = {
        "baseline": {"harmful_projection": base_h, "control_projection": base_s,
                     "separation": baseline_sep},
        "edited": {"harmful_projection": edited_h, "control_projection": edited_s,
                   "separation": edited_sep},
        "separation_delta": edited_sep - baseline_sep,
        "separation_removed_pct": 100 * (baseline_sep - edited_sep) / baseline_sep,
        "probe_layer": layer,
        "post_edit_own_direction_selected_layer": final_report["selected_layer"],
        "post_edit_own_direction_separation": final_report["separation_at_selected"],
        "per_pass": structural["iteration_trace"],
        "raw_report": "evidence/s3-compare/surgical-metrics.json",
    }
    (args.out / "distribution-metrics.json").write_text(
        json.dumps(distribution, sort_keys=True, indent=2) + "\n"
    )
    (args.out / "surgical-metrics.json").write_text(json.dumps({
        "schema": "ablagent.qwen36.surgical.v1",
        "geometry": geometry,
        "target_metrics": structural["targets"],
    }, sort_keys=True, indent=2) + "\n")

    summary = distribution["summary"]
    automatic = {
        "schema": "ablagent.qwen36.automatic-gate.v1",
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "max_nll_relative_delta": 0.10,
            "max_js": 0.02,
            "min_top1_agreement": 0.85,
        },
        "observed": {
            "nll_relative_delta": summary["nll_relative_delta"],
            "jensen_shannon": summary["jensen_shannon"],
            "top1_agreement": summary["top1_agreement"],
        },
    }
    observed, thresholds = automatic["observed"], automatic["thresholds"]
    automatic["passed"] = (
        observed["nll_relative_delta"] <= thresholds["max_nll_relative_delta"]
        and observed["jensen_shannon"] <= thresholds["max_js"]
        and observed["top1_agreement"] >= thresholds["min_top1_agreement"]
    )
    automatic["disposition"] = "pass" if automatic["passed"] else "fail"
    (args.out / "automatic-gate.json").write_text(
        json.dumps(automatic, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps({
        "distribution": summary,
        "geometry": geometry,
        "automatic_gate": automatic,
    }, indent=2), flush=True)
    if not automatic["passed"]:
        raise RuntimeError("automatic distribution gate failed")
    print("MEASUREMENT_DONE", flush=True)


if __name__ == "__main__":
    main()
