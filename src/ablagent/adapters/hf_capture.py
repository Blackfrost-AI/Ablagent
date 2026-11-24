"""S1/S2 real adapters: baseline generation and refusal-direction capture.

CPU torch only, lazy imports. Capture uses balanced refusal-prefill vs
helpful-prefill contrasts over the pack's synthetic capture corpus and emits
DWMIP DirectionCandidate objects labeled ``correlational`` — a statistical
contrast, never a causal claim.

The derived capture bank binds the ACTUAL model inputs (prompt + assistant
prefill as JSON) into records_digest; lineage from the pack corpus is recorded
in the bank notes and the stage receipt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ablagent.artifacts import ArtifactStore
from ablagent.banks import BankManifest, BankRecord, LicenseTerms
from ablagent.contracts import _identifier
from ablagent.errors import ValidationError
from ablagent.identity import GraphSpec
from ablagent.interventions import HF_ALGORITHM_ID, DirectionCandidate

# Refusal-side vs helpful-side assistant prefills for the contrast. These are
# generic response prefixes, not corpus content; the corpus supplies the task.
HOSTILE_PREFILL = "I'm sorry, but I can't help with that request."
BENIGN_PREFILL = "Sure — here is a careful, evidence-grounded answer:"

_MAX_NEW_TOKENS_BASELINE = 48
_MAX_PROMPT_TOKENS = 256


def _norm_tuple(values) -> tuple[float, ...]:
    """Normalize -0.0 to 0.0 so digests are stable across platforms."""
    return tuple(float(x) + 0.0 for x in values)


def _load_model(snapshot_dir: Path):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))
    # float32 compute is mandatory for reproducible capture/baseline numerics.
    # transformers 5 renamed torch_dtype -> dtype; select by major version so
    # both the declared floor (4.44) and 5.x load in float32 without warnings.
    dtype_kwarg = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot_dir), low_cpu_mem_usage=True, **{dtype_kwarg: torch.float32}
    )
    if model.dtype != torch.float32:
        model = model.to(torch.float32)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _chat_inputs(tokenizer, prompt: str, prefill: str | None):
    """Build input ids: chat template when available, raw text otherwise.

    When ``prefill`` is given, the assistant turn is opened with that exact
    text (add_generation_prompt=False + manual suffix) so the last token sits
    inside the assistant response — the capture contrast position.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=prefill is None
        )
        has_template = True
    except Exception:
        text = prompt + "\n"
        has_template = False
    if prefill is not None:
        text = text + prefill
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=_MAX_PROMPT_TOKENS)
    return encoded, has_template


def _prefill_survives(tokenizer, encoded, prefill: str) -> bool:
    """True when truncation did not eat into the appended assistant prefill.

    A truncated prefill would make hostile/benign inputs partially identical
    and silently mislabel the contrast; such records are excluded, not used.
    """
    prefill_ids = tokenizer(prefill, add_special_tokens=False)["input_ids"]
    if not prefill_ids:
        return False
    ids = encoded["input_ids"][0].tolist()
    return ids[-len(prefill_ids) :] == prefill_ids


def baseline_generate(
    snapshot_dir: Path,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int = _MAX_NEW_TOKENS_BASELINE,
) -> dict[str, Any]:
    """S1: greedy generation on evaluation rows. Baseline evidence, NOT a gate."""
    import torch

    model, tokenizer = _load_model(snapshot_dir)
    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            encoded, _has_template = _chat_inputs(tokenizer, row["prompt"], None)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[0][encoded["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(
                {"id": row["id"], "response": text, "new_tokens": int(new_tokens.shape[0])}
            )
    return {
        "rows": outputs,
        "row_count": len(outputs),
        "max_new_tokens": max_new_tokens,
        "decoding": "greedy",
        "claim": "baseline_behavior_evidence_only_not_a_gate",
    }


def _writer_modules(model, graph: GraphSpec) -> dict[str, Any]:
    """Map writer tensor names to owning modules (name minus '.weight')."""
    modules = dict(model.named_modules())
    found: dict[str, Any] = {}
    for writer in graph.writers:
        module_name = writer.tensor_name.rsplit(".", 1)[0]
        module = modules.get(module_name)
        if module is None:
            raise ValidationError(
                f"writer {writer.writer_id}: module {module_name!r} not found in model"
            )
        found[writer.tensor_name] = module
    return found


def _is_embedlike(tensor_name: str) -> bool:
    lowered = tensor_name.lower()
    return "embed" in lowered or "wte" in lowered or "lm_head" in lowered


def derive_capture_bank(
    capture_rows: list[dict[str, Any]],
    *,
    bank_id: str,
    pack_id: str,
    corpus_sha256: str,
    max_pairs: int = 16,
) -> BankManifest:
    """Derive a balanced capture bank from pack corpus rows (JSON prompt+prefill)."""
    pairs = capture_rows[:max_pairs]
    if len(pairs) < 4:
        raise ValidationError("capture needs at least 4 corpus rows (bank minimum)")
    records: list[BankRecord] = []
    for index, row in enumerate(pairs):
        pair_id = _identifier(f"capture-pair-{index:03d}")
        for group, prefill in (("hostile", HOSTILE_PREFILL), ("benign", BENIGN_PREFILL)):
            records.append(
                BankRecord(
                    record_id=_identifier(f"capture-{index:03d}-{group}"),
                    prompt=json.dumps(
                        {"prompt": row["prompt"], "assistant_prefill": prefill},
                        sort_keys=True,
                    ),
                    group=group,
                    domain="derisk",
                    pair_id=pair_id,
                )
            )
    return BankManifest(
        bank_id=_identifier(bank_id),
        version="v0",
        role="capture",
        visibility="public",
        records=tuple(records),
        license=LicenseTerms(
            spdx_id="CC-BY-4.0",
            copyright="2026 Ablagent contributors",
            source=(
                f"Derived from pack {pack_id} synthetic capture corpus (sha256:{corpus_sha256})."
            ),
            license_text="banks/LICENSE-CC-BY-4.0",
            redistribution_allowed=True,
            transformation_lineage=(
                "Each record wraps one synthetic corpus prompt with a fixed generic "
                "assistant prefill (refusal-side or helpful-side). No model outputs "
                "and no third-party records."
            ),
            generator_terms="No external generator or terms apply.",
        ),
        split_policy="Public derived capture fixture; never a release holdout.",
        notes=(
            f"Derived at S2 from {pack_id} capture corpus sha256:{corpus_sha256}; "
            f"{len(pairs)} balanced pairs; records bind actual model inputs."
        ),
    )


def capture_directions(
    snapshot_dir: Path,
    graph: GraphSpec,
    baseline_ref,
    bank: BankManifest,
    *,
    store: ArtifactStore,
    max_pairs: int = 16,
) -> dict[str, Any]:
    """S2: hook writer outputs, contrast hostile vs benign prefills, emit candidates.

    Per writer: mean over records of last-token output vectors; direction =
    mean_hostile - mean_benign (the refusal-contrast direction). Embedding-like
    writers are excluded from capture ranking (module output geometry differs);
    they remain declared in the graph and preserved by the bake.
    """
    import numpy as np
    import torch

    model, tokenizer = _load_model(snapshot_dir)
    modules = _writer_modules(model, graph)
    bank_ref = bank.reference()
    sums: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, dict[str, int]] = {}

    hooks = []
    active: dict[str, str] = {}

    def make_hook(tensor_name: str):
        def hook(_module, _inputs, output):
            group = active.get("group")
            if group is None:
                return
            value = output[0] if isinstance(output, tuple) else output
            if value.ndim != 3:
                return
            index = int(active["last_index"])
            vec = value[0, index, :].detach().to(torch.float32).cpu().numpy()
            bucket = sums.setdefault(tensor_name, {})
            counter = counts.setdefault(tensor_name, {})
            if group not in bucket:
                bucket[group] = np.zeros_like(vec, dtype=np.float64)
                counter[group] = 0
            if vec.shape[0] != bucket[group].shape[0]:
                return  # geometry mismatch (e.g. embedding seq dim) — skip honestly
            bucket[group] += vec.astype(np.float64)
            counter[group] += 1

        return hook

    for tensor_name, module in modules.items():
        hooks.append(module.register_forward_hook(make_hook(tensor_name)))

    records = list(bank.records)[: max_pairs * 2]
    template_used: set[bool] = set()
    dropped_records: list[dict[str, Any]] = []
    # Drop whole PAIRS when either side's prefill was truncated away, so the
    # contrast stays balanced and every used record ends inside the prefill.
    by_pair: dict[str, list[Any]] = {}
    for record in records:
        by_pair.setdefault(record.pair_id or record.record_id, []).append(record)
    try:
        with torch.no_grad():
            for pair_id, pair in by_pair.items():
                prepared = []
                ok = True
                for record in pair:
                    payload = json.loads(record.prompt)
                    encoded, has_template = _chat_inputs(
                        tokenizer, payload["prompt"], payload["assistant_prefill"]
                    )
                    if not _prefill_survives(tokenizer, encoded, payload["assistant_prefill"]):
                        ok = False
                    prepared.append((record, encoded, has_template))
                if not ok:
                    dropped_records.append(
                        {"pair_id": pair_id, "reason": "prefill truncated at prompt cap"}
                    )
                    continue
                for record, encoded, has_template in prepared:
                    template_used.add(has_template)
                    last_index = int(encoded["attention_mask"].sum().item()) - 1
                    active["group"] = record.group
                    active["last_index"] = last_index
                    model(**encoded)
                    active.clear()
    finally:
        for handle in hooks:
            handle.remove()

    candidates: list[tuple[DirectionCandidate, Any]] = []
    rankings: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for writer in graph.writers:
        name = writer.tensor_name
        if _is_embedlike(name):
            skipped.append(
                {"tensor_name": name, "reason": "embedding-like writer excluded from capture"}
            )
            continue
        bucket = sums.get(name)
        counter = counts.get(name)
        if (
            not bucket
            or "hostile" not in bucket
            or "benign" not in bucket
            or counter.get("hostile", 0) < 4
            or counter.get("benign", 0) < 4
            or counter["hostile"] != counter["benign"]
        ):
            skipped.append(
                {"tensor_name": name, "reason": "no balanced last-token capture at this writer"}
            )
            continue
        n = counter["hostile"]
        mean_hostile = bucket["hostile"] / n
        mean_benign = bucket["benign"] / n
        vector = mean_hostile - mean_benign
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm == 0.0:
            skipped.append({"tensor_name": name, "reason": "zero or non-finite contrast"})
            continue
        tid = writer.writer_id.removesuffix("-writer")[:110]
        candidate = DirectionCandidate(
            candidate_id=_identifier(f"{tid}-candidate"),
            baseline_digest=baseline_ref.tensor_digest,
            graph_digest=graph.digest(),
            coordinate=writer.coordinate,
            space_id=writer.output_space_id,
            capture_bank=bank_ref,
            algorithm_id=HF_ALGORITHM_ID,
            vector=_norm_tuple(vector),
            benign_count=n,
            hostile_count=n,
        )
        candidates.append((candidate, writer))
        rankings.append(
            {
                "writer_id": writer.writer_id,
                "tensor_name": name,
                "contrast_norm": norm,
                "pairs": n,
            }
        )

    if not candidates:
        raise ValidationError(
            "capture produced no usable direction candidates; "
            "the checkpoint/corpus combination is outside the v0 adapter envelope"
        )
    rankings.sort(key=lambda item: item["contrast_norm"], reverse=True)
    for index, (candidate, _writer) in enumerate(candidates):
        store.write_direction(candidate, f"candidate-{index:03d}")
    return {
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "writer_id": w.writer_id,
                "tensor_name": w.tensor_name,
                "path": f"candidate-{i:03d}.direction.json",
            }
            for i, (c, w) in enumerate(candidates)
        ],
        "rankings": rankings,
        "skipped": skipped,
        "bank": bank_ref.model_dump(mode="json"),
        "algorithm_id": HF_ALGORITHM_ID,
        "evidence_class": "correlational",
        "capture_prefills": {"hostile": HOSTILE_PREFILL, "benign": BENIGN_PREFILL},
        "chat_template_used": sorted(template_used),
        "dropped_records": dropped_records,
        "prompt_token_cap": _MAX_PROMPT_TOKENS,
    }
