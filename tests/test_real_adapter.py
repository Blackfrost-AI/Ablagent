"""Real-executor tests: tiny local HF checkpoints, no network.

The ``real`` marker gates tests that need torch/transformers (the [real]
extra). They build a 2-layer bf16 llama in a tmp dir and drive the actual
intake → capture → bake → export chain plus the full run_real pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ablagent.errors import ValidationError

real = pytest.mark.real

pytest.importorskip  # noqa: B018  (kept explicit below per fixture)


def _have_real_deps() -> bool:
    try:
        import safetensors  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


needs_real = pytest.mark.skipif(not _have_real_deps(), reason="[real] extra not installed")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """A real 2-layer bf16 llama with tied embeddings and a chat template."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(7)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)
    out = tmp_path_factory.mktemp("tiny-llama")
    model.save_pretrained(out, safe_serialization=True)
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=64, special_tokens=["<unk>", "<s>", "</s>"])
    tok.train_from_iterator(["hello world this is a tiny test corpus " * 20], trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
    )
    fast.chat_template = (
        "{% for m in messages %}{{ '<s>' + m['role'] + ': ' + m['content'] + '</s>' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<s>assistant: ' }}{% endif %}"
    )
    fast.save_pretrained(out)
    return out


# ---------------------------------------------------------------- parse/pin


def test_parse_model_ref_forms():
    from ablagent.adapters.hf_intake import parse_model_ref

    assert parse_model_ref("https://huggingface.co/org/model") == ("org/model", None)
    url = "https://huggingface.co/org/model/tree/abc123"
    assert parse_model_ref(url) == ("org/model", "abc123")
    assert parse_model_ref("org/model") == ("org/model", None)
    with pytest.raises(ValidationError):
        parse_model_ref("mock://derisk")
    with pytest.raises(ValidationError):
        parse_model_ref("not a model ref")


def test_parse_local_dir(tmp_path):
    from ablagent.adapters.hf_intake import parse_model_ref

    repo, rev = parse_model_ref(str(tmp_path))
    assert repo == str(tmp_path.resolve()) and rev is None


# ---------------------------------------------------------------- intake


@real
@needs_real
def test_intake_surface_and_ties(tiny_checkpoint, tmp_path):
    from ablagent.adapters.hf_intake import intake
    from ablagent.artifacts import ArtifactStore, load_tensors
    from ablagent.constants import HF_ADAPTER_ID

    receipt = intake(
        str(tiny_checkpoint), snapshot_dir=tmp_path / "snap", store_root=tmp_path / "art"
    )
    assert receipt["adapter_id"] == HF_ADAPTER_ID
    prov = receipt["provenance"]
    assert prov["tie_word_embeddings"] is True
    # HF serializes tied checkpoints WITHOUT the alias; exclusion covers the
    # case where an alias IS present in the shards (see unit test below).
    assert all(dtype == "bfloat16" for dtype in prov["original_dtypes"].values())
    store = ArtifactStore(tmp_path / "art")
    graph = store.load_graph("baseline.graph.json")
    names = {w.tensor_name for w in graph.writers}
    assert "lm_head.weight" not in names
    assert "model.embed_tokens.weight" in names
    # every writer is float32 in the editable surface
    tensors = load_tensors(store.path(receipt["baseline"]["path"]))
    assert all(t.dtype.name == "float32" for t in tensors.values())
    # every writer declares rank_one_output and matching input/output spaces
    assert all(w.operations == ("rank_one_output",) for w in graph.writers)


@real
@needs_real
def test_tied_alias_excluded_when_present_in_shards():
    import torch
    from transformers import LlamaConfig

    from ablagent.adapters.hf_intake import build_editable_surface

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    embed = torch.randn(64, 32)
    tensors = {
        "model.embed_tokens.weight": embed,
        "lm_head.weight": embed.clone(),  # alias present in shards
        "model.layers.0.mlp.gate_proj.weight": torch.randn(64, 32),
    }
    _graph, surface, prov = build_editable_surface(tensors, cfg, "hf-tie-test")
    assert prov["excluded_aliases"] == {"lm_head.weight": "model.embed_tokens.weight"}
    assert "lm_head.weight" not in surface
    assert "model.embed_tokens.weight" in surface


@real
@needs_real
def test_intake_rejects_non_safetensors_dir(tmp_path):
    from ablagent.adapters.hf_intake import intake

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    with pytest.raises(ValidationError, match="safetensors"):
        intake(str(empty), snapshot_dir=tmp_path / "snap", store_root=tmp_path / "art")


@real
@needs_real
def test_envelope_cap(tiny_checkpoint, tmp_path, monkeypatch):
    from ablagent.adapters import hf_intake

    monkeypatch.setattr(hf_intake, "_MAX_SURFACE_BYTES", 1024)
    with pytest.raises(ValidationError, match="resource envelope"):
        hf_intake.intake(
            str(tiny_checkpoint), snapshot_dir=tmp_path / "snap", store_root=tmp_path / "art"
        )


@real
@needs_real
def test_nonfinite_tensor_rejected():
    import torch
    from transformers import LlamaConfig

    from ablagent.adapters.hf_intake import build_editable_surface

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    tensors = {"model.layers.0.mlp.gate_proj.weight": torch.full((64, 32), float("nan"))}
    with pytest.raises(ValidationError, match="non-finite"):
        build_editable_surface(tensors, cfg, "hf-nan-test")


# ---------------------------------------------------------------- full run


@real
@needs_real
def test_run_real_end_to_end(tiny_checkpoint, tmp_path):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablagent.packs import load_pack
    from ablagent.pipeline_real import run_real

    pack = load_pack(Path(__file__).parents[1] / "packs/derisk")
    result = run_real(
        pack,
        model=str(tiny_checkpoint),
        runs_root=tmp_path / "runs",
        alpha=1.0,
        max_pairs=8,
        baseline_rows=4,
    )
    run_dir = Path(result["run_directory"])

    # Completed vs honestly skipped stages.
    assert result["completed_stages"] == [
        "S0_INTAKE",
        "S1_BASELINE",
        "S2_DIRECTIONS",
        "S3_DERISK",
        "S8_PRODUCTIZE",
        "S9_CARD_DRAFT",
    ]
    assert set(result["skipped_stages"]) == {
        "S4_QUANT",
        "S5_SERVE",
        "S6_DOMAIN_VERIFY",
        "S7_CAPABILITY_VERIFY",
        "SR_RELEASE_REDLINE",
        "S10_DELIVER",
    }
    assert result["release"] == "not_eligible"

    # Control plane: every stage release stays not_eligible; skips carry reasons.
    control = json.loads((run_dir / "control/CONTROL.json").read_text())
    assert all(s["release"] == "not_eligible" for s in control["stages"].values())
    for stage in result["skipped_stages"]:
        assert control["stages"][stage]["execution"] == "skipped"

    # Manifest re-verifies independently.
    from ablagent.artifacts import ArtifactStore

    store = ArtifactStore(run_dir / "artifacts")
    verified = store.verify_compiled(result["manifest"])
    assert verified.structural_check.passed

    # Export: loads as a real HF model, generates, exactly one tensor changed,
    # original dtype restored, ties preserved.
    export = Path(result["export_dir"])
    tok = AutoTokenizer.from_pretrained(export)
    model = AutoModelForCausalLM.from_pretrained(export)
    model.eval()
    enc = tok("hello world", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id
        )
    assert out.shape[1] > enc["input_ids"].shape[1]
    src = load_file(str(run_dir / "hf-snapshot/model.safetensors"))
    dst = load_file(str(export / "model.safetensors"))
    changed = [k for k in src if not torch.equal(src[k], dst[k])]
    assert changed == [result_selected_tensor(run_dir)]
    assert dst[changed[0]].dtype == torch.bfloat16
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    # Card draft is explicitly not a release (stored raw, not evidence-wrapped).
    card = next(
        json.loads(p.read_text())
        for p in sorted((run_dir / "evidence").glob("s9_card_draft-*.json"))
        if json.loads(p.read_text()).get("schema") == "ablagent.card-draft.v1"
    )
    assert card["status"] == "draft_experimental_not_a_release"
    assert card["release_eligibility"] == "not_eligible"
    assert card["intervention"]["evidence_class"] == "correlational"
    assert set(card["untested_stages"]) == set(result["skipped_stages"])


def result_selected_tensor(run_dir: Path) -> str:
    receipt = json.loads(next((run_dir / "evidence").glob("s3_derisk-bake-*.json")).read_text())
    return receipt["payload"]["tensor_name"]


@real
@needs_real
def test_run_real_writer_selection_and_alpha(tiny_checkpoint, tmp_path):
    from ablagent.packs import load_pack
    from ablagent.pipeline_real import run_real

    pack = load_pack(Path(__file__).parents[1] / "packs/derisk")
    result = run_real(
        pack,
        model=str(tiny_checkpoint),
        runs_root=tmp_path / "runs",
        alpha=0.5,
        writer="model.layers.0.self_attn.o_proj.weight",
        max_pairs=8,
        baseline_rows=2,
    )
    assert result["selected_writer"] == "model.layers.0.self_attn.o_proj.weight-writer"
    assert result["alpha"] == 0.5
    run_dir = Path(result["run_directory"])
    bake = json.loads(next((run_dir / "evidence").glob("s3_derisk-bake-*.json")).read_text())
    assert bake["payload"]["sign_test"]["passed"] is True
    assert bake["payload"]["tensor_name"] == "model.layers.0.self_attn.o_proj.weight"


@real
@needs_real
def test_run_real_rejects_unknown_writer(tiny_checkpoint, tmp_path):
    from ablagent.packs import load_pack
    from ablagent.pipeline_real import run_real

    pack = load_pack(Path(__file__).parents[1] / "packs/derisk")
    with pytest.raises(ValidationError, match="no captured candidate"):
        run_real(
            pack,
            model=str(tiny_checkpoint),
            runs_root=tmp_path / "runs",
            writer="does.not.exist.weight",
            max_pairs=8,
            baseline_rows=2,
        )


def test_run_real_alpha_validation(tmp_path):
    from ablagent.packs import load_pack
    from ablagent.pipeline_real import run_real

    pack = load_pack(Path(__file__).parents[1] / "packs/derisk")
    if not _have_real_deps():
        pytest.skip("[real] extra not installed")
    with pytest.raises(ValidationError, match="alpha"):
        run_real(pack, model="org/model", runs_root=tmp_path, alpha=1.5)


# ---------------------------------------------------------------- secret scan


def test_secret_scan_skips_binaries_and_hex_ids(tmp_path):
    from ablagent.security.secret_scan import scan_tree

    (tmp_path / "weights.safetensors").write_bytes(b"\x00sk-abcdefghijklmnopqrstuvwx\x00")
    (tmp_path / "receipt.json").write_text(
        json.dumps({"revision": "a" * 40})
    )  # git-style sha1, not a credential
    assert scan_tree(tmp_path) == ()
    (tmp_path / "leak.env").write_text("OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwx\n")
    findings = scan_tree(tmp_path)
    assert findings and all("leak.env" in f.path for f in findings)


@real
@needs_real
def test_conv1d_family_fails_closed():
    import torch
    from transformers import GPT2Config

    from ablagent.adapters.hf_intake import build_editable_surface

    cfg = GPT2Config(n_embd=32, n_layer=1, n_head=4, vocab_size=64)
    tensors = {"transformer.h.0.attn.c_proj.weight": torch.randn(32, 32)}
    with pytest.raises(ValidationError, match="Conv1D"):
        build_editable_surface(tensors, cfg, "hf-gpt2-test")


@real
@needs_real
def test_local_preflight_size_cap(tiny_checkpoint, tmp_path, monkeypatch):
    from ablagent.adapters import hf_intake

    monkeypatch.setattr(hf_intake, "_MAX_SURFACE_BYTES", 16)
    with pytest.raises(ValidationError, match="resource envelope"):
        hf_intake.intake(
            str(tiny_checkpoint), snapshot_dir=tmp_path / "snap", store_root=tmp_path / "art"
        )


def test_scan_tree_report_lists_unscanned(tmp_path):
    from ablagent.security.secret_scan import scan_tree_report

    (tmp_path / "w.safetensors").write_bytes(b"\x00\x01")
    (tmp_path / "ok.txt").write_text("nothing here\n")
    findings, unscanned = scan_tree_report(tmp_path)
    assert findings == ()
    assert any(u["reason"] == "binary-suffix" for u in unscanned)
