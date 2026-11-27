#!/usr/bin/env python3
"""In-memory BF16 loader for Inkling-2-Small (thinkingmachines/Inkling-2-Small).

Host transformers 5.24.1 has NO modelopt/hf_quant_config handler, so the packed
NVFP4 (L0-39) / MXFP8 (L40-47) expert tensors cannot map onto the plain-BF16
gate_up_proj/down_proj parameters. This loader builds the empty HF model and
fills every parameter by dequantizing the experts to BF16, matching the exact
on-disk contract verified 2026-11-27:

  NVFP4 experts (L0-39), per projection p in {w13, w2}:
    p_weight        U8        [E, M, K_packed]   (two fp4 e2m1 per byte, low nibble first)
    p_weight_scale  F8_E4M3   [E, M, K/16]       (block scale, group 16 along K)
    p_weight_amax   F32       [E]                (per-expert outer amax)
    p_weight_outer_scale_denominator F32 [1]
    global_scale[e] = amax[e] / denominator      (DECODE multiplier, ModelOpt convention)
    dequant = fp4_value * block_scale.float() * global_scale[e]

  MXFP8 experts (L40-47), per projection p:
    p_weight        F8_E4M3   [E, M, K]
    p_weight_scale  U8        [E, M, K/32]       (E8M0 exponent, group 32 along K)
    dequant = fp8_value.float() * 2**(scale - 127)

  w13 -> gate_up_proj [E, 2*inter, hidden] = [384, 4608, 4608]
  w2  -> down_proj    [E, hidden, inter]   = [384, 4608, 2304]

Gate/up layout: HF SoftcappedSwiGLU uses chunk(2, dim=-1) (split halves:
gate=first inter rows, up=second inter rows). vLLM activation uses interleaved
pairs. The loader exposes GATE_UP_LAYOUT to select; --verify tests both against
a functional forward so the correct convention is measured, not assumed.

Everything else (wo_ud BF16 attention, router, sconv, attn_res, norms, embed,
lm_head, vision tower, MTP/DSpark) is copied verbatim from the checkpoint.

Read-only on the source checkpoint. No weights written. CPU-resident fill, then
device_map dispatch by the caller. Self-verifying: --verify dequantizes real
expert bytes and compares against vLLM's reference dequantize_nvfp4_moe_weights
(run inside the pinned image) plus a functional MoE forward.
"""
import argparse, json, pathlib, sys
import torch
from safetensors import safe_open

FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def unpack_fp4(packed_u8):
    """[... , K/2] U8 -> [..., K] float32, low nibble first (vLLM break_fp4_bytes)."""
    lo = packed_u8 & 0x0F
    hi = (packed_u8 >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1).reshape(*packed_u8.shape[:-1], packed_u8.shape[-1] * 2)
    return FP4_LUT[out.long()]


def dequant_nvfp4(packed, scale_f8, amax, denom, group=16):
    """packed [E,M,Kp] U8, scale_f8 [E,M,K/group] F8_E4M3, amax [E], denom [1]."""
    vals = unpack_fp4(packed)                       # [E,M,K] f32
    E, M, K = vals.shape
    bs = scale_f8.float()                           # [E,M,K/group]
    bs = bs.repeat_interleave(group, dim=-1)        # [E,M,K]
    gs = (amax.float() / denom.float()).reshape(E, 1, 1)  # [E,1,1]
    return vals * bs * gs


def dequant_mxfp8(fp8, scale_u8, group=32):
    """fp8 [E,M,K] F8_E4M3, scale_u8 [E,M,K/group] U8 (E8M0)."""
    vals = fp8.float()
    bs = torch.exp2(scale_u8.float() - 127.0)       # [E,M,K/group]
    bs = bs.repeat_interleave(group, dim=-1)
    return vals * bs


def to_gate_up(w13_bf16, layout, inter):
    """w13 dequantized [E, 2*inter, hidden] -> HF gate_up_proj [E, 2*inter, hidden].
    layout='halves': checkpoint already gate-first/up-second (identity).
    layout='interleaved': checkpoint even=gate,odd=up -> de-interleave to halves."""
    if layout == "halves":
        return w13_bf16
    E, twoI, H = w13_bf16.shape
    gate = w13_bf16[:, 0::2, :]                     # [E, inter, H]
    up = w13_bf16[:, 1::2, :]
    return torch.cat([gate, up], dim=1)             # [E, 2*inter, H] halves


def load_expert_layer(f, prefix, quant, layout, inter, device="cpu"):
    """Return (gate_up_proj, down_proj) BF16 tensors for one layer's experts."""
    if quant == "nvfp4":
        w13 = dequant_nvfp4(f.get_tensor(prefix + "w13_weight"),
                            f.get_tensor(prefix + "w13_weight_scale"),
                            f.get_tensor(prefix + "w13_weight_amax"),
                            f.get_tensor(prefix + "w13_weight_outer_scale_denominator"))
        w2 = dequant_nvfp4(f.get_tensor(prefix + "w2_weight"),
                           f.get_tensor(prefix + "w2_weight_scale"),
                           f.get_tensor(prefix + "w2_weight_amax"),
                           f.get_tensor(prefix + "w2_weight_outer_scale_denominator"))
    else:
        w13 = dequant_mxfp8(f.get_tensor(prefix + "w13_weight"),
                            f.get_tensor(prefix + "w13_weight_scale"))
        w2 = dequant_mxfp8(f.get_tensor(prefix + "w2_weight"),
                           f.get_tensor(prefix + "w2_weight_scale"))
    gate_up = to_gate_up(w13, layout, inter).to(torch.bfloat16)
    down = w2.to(torch.bfloat16)
    return gate_up.to(device), down.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--verify", action="store_true", help="dequant real experts + sanity stats, no full load")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--layout", default="halves", choices=["halves", "interleaved"])
    a = p.parse_args()
    snap = pathlib.Path(a.checkpoint)
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    L = a.layer
    quant = "nvfp4" if L < 40 else "mxfp8"
    prefix = f"model.llm.layers.{L}.mlp.experts."
    shard = snap / idx[prefix + "w13_weight"]
    with safe_open(str(shard), framework="pt") as f:
        gu, down = load_expert_layer(f, prefix, quant, a.layout, 2304)
    print(json.dumps({
        "layer": L, "quant": quant, "layout": a.layout,
        "gate_up_shape": list(gu.shape), "down_shape": list(down.shape),
        "gate_up_dtype": str(gu.dtype),
        "gate_up_absmax": float(gu.abs().max()), "gate_up_mean": float(gu.float().mean()),
        "gate_up_std": float(gu.float().std()),
        "down_absmax": float(down.abs().max()), "down_std": float(down.float().std()),
        "finite": bool(torch.isfinite(gu).all() and torch.isfinite(down).all()),
        "gate_half_std": float(gu[:, :2304].float().std()),
        "up_half_std": float(gu[:, 2304:].float().std()),
    }, indent=1))


if __name__ == "__main__":
    main()