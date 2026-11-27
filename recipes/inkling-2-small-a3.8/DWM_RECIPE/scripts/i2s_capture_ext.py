"""Inkling-2-Small capture worker extension (importable module, no function pickling).

Mixed into the vLLM GPU worker via worker_extension_cls="i2s_capture_ext.I2SCaptureExt".
Methods invoked by name through LLM.collective_rpc("method", args=...) — string method
names are resolved on the worker, so nothing is pickled across the spawn boundary.

Hook design (verified against nightly image source):
- attention_writer: layer.attn.forward returns this rank's PARTIAL [T,4608]
  (wo_ud RowParallelLinear reduce_results=False). SUM across TP ranks == full wo_ud OUT.
- mlp_output / moe_experts: layer.mlp.forward (Inkling2MoE) returns FULL all-reduced
  bf16 [T,4608] (reduce=True, ep_size=1). Identical (routed-only, no shared experts).
- layer_output: fused block-attn-residual; no clean module output -> absent.
"""
import os
import numpy as np
import torch

PHYSICAL = 48
HIDDEN = 4608


class I2SCaptureExt:
    """Worker extension: installs forward hooks and dumps per-row partials to disk."""

    def i2s_install(self, partial_dir):
        model = self.model_runner.get_model()
        trunk = model.model
        layers = trunk.layers
        assert len(layers) == PHYSICAL, f"decoder len {len(layers)}"
        # Force the NON-FUSED decode path for clean hookable module outputs:
        # conv_mix = (allreduce_workspace is not None) and has_cache. With the
        # workspace disabled and short prompts (sp=None), the model runs
        #   attn_reduced = all_reduce(attn_out); mlp_out = layer.mlp(mlp_in[0])
        # where layer.mlp(reduce=True) returns the FULL all-reduced bf16 [T,H].
        # The fused path instead returns unreduced per-expert (down, src2dst)
        # rows (zeros on ranks with no routed experts) -> unusable for capture.
        # This is an activation-capture-only change; no weights are modified.
        self._i2s_saved_ws = trunk.allreduce_workspace
        self._i2s_saved_sp = trunk.seq_parallel
        trunk.allreduce_workspace = None
        trunk.seq_parallel = None
        state = {"attn": {}, "mlp": {}, "visits": {}}
        handles = []

        def last_tok(v):
            if isinstance(v, tuple):
                v = v[0]
            if v.ndim == 3:
                return v[0, -1, :]
            if v.ndim == 2:
                return v[-1, :]
            raise RuntimeError(f"shape {tuple(v.shape)}")

        for i, layer in enumerate(layers):
            def attn_hook(mod, args, output, li=i):
                state["attn"][li] = last_tok(output).detach().float().cpu().numpy()
                state["visits"][li] = state["visits"].get(li, 0) + 1

            def mlp_hook(mod, args, output, li=i):
                state["mlp"][li] = last_tok(output).detach().float().cpu().numpy()

            handles.append(layer.attn.register_forward_hook(attn_hook))
            handles.append(layer.mlp.register_forward_hook(mlp_hook))

        self._i2s_state = state
        self._i2s_handles = handles
        self._i2s_pdir = partial_dir
        return {"layers": len(layers), "hidden": HIDDEN,
                "attn_type": type(layers[0].attn).__name__,
                "mlp_type": type(layers[0].mlp).__name__,
                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}

    def i2s_grab(self, row_idx):
        st = self._i2s_state
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        attn = np.stack([st["attn"][l] for l in range(PHYSICAL)])
        mlp = np.stack([st["mlp"][l] for l in range(PHYSICAL)])
        pdir = self._i2s_pdir
        os.makedirs(pdir, exist_ok=True)
        np.save(f"{pdir}/row{row_idx:02d}_rank{rank}_attn.npy", attn)
        np.save(f"{pdir}/row{row_idx:02d}_rank{rank}_mlp.npy", mlp)
        visits = dict(st["visits"])
        st["visits"].clear(); st["attn"].clear(); st["mlp"].clear()
        return {"rank": rank, "visits": visits}

    def i2s_remove(self):
        for h in getattr(self, "_i2s_handles", []):
            h.remove()
        self._i2s_handles = []
        model = self.model_runner.get_model()
        trunk = model.model
        if hasattr(self, "_i2s_saved_ws"):
            trunk.allreduce_workspace = self._i2s_saved_ws
            trunk.seq_parallel = self._i2s_saved_sp
        return True