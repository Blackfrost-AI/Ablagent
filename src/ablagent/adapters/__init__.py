"""Real-stage adapters for dense HuggingFace checkpoints (CPU torch).

Only imported by the explicit `--executor real` path. The mock and synthetic
paths never import torch. Every adapter keeps the DWMIP digest chain and
fail-closed semantics of the synthetic core unchanged.
"""
