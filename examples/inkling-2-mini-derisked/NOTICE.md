# NOTICE — vendored example material

This directory contains material vendored from
`Blackfrost-AI/Inkling-2-Mini-Derisked` (Hugging Face, license tag `other`),
derived from the upstream base model `thinkingmachines/Inkling-2-Mini`
(license: other). The repository's Apache-2.0 license covers the DWMIP
runtime and Blackfrost-authored code/documentation; it does **not**
automatically extend to every file here. Per-file scope:

| Material | Files | License / terms |
|---|---|---|
| Kit scripts | `DWM_RECIPE/scripts/*.py`, `deployment-kit/scripts/serve.sh` | Apache-2.0 (Blackfrost-authored code) |
| Kit documentation | `DWM_RECIPE/RUNBOOK.md`, `deployment-kit/*.md` | Apache-2.0 (Blackfrost-authored documentation) |
| hard32 prompt pairs | `DWM_RECIPE/scripts/v1_corpus_hard32.jsonl` | CC-BY-4.0 (Blackfrost-authored; same content license as `banks/`) |
| Activation-derived artifacts | `directions.npz`, `raw-capture.npz` (fetch-only), `direction-battery.json`, `capture-metadata.json`, `s2-summary.json`, `s3-edit-receipt-a25.json` | Blackfrost-published research artifacts derived from the upstream model. **No additional license is granted by this repository.** Whether activation-derived vectors fall under the upstream model license is unresolved; review the upstream terms before redistribution or commercial use. |

Model weights are not vendored; obtain them from the HF repository under its
published terms. The 443-row measurement corpus, internal endpoints, and
credentials are excluded entirely.

The hard32 pairs were authored by Blackfrost for this research program; they
are not verbatim rows of third-party benchmark corpora.