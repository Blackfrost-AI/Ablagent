# Third-party notices

This method uses, but does not redistribute, the following third-party works:

- `thinkingmachines/Inkling-2-Small` (upstream base checkpoint, Apache-2.0 at
  the pinned revision) and the derived
  `Blackfrost-Research/Inkling-2-Small-Derisked` release, governed by the
  upstream model terms and notices.
- The hard32 contrast bank and the 443-row false-refusal corpus derive from
  published research prompt sets (AdvBench, StrongREJECT, XSTest lineage);
  see `benchmarks/PROVENANCE.md` in the parent repository for per-corpus
  upstream attribution and licenses.
- Arditi et al., "Refusal in Language Models Is Mediated by a Single
  Direction," NeurIPS 2024 (arXiv:2406.11717) — foundational research
  citation; the paper is not relicensed here.
- PyTorch, NumPy, Transformers, safetensors, Hugging Face Hub, vLLM, CUDA,
  Docker, and the pinned vLLM container image, each governed by its own
  license.

The Apache-2.0 license in this package covers Blackfrost AI-authored content
only.