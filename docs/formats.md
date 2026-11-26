# DWM across precision and quantization formats

**Format-aware DWM research: experiments from BF16 to 2-bit IQ.** Quantization is part of the intervention surface, not an afterthought—and not a reason to assume a weight edit survives requantization or preserves behavior.

The evidence below separates three things that are often conflated:

- **What the current CLI executes:** small dense float `safetensors` checkpoints on CPU.
- **What Blackfrost research has demonstrated on a specific model arm:** editing, scale/codec qualification, requantization, serialization, or serving.
- **What remains unproven:** preserved refusal-removal behavior, retained capability, safety/release gates, and transfer to another checkpoint or backend.

**A prior format result is not a promise that this repository's CLI directly edits that codec.** It is research experience that helps identify the right capture/edit/verification path for a future adapter or an explicitly qualified agent-led run. No codec is inferred from a filename.

## Research coverage by path

| Format / path | Research evidence | What it does **not** prove |
|---|---|---|
| **BF16** | Dense float edits and capture are the reference path for multiple model investigations; the public CLI executes the qualified dense subset. A finalized [GLM BF16 recipe](../recipes/glm-5.3-bf16-a2.2/README.md) is preserved in the recipe library. | BF16 success does not transfer automatically to another architecture or quantized derivative. |
| **FP8 / MXFP8** | MXFP8 E8M0 scale reconstruction and requantization round-trips were tested on MiniMax M3.1 Flash; the specific edit experiment was **underpowered/unvalidated**, not a successful DWM outcome. | Codec or scale correctness does not establish native FP8 edit efficacy, serving or behavior. |
| **NVFP4** | Bolt-74B research used a BF16 edit followed by routed-experts-only NVFP4 post-training quantization and served the derivative. A separate FP8/NVFP4 requantization round-trip study reached scale/value-level checks but not a validated intervention. The public GLM recipe documents NVFP4 serialization. | These are **not** proof of native NVFP4 tensor editing or preserved behavior. BF16→NVFP4 and native NVFP4 are different paths. |
| **GGUF / Q8_0 / IQ2_XXS** | Bolt-24GB research edited an IQ2_XXS surface with **Q8_0 promotion**, preserved selected Q8_0 writers byte-exactly, passed the codec checks and served the artifact. Sage-S2.1-Flash research edited routed IQ2_XXS surfaces and measured poor directional retention; behavioral change was attributed principally to shared Q8 writers, with refusal still observed at α3.5. | GGUF is a container, not a universal quantization support claim. Promotion, packing and kernel constraints matter. IQ edit retention and behavioral efficacy are checkpoint-specific; these results do not prove arbitrary IQ variants or capability preservation. |
| **EXL3** | **Not yet proven for DWM editing.** A separate native-capture investigation is not completed edit support. | Do not select EXL3 on the basis of the BF16, FP8/NVFP4 or GGUF/IQ results. It requires a dedicated decode/write/readback/serve/behavior qualification. |

These are **private historical run-record claims**, summarized by the operator for orientation, except where a public recipe is linked. They are not bundled reproduction packages, current live services, independent replication or release evidence. They must not be cited as measured results produced by the open-source executor. No private model artifacts, prompts, endpoints or weights are included here.

## What changes when weights are quantized?

A DWM direction is defined in a coordinate space. A codec determines how values in that space are represented. Moving between them can change the edit itself.

1. **Establish the physical representation.** Record container, tensor name and shape, storage type, block/group size, scales, zero points, permutation/packing and backend expectations. `GGUF`, `FP8` and `NVFP4` do not each answer all of those questions.
2. **Choose and declare the edit path.** Distinguish editing full precision before quantization from native quantized editing, dequantize→edit→requantize, and promoting selected writers to a higher-precision representation. Do not silently mix paths.
3. **Capture in the intended coordinate space.** A compatible hidden size is insufficient. Check fused modules, routed/shared experts, tied weights and the writer actually consumed by the serving engine.
4. **Verify the round-trip and the write.** Compare declared writers and non-target invariants. Test scale reconstruction and readback separately from the intervention. A byte-exact tensor write proves only that write.
5. **Measure directional retention.** Requantization can absorb or distort a low-precision edit. Tensor error and directional survival are different measurements; neither is a behavioral score.
6. **Serve the exact derivative.** Pin the engine, kernels and quant path. Acceptance of a file by a loader is not serving readiness.
7. **Evaluate behavior and capability independently.** A serving result is not refusal-removal efficacy; a refusal metric is not capability preservation. The release-only operator holdout remains separate.

## Runtime support versus format research

| Capability | Current repository status |
|---|---|
| Dense BF16/F16 checkpoint intake | Qualified small dense CPU executor; float32 intermediate computation and export restoration |
| Generic safetensors support | **Not implied:** graph/writer/dtype/layout and resource limits apply |
| FP8, NVFP4, GGUF and IQ direct codecs | **Not implemented** in the current CLI; no extension-based codec detection |
| Requantization or selective promotion | **Research operation**, not a shipped generic codec API |
| vLLM/SGLang/llama.cpp readiness | **Not implemented** by the current CLI; [environment guidance](serving-stacks.md) is descriptive |
| EXL3 | **Unproven for DWM editing and unsupported by the CLI** |
| Quantized-derivative behavior/capability/release | **Unmeasured by the current pipeline**; S4/S5/S6/S7 remain skipped |

## What an adapter must prove

A future adapter should publish an exact contract: accepted container and codec, supported writer operations, packing/readback semantics, resource envelope and backend requirements. Its evidence must cover capture, edit, serialization, serving, behavior and capability separately. Negative results and rejected arms belong in the evidence record.

Until then, **unsupported formats fail closed**. The research goal is to help users navigate their quantization path without pretending every quant has identical editable surfaces—or that a successful file transformation establishes a successful DWM outcome.

See [architecture navigation](architecture-navigation.md), the [complete support matrix](support-matrix.md), [serving environments](serving-stacks.md), and the [measurement protocol](../benchmarks/README.md).
