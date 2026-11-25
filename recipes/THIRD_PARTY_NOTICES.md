# Third-party notices

## Foundational research

Arditi et al., “Refusal in Language Models Is Mediated by a Single Direction,”
NeurIPS 2024, DOI `10.52202/079017-4322`, established the central
refusal-direction result that informs this research lineage. The authors'
accompanying `andyrdt/refusal_direction` source repository is Apache-2.0.

This acknowledgment is a citation and statement of inspiration. The paper is
not relicensed by this repository. Any directly reused third-party material
must retain its own attribution and license notice in the recipe that uses it.

## Models and runtimes

The recipes reference separately distributed Qwen and GLM model checkpoints.
Those weights, tokenizers, templates, configurations, and derived model
artifacts retain their upstream licenses and acceptable-use terms.

Runtime dependencies including PyTorch, Transformers, safetensors, Hugging
Face Hub, vLLM, SGLang, CUDA, and container base images retain their respective
licenses. Each recipe's `MODEL-NOTICE.md`, `THIRD_PARTY_NOTICES.md`, lock files,
and upstream checkpoint notices control where they are more specific.

## Nested packages

Every recipe is an exact snapshot of a Blackfrost repository that was public
at import time. Nested license and notice files are preserved. The Qwen 9B
package did not expose a GitHub-detected license at import; Blackfrost licenses
its Blackfrost-authored contents in this monorepo under the root Apache-2.0
license. Third-party model and dependency terms remain separate.

