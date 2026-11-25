# Ablagent benchmark corpora — notices

## Third-party research datasets

- **AdvBench / llm-attacks** — Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models*. © 2023 Andy Zou et al. MIT. <https://github.com/llm-attacks/llm-attacks> · license text: `LICENSES/AdvBench-MIT.txt`.
- **StrongREJECT** — Souly et al., *StrongREJECT: A Benchmark for Measuring Faithful Refusal*, 2024. © 2024 Alexandra Souly. MIT. <https://github.com/alexandrasouly/strongreject> · license text: `LICENSES/StrongREJECT-MIT.txt`.
- **XSTest** — Röttger et al., *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*, NAACL 2024. <https://github.com/paul-rottger/xstest> · <https://aclanthology.org/2024.naacl-long.301/> · CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/legalcode> · full legal code: `LICENSES/CC-BY-4.0-legalcode.txt`.

Selected XSTest prompts were adapted: the subset was re-keyed into the flat
composite corpus schema, category labels normalized, and the rows given
anonymous composite IDs. Attribution is at dataset level; no per-row upstream
identifiers are present. This adaptation is provided under the same CC BY 4.0
license with the notices above retained. The upstream authors do not endorse
this repository or its DWM method. No additional restrictions are applied to
the CC BY 4.0 material.

## First-party material

`CYBER-REASONING-10-v1` and all Ablagent-authored documentation, tooling,
schemas, and fixtures in `benchmarks/` are © 2026 Blackfrost AI, licensed under
the repository [Apache-2.0 license](../LICENSE), except where a file states
otherwise. Prompt files contain questions only; no responses or answers are
distributed.

## Safety boundary

An operator-controlled, release-only red-line corpus is intentionally absent
from this repository. The synthetic development fixture under `packs/derisk/` is
a separate public fixture with deprecated legacy vocabulary; see
[`packs/derisk/NAMESPACE-NOTICE.md`](../packs/derisk/NAMESPACE-NOTICE.md).
