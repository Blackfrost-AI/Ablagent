# Benchmark corpus provenance

These are **prompt-only** research corpora. This repository ships no model
responses, scores, judgments, or generated answers for them. All three files
under `corpora/` are vendored byte-for-byte from pinned public-research
material/first-party probes; `validate.py` enforces their SHA-256 digests and
that no other corpus or answer-like column can be silently added.

## License scope and attribution

| Dataset | Rows in 443 | Upstream | Terms |
|---|---|---|---|
| AdvBench `harmful_behaviors` | 146 | [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks) — Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models* | MIT — verbatim text in [LICENSES/AdvBench-MIT.txt](LICENSES/AdvBench-MIT.txt), copyright © 2023 Andy Zou et al. |
| StrongREJECT | 149 | [alexandrasouly/strongreject](https://github.com/alexandrasouly/strongreject) — Souly et al., *StrongREJECT: A Benchmark for Measuring Faithful Refusal*, 2024 | MIT — verbatim text in [LICENSES/StrongREJECT-MIT.txt](LICENSES/StrongREJECT-MIT.txt), copyright © 2024 Alexandra Souly |
| XSTest | 148 (75 safe, 73 unsafe) | [paul-rottger/xstest](https://github.com/paul-rottger/xstest) — Röttger et al., *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*, NAACL 2024, [ACL Anthology](https://aclanthology.org/2024.naacl-long.301/) | CC BY 4.0 — full legal code in [LICENSES/CC-BY-4.0-legalcode.txt](LICENSES/CC-BY-4.0-legalcode.txt) |

Attribution requirements are satisfied by this notice, the dataset-level
citations above, the links to each upstream repository/paper, and the verbatim
license texts. Redistribution and commercial use are permitted by MIT and
CC BY 4.0; CC BY 4.0 has **no ShareAlike obligation**.

**Modification/adaptation notice (CC BY 4.0 §3):** the XSTest rows here are a
**selected subset**, rows were re-keyed into the flat `id,dataset,prompt,label`
schema of the composite 443 corpus, and category labels were normalized. Prompt
text is preserved; the arrangement is a derivative. The upstream dataset's
safe/unsafe labels were retained as `label`. No per-row upstream identifier is
present in the composite corpus; **per-row source attribution is therefore not
claimed** — attribution is at dataset level. No endorsement by the upstream
authors is implied; the disclaimer and license link above apply.

The cyber-10 probes are first-party authored research material published
under this repository's [Apache-2.0 license](../LICENSE); they are not derived
from the upstream datasets. Their prompts deliberately ask for reasoning on
offensive-security topics; they are **questions, not instructions or answers**.

Full Notices: [NOTICE.md](NOTICE.md).

## 443 evaluation corpus

`R1-FALSE-REFUSAL-443-v1` is a composite, prompt-only false-refusal and
negative-control corpus:

- 146 AdvBench harmful-behavior prompts (`label: harmful`)
- 149 StrongREJECT forbidden prompts (`label: harmful`)
- 148 XSTest prompts (`label: safe` or `label: unsafe`)
- Every row carries a stable `id` of the form `r1fr443-NNNN`, a `dataset`
  field, `prompt`, and `label`. No response, rubric, expected answer, or score.

Selection note: the original 450-row three-way composite (150 each) had seven
rows removed during its curation for safety categorization; the resulting
443-row prompt set is the measurement corpus used in published Blackfrost
run-record reports. That removal is a historical curation fact, not a policy
artifact of this repository. **An operator-controlled, release-only red-line
corpus is intentionally absent and was not opened or copied**; it is not part
of the 443, the smoke-50, or any public fixture.

Historical figures such as `266/295 harmful-prompt compliance (90.2%)` and
smoke-50 alpha-ladder numbers elsewhere in this repository are **run-record
claims** about specific past DWM builds. They are not measurements produced by
this repository, and no code here reproduces them without a served model,
frozen responses, and an operator-run evaluation. See [protocol/SCORING.md](protocol/SCORING.md).

## Smoke-50

`R1-SMOKE-50-v1` is the **first 50 prompt strings** of the 443 corpus in
order, all AdvBench harmful rows, in the slim `dataset,prompt,label` schema
(no `id`). It is a fast wiring/serve smoke sequence, **not an independent
sample**. Any tuning performed on it contaminates the 443 evaluation split;
record such tuning explicitly.

## Cyber-10

`CYBER-REASONING-10-v1` is ten first-party security reasoning probes spanning
malware analysis, detection engineering, wallet/Bech32 bug hunting, incident
triage, and beacon/cron analysis. Rows carry `id,topic,prompt`. No expected
answers, refusal labels, or score key are shipped: the probe is **qualitative
only**. A historical run record notes 6/10 answered, 0/10 refused, 4/10
degenerate (reasoning loops) on one model build — that is a claim about that
run, not a property of this corpus.

## Rebuild / verification receipt

Transformation applied at import: **none** — each source file was copied
byte-for-byte; the on-disk SHA-256 values in [REGISTRY.yaml](REGISTRY.yaml)
are the source digests. Verify with:

```bash
python benchmarks/validate.py --strict
```

The verifier checks all three digests, row counts, exact field schemas,
corpus-ID integrity, that smoke-50's prompt sequence equals the first 50
prompts of the 443, that no answer/score/credential/private-path/protected
namespace fields exist, and that the license files are present. A change to
any corpus byte, license text, or registry entry requires a new versioned
corpus id and an updated registry; do not mutate these files in place.
