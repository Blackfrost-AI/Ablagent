# Bank roles, licensing, and release holdouts

## Roles

| Role | Purpose | Public bundling |
|---|---|---|
| `capture` | Balanced paired observations for candidate derivation | If explicitly licensed |
| `calibration` | Geometry, thresholds, and candidate selection | Separate from capture/evaluation |
| `evaluation` | Held-out behavioral or synthetic comparisons | If explicitly licensed |
| `mock_release_redline` | Inert public policy-schema fixtures | Yes; never a real gate measurement |
| `release_redline` | Independent private release evaluation | Never |

The v0 capture manifest requires at least four complete benign/hostile pairs with balanced counts. This is a minimal synthetic fixture contract, not a claim that four pairs are sufficient for language-model research.

The `prompt` field in the synthetic capture bank is a numeric JSON input. Labels are fixture semantics, not moral or safety judgments.

## Separation and contamination

Do not reuse a capture bank as evaluation or release evidence. Track source families, pair identities, transformations, and split policy in each manifest. Freeze selection rules and retain every trial. Use external tooling for deduplication, paraphrase-robust overlap checks, and private-bank access controls; v0 does not claim to solve contamination detection.

## Provenance is more than a URL

Every bundled bank declares:

- Source and version/lineage.
- Copyright, SPDX identifier, and license-text path.
- Explicit redistribution permission.
- Transformation/annotation lineage.
- Generator and applicable output terms where relevant.

Public availability does not establish redistribution permission. Generated content is not automatically unencumbered. Unknown permission blocks public bundling. v0 permits only reviewed `CC0-1.0` and `CC-BY-4.0` public packs; this is a conservative default, not a general legal compatibility decision.

## Private gates

Keep actual red-line prompts, approval keys, and protected evaluation history outside this repository. A visibility field guides eligibility only; it is not encryption or access control. Public code cannot prevent modified forks from bypassing gates. Private infrastructure can only govern an organization's own official release attestation.
