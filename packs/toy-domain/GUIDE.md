# Toy domain pack — guide

This is the reference example of a complete, minimal domain pack. It runs
end-to-end on CPU with the mock executor (no model, no GPU) and is used by
the test suite and the getting-started flow.

## What's in it

- `PACK.yaml` — valid manifest with pinned corpus hash, red-lines, four
  stage blocks (S0, S3, S6, S7)
- `corpora/toy-corpus-1.jsonl` — 4 prompts: 2 specialize (logistics-style
  questions), 2 baseline (general knowledge)

## How to run it (once the builder lands M2b)

```sh
ablagent run --pack toy-domain --model mock://tiny --executor mock
```

## How to copy it for a new domain

1. `cp -r packs/toy-domain packs/<your-domain>`
2. Edit `PACK.yaml`: new `pack_id`, `domain`, `title`, `summary`, `purpose`
3. Replace corpora; recompute hashes (`shasum -a 256 corpora/*.jsonl`);
   pin them under `corpora_sha256`
4. Fill in the stage blocks your domain needs (probes, gates, scripts)
5. `ablagent doctor` — must pass

## Why 4 prompts

Small enough to read in one glance, complete enough to exercise: intake
hash verification, capture corpus references, domain-verify, and the
capability floor. Real packs use ≥100 prompts per their GUIDE.
