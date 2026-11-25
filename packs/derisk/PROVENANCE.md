# Derisk provenance

All corpora are synthetic, deterministic development fixtures generated during
the original synthetic-pack import (the generator is not shipped in the
open-source checkout); no AdvBench, StrongREJECT, XSTest, or other third-party
prompt text is copied. Counts and pins are intentionally explicit.

- S6 safe-capability rows: 443
- SR synthetic development fixture rows: 9 across four categories (legacy
  `holdout`/`release_redline` naming — deprecated; see
  [NAMESPACE-NOTICE.md](NAMESPACE-NOTICE.md))
- S2 capture rows: 32 direction-pair fixtures
- License: Apache-2.0 repository fixture
- Red-line enforcement: intentionally deferred; fixture labels are inert until
  quality gates are proven. A real, publishable measurement corpus lives under
  [`benchmarks/`](../../benchmarks/README.md).
