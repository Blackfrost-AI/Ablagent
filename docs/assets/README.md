# Presentation assets

Blackfrost brand artwork commissioned by the operator for this repository, licensed under the repository's [Apache-2.0 license](../../LICENSE). No third-party logos, photographs, or model artifacts are included. Font files are not distributed.

- `ablagent-banner.png`: 1280 × 720 README header — ice-circuit snowflake, ABLAGENT wordmark, "Domain Weight Modification · Evidence-First Runtime".
- `ablagent-workflow.png`: 1280 × 720 architecture illustration (observations / rank-one updates → DWM core → weight bank / agent runtime / feedback loop / evaluation). Brand artwork, not a literal stage diagram; the exact executable stage sequence is the numbered list beside it in the main README.
- `ablagent-social.png`: 1280 × 640 GitHub social-preview artwork (Blackfrost AI lockup).
- `ablagent-site-og.png`: 1200 × 630 Open Graph / Twitter link-preview image for the GitHub Pages site (`docs/index.html`).
- `ablagent-logo.png`: 512 × 512 square ice-circuit mark (site brand, apple-touch-icon).
- `favicon.png`: 64 × 64 site favicon.

## Sources and regeneration

The current renders are prepared from operator-supplied Blackfrost source art (1792 × 1008 JPEGs and a 1408 × 1408 square mark) by [`scripts/prepare_brand_assets.py`](../../scripts/prepare_brand_assets.py). It center-crops to each target aspect ratio and resizes with Lanczos; Pillow is a dev-only tool, not a runtime dependency, and the script makes no network calls. The source JPEGs are operator assets kept outside the repository; the committed PNGs are the canonical distributable renders.

The earlier programmatic renders ([`scripts/render_readme_art.py`](../../scripts/render_readme_art.py), [`scripts/render_site_og.py`](../../scripts/render_site_og.py)) are retained as history. **Do not run them to "refresh" assets — they would overwrite the Blackfrost brand renders with the old placeholder art.**

## GitHub social preview

An authorized repository administrator can open **Settings → General → Social preview**, upload `ablagent-social.png`, and save. This is a manual setup asset, not an already-configured remote setting. The repository was made public by operator decision on 2026-11-24.

## GitHub Pages site

`docs/index.html` is a fully static, dependency-free site (no build step, no CDN, no tracking). The `og:image`/`twitter:image` tags point at `https://blackfrost-ai.github.io/Ablagent/assets/ablagent-site-og.png`. Pages is enabled from `main` `/docs` and live at <https://blackfrost-ai.github.io/Ablagent/>. **A published Pages site is public to anyone with the URL** — it publishes everything under `docs/`; the repository itself was made public on 2026-11-24.

The README uses local PNGs with opaque dark backgrounds, so artwork colors do not depend on GitHub's light/dark theme. Important content is repeated in Markdown for narrow screens and assistive technology. No externally hosted badges or tracking images are used.