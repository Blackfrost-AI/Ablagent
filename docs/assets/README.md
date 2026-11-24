# Presentation assets

Original Ablagent artwork, licensed under the repository's [Apache-2.0 license](../../LICENSE). No third-party logos, photographs, or model artifacts are included. Font files are not distributed.

- `ablagent-banner.png`: 1280 × 480 README header.
- `ablagent-workflow.png`: 1280 × 420 synthetic workflow illustration. Its information is repeated as accessible text in the main README.
- `ablagent-social.png`: 1280 × 640 social-preview artwork.
- `ablagent-site-og.png`: 1200 × 630 Open Graph / Twitter link-preview image for the GitHub Pages site (`docs/index.html`).

Editable sources: [`scripts/render_readme_art.py`](../../scripts/render_readme_art.py) (README art) and
[`scripts/render_site_og.py`](../../scripts/render_site_og.py) (site OG image).

Regenerate from the repository root with Python and Pillow installed in a separate tooling environment (Pillow is not a runtime dependency):

```bash
python scripts/render_readme_art.py --font /path/to/local-sans-serif.ttf
python scripts/render_site_og.py
```

The supplied renders use Helvetica from the local system. A different licensed local sans-serif font may alter wrapping and spacing; visually inspect all three renders after regeneration. The renderer has no network calls.

## GitHub social preview

An authorized repository administrator can open **Settings → General → Social preview**, upload `ablagent-social.png`, and save. This is a manual setup asset, not an already-configured remote setting. The repository was made public by operator decision on 2026-11-24.

## GitHub Pages site

`docs/index.html` is a fully static, dependency-free site (no build step, no CDN, no tracking). The `og:image`/`twitter:image` tags point at `https://blackfrost-ai.github.io/Ablagent/assets/ablagent-site-og.png`. Pages is enabled from `main` `/docs` and live at <https://blackfrost-ai.github.io/Ablagent/>. **A published Pages site is public to anyone with the URL** — it publishes everything under `docs/`; the repository itself was made public on 2026-11-24.

The README uses local PNGs with opaque dark backgrounds, so artwork colors do not depend on GitHub's light/dark theme. Important content is repeated in Markdown for narrow screens and assistive technology. No externally hosted badges or tracking images are used.
