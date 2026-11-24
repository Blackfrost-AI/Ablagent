# Presentation assets

Original Ablagent artwork, licensed under the repository's [Apache-2.0 license](../../LICENSE). No third-party logos, photographs, or model artifacts are included. Font files are not distributed.

- `ablagent-banner.png`: 1280 × 480 README header.
- `ablagent-workflow.png`: 1280 × 420 synthetic workflow illustration. Its information is repeated as accessible text in the main README.
- `ablagent-social.png`: 1280 × 640 social-preview artwork.

Editable source: [`scripts/render_readme_art.py`](../../scripts/render_readme_art.py).

Regenerate from the repository root with Python and Pillow installed in a separate tooling environment (Pillow is not a runtime dependency):

```bash
python scripts/render_readme_art.py --font /path/to/local-sans-serif.ttf
```

The supplied renders use Helvetica from the local system. A different licensed local sans-serif font may alter wrapping and spacing; visually inspect all three renders after regeneration. The renderer has no network calls.

## GitHub social preview

An authorized repository administrator can open **Settings → General → Social preview**, upload `ablagent-social.png`, and save. This is a manual setup asset, not an already-configured remote setting. Keep the repository private until a separate publication decision.

The README uses local PNGs with opaque dark backgrounds, so artwork colors do not depend on GitHub's light/dark theme. Important content is repeated in Markdown for narrow screens and assistive technology. No externally hosted badges or tracking images are used.
