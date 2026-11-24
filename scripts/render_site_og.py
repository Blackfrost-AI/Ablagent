#!/usr/bin/env python3
"""Render the site social/OG image (1200x630) into docs/assets/.

Original artwork, Apache-2.0, same palette as render_readme_art.py.
Requires Pillow (dev-only; not a runtime dependency):
    .venv/bin/python scripts/render_site_og.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "ablagent-site-og.png"
W, H = 1200, 630

BG = (11, 20, 32)
PANEL = (20, 35, 51)
LINE = (49, 72, 93)
INK = (237, 245, 250)
MUTED = (168, 188, 205)
ACCENT = (116, 227, 208)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    mac = "/System/Library/Fonts/Supplemental/Arial{}.ttf".format(" Bold" if bold else "")
    dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf".format("-Bold" if bold else "")
    candidates = [mac, "/System/Library/Fonts/Helvetica.ttc", dejavu]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def hex_graph(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    import math

    pts = [
        (
            cx + r * math.cos(math.radians(60 * i - 90)),
            cy + r * math.sin(math.radians(60 * i - 90)),
        )
        for i in range(6)
    ]
    for i in range(6):
        draw.line([pts[i], pts[(i + 1) % 6]], fill=LINE, width=3)
    for i in range(6):
        draw.line([pts[i], pts[(i + 3) % 6]], fill=LINE, width=2)
    draw.line([pts[4], pts[0]], fill=ACCENT, width=5)
    for x, y in pts:
        draw.ellipse([x - 9, y - 9, x + 9, y + 9], fill=ACCENT)
    draw.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], fill=INK)


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # subtle panel band + border
    draw.rectangle([0, 0, W, H], outline=LINE, width=2)
    draw.rectangle([40, 40, W - 40, H - 40], outline=LINE, width=1)

    hex_graph(draw, 990, 315, 170)

    draw.text((80, 78), "DWM RESEARCH · DWMIP v0 PROTOCOL", font=font(22, bold=True), fill=ACCENT)
    draw.text((80, 130), "Ablagent", font=font(88, bold=True), fill=INK)
    draw.text(
        (80, 248), "Weight modification you can audit.", font=font(40, bold=True), fill=ACCENT
    )

    lines = [
        "Evidence-first, architecture-aware Domain Weight Modification.",
        "From abliteration research to a declared, digest-bound experiment:",
        "every gate has a purpose. No score compensates for a missing redline.",
    ]
    y = 330
    for line in lines:
        draw.text((80, y), line, font=font(26), fill=MUTED)
        y += 42

    # gate chips
    chips = ["verify first", "structural", "evaluation", "bank_publication", "redline x6"]
    x = 80
    f = font(20, bold=True)
    for chip in chips:
        w = draw.textlength(chip, font=f) + 36
        draw.rounded_rectangle([x, 490, x + w, 534], radius=22, fill=PANEL, outline=LINE, width=2)
        draw.text((x + 18, 500), chip, font=f, fill=ACCENT)
        x += w + 14

    draw.text(
        (80, 560),
        "Blackfrost-AI/Ablagent · synthetic reference runtime + real-model worked example",
        font=font(20),
        fill=MUTED,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {W}x{H})")


if __name__ == "__main__":
    main()
