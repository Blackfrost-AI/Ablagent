"""Render original README artwork. Optional tooling: Python + Pillow (not runtime).

Usage: python3 scripts/render_readme_art.py --font /path/to/sans-serif.ttf
Artwork and this source are licensed under the repository's Apache-2.0 license.
"""

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
BG = "#0b1420"
INK = "#edf5fa"
MUTED = "#a8bccd"
ACCENT = "#74e3d0"


def render(font_path):
    OUT.mkdir(parents=True, exist_ok=True)

    def font(size):
        return ImageFont.truetype(font_path, size)

    def hero(width, height, filename):
        image = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(image)
        # An abstract writer space with one highlighted intervention direction.
        cx, cy = width * 0.81, height * 0.49
        radius = height * 0.29
        nodes = [
            (cx + math.cos(i * math.pi / 3) * radius, cy + math.sin(i * math.pi / 3) * radius)
            for i in range(6)
        ]
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                draw.line((a, b), fill="#24394c", width=2)
        draw.line((nodes[3], nodes[0]), fill=ACCENT, width=4)
        for x, y in nodes:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=ACCENT)
        draw.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=INK)
        left = 64
        draw.text((left, height * 0.15), "DWM RESEARCH / DWMIP v0", font=font(20), fill=ACCENT)
        draw.text((left - 4, height * 0.28), "Ablagent", font=font(88), fill=INK)
        draw.text((left, height * 0.57), "Architecture-aware research.", font=font(28), fill=INK)
        draw.text((left, height * 0.67), "Evidence-bound experiments.", font=font(28), fill=MUTED)
        draw.line((left, height - 52, width - left, height - 52), fill="#2b4053", width=1)
        draw.text(
            (left, height - 37),
            "RESEARCH PROTOCOL + SYNTHETIC REFERENCE RUNTIME",
            font=font(16),
            fill=MUTED,
        )
        image.save(OUT / filename)

    hero(1280, 480, "ablagent-banner.png")
    hero(1280, 640, "ablagent-social.png")
    image = Image.new("RGB", (1280, 420), BG)
    draw = ImageDraw.Draw(image)
    draw.text((48, 30), "An experiment you can inspect", font=font(32), fill=INK)
    draw.text((48, 80), "The executable synthetic reference path", font=font(20), fill=MUTED)
    labels = [
        ("01 / DECLARE", "Graph + spaces", "Explicit writer identity"),
        ("02 / PROPOSE", "Bounded plan", "No execution authority"),
        ("03 / COMPILE", "Selected update", "Baseline + direction"),
        ("04 / VERIFY", "Replay evidence", "Artifacts + trial lineage"),
    ]
    for i, (label, title, sub) in enumerate(labels):
        x = 48 + i * 302
        draw.rounded_rectangle(
            (x, 145, x + 278, 290), radius=12, fill="#142333", outline="#31485d", width=2
        )
        draw.text((x + 18, 163), label, font=font(18), fill=ACCENT)
        draw.text((x + 18, 204), title, font=font(25), fill=INK)
        draw.text((x + 18, 250), sub, font=font(17), fill=MUTED)
        if i < 3:
            draw.line((x + 281, 218, x + 298, 218), fill=ACCENT, width=2)
    draw.text(
        (48, 331),
        "Then evaluate gates: missing evidence blocks eligibility.",
        font=font(24),
        fill=INK,
    )
    draw.text(
        (48, 374),
        "Synthetic conformance is not model efficacy, "
        "safety certification, or release authorization.",
        font=font(19),
        fill=MUTED,
    )
    image.save(OUT / "ablagent-workflow.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", required=True, help="Local sans-serif TrueType/OpenType font")
    render(parser.parse_args().font)
