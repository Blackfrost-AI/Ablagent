#!/usr/bin/env python3
"""Prepare Blackfrost brand assets for docs/assets/ from operator-supplied art.

Sources (operator-provided, Apache-2.0 with the repository):
  .scratch/brand-src-og.jpg      1792x1008  OG_social variant, Blackfrost AI lockup
  .scratch/brand-src-logo.jpg    1408x1408  ice-circuit snowflake + ABLAGENT wordmark
  .scratch/brand-src-arch.jpg    1792x1008  architecture illustration (DWM core hexes)
  .scratch/brand-src-banner.jpg  1792x1008  snowflake + ABLAGENT hero

Outputs:
  ablagent-site-og.png   1200x630  center-crop of OG source (1.905 vs 1.778 ratio)
  ablagent-social.png    1280x640  center-crop of OG source (2:1)
  ablagent-banner.png    1280x720  full banner art, 16:9
  ablagent-workflow.png  1280x720  architecture illustration, 16:9
  ablagent-logo.png      512x512   square mark
  favicon.png            64x64     square mark

Dev-only tool (Pillow); not a runtime dependency. No network calls.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / ".scratch"
OUT = ROOT / "docs" / "assets"


def center_crop(img: Image.Image, ratio: float) -> Image.Image:
    w, h = img.size
    if w / h > ratio:  # too wide: trim sides
        new_w = round(h * ratio)
        x = (w - new_w) // 2
        box = (x, 0, x + new_w, h)
    else:  # too tall: trim top/bottom evenly
        new_h = round(w / ratio)
        y = (h - new_h) // 2
        box = (0, y, w, y + new_h)
    return img.crop(box)


def emit(src_name: str, out_name: str, size: tuple[int, int]) -> None:
    img = Image.open(SRC / src_name).convert("RGB")
    img = center_crop(img, size[0] / size[1])
    img = img.resize(size, Image.LANCZOS)
    img.save(OUT / out_name, optimize=True)
    print(f"{out_name}: {size[0]}x{size[1]} <- {src_name}")


def main() -> None:
    emit("brand-src-og.jpg", "ablagent-site-og.png", (1200, 630))
    emit("brand-src-og.jpg", "ablagent-social.png", (1280, 640))
    emit("brand-src-banner.jpg", "ablagent-banner.png", (1280, 720))
    emit("brand-src-arch.jpg", "ablagent-workflow.png", (1280, 720))
    emit("brand-src-logo.jpg", "ablagent-logo.png", (512, 512))
    emit("brand-src-logo.jpg", "favicon.png", (64, 64))


if __name__ == "__main__":
    main()
