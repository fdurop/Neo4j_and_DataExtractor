"""Prepare small or rotated table images for specialist recognition."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class TableImageVariant:
    path: str
    rotation: int
    scale: float


class TableImagePreprocessor:
    """Create high-resolution candidates, trying rotations only on demand."""

    def __init__(self, target_short_side: int = 1600, max_long_side: int = 3200):
        self.target_short_side = target_short_side
        self.max_long_side = max_long_side

    def variants(self, image_path: str) -> Iterable[TableImageVariant]:
        for rotation in (0, 90, 270):
            yield self.prepare(image_path, rotation)

    def prepare(self, image_path: str, rotation: int = 0) -> TableImageVariant:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if rotation:
                image = image.rotate(rotation, expand=True, fillcolor="white")

            width, height = image.size
            short_side = max(1, min(width, height))
            long_side = max(width, height)
            scale = max(1.0, self.target_short_side / short_side)
            scale = min(scale, self.max_long_side / max(1, long_side))
            if scale > 1.01:
                image = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.Resampling.LANCZOS,
                )

            image = ImageOps.autocontrast(image, cutoff=0.5)
            image = ImageEnhance.Contrast(image).enhance(1.08)
            image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=135, threshold=3))

            stem, _ = os.path.splitext(image_path)
            output_path = f"{stem}_table_ready_r{rotation}.png"
            image.save(output_path, format="PNG", optimize=True)

        return TableImageVariant(output_path, rotation, round(scale, 4))
