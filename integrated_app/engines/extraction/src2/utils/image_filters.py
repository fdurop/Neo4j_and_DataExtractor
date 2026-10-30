from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


@dataclass
class ImageFilterDecision:
    keep: bool
    reason: str
    stats: Dict[str, float]


def _rgb_on_white(img: Image.Image) -> Image.Image:
    """Convert transparent teaching graphics without turning transparency black."""
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in {"RGBA", "LA"} or "transparency" in img.info:
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def _has_structured_foreground(gray: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """Protect sparse strokes before applying low-information heuristics.

    A mostly white (or black) canvas is normal for diagrams. Require a small
    coherent foreground component rather than a minimum percentage of ink.
    Eight-neighbour connectivity preserves one-pixel diagonal strokes. The
    absolute support/span checks reject isolated pixels and tiny specks; they
    make no claim about educational relevance. Stop once support is established
    to avoid traversing a large connected region unnecessarily.
    """
    height, width = gray.shape
    background = float(np.median(gray))
    foreground = np.abs(gray - background) >= 8.0
    foreground_pixels = int(foreground.sum())
    stats = {
        "foreground_pixels": float(foreground_pixels),
        "foreground_contrast_threshold": 8.0,
        "largest_examined_component_pixels": 0.0,
        "largest_examined_component_span": 0.0,
        "structured_foreground": 0.0,
    }
    if foreground_pixels < 32:
        return False, stats

    remaining = foreground.reshape(-1)
    # Bounded scanning chunks avoid materializing every foreground coordinate
    # in a large low-contrast image. Each visited pixel is cleared immediately.
    for offset in range(0, remaining.size, 65536):
        starts = np.flatnonzero(remaining[offset:offset + 65536]) + offset
        for start in starts:
            if not remaining[start]:
                continue
            remaining[start] = False
            pending = [int(start)]
            count = 0
            min_x = max_x = int(start) % width
            min_y = max_y = int(start) // width
            while pending:
                pixel = pending.pop()
                y, x = divmod(pixel, width)
                count += 1
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                span = max(max_x - min_x + 1, max_y - min_y + 1)
                if count >= 32 and span >= 12:
                    stats.update(largest_examined_component_pixels=float(count),
                                 largest_examined_component_span=float(span),
                                 structured_foreground=1.0)
                    return True, stats
                for next_y in range(max(0, y - 1), min(height, y + 2)):
                    for next_x in range(max(0, x - 1), min(width, x + 2)):
                        neighbour = next_y * width + next_x
                        if remaining[neighbour]:
                            remaining[neighbour] = False
                            pending.append(neighbour)
            stats["largest_examined_component_pixels"] = max(stats["largest_examined_component_pixels"], float(count))
            stats["largest_examined_component_span"] = max(stats["largest_examined_component_span"], float(span))
    return False, stats


def analyze_image(image_path: str) -> ImageFilterDecision:
    try:
        if not os.path.exists(image_path):
            return ImageFilterDecision(False, "missing_file", {})

        with Image.open(image_path) as img:
            img = _rgb_on_white(img)
            width, height = img.size
            arr = np.asarray(img, dtype=np.uint8)

        if width <= 0 or height <= 0:
            return ImageFilterDecision(False, "invalid_size", {"width": width, "height": height})

        gray = arr.mean(axis=2)
        mean = float(gray.mean())
        std = float(gray.std())
        black_ratio = float((gray < 18).mean())
        white_ratio = float((gray > 245).mean())
        mid_ratio = float(((gray >= 18) & (gray <= 245)).mean())

        edge_x = float(np.abs(np.diff(gray, axis=1)).mean()) if width > 1 else 0.0
        edge_y = float(np.abs(np.diff(gray, axis=0)).mean()) if height > 1 else 0.0
        edge_score = (edge_x + edge_y) / 2.0

        non_white = 1.0 - white_ratio
        area = width * height
        min_dim = min(width, height)

        stats = {
            "width": float(width),
            "height": float(height),
            "area": float(area),
            "mean": mean,
            "std": std,
            "black_ratio": black_ratio,
            "white_ratio": white_ratio,
            "mid_ratio": mid_ratio,
            "edge_score": edge_score,
        }

        if area < 9000 or min_dim < 60:
            return ImageFilterDecision(False, "too_small", stats)

        low_information_candidate = (
            std < 5.0 or black_ratio > 0.92 or white_ratio > 0.97
            or (non_white < 0.02 and area < 40000)
            or (edge_score < 1.2 and std < 18.0)
        )
        if low_information_candidate:
            supported, foreground_stats = _has_structured_foreground(gray)
            stats.update(foreground_stats)
            if supported:
                return ImageFilterDecision(True, "keep_structured_foreground", stats)

        if std < 5.0:
            return ImageFilterDecision(False, "flat_image", stats)

        if black_ratio > 0.92 or white_ratio > 0.97:
            return ImageFilterDecision(False, "mostly_black_or_white", stats)

        if non_white < 0.02 and area < 40000:
            return ImageFilterDecision(False, "almost_blank", stats)

        if edge_score < 1.2 and std < 18.0:
            return ImageFilterDecision(False, "low_information", stats)

        return ImageFilterDecision(True, "keep", stats)

    except Exception as exc:
        return ImageFilterDecision(False, f"analysis_failed:{exc}", {})


def should_keep_image(image_path: str) -> Tuple[bool, str, Dict[str, float]]:
    decision = analyze_image(image_path)
    return decision.keep, decision.reason, decision.stats


def image_signature(image_path: str) -> str:
    """Stable signature used for repeated logo/watermark removal."""
    try:
        suffix = os.path.splitext(image_path)[1].lower()
        if suffix in {".emf", ".wmf", ".svg"}:
            with open(image_path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
            return f"vector:{suffix}:{digest}"

        with Image.open(image_path) as img:
            img = _rgb_on_white(img)
            width, height = img.size
            small = img.convert("L").resize((16, 16))
            values = np.asarray(small, dtype=np.float32).reshape(-1)
            avg = float(values.mean())
            bits = "".join("1" if value > avg else "0" for value in values)
            ahash = hex(int(bits, 2))[2:].zfill(64)
        file_size = os.path.getsize(image_path)
        return f"raster:{width}x{height}:{file_size}:{ahash}"
    except Exception:
        try:
            with open(image_path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
            return f"fallback:{digest}"
        except Exception:
            return f"missing:{image_path}"


def repeated_image_paths(image_paths: Iterable[str], min_count: int = 3) -> Dict[str, str]:
    """Return {path: signature} for images repeated enough to be layout noise."""
    path_signature = {path: image_signature(path) for path in image_paths if os.path.exists(path)}
    counts = Counter(path_signature.values())
    return {
        path: signature
        for path, signature in path_signature.items()
        if counts[signature] >= min_count
    }
