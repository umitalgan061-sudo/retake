"""Conservative photographic artifact checks for optional diffusion renders.

A high ArcFace score is necessary but insufficient: a generator may produce
smooth, clipped or harsh skin with a numerically high resemblance score. These
simple non-ML checks compare the candidate with the *already composited* face
at two cheek regions. They reject clear texture regressions but never assert
that an image is authentic or that the depicted identity is proven.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _cheek_mask(shape: tuple[int, int], bounds: dict[str, Any]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    x, y = float(bounds["x"]), float(bounds["y"])
    fw, fh = float(bounds["width"]), float(bounds["height"])
    if any(not math.isfinite(v) for v in (x, y, fw, fh)) or fw < 1 or fh < 1:
        raise ValueError("Yüz sınırları geçersiz.")
    # Stay away from eyes, nose, lips, jaw, hair and crop border.
    for left, right in ((0.16, 0.38), (0.62, 0.84)):
        x0 = max(0, min(w, int(x + fw * left)))
        x1 = max(0, min(w, int(x + fw * right)))
        y0 = max(0, min(h, int(y + fh * 0.34)))
        y1 = max(0, min(h, int(y + fh * 0.64)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    return mask.astype(bool)


def photograph_texture_metrics(
    image: np.ndarray,
    bounds: dict[str, Any],
) -> dict[str, float] | None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        return None
    cheeks = _cheek_mask(image.shape[:2], bounds)
    if int(cheeks.sum()) < 80:
        return None  # too small to assess honestly
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high = grey - cv2.GaussianBlur(grey, (0, 0), 1.3)
    # RMS of photographic micro-detail; not dominated by a single sharp lip
    # or high-contrast iris because those areas aren't in the cheek mask.
    detail = math.sqrt(float(np.mean(high[cheeks] ** 2)))
    rgb = image[cheeks]
    clipped = float(np.mean(np.all(rgb >= 252, axis=1) | np.all(rgb <= 3, axis=1)))
    return {"detail": detail, "clipped": clipped}


def photographic_render_acceptable(
    base: np.ndarray,
    candidate: np.ndarray,
    bounds: dict[str, Any],
) -> tuple[bool, str]:
    """Reject strong skin smoothing, excessive sharpening and clipped skin."""
    if base is None or candidate is None or base.shape != candidate.shape:
        return False, "boyutlar eşleşmiyor"
    original = photograph_texture_metrics(base, bounds)
    rendered = photograph_texture_metrics(candidate, bounds)
    if original is None or rendered is None:
        return True, "doku alanı küçük; korumacı kontrol uygulanamadı"
    if original["detail"] >= 2.0 and rendered["detail"] < original["detail"] * 0.45:
        return False, "yüz derisi aşırı düzleştirilmiş"
    if original["detail"] >= 2.0 and rendered["detail"] > max(20.0, original["detail"] * 5.0):
        return False, "doğal olmayan aşırı keskinlik"
    if rendered["clipped"] > max(0.12, original["clipped"] + 0.08):
        return False, "ciltte aşırı siyah/beyaz kırpılması"
    return True, "doku korundu"


def check_render_file(
    base_path: Path, render_path: Path, bounds: dict[str, Any]
) -> tuple[bool, str]:
    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    render = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
    return photographic_render_acceptable(base, render, bounds)
