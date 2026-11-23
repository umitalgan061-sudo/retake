"""Image-based regional detail preservation check for Retake outputs.

This is *not* a biometric identity score. ArcFace measures global identity,
while this module checks whether the edited photo preserves photographic
high-frequency information (eyes, nose, mouth, lower face). Keeping detail
relative to the original capture helps avoid waxy/blurred faces. No fixed
beard, age, or person-specific assumptions are inferred from a single photo.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Relative positions within the detected face bounding rectangle, deliberately
# broad to accommodate modest head pose. Avoid clothing/background.
REGIONS = {
    "eyes": (0.12, 0.24, 0.88, 0.44),
    "nose": (0.30, 0.37, 0.70, 0.63),
    "mouth": (0.22, 0.57, 0.78, 0.78),
    "lower_face": (0.17, 0.72, 0.83, 0.92),
}


def choose_detail_safe_tie(
    identity_scores: list[float | None],
    detail_scores: list[float | None],
    incumbent: int,
    identity_tolerance: float = 0.003,
    minimum_detail_gain: float = 0.025,
) -> int:
    """Break an ArcFace near-tie with photographic detail, never replace a clear winner.

    Identity remains the primary safety gate. This narrow tie-break exists
    because a cosmetically smoother candidate can score virtually the same in
    ArcFace while visibly losing beard, skin and hair-adjacent texture.
    """
    if not (0 <= incumbent < len(identity_scores) == len(detail_scores)):
        return incumbent
    identity = identity_scores[incumbent]
    incumbent_detail = detail_scores[incumbent]
    if identity is None or incumbent_detail is None:
        return incumbent
    eligible = [
        index for index, (score, detail) in enumerate(zip(identity_scores, detail_scores))
        if score is not None and detail is not None and score >= identity - identity_tolerance
    ]
    if not eligible:
        return incumbent
    best = max(eligible, key=lambda index: (float(detail_scores[index]), float(identity_scores[index])))
    if float(detail_scores[best]) >= incumbent_detail + minimum_detail_gain:
        return best
    return incumbent


def _detail_energy(gray: np.ndarray) -> float:
    if min(gray.shape[:2]) < 8:
        return 0.0
    signal = gray.astype(np.float32)
    high = signal - cv2.GaussianBlur(signal, (0, 0), 1.25)
    return math.sqrt(float(np.mean(np.square(high))))


def regional_photo_detail(
    original: np.ndarray, edited: np.ndarray,
    bounds: dict[str, Any],
) -> tuple[float | None, dict[str, float]]:
    """Return a 0..1 photographic-detail score and per-region energy ratios.

    The ratio is bounded to avoid treating extra sharpening as more realistic.
    A low-light or motion-blurred original does not get unfairly penalized for
    having little baseline texture. Use for near-tie reranking, not as proof of
    matching facial features.
    """
    if (
        original is None or edited is None or original.shape != edited.shape
        or original.ndim != 3 or original.shape[2] < 3
    ):
        return None, {}
    height, width = original.shape[:2]
    try:
        x, y = float(bounds["x"]), float(bounds["y"])
        fw, fh = float(bounds["width"]), float(bounds["height"])
    except (TypeError, ValueError, KeyError):
        return None, {}
    if not all(math.isfinite(v) for v in (x, y, fw, fh)) or min(fw, fh) < 20:
        return None, {}

    source = cv2.cvtColor(original[:, :, :3], cv2.COLOR_BGR2GRAY)
    result = cv2.cvtColor(edited[:, :, :3], cv2.COLOR_BGR2GRAY)
    ratios: dict[str, float] = {}
    qualities: list[float] = []
    for name, (left, top, right, bottom) in REGIONS.items():
        x0 = max(0, min(width, int(x + fw * left)))
        x1 = max(0, min(width, int(x + fw * right)))
        y0 = max(0, min(height, int(y + fh * top)))
        y1 = max(0, min(height, int(y + fh * bottom)))
        if min(x1 - x0, y1 - y0) < 8:
            continue
        baseline = _detail_energy(source[y0:y1, x0:x1])
        changed = _detail_energy(result[y0:y1, x0:x1])
        if baseline < 1.25:
            continue
        ratio = (changed + 0.25) / (baseline + 0.25)
        ratios[name] = round(ratio, 4)
        # Penalize loss (waxy), but also oversharpening (halos), equally.
        quality = math.exp(-0.72 * abs(math.log(max(1e-6, ratio))))
        qualities.append(quality)
    if not qualities:
        return None, ratios
    return round(sum(qualities) / len(qualities), 5), ratios


def check_photo_detail(
    original_path: Path, result_path: Path,
    bounds: dict[str, Any],
) -> tuple[float | None, dict[str, float]]:
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    result = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
    return regional_photo_detail(original, result, bounds)

