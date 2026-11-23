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
    """Resolve only valid ArcFace near-ties using natural photo microdetail.

    The selected identity score must remain within a narrow cosine tolerance
    of both the incumbent and the strongest measurable candidate. A missing
    score, an invalid quality metric, or an inconclusive gain never replaces
    the current winner. This is a quality heuristic, not a realism percentage.
    """
    if (
        not (0 <= incumbent < len(identity_scores) == len(detail_scores))
        or not math.isfinite(identity_tolerance)
        or not 0 <= identity_tolerance <= 0.01
        or not math.isfinite(minimum_detail_gain)
        or not 0 <= minimum_detail_gain <= 1
    ):
        return incumbent

    def identity(value: float | None) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) and -1 <= value <= 1 else None

    def quality(value: float | None) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) and 0 <= value <= 1 else None

    current_identity = identity(identity_scores[incumbent])
    current_detail = quality(detail_scores[incumbent])
    if current_identity is None or current_detail is None:
        return incumbent

    verified_scores = [identity(score) for score in identity_scores]
    highest_identity = max(score for score in verified_scores if score is not None)
    if current_identity < highest_identity - identity_tolerance:
        # An existing identity winner that is not itself near the highest
        # measurement cannot be quietly replaced by an appearance-only vote.
        return incumbent

    contenders = [
        index for index, score in enumerate(verified_scores)
        if score is not None
        and score >= current_identity - identity_tolerance
        and score >= highest_identity - identity_tolerance
        and quality(detail_scores[index]) is not None
    ]
    if not contenders:
        return incumbent
    best = max(
        contenders,
        key=lambda index: (float(detail_scores[index]), float(verified_scores[index]))
    )
    if quality(detail_scores[best]) >= current_detail + minimum_detail_gain:
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

