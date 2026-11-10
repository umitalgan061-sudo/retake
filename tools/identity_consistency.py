from __future__ import annotations

from math import sqrt
from statistics import median
from typing import Sequence


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    if not values or any(not (-float("inf") < value < float("inf")) for value in values):
        raise ValueError("ArcFace embedding boş veya sonlu olmayan değer içeriyor.")
    norm = sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("ArcFace embedding normu sıfır.")
    return [value / norm for value in values]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("ArcFace embedding boyutları eşleşmiyor.")
    value = sum(a * b for a, b in zip(left, right))
    return max(-1.0, min(1.0, value))


def analyze_identity_embeddings(
    embeddings: Sequence[Sequence[float]],
    file_names: Sequence[str] | None = None
) -> dict[str, object]:
    """Robust within-set consistency screening; this is not biometric proof."""
    if len(embeddings) < 3:
        raise ValueError("Kimlik tutarlılığı için en az üç embedding gerekli.")
    normalized = [_normalize(vector) for vector in embeddings]
    if len({len(vector) for vector in normalized}) != 1:
        raise ValueError("ArcFace embedding vektörlerinin boyutları eşleşmiyor.")

    count = len(normalized)
    similarities = [[1.0] * count for _ in range(count)]
    pairwise: list[float] = []
    for left in range(count):
        for right in range(left + 1, count):
            score = _cosine(normalized[left], normalized[right])
            similarities[left][right] = score
            similarities[right][left] = score
            pairwise.append(score)

    per_reference = [
        median(similarities[index][other] for other in range(count) if other != index)
        for index in range(count)
    ]
    global_median = median(pairwise)
    medoid_index = max(range(count), key=lambda index: per_reference[index])
    center = median(per_reference)
    mad = median(abs(score - center) for score in per_reference)
    cutoff = max(0.25, center - max(0.12, 4.0 * 1.4826 * mad))

    # Only strong, isolated deviations are highlighted. They remain in the
    # appearance profile and keep one identity vote; they are never silently dropped.
    outlier_indices = [
        index for index, score in enumerate(per_reference)
        if score < cutoff and score < 0.35 and center - score >= 0.15
    ]
    # Block only a severely incoherent collection; normal pose/lighting changes
    # should not be rejected by an uncalibrated universal 0.75 threshold.
    hard_conflict = global_median < 0.15
    names = list(file_names or [f"reference-{index + 1:02d}" for index in range(count)])
    if len(names) != count:
        raise ValueError("ArcFace dosya adları ve embedding sayısı eşleşmiyor.")

    return {
        "count": count,
        "median_similarity": round(float(global_median), 6),
        "medoid_index": medoid_index,
        "reference_median_similarities": [round(float(score), 6) for score in per_reference],
        "adaptive_cutoff": round(float(cutoff), 6),
        "outlier_indices": outlier_indices,
        "outlier_files": [names[index] for index in outlier_indices],
        "hard_conflict": hard_conflict,
        "method": "arcface-medoid-median-v1"
    }
