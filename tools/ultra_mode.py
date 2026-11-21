"""Retake's always-on Ultra identity search policy.

Never feed generated pixels back into the source identity model. Each pass
receives the *original* target and same verified 25 references; only model
weight, compositing/detail strength and head fitting vary. Select the highest
verified ArcFace cosine result, preferring photorealistic texture on a near tie.

240 seconds is a search *budget*, not a mandatory delay, a real-time
guarantee, or a reason to discard a good first candidate. A CUDA operation
already in progress is allowed to finish before checking the budget again.
"""

from __future__ import annotations

import math
from typing import Any

ULTRA_VARIATIONS = (0, 1, 2, 3)
ULTRA_SEARCH_BUDGET_SECONDS = 240
ULTRA_SIMILARITY_GAIN = 0.008
ULTRA_REALISM_TIE_LOSS = 0.003


def with_variation(parsed: tuple, variation: int) -> tuple:
    """Return a fresh validated-job payload; never mutate the original target."""
    if not isinstance(parsed, tuple) or len(parsed) != 7:
        raise ValueError("Ultra işlem isteği eksik.")
    values = list(parsed)
    values[6] = {**values[6], "variation": variation}
    return tuple(values)


def score_from_headers(headers: dict[str, str]) -> float | None:
    try:
        score = float(headers.get("X-Retake-Similarity", ""))
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) and -1.0 <= score <= 1.0 else None


def _photo_detail(headers: dict[str, str]) -> int:
    # A real, non-averaged reference texture or identity-safe diffusion may
    # preserve fine skin detail. Only use this as a tiebreaker; the engine has
    # already checked ArcFace and rejected poor photographic texture.
    return (
        int(headers.get("X-Retake-Realism") == "on")
        + int(headers.get("X-Retake-Texture-Source", "average") != "average")
        + int(headers.get("X-Retake-Real-Face") == "on")
    )


def prefer_ultra_candidate(
    candidate: dict[str, str], best: dict[str, str] | None
) -> bool:
    if best is None:
        return score_from_headers(candidate) is not None
    new_score = score_from_headers(candidate)
    old_score = score_from_headers(best)
    if new_score is None:
        return False
    if old_score is None:
        return True
    if new_score > old_score + ULTRA_SIMILARITY_GAIN:
        return True
    # Never trade more than 0.003 absolute cosine similarity for an optional
    # realism/detail benefit; 0.003 is smaller than the engine's rescue gate.
    return (
        new_score >= old_score - ULTRA_REALISM_TIE_LOSS
        and _photo_detail(candidate) > _photo_detail(best)
    )


def should_continue_ultra(
    completed_rounds: int, elapsed_seconds: float,
    total_rounds: int = len(ULTRA_VARIATIONS),
    budget_seconds: float = ULTRA_SEARCH_BUDGET_SECONDS
) -> bool:
    return completed_rounds < total_rounds and elapsed_seconds < budget_seconds
