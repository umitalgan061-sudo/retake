"""Always-on seven-minute Retake Ultra search policy.

One identity transfer always starts from the unmodified original crop and the
verified 25 reference photographs. Results are *never* recursively used as a
new target. Search across distinct identity/model-weight and head-fit options,
keeping the best verified result; one candidate is returned to the browser.

The seven-minute minimum is a user-requested elapsed-time contract, not a
promise that extra model calls magically improve identity. CUDA passes can
take longer than seven minutes, and verified jobs may spend the remaining
interval in the clearly labelled final waiting phase.
"""
from __future__ import annotations

import math
from typing import Any

# 24 genuinely different blend/weight/head-shape candidates. The execution
# budget cuts off new CUDA rounds after 420s; active CUDA work isn't cancelled.
ULTRA_VARIATIONS = tuple(range(24))
ULTRA_MIN_SECONDS = 420
ULTRA_SEARCH_BUDGET_SECONDS = 420
ULTRA_REALISM_TIE_LOSS = 0.002
ULTRA_SCORE_EPS = 0.001
ULTRA_DETAIL_GAIN = 0.06


def with_variation(parsed: tuple, variation: int) -> tuple:
    """Deep-copy only the metadata dictionary, preserving original image bytes."""
    if not isinstance(parsed, tuple) or len(parsed) != 7:
        raise ValueError("Ultra işlem isteği eksik.")
    if not isinstance(variation, int) or variation not in ULTRA_VARIATIONS:
        raise ValueError("Ultra varyasyonu geçersiz.")
    if not isinstance(parsed[6], dict):
        raise ValueError("Ultra metadata alanı geçersiz.")
    values = list(parsed)
    values[6] = {**parsed[6], "variation": variation}
    return tuple(values)


def score_from_headers(headers: dict[str, str]) -> float | None:
    try:
        score = float(headers.get("X-Retake-Similarity", ""))
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) and -1.0 <= score <= 1.0 else None


def detail_from_headers(headers: dict[str, str]) -> float | None:
    try:
        score = float(headers.get("X-Retake-Detail-Score", ""))
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def _photo_detail(headers: dict[str, str]) -> int:
    return (
        int(headers.get("X-Retake-Realism") == "on")
        + int(headers.get("X-Retake-Texture-Source", "average") != "average")
        + int(headers.get("X-Retake-Real-Face") == "on")
    )


def prefer_ultra_candidate(
    candidate: dict[str, str], best: dict[str, str] | None
) -> bool:
    """Identity-first with *bounded* regional sharpness / realism tie-breaking.

    ArcFace cosine is NOT a calibrated percentage or proof of identity.
    Do not discard a measurable gain solely because it is smaller than an
    arbitrary old threshold. Also don't trade meaningful identity for beauty.
    """
    new = score_from_headers(candidate)
    if new is None:
        return False
    if best is None:
        return True
    old = score_from_headers(best)
    if old is None:
        return True
    if new > old + ULTRA_REALISM_TIE_LOSS:
        return True
    if old > new + ULTRA_REALISM_TIE_LOSS:
        return False

    new_detail = detail_from_headers(candidate)
    old_detail = detail_from_headers(best)
    if new_detail is not None and old_detail is not None:
        if new_detail > old_detail + ULTRA_DETAIL_GAIN and new >= old - ULTRA_REALISM_TIE_LOSS:
            return True
        if old_detail > new_detail + ULTRA_DETAIL_GAIN and new <= old + ULTRA_SCORE_EPS:
            return False

    if new > old + ULTRA_SCORE_EPS:
        return True
    if old > new + ULTRA_SCORE_EPS:
        # A near tie can still select a reliably more photographic candidate.
        return _photo_detail(candidate) > _photo_detail(best)
    return _photo_detail(candidate) > _photo_detail(best)


def should_continue_ultra(
    completed_rounds: int, elapsed_seconds: float,
    total_rounds: int = len(ULTRA_VARIATIONS),
    budget_seconds: float = ULTRA_SEARCH_BUDGET_SECONDS
) -> bool:
    return 0 <= completed_rounds < total_rounds and elapsed_seconds < budget_seconds
