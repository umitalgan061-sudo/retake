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
# Require a visible 2.5-percentage-point residual halo improvement in an
# identity near-tie before preferring a cleaner photographic boundary.
ULTRA_RIM_GAIN = 0.025
# Photographic naturalness (tools/photo_naturalness.py, 0-100). A candidate
# that clearly looks less like a photograph may not win on a small ArcFace
# lead; one below the floor loses to a natural one that is nearly as close.
ULTRA_NATURALNESS_GAIN = 6.0         # points that count as clearly more natural
ULTRA_NATURALNESS_BAND = 0.010       # ArcFace lead such a gain outweighs
ULTRA_NATURALNESS_FLOOR = 65.0       # below this a result reads as edited
ULTRA_NATURALNESS_FLOOR_BAND = 0.030 # ArcFace lead the floor outweighs
# NVIDIA vision judge (tools/nvidia_judge.py, 0-10). It re-ranks only the
# identity near-ties; both order-reversed passes must prefer the switch.
ULTRA_JUDGE_GAIN = 1.0               # realism points (of 10) a switch needs
ULTRA_JUDGE_POOL = 3                 # candidates shown to the judge


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


def rim_from_headers(headers: dict[str, str]) -> float | None:
    """Uncalibrated residual dark seam fraction (lower is visually better)."""
    try:
        value = float(headers.get("X-Retake-Rim-Fraction", ""))
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def naturalness_from_headers(headers: dict[str, str]) -> float | None:
    try:
        value = float(headers.get("X-Retake-Naturalness", ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0.0 <= value <= 100.0 else None


def _naturalness_decision(candidate: dict[str, str], best: dict[str, str], new: float, old: float) -> bool | None:
    """True/False when naturalness settles the comparison, None otherwise."""
    new_nat = naturalness_from_headers(candidate)
    old_nat = naturalness_from_headers(best)
    if new_nat is None or old_nat is None:
        return None
    floor = ULTRA_NATURALNESS_FLOOR
    if new_nat >= floor > old_nat and new >= old - ULTRA_NATURALNESS_FLOOR_BAND:
        return True
    if old_nat >= floor > new_nat and old >= new - ULTRA_NATURALNESS_FLOOR_BAND:
        return False
    if abs(new - old) <= ULTRA_NATURALNESS_BAND:
        if new_nat >= old_nat + ULTRA_NATURALNESS_GAIN:
            return True
        if old_nat >= new_nat + ULTRA_NATURALNESS_GAIN:
            return False
    return None


def identity_slack(headers: dict[str, str], strongest: dict[str, str] | None) -> float:
    """How far below the strongest-identity round a candidate may stay.

    Normally a near-tie (ULTRA_REALISM_TIE_LOSS). A candidate that looks
    clearly more like a photograph than the strongest round earns the same
    room as the naturalness rule in prefer_ultra_candidate.
    """
    if strongest is None:
        return ULTRA_REALISM_TIE_LOSS
    mine = naturalness_from_headers(headers)
    theirs = naturalness_from_headers(strongest)
    if mine is None or theirs is None:
        return ULTRA_REALISM_TIE_LOSS
    if mine >= ULTRA_NATURALNESS_FLOOR > theirs:
        return ULTRA_NATURALNESS_FLOOR_BAND
    if mine >= theirs + ULTRA_NATURALNESS_GAIN:
        return ULTRA_NATURALNESS_BAND
    return ULTRA_REALISM_TIE_LOSS


def judge_pool(best: tuple, others: list[tuple], limit: int = ULTRA_JUDGE_POOL) -> list[tuple]:
    """The current best first, then the strongest identity near-ties.

    Items are (image bytes, headers, label). Only candidates within
    ULTRA_NATURALNESS_BAND ArcFace of the best and not clearly less natural
    by Retake's own measurement are shown; duplicates are skipped.
    """
    best_identity = score_from_headers(best[1])
    if best_identity is None:
        return [best]
    best_nat = naturalness_from_headers(best[1])
    pool = [best]
    seen = {best[0]}
    ranked = sorted(others, key=lambda item: score_from_headers(item[1]) or -1.0, reverse=True)
    for item in ranked:
        if len(pool) >= limit:
            break
        identity = score_from_headers(item[1])
        if item[0] in seen or identity is None or identity < best_identity - ULTRA_NATURALNESS_BAND:
            continue
        nat = naturalness_from_headers(item[1])
        if nat is not None and (nat < ULTRA_NATURALNESS_FLOOR
                                or (best_nat is not None and nat < best_nat - ULTRA_NATURALNESS_GAIN)):
            continue
        pool.append(item)
        seen.add(item[0])
    return pool


def judge_pick(passes: list[list[float]], gain: float = ULTRA_JUDGE_GAIN) -> int:
    """Index of the pool entry the judge's passes justify (0 = keep best).

    A switch needs two passes (the second with the order reversed), each
    ranking the candidate above the current best, and a mean lead of `gain`.
    """
    if len(passes) < 2 or any(len(row) != len(passes[0]) for row in passes):
        return 0
    count = len(passes[0])
    means = [sum(row[index] for row in passes) / len(passes) for index in range(count)]
    choice = 0
    for index in range(1, count):
        if all(row[index] > row[0] for row in passes) and means[index] >= means[0] + gain \
                and means[index] > means[choice]:
            choice = index
    return choice


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
    decided = _naturalness_decision(candidate, best, new, old)
    if decided is not None:
        return decided
    if new > old + ULTRA_REALISM_TIE_LOSS:
        return True
    if old > new + ULTRA_REALISM_TIE_LOSS:
        return False

    # ArcFace gets first priority; in a tight identity tie, favor a genuinely
    # less visible hairline/jaw outline BEFORE generic texture sharpness.
    # Never treat a missing/invalid quality measurement as a perfect seam.
    new_rim = rim_from_headers(candidate)
    old_rim = rim_from_headers(best)
    if new_rim is not None and old_rim is not None:
        if new_rim + ULTRA_RIM_GAIN < old_rim:
            return True
        if old_rim + ULTRA_RIM_GAIN < new_rim:
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
