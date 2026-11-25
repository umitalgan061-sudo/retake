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


def _photo_detail(headers: dict[str, str]) -> int:
    return (
        int(headers.get("X-Retake-Realism") == "on")
        + int(headers.get("X-Retake-Texture-Source", "average") != "average")
        + int(headers.get("X-Retake-Real-Face") == "on")
    )


# Identity sufficiency. The person's own photos reach a typical ArcFace
# similarity to the identity centroid (leave-one-out "photo level"). A result
# at that level is already as recognisably this person as a real photo of
# them; chasing a higher cosine beyond it mostly rewards swappers that pull
# the ArcFace embedding (InSwapper is driven by ArcFace) at the cost of
# blurred eyes, waxy skin and halos. Above the level, realism decides.
ULTRA_SUFFICIENCY_TOLERANCE = 0.02


def photo_level_from_headers(headers: dict[str, str] | None) -> float | None:
    if not headers:
        return None
    try:
        value = float(headers.get("X-Retake-Identity-Photo-Level", ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0.40 <= value <= 0.99 else None


def identity_sufficiency(headers: dict[str, str] | None) -> float | None:
    level = photo_level_from_headers(headers)
    return None if level is None else min(0.92, level - ULTRA_SUFFICIENCY_TOLERANCE)


def is_identity_sufficient(headers: dict[str, str] | None) -> bool:
    threshold = identity_sufficiency(headers)
    score = score_from_headers(headers) if headers else None
    return threshold is not None and score is not None and score >= threshold


def identity_acceptable(headers: dict[str, str], peak_identity: float) -> bool:
    """Near the strongest candidate, or already at the person's photo level."""
    score = score_from_headers(headers)
    return score is not None and (score >= peak_identity - ULTRA_REALISM_TIE_LOSS or is_identity_sufficient(headers))


def realism_order(headers: dict[str, str]) -> tuple:
    """Sort key among identity-sufficient candidates (larger is better)."""
    rim = rim_from_headers(headers)
    detail = detail_from_headers(headers)
    return (
        headers.get("X-Retake-Identity-Reinforced") != "on",
        -(rim if rim is not None else 1.0),
        detail if detail is not None else 0.0,
        _photo_detail(headers),
        score_from_headers(headers) or -1.0,
    )


def prefer_sufficient_candidate(candidate: dict[str, str], best: dict[str, str]) -> bool:
    # 1. A 256 px render beats the 128 px InSwapper reinforcement.
    new_reinforced = candidate.get("X-Retake-Identity-Reinforced") == "on"
    old_reinforced = best.get("X-Retake-Identity-Reinforced") == "on"
    if new_reinforced != old_reinforced:
        return not new_reinforced
    # 2. A visibly cleaner hairline/jaw outline.
    new_rim, old_rim = rim_from_headers(candidate), rim_from_headers(best)
    if new_rim is not None and old_rim is not None:
        if new_rim + ULTRA_RIM_GAIN < old_rim:
            return True
        if old_rim + ULTRA_RIM_GAIN < new_rim:
            return False
    # 3. More of the real photo's eye/nose/mouth detail.
    new_detail, old_detail = detail_from_headers(candidate), detail_from_headers(best)
    if new_detail is not None and old_detail is not None and abs(new_detail - old_detail) > ULTRA_DETAIL_GAIN / 2:
        return new_detail > old_detail
    if _photo_detail(candidate) != _photo_detail(best):
        return _photo_detail(candidate) > _photo_detail(best)
    # 4. Only then identity, as a tie-breaker.
    return (score_from_headers(candidate) or -1.0) > (score_from_headers(best) or -1.0) + ULTRA_SCORE_EPS


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
    if is_identity_sufficient(candidate) and is_identity_sufficient(best):
        return prefer_sufficient_candidate(candidate, best)
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


# Search aspirations, not identity guarantees. Median pairwise reference
# cosine is an imperfect guide, not a calibrated threshold for a person.
ULTRA_EXTRA_FINALISTS = 2
ULTRA_EXTRA_START_BUDGET_SECONDS = 240
ULTRA_PROFILE_MARGIN = .065

def _header_cosine(headers, name):
    try:
        value = float(headers.get(name, ""))
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and -1 <= value <= 1 else None

def identity_aspiration(headers):
    median = _header_cosine(headers, "X-Retake-Identity-Median-Similarity")
    before = _header_cosine(headers, "X-Retake-Similarity-Before")
    if median is None or median < .30:
        return None
    # Never demand an impossible fixed score; preserve an already strong
    # original rather than making larger edits merely for the sake of edits.
    target = min(.93, max(.55, median + ULTRA_PROFILE_MARGIN))
    if before is not None:
        target = max(target, before)
    return target

def identity_gain(headers):
    after = score_from_headers(headers)
    before = _header_cosine(headers, "X-Retake-Similarity-Before")
    return None if after is None or before is None else after - before

def needs_extra_finalist(headers):
    # At the person's own photo level there is nothing left to chase.
    if is_identity_sufficient(headers):
        return False
    target = identity_aspiration(headers)
    score = score_from_headers(headers)
    return target is not None and score is not None and score < target - ULTRA_SCORE_EPS

def choose_extra_variations(round_scores, winner_round, count=ULTRA_EXTRA_FINALISTS):
    """Try a different verified variation, then a distinct geometry/weight.

    Only metadata is retained here; the next pass starts on the original.
    Prefer a verified alternative from another swapper-weight family.
    """
    if not 1 <= winner_round <= len(ULTRA_VARIATIONS):
        return []
    winner = ULTRA_VARIATIONS[winner_round - 1]
    families = (1.0,1.0,1.0,.95,1.0,1.0,.90,1.0,.95,.95,.95,.95,.95,.95,.95,.95,.90,.90,.95,.95,.90,.90,.90,.90)
    others = [(variation, score) for variation, score in round_scores.items()
              if variation in ULTRA_VARIATIONS and variation != winner
              and isinstance(score, (float, int)) and math.isfinite(score)]
    others.sort(key=lambda item: (families[item[0]] != families[winner], item[1]), reverse=True)
    choices = [variation for variation, _ in others[:count]]
    # If the time budget allowed only one original round, use one of the
    # different weight/geometry options that has not already been evaluated.
    for variation in (3, 6, 2, 10, 7):
        if len(choices) >= count:
            break
        if variation != winner and variation not in choices and variation not in round_scores:
            choices.append(variation)
    return choices


def defer_flux_retry(outcome, diagnostics):
    """Spend the first extra finalist on identity instead of repeated bad diffusion."""
    if outcome != "identity-loss":
        return False
    try:
        identity = float(diagnostics.get("X-Retake-FLUX-Identity-Delta", ""))
        detail = float(diagnostics.get("X-Retake-FLUX-Detail-Delta", ""))
    except (TypeError,ValueError):
        return False
    return math.isfinite(identity) and math.isfinite(detail) and identity < -.03 and detail <= 0
