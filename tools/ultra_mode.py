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
from dataclasses import dataclass
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


def sufficiency_from_level(level: Any) -> float | None:
    """ArcFace score from which a result counts as this person's photo level."""
    if isinstance(level, bool):
        return None
    try:
        value = float(level)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not 0.40 <= value <= 0.99:
        return None
    return min(0.92, value - ULTRA_SUFFICIENCY_TOLERANCE)


def identity_sufficiency(headers: dict[str, str] | None) -> float | None:
    return sufficiency_from_level(photo_level_from_headers(headers))


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
        fine_quality_from_headers(headers) or 0.0,
        reference_appearance_from_headers(headers) or 0.0,
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
    # 3. Eyes, brows and lips measurably closer to the photo (v26).
    fine = fine_preference(candidate, best)
    if fine is not None:
        return fine
    appearance = reference_appearance_preference(candidate, best)
    if appearance is not None:
        return appearance
    # 4. More of the real photo's eye/nose/mouth detail.
    new_detail, old_detail = detail_from_headers(candidate), detail_from_headers(best)
    if new_detail is not None and old_detail is not None and abs(new_detail - old_detail) > ULTRA_DETAIL_GAIN / 2:
        return new_detail > old_detail
    if _photo_detail(candidate) != _photo_detail(best):
        return _photo_detail(candidate) > _photo_detail(best)
    # 5. Only then identity, as a tie-breaker.
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
    new_sufficient = is_identity_sufficient(candidate)
    old_sufficient = is_identity_sufficient(best)
    if new_sufficient and old_sufficient:
        return prefer_sufficient_candidate(candidate, best)
    if new_sufficient != old_sufficient and identity_sufficiency(candidate) == identity_sufficiency(best):
        return new_sufficient
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

    # Inside the identity tie band: more natural finest features first.
    fine = fine_preference(candidate, best)
    if fine is True and new >= old - ULTRA_REALISM_TIE_LOSS:
        return True
    if fine is False and new <= old + ULTRA_SCORE_EPS:
        return False

    appearance = reference_appearance_preference(candidate, best)
    if appearance is True and new >= old - ULTRA_REALISM_TIE_LOSS:
        return True
    if appearance is False and new <= old + ULTRA_SCORE_EPS:
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
    # Use the measured leave-one-out reference level when it exists. The
    # old pairwise median is on a different scale and could report a
    # completed search while the output remained below photo sufficiency.
    target = identity_sufficiency(headers)
    if target is None:
        if median is None or median < .30:
            return None
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

def choose_extra_variations(round_scores, winner_round, count=ULTRA_EXTRA_FINALISTS, fallback=None):
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
    # A learned schedule supplies the next most promising untried variants.
    untried = [v for v in (fallback or ()) if v in ULTRA_VARIATIONS] or [3, 6, 2, 10, 7]
    for variation in untried:
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


# --- Quality depth (v23) ---------------------------------------------------
# A bounded seven-minute search can only try a few of the 24 variants. When
# the person prefers the best photo over a quick one, the engine may search
# longer, refine the winner's continuous settings, and give more than one
# finalist the photographic realism pass. Gates and selection rules are the
# same at every depth; depth only buys more measured candidates.

@dataclass(frozen=True)
class UltraDepth:
    name: str
    search_seconds: float          # no new search round starts after this
    extra_finalists: int           # bounded second search when under target
    extra_start_seconds: float
    realism_finalists: int         # distinct candidates given the realism pass
    refinements: int               # compass-search steps around the winner
    refine_seconds: float          # no new refinement starts after this
    expected_minutes: tuple[int, int]
    client_timeout_seconds: int


ULTRA_DEPTHS: dict[str, UltraDepth] = {
    "standard": UltraDepth("standard", ULTRA_SEARCH_BUDGET_SECONDS, ULTRA_EXTRA_FINALISTS,
                           ULTRA_EXTRA_START_BUDGET_SECONDS, 1, 0, 0, (8, 15), 45 * 60),
    "deep": UltraDepth("deep", 1500, 3, 600, 2, 3, 600, (25, 40), 90 * 60),
    "max": UltraDepth("max", 3600, 4, 900, 3, 6, 1200, (60, 95), 150 * 60),
}
# Old browsers send no depth: they keep the seven-minute behaviour.
ULTRA_DEFAULT_DEPTH = "standard"


def ultra_depth(name: Any) -> UltraDepth:
    return ULTRA_DEPTHS.get(name) if isinstance(name, str) and name in ULTRA_DEPTHS else ULTRA_DEPTHS[ULTRA_DEFAULT_DEPTH]


def depth_summary() -> dict[str, dict[str, Any]]:
    """Health payload: what each depth will do on this engine."""
    return {
        name: {
            "search_seconds": int(depth.search_seconds),
            "extra_finalists": depth.extra_finalists,
            "realism_finalists": depth.realism_finalists,
            "refinements": depth.refinements,
            "expected_minutes": list(depth.expected_minutes),
            "client_timeout_seconds": depth.client_timeout_seconds,
        }
        for name, depth in ULTRA_DEPTHS.items()
    }


# Refinement: the 24 variants are a coarse grid. Around the winner, step the
# continuous settings (lower-face detail strength, head-fit strength) and
# keep a step only when the normal selection rules prefer it. The swapper
# weight is not changed, so the cached FaceFusion passes are reused and a
# refinement costs compositing and scoring, not new identity passes.
REFINE_DETAIL_STEP = 0.06
REFINE_HEAD_STEP = 0.30
REFINE_DETAIL_RANGE = (0.04, 0.40)
REFINE_HEAD_RANGE = (0.0, 1.50)
REFINE_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
REFINE_MIN_SCALE = 0.5


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


def normalize_refinement(params: Any) -> tuple[float, float, float] | None:
    """A refinement is (swapper weight, detail strength, head-fit strength)."""
    if not isinstance(params, (tuple, list)) or len(params) != 3:
        return None
    values = []
    for value in params:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        values.append(float(value))
    weight, detail, head = values
    if not 0.5 <= weight <= 1.0:
        return None
    if not REFINE_DETAIL_RANGE[0] - 1e-9 <= detail <= REFINE_DETAIL_RANGE[1] + 1e-9:
        return None
    if not REFINE_HEAD_RANGE[0] - 1e-9 <= head <= REFINE_HEAD_RANGE[1] + 1e-9:
        return None
    return round(weight, 2), round(detail, 3), round(head, 3)


class RefinePlanner:
    """Compass search: keep a direction while it wins, halve the step when
    every neighbour has been measured. Pure bookkeeping, no image work."""

    def __init__(self, start: Any, tried: Any = (), budget: int = 0) -> None:
        best = normalize_refinement(start)
        if best is None:
            # Grid variants may sit outside the refinement box; clamp once.
            weight, detail, head = (float(value) for value in start)
            best = (round(weight, 2), round(_clamp(detail, REFINE_DETAIL_RANGE), 3),
                    round(_clamp(head, REFINE_HEAD_RANGE), 3))
        self.best = best
        self.tried = {best}
        for params in tried:
            try:
                weight, detail, head = (float(value) for value in params)
            except (TypeError, ValueError):
                continue
            self.tried.add((round(weight, 2), round(detail, 3), round(head, 3)))
        self.budget = max(0, int(budget))
        self.used = 0
        self.direction = 0
        self.scale = 1.0
        self._pending: tuple[tuple[float, float, float], int] | None = None

    def _step(self, index: int) -> tuple[float, float, float]:
        weight, detail, head = self.best
        d_sign, h_sign = REFINE_DIRECTIONS[index]
        return (
            weight,
            round(_clamp(detail + d_sign * REFINE_DETAIL_STEP * self.scale, REFINE_DETAIL_RANGE), 3),
            round(_clamp(head + h_sign * REFINE_HEAD_STEP * self.scale, REFINE_HEAD_RANGE), 3),
        )

    def next(self) -> tuple[float, float, float] | None:
        if self.used >= self.budget:
            return None
        while self.scale >= REFINE_MIN_SCALE:
            for offset in range(len(REFINE_DIRECTIONS)):
                index = (self.direction + offset) % len(REFINE_DIRECTIONS)
                candidate = self._step(index)
                if candidate not in self.tried:
                    self.tried.add(candidate)
                    self.used += 1
                    self._pending = (candidate, index)
                    return candidate
            self.scale /= 2
        return None

    def feedback(self, params: Any, improved: bool) -> None:
        if self._pending is None or normalize_refinement(params) != self._pending[0]:
            return
        candidate, index = self._pending
        self._pending = None
        if improved:
            self.best = candidate
            self.direction = index  # momentum: try further the same way
        else:
            self.direction = (index + 1) % len(REFINE_DIRECTIONS)


def with_refinement(parsed: tuple, variation: int, params: Any = None,
                    realism_seed: int | None = None) -> tuple:
    """A variant request with optional refined settings and realism seed.

    Only the engine sets these keys: client metadata is validated into a
    whitelist, so a browser cannot inject either of them.
    """
    request = with_variation(parsed, variation)
    if params is not None:
        normalized = normalize_refinement(params)
        if normalized is None:
            raise ValueError("Ultra ince ayar değeri geçersiz.")
        request[6]["variationParams"] = normalized
    if realism_seed is not None:
        if isinstance(realism_seed, bool) or not isinstance(realism_seed, int) or not 0 <= realism_seed < 2 ** 31:
            raise ValueError("Gerçekçilik tohumu geçersiz.")
        request[6]["realismSeedBase"] = realism_seed
    return request


REALISM_SEED_BASE = 7
REALISM_SEED_STRIDE = 100


def finalist_specs(records: list, best_spec: tuple, peak_identity: float, count: int) -> list[tuple]:
    """The winner first, then the next distinct candidates the normal rules
    prefer, for the realism pass. A spec is (variation, params or None)."""
    if count <= 0:
        return []
    specs = [best_spec]
    remaining = [
        (spec, headers) for spec, headers in records
        if spec != best_spec and score_from_headers(headers) is not None
        and identity_acceptable(headers, peak_identity)
    ]
    # Spend a limited expensive realism pass on a genuinely distinct
    # regional/fine-detail peak as well as the default identity winner.
    # An already sufficient winner cannot be replaced by a sublevel spec.
    incumbent = next((headers for spec,headers in records if spec == best_spec), {})
    if is_identity_sufficient(incumbent):
        remaining = [(spec,h) for spec,h in remaining if is_identity_sufficient(h)]
    for measure in (reference_appearance_from_headers, fine_quality_from_headers):
        if len(specs) >= count or not remaining:
            break
        baseline = measure(incumbent)
        if baseline is None:
            continue
        measured = [(measure(h),spec) for spec,h in remaining if measure(h) is not None]
        if measured:
            value,spec = max(measured,key=lambda row:row[0])
            if value > baseline + .075:
                specs.append(spec)
                remaining = [item for item in remaining if item[0]!=spec]
    while remaining and len(specs) < count:
        chosen_index = 0
        for index in range(1, len(remaining)):
            if prefer_ultra_candidate(remaining[index][1], remaining[chosen_index][1]):
                chosen_index = index
        spec, _ = remaining.pop(chosen_index)
        if spec not in specs:
            specs.append(spec)
        remaining = [item for item in remaining if item[0] != spec]
    return specs


# --- Job diagnostics (v26) -------------------------------------------------
# One header that says where the time went, so a long job can be tuned from
# a copied report instead of a console log. Format (seconds, rounded):
#   search=19x72;refine=4x31;realism=2x210;extra=1x65;judge=40;hold=0;total=2210
TIMING_PHASES = ("search", "refine", "realism", "extra")


def format_ultra_timing(phases: dict[str, list[float]], judge_seconds: float,
                        hold_seconds: float, total_seconds: float) -> str:
    parts = []
    for name in TIMING_PHASES:
        durations = [value for value in phases.get(name, []) if math.isfinite(value) and value >= 0]
        if durations or name == "search":
            mean = sum(durations) / len(durations) if durations else 0.0
            parts.append(f"{name}={len(durations)}x{round(mean)}")
    for name, value in (("judge", judge_seconds), ("hold", hold_seconds), ("total", total_seconds)):
        parts.append(f"{name}={round(max(0.0, value)) if math.isfinite(value) else 0}")
    return ";".join(parts)


# Finest features in selection: once identity is equal (or sufficient), the
# candidate whose eyes, brows and lips are measurably closer to the photo's
# own detail is the more natural one. Box-region detail stays the next rule.
ULTRA_FINE_GAIN = 0.03


def fine_quality_from_headers(headers: dict[str, str]) -> float | None:
    try:
        value = float(headers.get("X-Retake-Fine-Quality", ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def fine_preference(candidate: dict[str, str], best: dict[str, str]) -> bool | None:
    """True/False when the finest features clearly differ, else None."""
    new, old = fine_quality_from_headers(candidate), fine_quality_from_headers(best)
    if new is None or old is None or abs(new - old) <= ULTRA_FINE_GAIN:
        return None
    return new > old


def reference_appearance_from_headers(headers):
    try:
        value = float(headers.get("X-Retake-Reference-Appearance", ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def reference_appearance_preference(candidate, best):
    """Measured lower-face contrast/texture fit, not another identity score."""
    new = reference_appearance_from_headers(candidate)
    old = reference_appearance_from_headers(best)
    if new is None or old is None or abs(new-old) <= .075:
        return None
    return new > old
