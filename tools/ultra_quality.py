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


def read_delivered_pair(original_path, result_path):
    """Mirror the binary preblended transport: hidden RGB is not delivered."""
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    result = cv2.imread(str(result_path), cv2.IMREAD_UNCHANGED)
    if original is None or result is None or result.shape[:2] != original.shape[:2]:
        return original, None
    if result.ndim == 3 and result.shape[2] == 4:
        result = np.where(result[:,:,3:4]>0,result[:,:,:3],original)
    elif result.ndim == 2:
        result = cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)
    return original,result


def check_photo_detail(
    original_path: Path, result_path: Path,
    bounds: dict[str, Any],
) -> tuple[float | None, dict[str, float]]:
    original, result = read_delivered_pair(original_path,result_path)
    return regional_photo_detail(original, result, bounds)



# A high cosine score must not rescue obvious detail destruction. Thresholds
# are deliberately conservative heuristics, not identity or authenticity tests.
DETAIL_FLOORS = {"eyes": .70, "nose": .55, "mouth": .70, "lower_face": .50}
DETAIL_CEILINGS = {"eyes": 2.5, "nose": 3.0, "mouth": 2.5}

def detail_gate(score, ratios):
    if score is None or not ratios:
        return False, "unmeasured"
    for name, ratio in ratios.items():
        if not math.isfinite(ratio) or ratio < DETAIL_FLOORS.get(name, .50):
            return False, name + "-blur"
        if ratio > DETAIL_CEILINGS.get(name, float("inf")):
            return False, name + "-oversharpen"
    return True, "passed"

def check_detail_gate(original_path, result_path, bounds):
    """(accepted, reason, score, ratios) for a result file against the original.

    A soft original (slightly out of focus, re-compressed, upscaled) has no
    region with enough texture to compare (regional_photo_detail skips
    them); there is nothing to call blurred then, so it passes as
    "soft-original" instead of rejecting every candidate.
    """
    # Measured on the delivered pixels (binary alpha), like check_photo_detail.
    original, result = read_delivered_pair(original_path, result_path)
    if original is None or result is None or original.shape != result.shape:
        return False, "unreadable", None, {}
    score, ratios = regional_photo_detail(original, result, bounds)
    if score is None and not ratios:
        return True, "soft-original", None, {}
    accepted, reason = detail_gate(score, ratios)
    return accepted, reason, score, ratios

def balance_with_original(original_path, candidate_path, output_path, strength):
    """A separately verified alternative, not a post-score pixel mutation.

    Recover native texture and color using the original pixels. Never change
    pixels outside the pre-existing matte or change alpha. Allow genuine
    reference beard/jaw edits to survive partial blending; no fixed beard rule.
    """
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    candidate = cv2.imread(str(candidate_path), cv2.IMREAD_UNCHANGED)
    if original is None or candidate is None or candidate.ndim != 3 or candidate.shape[2] != 4 or candidate.shape[:2] != original.shape[:2]:
        raise ValueError("Invalid source or candidate matte")
    if not 0 < strength <= .65:
        raise ValueError("Invalid recovery strength")
    result = candidate.copy()
    weight = (candidate[:, :, 3].astype(np.float32) / 255.0 * strength)[:, :, None]
    result[:, :, :3] = np.clip(np.rint(candidate[:, :, :3] * (1 - weight) + original * weight), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(output_path), result):
        raise OSError("Detail recovery candidate could not be written")
    return output_path


# InSwapper 128 draws the face at 128 px. On a large face the upscale blurs
# eyes and lips and leaves ghosted double lids, while ArcFace still rises
# (the swapper is driven by the same ArcFace embedding family it is scored
# with). Reinforcement must therefore not cost visible eye/mouth detail
# relative to the 256 px pass it would replace.
REINFORCE_MIN_DETAIL = {"eyes": 0.85, "mouth": 0.85}


def reinforcement_detail_decision(ratios: dict[str, float]) -> tuple[bool, str]:
    """Accept only when every measurable eye/mouth region keeps its detail.

    Unmeasurable regions (tiny or textureless faces, where a 128 px render
    loses little) keep the previous ArcFace-only behavior.
    """
    measured = [name for name in REINFORCE_MIN_DETAIL if name in ratios]
    for name in measured:
        ratio = ratios[name]
        if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < REINFORCE_MIN_DETAIL[name]:
            return False, name + "-blur"
    return True, "passed" if measured else "unmeasured"


def check_reinforcement_detail(single_path: Path, reinforced_path: Path, bounds: dict[str, Any]) -> tuple[bool, str, dict[str, float]]:
    single = cv2.imread(str(single_path), cv2.IMREAD_COLOR)
    reinforced = cv2.imread(str(reinforced_path), cv2.IMREAD_COLOR)
    if single is None or reinforced is None or single.shape != reinforced.shape:
        return False, "unreadable", {}
    _score, ratios = regional_photo_detail(single, reinforced, bounds)
    accepted, reason = reinforcement_detail_decision(ratios)
    return accepted, reason, ratios

# A regional reference transfer (beard / full face / jaw) must earn its place:
# either a real identity gain or no loss of photographic naturalness
# (tools/photo_naturalness.py). A low-opacity reference beard ties ArcFace
# but reads as a grey, ghosted shadow over the jaw and lips.
REGIONAL_IDENTITY_GAIN = 0.006
REGIONAL_NATURALNESS_SLACK = 1.5     # naturalness points (of 100) a regional may lose
REGIONAL_LABELS = ("identity-beard", "identity-full", "identity-jaw")


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def regional_earns_place(index, current, identity_scores, naturalness_scores) -> bool:
    if naturalness_scores is None:
        return True   # legacy callers: identity/detail rules only
    if identity_scores[index] >= identity_scores[current] + REGIONAL_IDENTITY_GAIN:
        return True
    mine, theirs = naturalness_scores[index], naturalness_scores[current]
    return _finite(mine) and _finite(theirs) and mine >= theirs - REGIONAL_NATURALNESS_SLACK


def select_verified_regional_candidate(identity_scores, detail_scores, labels, incumbent, naturalness_scores=None):
    """Prefer measured regional detail only within the verified identity peak."""
    if not (len(identity_scores) == len(detail_scores) == len(labels)):
        return incumbent
    if naturalness_scores is not None and len(naturalness_scores) != len(labels):
        naturalness_scores = None
    valid = [i for i, score in enumerate(identity_scores)
             if isinstance(score, (int, float)) and not isinstance(score, bool)
             and math.isfinite(score) and -1 <= score <= 1]
    if not valid:
        return incumbent
    peak = max(valid, key=lambda i: identity_scores[i])
    eligible = [i for i in valid if identity_scores[i] >= identity_scores[peak] - .003]
    start = incumbent if incumbent in eligible else peak
    current = choose_detail_safe_tie(identity_scores, detail_scores, start)
    if (current != start and labels[current] in REGIONAL_LABELS
            and not regional_earns_place(current, start, identity_scores, naturalness_scores)):
        current = start
    if labels[current] not in ("identity-beard", "identity-full", "identity-jaw"):
        baseline = detail_scores[current]
        if isinstance(baseline, (float, int)) and math.isfinite(baseline):
            regional = [i for i in eligible
                        if labels[i] in ("identity-beard", "identity-full", "identity-jaw")
                        and isinstance(detail_scores[i], (float, int))
                        and math.isfinite(detail_scores[i])
                        and detail_scores[i] >= baseline - .025
                        and regional_earns_place(i, current, identity_scores, naturalness_scores)]
            if regional:
                current = max(regional, key=lambda i: (identity_scores[i], detail_scores[i]))
    return current

def regional_edit_statistics(original_path, edited_path, bounds):
    """Native-pixel edit evidence, independent of source texture energy.

    A change is not a quality or identity gain. In particular, smooth lips
    and small beards may have insufficient high-frequency baseline energy
    but still allow a valid measurement of the delivered pixel difference.
    """
    original, edited = read_delivered_pair(original_path,edited_path)
    if original is None or edited is None or original.shape != edited.shape:
        return {}
    height, width = original.shape[:2]
    try:
        x, y, fw, fh = (float(bounds[key]) for key in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError):
        return {}
    if not all(math.isfinite(v) for v in (x,y,fw,fh)) or min(fw,fh) < 20:
        return {}
    regions = {**REGIONS, "moustache": (.27,.50,.73,.62), "chin": (.29,.76,.71,.92),
               "forehead": (.24,.10,.76,.25), "brows": (.12,.19,.88,.30),
               "left_cheek": (.12,.43,.34,.68), "right_cheek": (.66,.43,.88,.68),
               "left_jaw": (.12,.69,.35,.87), "right_jaw": (.65,.69,.88,.87)}
    result = {}
    for name, (left,top,right,bottom) in regions.items():
        x0,x1 = max(0,int(x+fw*left)), min(width,int(x+fw*right))
        y0,y1 = max(0,int(y+fh*top)), min(height,int(y+fh*bottom))
        if min(x1-x0,y1-y0) < 4:
            continue
        before = original[y0:y1,x0:x1].astype(np.int16)
        after = edited[y0:y1,x0:x1].astype(np.int16)
        values = np.max(np.abs(after-before),axis=2)
        result[name] = round(float(np.mean(values)),3)
    return result


def select_sufficient_realism(identity_scores, detail_scores, chosen, threshold, minimum_detail_gain=0.025,
                              naturalness_scores=None):
    """Inside one render: once identity reaches the person's photo level,
    pick the identity-sufficient candidate that keeps the most real photo
    detail (e.g. no-head-warp, photo-balance) instead of the highest cosine.

    Below the level, or without a measured level, the identity winner stays.
    """
    def valid(value, low, high):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and low <= value <= high

    if (
        threshold is None or not valid(threshold, -1, 1)
        or not (0 <= chosen < len(identity_scores) == len(detail_scores))
        or not valid(identity_scores[chosen], -1, 1) or identity_scores[chosen] < threshold
        or not valid(detail_scores[chosen], 0, 1)
    ):
        return chosen
    if naturalness_scores is not None and len(naturalness_scores) != len(identity_scores):
        naturalness_scores = None
    contenders = [
        index for index, score in enumerate(identity_scores)
        if valid(score, -1, 1) and score >= threshold and valid(detail_scores[index], 0, 1)
        # More regional energy is not realism when the photo reads as edited.
        and (naturalness_scores is None or index == chosen
             or not (valid(naturalness_scores[index], 0, 100) and valid(naturalness_scores[chosen], 0, 100))
             or naturalness_scores[index] >= naturalness_scores[chosen] - REGIONAL_NATURALNESS_SLACK)
    ]
    best = max(contenders, key=lambda index: (detail_scores[index], identity_scores[index]))
    return best if detail_scores[best] >= detail_scores[chosen] + minimum_detail_gain else chosen


# The 128 px reinforcement carries identity in its coarse structure and loses
# only the finest detail when upscaled onto a large face. Its low frequencies
# plus the 256 px pass's own fine band keep both (classic frequency split).
REINFORCE_RENDER_EDGE = 128.0
REINFORCE_SPLIT_SCALE = 0.5      # sigma = scale * face width / 128 px
REINFORCE_SPLIT_RANGE = (1.0, 4.0)


def reinforcement_split_sigma(face_width: float) -> float:
    low, high = REINFORCE_SPLIT_RANGE
    if not math.isfinite(face_width) or face_width <= 0:
        return low
    return float(min(high, max(low, REINFORCE_SPLIT_SCALE * face_width / REINFORCE_RENDER_EDGE)))


def fuse_reinforcement_detail(single_path: Path, reinforced_path: Path, output_path: Path,
                              bounds: dict[str, Any]) -> float:
    """Write reinforced low frequencies + single-pass fine detail; returns sigma.

    Both are whole swapper outputs of the same untouched target, so pixels
    the swappers left alone stay identical (same in both, same in the sum).
    """
    single = cv2.imread(str(single_path), cv2.IMREAD_UNCHANGED)
    reinforced = cv2.imread(str(reinforced_path), cv2.IMREAD_UNCHANGED)
    if single is None or reinforced is None or single.shape != reinforced.shape:
        raise ValueError("Pekiştirme ayrıntı birleşimi için görüntüler okunamadı.")
    try:
        width = float(bounds["width"])
    except (KeyError, TypeError, ValueError):
        width = float(single.shape[1]) / 3
    sigma = reinforcement_split_sigma(width)
    s = single.astype(np.float32)
    r = reinforced.astype(np.float32)
    fused = cv2.GaussianBlur(r, (0, 0), sigma) + (s - cv2.GaussianBlur(s, (0, 0), sigma))
    if fused.ndim == 3 and fused.shape[2] == 4:
        fused[:, :, 3] = r[:, :, 3]
    if not cv2.imwrite(str(output_path), np.clip(np.rint(fused), 0, 255).astype(np.uint8)):
        raise OSError("Pekiştirme ayrıntı birleşimi yazılamadı.")
    return sigma


def reinforcement_wanted(single, double, threshold, minimum_gain=0.008):
    """128 px InSwapper reinforcement only when identity is still short of the
    person's photo level and ArcFace actually rises."""
    def valid(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    if not valid(single) or not valid(double):
        return False, "unscored"
    if threshold is not None and valid(threshold) and single >= threshold:
        return False, "identity-sufficient"
    if double <= single + minimum_gain:
        return False, "no-identity-gain"
    return True, "gain"


def _ssim_luma(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda image: cv2.GaussianBlur(image, (0, 0), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(np.mean(ssim_map))


def self_fidelity(original_path: Path, edited_path: Path, bounds: dict[str, Any]) -> dict[str, float]:
    """Ground-truth realism for a self-test: the target is one of the
    person's own reference photos, so its original face is the right answer.

    Returns face-region luma SSIM (1 = structurally identical) and mean CIE76
    colour difference ΔE (0 = identical). Measurement only, never selection:
    choosing by it would turn the self-test into an oracle.
    """
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    edited = cv2.imread(str(edited_path), cv2.IMREAD_COLOR)
    if original is None or edited is None or original.shape != edited.shape:
        return {}
    try:
        x, y, fw, fh = (float(bounds[key]) for key in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError):
        return {}
    if not all(math.isfinite(value) for value in (x, y, fw, fh)) or min(fw, fh) < 16:
        return {}
    height, width = original.shape[:2]
    x0, y0 = max(0, int(x - fw * 0.05)), max(0, int(y - fh * 0.05))
    x1, y1 = min(width, int(math.ceil(x + fw * 1.05))), min(height, int(math.ceil(y + fh * 1.05)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return {}
    a, b = original[y0:y1, x0:x1], edited[y0:y1, x0:x1]
    ssim = _ssim_luma(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    lab_a = cv2.cvtColor(a.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab_b = cv2.cvtColor(b.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    delta_e = float(np.mean(np.sqrt(np.sum((lab_a - lab_b) ** 2, axis=2))))
    return {"ssim": round(max(-1.0, min(1.0, ssim)), 4), "delta_e": round(delta_e, 3)}
