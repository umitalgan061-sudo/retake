"""Mesh-guided finishing of the finest facial features (Retake v24).

``match_photo_texture`` matches *one* sharpness ratio over the whole face.
Swaps go soft (or ring) in specific places — eyelashes and lid creases, brow
hair, lip lines, nostril rims, teeth — and the iris often loses its
catchlight, which makes the eyes look dead. A single face-wide ratio cannot
fix that: it sharpens the cheeks while the eyes stay soft.

This module measures every feature on the browser's 478-point target mesh
against the original photo and corrects only that feature:

* soft feature → restore the result's *own* fine and mid detail (bounded
  unsharp gain). No texture is copied from the target, so the target
  person's features cannot leak into the new face;
* over-sharpened feature (halos, ringing) → soften towards the photo's focus;
* lost catchlight → the original photo's specular highlight is put back on
  the iris. A catchlight is the scene's light source, not identity;
* grey, yellow or blown-out eye whites → the sclera's tone is pulled
  towards the photo's own sclera under the same light (bounded Lab shift).

Feature positions in the result are re-aligned with phase correlation, so
head-fit and jaw warps (a few percent of the face width) are tolerated. The
engine keeps a refinement only if ArcFace and the regional detail gate still
verify it. A heuristic finishing step, not an authenticity measure.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MESH_POINTS = 478

# MediaPipe Face Mesh indices. "_r" is the subject's right (image left).
FEATURES: dict[str, tuple[int, ...]] = {
    "eye_r": (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    "eye_l": (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
    "brow_r": (46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
    "brow_l": (276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
    "lips": (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185),
    "nose_tip": (1, 4, 98, 2, 327, 129, 358, 64, 294),
    "mouth_inner": (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191),
}
# Point sets that are not an ordered outline: use their convex hull.
HULL_FEATURES = frozenset({"brow_r", "brow_l", "nose_tip"})
# Dilation (fraction of the feature's larger side): eyes include lashes and
# the lid crease, brows their hair, lips the vermilion border.
MARGINS = {"eye": 0.30, "brow": 0.15, "lips": 0.10, "nose": 0.06, "mouth": 0.0}
# (iris centre, iris ring, (upper lid, lower lid, corner, corner)).
IRISES = {
    "eye_r": (468, (469, 470, 471, 472), (159, 145, 33, 133)),
    "eye_l": (473, (474, 475, 476, 477), (386, 374, 362, 263)),
}
FEATURE_LABELS = {
    "eye_r": "sağ göz", "eye_l": "sol göz", "brow_r": "sağ kaş", "brow_l": "sol kaş",
    "lips": "dudak", "nose_tip": "burun ucu", "mouth_inner": "dişler",
}

FINE_SIGMA = 1.0          # lashes, brow hair, pores, sensor grain
MID_SIGMA = 2.4           # lid crease, lip lines, nostril rim
RESTORE_BELOW = 0.90      # softer than 90 % of the photo's detail → restore
SOFTEN_ABOVE = 1.70       # above 170 % → ringing/halo, soften
RESTORE_TARGET = 1.0
SOFTEN_TARGET = 1.25
MAX_RESTORE_GAIN = 1.2    # at most 2.2× the result's own band
MAX_SOFTEN = 0.6
BAND_CLIP = 24.0
MIN_BASELINE = 1.0        # the photo itself has no detail here: nothing to match
MIN_FEATURE_PIXELS = 48
MAX_SHIFT_RATIO = 0.08    # of the face width
MIN_SHIFT_RESPONSE = 0.10
# Catchlights
MIN_IRIS_RADIUS = 2.5
MIN_EYE_OPENNESS = 0.18
HIGHLIGHT_PEAK = 185.0
HIGHLIGHT_CONTRAST = 55.0
MAX_HIGHLIGHT_FRACTION = 0.25
PRESENT_PEAK_LOSS = 25.0
PRESENT_AREA_FRACTION = 0.35
CATCHLIGHT_OPACITY = 0.92
# A catchlight only goes onto a dark iris/pupil: anywhere brighter means
# the alignment is off, and a white dot on the eye white or skin would show.
MAX_DESTINATION_LUMA = 140.0
# Inside the iris only: a wider disk reaches the (bright) sclera.
IRIS_DISK_SCALE = 0.9
# Sclera (eye white) tone, in Lab: trigger, strength and hard caps.
SCLERA_TRIGGER_L = 6.0
SCLERA_TRIGGER_AB = 5.0
SCLERA_STRENGTH = 0.7
SCLERA_MAX_L = 18.0
SCLERA_MAX_AB = 10.0
MIN_SCLERA_PIXELS = 30


def _kind(name: str) -> str:
    return name.split("_", 1)[0]


def _luma(image: np.ndarray) -> np.ndarray:
    return image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299


def valid_mesh(landmarks: Any, shape: tuple[int, int]) -> np.ndarray | None:
    try:
        points = np.asarray(landmarks, np.float32)
    except (TypeError, ValueError):
        return None
    if points.shape != (MESH_POINTS, 2) or not np.isfinite(points).all():
        return None
    height, width = shape
    margin = 0.25 * max(height, width)
    if (points[:, 0] < -margin).any() or (points[:, 0] > width + margin).any() \
            or (points[:, 1] < -margin).any() or (points[:, 1] > height + margin).any():
        return None
    return points


def feature_mask(points: np.ndarray, name: str, shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Feathered 0..1 mask of one feature and its pixel bounding box."""
    outline = np.round(points[list(FEATURES[name])]).astype(np.int32)
    if name in HULL_FEATURES:
        outline = cv2.convexHull(outline).reshape(-1, 2)
    binary = np.zeros(shape, np.uint8)
    cv2.fillPoly(binary, [outline], 255)
    x, y, w, h = cv2.boundingRect(outline)
    size = max(w, h)
    grow = int(round(MARGINS[_kind(name)] * size))
    if grow > 0:
        binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    sigma = max(1.0, 0.3 * grow) if grow > 0 else 0.8
    mask = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (0, 0), sigma)
    return mask, (x - grow, y - grow, w + 2 * grow, h + 2 * grow)


def estimate_shift(target_luma: np.ndarray, result_luma: np.ndarray,
                   box: tuple[int, int, int, int], max_shift: float) -> tuple[float, float]:
    """Where the feature sits in the result relative to the photo (px)."""
    height, width = target_luma.shape
    x, y, w, h = box
    pad_x, pad_y = max(8, w // 2), max(8, h // 2)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return 0.0, 0.0
    a = np.ascontiguousarray(target_luma[y0:y1, x0:x1], np.float32)
    b = np.ascontiguousarray(result_luma[y0:y1, x0:x1], np.float32)
    window = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), window)
    except cv2.error:
        return 0.0, 0.0
    if not (math.isfinite(dx) and math.isfinite(dy)) or response < MIN_SHIFT_RESPONSE \
            or math.hypot(dx, dy) > max_shift:
        return 0.0, 0.0
    return float(dx), float(dy)


def _translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return image
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _band(luma: np.ndarray, low: float, high: float) -> np.ndarray:
    fine = luma if low <= 0 else cv2.GaussianBlur(luma, (0, 0), low)
    return fine - cv2.GaussianBlur(luma, (0, 0), high)


def _energy(band: np.ndarray, weight: np.ndarray) -> float:
    total = float(weight.sum())
    if total < 1.0:
        return 0.0
    return math.sqrt(float((band * band * weight).sum()) / total)


def _ratio(result: float, original: float) -> float:
    return (result + 0.25) / (original + 0.25)


def _quality(ratio: float) -> float:
    return math.exp(-0.72 * abs(math.log(max(1e-6, ratio))))


def _measure(target_luma: np.ndarray, result_luma: np.ndarray, masks: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[str, float]]:
    target_fine, target_mid = _band(target_luma, 0, FINE_SIGMA), _band(target_luma, FINE_SIGMA, MID_SIGMA)
    result_fine, result_mid = _band(result_luma, 0, FINE_SIGMA), _band(result_luma, FINE_SIGMA, MID_SIGMA)
    measured = {}
    for name, (target_mask, result_mask) in masks.items():
        baseline_fine = _energy(target_fine, target_mask)
        if baseline_fine < MIN_BASELINE:
            continue
        measured[name] = {
            "fine": _ratio(_energy(result_fine, result_mask), baseline_fine),
            "mid": _ratio(_energy(result_mid, result_mask), _energy(target_mid, target_mask)),
        }
    return measured


def _iris_geometry(points: np.ndarray, eye: str) -> tuple[np.ndarray, float, float]:
    centre_index, ring, (upper, lower, corner_a, corner_b) = IRISES[eye]
    centre = points[centre_index]
    radius = float(np.mean(np.linalg.norm(points[list(ring)] - centre, axis=1)))
    width = float(np.linalg.norm(points[corner_a] - points[corner_b]))
    openness = float(np.linalg.norm(points[upper] - points[lower])) / max(width, 1e-6)
    return centre, radius, openness


def _disk(shape: tuple[int, int], centre: np.ndarray, radius: float) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, (int(round(centre[0])), int(round(centre[1]))), max(1, int(round(radius))), 1, -1)
    return mask.astype(bool)


def restore_catchlight(target: np.ndarray, result: np.ndarray, points: np.ndarray, eye: str,
                       shift: tuple[float, float], weight: np.ndarray) -> tuple[np.ndarray, str]:
    """Put the photo's own specular highlight back on a dull iris."""
    centre, radius, openness = _iris_geometry(points, eye)
    if radius < MIN_IRIS_RADIUS or openness < MIN_EYE_OPENNESS:
        return result, "closed"
    shape = target.shape[:2]
    disk = _disk(shape, centre, radius * IRIS_DISK_SCALE)
    if int(disk.sum()) < 8:
        return result, "small"
    target_luma = _luma(target.astype(np.float32))
    values = target_luma[disk]
    median, peak = float(np.median(values)), float(values.max())
    if peak < HIGHLIGHT_PEAK or peak - median < HIGHLIGHT_CONTRAST:
        return result, "none-in-photo"
    threshold = max(170.0, median + 0.6 * (peak - median))
    highlight = disk & (target_luma >= threshold)
    area = int(highlight.sum())
    if area == 0 or area > MAX_HIGHLIGHT_FRACTION * int(disk.sum()):
        return result, "not-specular"
    dx, dy = shift
    shifted_disk = _disk(shape, centre + np.float32([dx, dy]), radius * IRIS_DISK_SCALE)
    result_luma = _luma(result)
    result_values = result_luma[shifted_disk]
    if result_values.size and float(result_values.max()) >= peak - PRESENT_PEAK_LOSS \
            and int((result_values >= threshold).sum()) >= PRESENT_AREA_FRACTION * area:
        return result, "present"
    destination = _translate(highlight.astype(np.float32), dx, dy) > 0.5
    if not destination.any() or float(np.median(result_luma[destination])) > MAX_DESTINATION_LUMA:
        return result, "misaligned"
    alpha = cv2.dilate(highlight.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6) * CATCHLIGHT_OPACITY
    alpha = _translate(alpha, dx, dy) * weight
    light = _translate(target.astype(np.float32), dx, dy)
    restored = result * (1.0 - alpha[:, :, None]) + light * alpha[:, :, None]
    return restored, "restored"


def _sclera_mask(points: np.ndarray, eye: str, shape: tuple[int, int]) -> np.ndarray:
    """Eye opening without its rim (lashes) and without the iris."""
    centre, radius, _ = _iris_geometry(points, eye)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(points[list(FEATURES[eye])]).astype(np.int32)], 1)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    cv2.circle(mask, (int(round(centre[0])), int(round(centre[1]))), max(1, int(round(radius * 1.1))), 0, -1)
    return mask.astype(bool)


def _bright_half_lab(lab: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    values = lab[mask]
    if len(values) < MIN_SCLERA_PIXELS:
        return None
    # The sclera is the brighter half of the opening; lid shadow and stray
    # lashes are the darker half.
    keep = values[:, 0] >= np.median(values[:, 0])
    return values[keep].mean(axis=0) if keep.sum() >= MIN_SCLERA_PIXELS // 2 else None


def correct_sclera(target: np.ndarray, result: np.ndarray, points: np.ndarray, eye: str,
                   shift: tuple[float, float], weight: np.ndarray) -> tuple[np.ndarray, str]:
    """Match the eye white's tone to the photo's own sclera, bounded."""
    _, radius, openness = _iris_geometry(points, eye)
    if radius < MIN_IRIS_RADIUS or openness < MIN_EYE_OPENNESS:
        return result, "closed"
    shape = target.shape[:2]
    target_mask = _sclera_mask(points, eye, shape)
    result_mask = _translate(target_mask.astype(np.float32), *shift) > 0.5
    if int(target_mask.sum()) < MIN_SCLERA_PIXELS or int(result_mask.sum()) < MIN_SCLERA_PIXELS:
        return result, "small"
    ys, xs = np.nonzero(target_mask | result_mask)
    pad = 4
    y0, y1 = max(0, ys.min() - pad), min(shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(shape[1], xs.max() + pad + 1)
    to_lab = lambda image: cv2.cvtColor(np.clip(image[y0:y1, x0:x1], 0, 255).astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    target_lab, result_lab = to_lab(target), to_lab(result)
    reference = _bright_half_lab(target_lab, target_mask[y0:y1, x0:x1])
    current = _bright_half_lab(result_lab, result_mask[y0:y1, x0:x1])
    if reference is None or current is None:
        return result, "small"
    difference = reference - current
    if abs(difference[0]) < SCLERA_TRIGGER_L and max(abs(difference[1]), abs(difference[2])) < SCLERA_TRIGGER_AB:
        return result, "matched"
    delta = np.float32([
        np.clip(difference[0] * SCLERA_STRENGTH, -SCLERA_MAX_L, SCLERA_MAX_L),
        np.clip(difference[1] * SCLERA_STRENGTH, -SCLERA_MAX_AB, SCLERA_MAX_AB),
        np.clip(difference[2] * SCLERA_STRENGTH, -SCLERA_MAX_AB, SCLERA_MAX_AB),
    ])
    feather = cv2.GaussianBlur(result_mask[y0:y1, x0:x1].astype(np.float32), (0, 0), 0.8)
    feather *= weight[y0:y1, x0:x1]
    corrected_lab = result_lab + delta[None, None, :] * feather[:, :, None]
    corrected = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR) * 255.0
    output = result.copy()
    blend = feather[:, :, None]
    output[y0:y1, x0:x1] = result[y0:y1, x0:x1] * (1.0 - blend) + corrected * blend
    return output, "corrected"


def refine_fine_features(target: np.ndarray, result: np.ndarray, landmarks: Any,
                         bounds: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (refined image with the result's channels, report)."""
    if target is None or result is None or target.ndim != 3 or result.ndim != 3 \
            or target.shape[:2] != result.shape[:2] or result.shape[2] not in (3, 4):
        return result, {"status": "unmeasured", "reason": "size"}
    shape = target.shape[:2]
    points = valid_mesh(landmarks, shape)
    if points is None:
        return result, {"status": "no-mesh"}
    try:
        face_width = float(bounds["width"])
    except (TypeError, ValueError, KeyError):
        return result, {"status": "unmeasured", "reason": "bounds"}
    if not math.isfinite(face_width) or face_width < 40:
        return result, {"status": "unmeasured", "reason": "bounds"}

    # Work on the features' region only (with room for dilation, blur and
    # the alignment search); everything else is returned untouched.
    used = sorted({index for indices in FEATURES.values() for index in indices}
                  | {index for iris, ring, _ in IRISES.values() for index in (iris, *ring)})
    feature_points = points[used]
    pad = int(math.ceil(0.15 * face_width + 4 * MID_SIGMA + 8))
    x0 = max(0, int(math.floor(feature_points[:, 0].min())) - pad)
    y0 = max(0, int(math.floor(feature_points[:, 1].min())) - pad)
    x1 = min(shape[1], int(math.ceil(feature_points[:, 0].max())) + pad + 1)
    y1 = min(shape[0], int(math.ceil(feature_points[:, 1].max())) + pad + 1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return result, {"status": "unmeasured", "reason": "region"}
    refined_region, report = _refine_region(
        target[y0:y1, x0:x1], result[y0:y1, x0:x1], points - np.float32([x0, y0]), face_width
    )
    if report.get("status") != "applied":
        return result, report
    output = result.copy()
    output[y0:y1, x0:x1, :3] = refined_region
    return output, report


def _refine_region(target: np.ndarray, result: np.ndarray, points: np.ndarray,
                   face_width: float) -> tuple[np.ndarray, dict[str, Any]]:
    shape = target.shape[:2]
    target_f = target[:, :, :3].astype(np.float32)
    colour = result[:, :, :3].astype(np.float32)
    # Never touch what the matte hides: there the result must stay the photo.
    weight = result[:, :, 3].astype(np.float32) / 255.0 if result.shape[2] == 4 else np.ones(shape, np.float32)
    target_luma, result_luma = _luma(target_f), _luma(colour)

    try:
        from .retake_detail import mouth_is_open
    except ImportError:
        from retake_detail import mouth_is_open
    names = [name for name in FEATURES if name != "mouth_inner"]
    if mouth_is_open(points):
        names.append("mouth_inner")

    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    shifts: dict[str, tuple[float, float]] = {}
    for name in names:
        mask, box = feature_mask(points, name, shape)
        if float((mask > 0.5).sum()) < MIN_FEATURE_PIXELS:
            continue
        shift = estimate_shift(target_luma, result_luma, box, MAX_SHIFT_RATIO * face_width)
        masks[name] = (mask, _translate(mask, *shift) * weight)
        shifts[name] = shift

    before = _measure(target_luma, result_luma, masks)
    if not before:
        return colour, {"status": "unmeasured", "reason": "no-detail"}

    gain_fine = np.zeros(shape, np.float32)
    gain_mid = np.zeros(shape, np.float32)
    soften = np.zeros(shape, np.float32)
    actions: dict[str, str] = {}
    for name, ratios in before.items():
        result_mask = masks[name][1]
        if ratios["fine"] < RESTORE_BELOW:
            gain = min(MAX_RESTORE_GAIN, RESTORE_TARGET / ratios["fine"] - 1.0)
            gain_fine = np.maximum(gain_fine, result_mask * gain)
            if ratios["mid"] < RESTORE_BELOW:
                gain_mid = np.maximum(gain_mid, result_mask * min(MAX_RESTORE_GAIN, RESTORE_TARGET / ratios["mid"] - 1.0) * 0.6)
            actions[name] = "restore"
        elif ratios["fine"] > SOFTEN_ABOVE:
            amount = min(MAX_SOFTEN, 1.0 - SOFTEN_TARGET / ratios["fine"])
            soften = np.maximum(soften, result_mask * amount)
            actions[name] = "soften"
        else:
            actions[name] = "keep"

    refined = colour
    if gain_fine.any() or gain_mid.any() or soften.any():
        blur_fine = cv2.GaussianBlur(colour, (0, 0), FINE_SIGMA)
        blur_mid = cv2.GaussianBlur(colour, (0, 0), MID_SIGMA)
        fine_band = np.clip(colour - blur_fine, -BAND_CLIP, BAND_CLIP)
        mid_band = np.clip(blur_fine - blur_mid, -BAND_CLIP, BAND_CLIP)
        refined = (colour + fine_band * gain_fine[:, :, None] + mid_band * gain_mid[:, :, None]
                   + (blur_fine - colour) * soften[:, :, None])

    catchlights: dict[str, str] = {}
    sclera: dict[str, str] = {}
    for eye in IRISES:
        if eye not in masks:
            catchlights[eye] = sclera[eye] = "unmeasured"
            continue
        # Tone first: a catchlight is painted on top of the corrected eye.
        refined, sclera[eye] = correct_sclera(target_f, refined, points, eye, shifts[eye], weight)
        refined, outcome = restore_catchlight(target_f, refined, points, eye, shifts[eye], weight)
        catchlights[eye] = outcome

    refined = np.clip(np.rint(refined), 0, 255)
    after = _measure(target_luma, _luma(refined), masks)
    quality_before = sum(_quality(item["fine"]) for item in before.values()) / len(before)
    quality_after = sum(_quality(after[name]["fine"]) for name in before if name in after) / max(1, len(after))
    restored = sum(1 for outcome in catchlights.values() if outcome == "restored")
    toned = sum(1 for outcome in sclera.values() if outcome == "corrected")
    changed = restored > 0 or toned > 0 or any(action != "keep" for action in actions.values())

    report = {
        "status": "applied" if changed else "unchanged",
        "features": {
            name: {
                "before": round(before[name]["fine"], 3),
                "after": round(after.get(name, before[name])["fine"], 3),
                "action": actions[name],
                "shift": [round(shifts[name][0], 2), round(shifts[name][1], 2)],
            }
            for name in before
        },
        "quality_before": round(quality_before, 4),
        "quality_after": round(quality_after, 4),
        "catchlights": catchlights,
        "catchlights_restored": restored,
        "sclera": sclera,
        "sclera_corrected": toned,
    }
    return refined.astype(np.uint8), report


def refine_fine_features_file(original_path: Path, composite_path: Path, output_path: Path,
                              bounds: dict[str, Any], landmarks: Any) -> dict[str, Any]:
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    composite = cv2.imread(str(composite_path), cv2.IMREAD_UNCHANGED)
    if original is None or composite is None:
        return {"status": "unmeasured", "reason": "unreadable"}
    if composite.ndim == 2:
        composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
    refined, report = refine_fine_features(original, composite, landmarks, bounds)
    if report.get("status") == "applied" and not cv2.imwrite(str(output_path), refined):
        raise OSError("İnce bölge iyileştirmesi yazılamadı.")
    return report


# Acceptance in the engine: identity must hold and the features must be
# measurably closer to the photo's own detail (or a catchlight came back).
FINE_IDENTITY_TOLERANCE = 0.003
FINE_MIN_QUALITY_GAIN = 0.01


def accept_fine_refinement(report: dict[str, Any], refined_score: float | None,
                           current_score: float | None, detail_gate_passed: bool) -> tuple[bool, str]:
    if report.get("status") != "applied":
        return False, str(report.get("status", "unmeasured"))
    if not detail_gate_passed:
        return False, "detail-gate"
    if refined_score is None or current_score is None or not math.isfinite(refined_score):
        return False, "identity-unmeasured"
    if refined_score < current_score - FINE_IDENTITY_TOLERANCE:
        return False, "identity-loss"
    gain = float(report.get("quality_after", 0.0)) - float(report.get("quality_before", 0.0))
    if (gain < FINE_MIN_QUALITY_GAIN and int(report.get("catchlights_restored", 0)) == 0
            and int(report.get("sclera_corrected", 0)) == 0):
        return False, "no-gain"
    return True, "applied"


def describe_report(report: dict[str, Any]) -> str:
    """Header value: name:before>after per feature, e.g. eye_r:0.712>0.981."""
    features = report.get("features") or {}
    return ",".join(
        f"{name}:{item['before']:.3f}>{item['after']:.3f}" for name, item in features.items()
        if isinstance(item, dict) and name in FEATURES
    )
