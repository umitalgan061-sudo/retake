"""Photographic naturalness score (0-100) for an edited face.

A real photograph is self-consistent: the face and its surroundings share one
light, one sensor's grain and sharpness, one skin tone, and nothing marks
where one ends. A replaced face that "looks pasted" breaks at least one of
these. Each check compares the edited photo with the *original* photo, where
the face and its surroundings were consistent by construction:

    border     no step in brightness or colour where the edit ends
    tone       the face keeps the original face-to-neck/ear skin relation
    grain      the face's sensor noise matches the surrounding photo
    detail     facial micro-detail is neither waxy nor over-sharpened
    light      the face is lit from where the original face was lit

The original photo scores 100 on every check. It is an image-consistency
measurement, not a deepfake detector and not proof of authenticity; identity
is measured separately (ArcFace).
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

WEIGHTS = {"border": 0.25, "tone": 0.20, "grain": 0.15, "detail": 0.20, "light": 0.20}
# Error that halves a check's score (calibrated on the user's photos and on
# deliberately broken composites; see tests/test_photo_naturalness.py).
HALF_SCORE = {
    "border": 1.6,     # low-pass jump per 100 px of face width (Lab units)
    "tone": 5.0,       # Lab a/b distance of the face-to-skin relation
    "grain": 0.45,     # |log| ratio of face/surround noise vs original
    "detail": 0.55,    # |log| ratio of facial micro-detail vs original
    "light": 0.35,     # 1 - correlation of the face's broad shading
}
CHANGE_THRESHOLD = 3
UPPER_FACE_SHARE = 0.65
BORDER_PERCENTILE = 80
GRAIN_SIGMA = 0.7


def _lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)


def _score(error: float, half: float) -> float:
    return float(100.0 * 0.5 ** (max(0.0, error) / half))


def changed_mask(result: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Where the edit changed the photo (filled, uint8 0/1)."""
    changed = (np.max(cv2.absdiff(result, original), axis=2) >= CHANGE_THRESHOLD).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed, 8)
    if count <= 1:
        return changed
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == largest).astype(np.uint8)
    # Fill holes (eyes or mouth the edit left untouched).
    flood = region.copy()
    pad = np.zeros((region.shape[0] + 2, region.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 1)
    return region | (1 - flood)


def _rings(mask: np.ndarray, face_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(face core, inner band, outer ring) boolean masks."""
    k = max(3, int(face_width * 0.04)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.erode(mask, kernel, iterations=2) > 0
    wide = max(5, int(face_width * 0.22)) | 1
    outer = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wide, wide))) > 0) & (mask == 0)
    band = (cv2.dilate(mask, kernel) > 0) & ~(cv2.erode(mask, kernel) > 0)
    return core, band, outer


def border_error(
    result: np.ndarray, original: np.ndarray, mask: np.ndarray, face_width: float,
    within: np.ndarray | None = None
) -> float:
    """Low-pass step of the change (L, a, b) across the edit's edge, per 100 px of face width.

    Only where the original photo is smooth across the edge (skin continuing
    into skin): where the edit ends on an edge the photo already had (the
    hairline, a collar, the beard line) a change in tone is not a seam.
    """
    edge = cv2.Canny(mask * 255, 50, 150) > 0
    if not edge.any():
        return 0.0
    band = cv2.dilate(edge.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if within is not None and (band & within).sum() >= 30:
        band &= within
    lab_original = _lab(original)
    broad = cv2.GaussianBlur(lab_original[:, :, 0], (0, 0), 3.0)
    texture = np.hypot(cv2.Sobel(broad, cv2.CV_32F, 1, 0), cv2.Sobel(broad, cv2.CV_32F, 0, 1))
    smooth = band & (texture <= np.percentile(texture[band], 50))
    if smooth.sum() >= 30:
        band = smooth
    change = cv2.GaussianBlur(_lab(result) - lab_original, (0, 0), 3.0)
    weights = (1.0, 1.6, 1.6)   # colour steps read worse than brightness steps
    steps = []
    for channel, weight in enumerate(weights):
        plane = change[:, :, channel]
        gradient = np.hypot(cv2.Sobel(plane, cv2.CV_32F, 1, 0), cv2.Sobel(plane, cv2.CV_32F, 0, 1))
        # 80th percentile: a seam runs along the edge; a few hair strands
        # crossing the edge are isolated points and must not decide.
        steps.append(weight * float(np.percentile(gradient[band], BORDER_PERCENTILE)))
    return max(steps) * 100.0 / max(face_width, 1.0)


def tone_error(result: np.ndarray, original: np.ndarray, core: np.ndarray, outer: np.ndarray) -> float | None:
    """Change of the face-to-surrounding-skin colour relation (Lab a/b)."""
    lab_o, lab_r = _lab(original), _lab(result)
    if core.sum() < 50 or outer.sum() < 50:
        return None
    face_o = np.median(lab_o[core][:, 1:], axis=0)
    # Skin around the face (neck, ears, cheeks the edit left): chroma near the
    # original face's, and not much darker (hair, shadow) or brighter (sky).
    lum_o = float(np.median(lab_o[core][:, 0]))
    chroma = np.linalg.norm(lab_o[:, :, 1:] - face_o, axis=2)
    skin = outer & (chroma < 12.0) & (np.abs(lab_o[:, :, 0] - lum_o) < 45.0)
    if skin.sum() < 40:
        return None
    relation_o = face_o - np.median(lab_o[skin][:, 1:], axis=0)
    face_r = np.median(lab_r[core][:, 1:], axis=0)
    relation_r = face_r - np.median(lab_r[skin][:, 1:], axis=0)
    return float(np.linalg.norm(relation_r - relation_o))


def _noise_level(gray: np.ndarray, region: np.ndarray) -> float | None:
    """Fine-grain level (sensor noise + pore texture) on the flat pixels of a region.

    Mean |residual| below ~1 px, measured where the broad image is flat, so
    edges and features do not count. Median-based estimators read ~0 on phone
    JPEGs; this one separates real skin grain from smoothed skin.
    """
    if region.sum() < 80:
        return None
    residual = gray - cv2.GaussianBlur(gray, (0, 0), GRAIN_SIGMA)
    broad = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gradient = np.hypot(cv2.Sobel(broad, cv2.CV_32F, 1, 0), cv2.Sobel(broad, cv2.CV_32F, 0, 1))
    flat = region & (gradient < np.percentile(gradient[region], 50))
    if flat.sum() < 40:
        return None
    return float(np.mean(np.abs(residual[flat]))) + 0.05


def grain_error(result: np.ndarray, original: np.ndarray, core: np.ndarray, outer: np.ndarray) -> float | None:
    gray_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    values = [_noise_level(gray, region) for gray, region in
              ((gray_o, core), (gray_o, outer), (gray_r, core), (gray_r, outer))]
    if any(value is None for value in values):
        return None
    face_o, ring_o, face_r, ring_r = values
    return abs(math.log((face_r / ring_r) / (face_o / ring_o)))


def detail_error(result: np.ndarray, original: np.ndarray, core: np.ndarray) -> float | None:
    """|log| change of facial micro-detail energy (waxy or over-sharpened)."""
    if core.sum() < 80:
        return None
    gray_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high_o = gray_o - cv2.GaussianBlur(gray_o, (0, 0), 1.25)
    high_r = gray_r - cv2.GaussianBlur(gray_r, (0, 0), 1.25)
    energy_o = math.sqrt(float(np.mean(high_o[core] ** 2)))
    energy_r = math.sqrt(float(np.mean(high_r[core] ** 2)))
    if energy_o < 0.6:
        return None   # a very soft original has no texture level to compare with
    return abs(math.log((energy_r + 0.25) / (energy_o + 0.25)))


def light_error(result: np.ndarray, original: np.ndarray, core: np.ndarray, face_width: float) -> float | None:
    """1 - correlation of the face's broad shading with the original's."""
    if core.sum() < 200:
        return None
    sigma = max(2.0, face_width * 0.10)
    broad = lambda image: cv2.GaussianBlur(_lab(image)[:, :, 0], (0, 0), sigma)
    a, b = broad(result)[core], broad(original)[core]
    a, b = a - a.mean(), b - b.mean()
    spread = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if spread < 1e-6 or float(np.std(b)) < 1.5:
        return None   # flat light: no direction to compare
    return 1.0 - float((a * b).sum()) / spread


def naturalness(
    result: np.ndarray,
    original: np.ndarray,
    face_width: float | None = None,
    mask: np.ndarray | None = None,
    region: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """{'score': 0..100 | None, 'checks': {name: 0..100}, 'errors': {...}}."""
    if result is None or original is None or result.shape != original.shape or result.ndim != 3:
        return {"score": None, "checks": {}, "errors": {}}
    result = result[:, :, :3]
    original = original[:, :, :3]
    if mask is None:
        changed = None
        if region is not None:
            # Only the face's neighbourhood counts: a re-saved photo differs
            # by a few levels everywhere.
            x, y, w, h = region
            limit = np.zeros(result.shape[:2], np.uint8)
            cv2.rectangle(limit, (int(x - 0.35 * w), int(y - 0.45 * h)), (int(x + 1.35 * w), int(y + 1.3 * h)), 1, -1)
            blurred_r = cv2.GaussianBlur(result, (0, 0), 1.5)
            blurred_o = cv2.GaussianBlur(original, (0, 0), 1.5)
            changed = changed_mask(blurred_r, blurred_o) & limit
            if face_width is None:
                face_width = float(w)
        mask = changed if changed is not None else changed_mask(result, original)
    else:
        mask = (mask > 0).astype(np.uint8)
    if int(mask.sum()) < 100:
        return {"score": 100.0, "checks": {name: 100.0 for name in WEIGHTS}, "errors": {}}
    if face_width is None:
        x, y, w, h = cv2.boundingRect(mask)
        face_width = float(max(w, 1))
    core, _, outer = _rings(mask, face_width)
    # Seams and light are read on the upper face (forehead, temples, cheeks):
    # below the mouth a new beard is real content, it may end on the jaw line
    # and darkens the broad shading. The jaw rim has its own detector.
    x, y, w, h = cv2.boundingRect(mask)
    upper = np.zeros(mask.shape, bool)
    upper[: y + int(h * UPPER_FACE_SHARE)] = True
    errors = {
        "border": border_error(result, original, mask, face_width, upper),
        "tone": tone_error(result, original, core, outer),
        "grain": grain_error(result, original, core, outer),
        "detail": detail_error(result, original, core),
        "light": light_error(result, original, core & upper, face_width),
    }
    checks = {name: round(_score(error, HALF_SCORE[name]), 1) for name, error in errors.items() if error is not None}
    if not checks:
        return {"score": None, "checks": {}, "errors": errors}
    total_weight = sum(WEIGHTS[name] for name in checks)
    # Weighted geometric mean: one broken check cannot be averaged away.
    log_mean = sum(WEIGHTS[name] * math.log(max(checks[name], 1.0) / 100.0) for name in checks) / total_weight
    return {
        "score": round(100.0 * math.exp(log_mean), 1),
        "checks": checks,
        "errors": {name: (None if value is None else round(float(value), 4)) for name, value in errors.items()},
    }


WORDS = {
    "border": "kenar",
    "tone": "ten uyumu",
    "grain": "gren",
    "detail": "detay",
    "light": "ışık",
}


def weakest(report: dict[str, Any]) -> str | None:
    checks = report.get("checks") or {}
    if not checks:
        return None
    name = min(checks, key=checks.get)
    return WORDS.get(name, name) if checks[name] < 70 else None
