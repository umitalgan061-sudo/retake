"""Pose-matched *real* lower-face transfer, with identity-safe fallbacks.

Unlike Retake's older beard control (which only darkens a colour field), this
moves high-frequency stubble, moustache and chin detail from a real reference
photo through its 478-point MediaPipe geometry. It intentionally leaves lips,
eyes, hair, clothing and the original photo outside the swapped region alone.

The candidate is approved by subsequent ArcFace and compositing gates. In
particular, FaceFusion's semantic parser has no dedicated "beard" choice:
we cannot gate the transplant solely on where the FaceFusion swap already
changed pixels, or the target person's original beard would remain.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    from .retake_detail import (
        INNER_MOUTH, _changed_mask, _prepare_candidate,
        candidate_fit, match_broad_light
    )
    from .retake_appearance import LOWER_FACE, LIPS
except ImportError:
    from retake_detail import (
        INNER_MOUTH, _changed_mask, _prepare_candidate,
        candidate_fit, match_broad_light
    )
    from retake_appearance import LOWER_FACE, LIPS

# Include the upper-lip moustache without reaching the nostrils or lower nose.
MOUSTACHE = (98, 2, 327, 425, 291, 0, 61, 205)
MIN_LOWER_FIT = -0.12


def lower_face_mask(
    target_points: np.ndarray,
    source_points: np.ndarray,
    shape: tuple[int, int],
    face_width: float,
) -> np.ndarray:
    """Soft 0..1 mask for beard/cheek-jaw/chin and moustache only."""
    pts = np.asarray(target_points, dtype=np.float32)
    source = np.asarray(source_points, dtype=np.float32)
    if pts.shape != (478, 2) or source.shape != (478, 2):
        raise ValueError("Alt yüz aktarımı için 478 yüz işaret noktası gerekir.")
    height, width = shape
    lower = np.zeros((height, width), np.uint8)
    cv2.fillPoly(lower, [np.round(pts[LOWER_FACE]).astype(np.int32)], 255)
    cv2.fillPoly(lower, [np.round(pts[list(MOUSTACHE)]).astype(np.int32)], 255)
    if np.count_nonzero(lower) < max(25, int(face_width * face_width * 0.02)):
        raise ValueError("Alt yüz/sakal alanı çok küçük.")

    # Lips and any visible mouth opening must remain from the swap so a
    # different smile/teeth position cannot be pasted into the photograph.
    mouth = np.zeros_like(lower)
    cv2.fillPoly(mouth, [np.round(pts[LIPS]).astype(np.int32)], 255)
    cv2.fillPoly(mouth, [np.round(pts[INNER_MOUTH]).astype(np.int32)], 255)
    margin = int(max(3, face_width * 0.025)) | 1
    mouth = cv2.dilate(mouth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin)))

    region = ((lower > 0) & (mouth == 0)).astype(np.uint8)
    inward = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    feather = max(2.0, face_width * 0.042)
    t = np.clip(inward / feather, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def choose_and_transfer_lower_face(
    swapped: np.ndarray,
    target: np.ndarray,
    candidates: list[tuple[np.ndarray, np.ndarray]],
    target_points: np.ndarray,
    face_width: float,
    *,
    strength: float = 0.90,
    moustache_scale: float = 1.0,
    jaw_scale: float = 1.0,
    alternate: bool = False,
) -> tuple[np.ndarray, int, list[dict[str, float]]]:
    """Return an additional lower-face candidate using real photo structure.

    Incompatible pose or missing facial hair should not introduce an
    unnatural face-shaped patch. The landmark geometry excludes hair, eyes,
    lips and clothing; existing FaceFusion changes guide the opacity but
    MUST NOT be the only gate (its parser has no separate beard label).
    The final candidate must pass ArcFace and seam screening upstream.
    Unmodelled hands over the beard remain a limitation.
    """
    if not candidates or not 0.0 < strength <= 1.0 or not all(0.75 <= value <= 1.25 for value in (moustache_scale, jaw_scale)):
        raise ValueError("Alt yüz aktarma adayı veya gücü geçersiz.")
    if swapped.shape != target.shape or swapped.ndim != 3 or swapped.shape[2] != 3:
        raise ValueError("Hedef ve aktarım görseli aynı BGR boyutunda olmalı.")
    points = np.asarray(target_points, np.float32)
    if points.shape != (478, 2):
        raise ValueError("Hedef yüz işaret noktaları geçersiz.")

    allowed = _changed_mask(swapped, target, face_width)
    fits: list[dict[str, float]] = []
    ranked = []
    for index, candidate in enumerate(candidates):
        reference, reference_points = candidate
        source_points = np.asarray(reference_points, np.float32)
        if source_points.shape != (478, 2):
            continue
        # Do not use the swap's change mask here: FaceFusion marks only
        # skin/eyes/nose/lips, frequently leaving original beard pixels
        # untouched. The geometry and mouth exclusion supply the safe region.
        warped, coverage, full_weight = _prepare_candidate(
            swapped, target, candidate, points, face_width,
            np.ones_like(allowed, dtype=np.float32)
        )
        transfer = lower_face_mask(points, source_points, target.shape[:2], face_width)
        # Allow substantial real reference hair transfer even where the
        # semantic swap did not edit pixels, while preserving its evidence
        # of trusted changed pixels and retaining a feathered jaw margin.
        confidence = 0.80 + 0.20 * np.clip(allowed, 0.0, 1.0)
        weight = full_weight * transfer * confidence
        if int(np.count_nonzero(weight > 0.1)) < max(64, int(face_width * face_width * 0.025)):
            continue
        filled = warped * coverage[:, :, None] + swapped.astype(np.float32) * (1.0 - coverage[:, :, None])
        fit = candidate_fit(filled, weight, swapped, target, face_width)
        fit["score"] -= 0.025 * index  # prefer the best pose from the browser on close fits
        fits.append(fit)
        if fit["structure"] >= MIN_LOWER_FIT:
            ranked.append((fit["score"], index, warped, weight, fit["structure"]))
    if not ranked:
        raise ValueError("Referans sakal/çene pozları hedefle yeterince uyuşmuyor.")
    ranked.sort(key=lambda row: row[0], reverse=True)
    best = ranked[0]
    if alternate and len(ranked)>1 and ranked[1][0] >= best[0]-.08:
        best=ranked[1]
    _, selected, warped, weight, _ = best
    warped = match_broad_light(warped, swapped, weight, face_width)
    # Independent moustache vs jaw strengths, with a soft transition and
    # the existing lip-exclusion and face-coverage masks still enforced.
    amount = zoned_transfer_amount(weight, points, face_width, strength, moustache_scale, jaw_scale)
    merged = swapped.astype(np.float32) * (1.0 - amount[:, :, None]) + warped * amount[:, :, None]
    return np.clip(np.rint(merged), 0, 255).astype(np.uint8), selected, fits


def zoned_transfer_amount(weight, points, face_width, strength, moustache_scale=1., jaw_scale=1.):
    if moustache_scale == jaw_scale:
        return np.clip(weight*strength*jaw_scale,0.,1.)
    region=np.zeros(weight.shape,np.uint8)
    cv2.fillPoly(region,[np.round(points[list(MOUSTACHE)]).astype(np.int32)],255)
    zone=cv2.GaussianBlur(region.astype(np.float32)/255.,(0,0),max(1.,face_width*.018))
    scale=jaw_scale+(moustache_scale-jaw_scale)*zone
    return np.clip(weight*strength*scale,0.,1.)
