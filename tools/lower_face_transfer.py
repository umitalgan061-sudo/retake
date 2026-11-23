"""Pose-matched *real* lower-face transfer, with identity-safe fallbacks.

Unlike Retake's older beard control (which only darkens a colour field), this
moves high-frequency stubble, moustache and chin detail from a real reference
photo through its 478-point MediaPipe geometry. It intentionally leaves lips,
eyes, hair, clothing and the original photo outside the swapped region alone.

This module proposes an additional candidate; ArcFace and compositing gates
decide whether it is suitable. It is never automatically the final result.
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
MIN_LOWER_FIT = 0.02


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
) -> tuple[np.ndarray, int, list[dict[str, float]]]:
    """Return an additional lower-face candidate using real photo structure.

    Incompatible pose or missing facial hair should not introduce an
    unnatural face-shaped patch. FaceFusion's own occlusion/region mask is
    honored and every resulting candidate must be ArcFace-screened upstream.
    """
    if not candidates or not 0.0 < strength <= 1.0:
        raise ValueError("Alt yüz aktarma adayı veya gücü geçersiz.")
    if swapped.shape != target.shape or swapped.ndim != 3 or swapped.shape[2] != 3:
        raise ValueError("Hedef ve aktarım görseli aynı BGR boyutunda olmalı.")
    points = np.asarray(target_points, np.float32)
    if points.shape != (478, 2):
        raise ValueError("Hedef yüz işaret noktaları geçersiz.")

    allowed = _changed_mask(swapped, target, face_width)
    fits: list[dict[str, float]] = []
    best = None
    for index, candidate in enumerate(candidates):
        reference, reference_points = candidate
        source_points = np.asarray(reference_points, np.float32)
        if source_points.shape != (478, 2):
            continue
        warped, coverage, full_weight = _prepare_candidate(
            swapped, target, candidate, points, face_width, allowed
        )
        transfer = lower_face_mask(points, source_points, target.shape[:2], face_width)
        weight = full_weight * transfer * np.clip(allowed, 0.0, 1.0)
        if int(np.count_nonzero(weight > 0.1)) < max(64, int(face_width * face_width * 0.025)):
            continue
        filled = warped * coverage[:, :, None] + swapped.astype(np.float32) * (1.0 - coverage[:, :, None])
        fit = candidate_fit(filled, weight, swapped, target, face_width)
        fit["score"] -= 0.025 * index  # prefer the best pose from the browser on close fits
        fits.append(fit)
        if best is None or fit["score"] > best[0]:
            best = (fit["score"], index, warped, weight, fit["structure"])
    if best is None or best[4] < MIN_LOWER_FIT:
        raise ValueError("Referans sakal/çene pozları hedefle yeterince uyuşmuyor.")
    _, selected, warped, weight, _ = best
    warped = match_broad_light(warped, swapped, weight, face_width)
    amount = np.clip(weight * strength, 0.0, 1.0)
    merged = swapped.astype(np.float32) * (1.0 - amount[:, :, None]) + warped * amount[:, :, None]
    return np.clip(np.rint(merged), 0, 255).astype(np.uint8), selected, fits
