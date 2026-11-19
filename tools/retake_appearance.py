"""The person's own beard and stubble, measured from the references.

Swap models render facial hair lighter and thinner than it is (a 256 px face,
smoothed). The 25-photo appearance profile (references aligned on the eyes
and averaged, sent by the browser) still carries the right *relative* tone:
how much darker, and of what colour, the lower face is than the cheeks. That
is independent of the scene light, so the swapped face's lower face is moved
towards the same relation. Hair-like pixels (darker than the cheek) take most
of the change; bare skin barely moves.

Measured on the user's photos (OpenCV Lab, lower face minus cheeks): the
references -70 L, the swap -46 L; corrected to -64 L with the ArcFace
similarity unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

# Canonical layout of the browser's aligned profile (createAlignedReferenceCanvas).
PROFILE_SIZE = (512, 682)          # width, height
PROFILE_EYES = ((168.0, 238.0), (344.0, 238.0))
BEARD_STRENGTH = 0.8
MAX_DARKEN = 45.0                  # Lab L units
MAX_CHROMA_SHIFT = 12.0
# The references' colour relation depends on their light (indoor shots make
# a beard look grey against the cheek); carried in full it greys the beard in
# a sunny photo. Darkness carries over, colour only a little.
CHROMA_SHARE = 0.25
MIN_CORRECTION = 4.0               # smaller differences are measurement noise

# MediaPipe face-mesh points.
LOWER_FACE = [2, 98, 205, 207, 214, 135, 138, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 367,
              364, 434, 427, 425, 327]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
CHEEKS = (50, 280)


def _lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)


def profile_regions(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(lower face, cheeks) boolean masks in the aligned profile."""
    height, width = shape
    sx, sy = width / PROFILE_SIZE[0], height / PROFILE_SIZE[1]
    (lx, ly), (rx, _) = PROFILE_EYES
    distance = rx - lx
    cx = (lx + rx) / 2.0
    point = lambda x, y: (int(round(x * sx)), int(round(y * sy)))
    lower = np.zeros((height, width), np.uint8)
    cv2.ellipse(lower, point(cx, ly + 1.42 * distance), (int(0.72 * distance * sx), int(0.50 * distance * sy)),
                0, 0, 360, 1, -1)
    lower[: point(0, ly + 0.95 * distance)[1]] = 0
    lips = np.zeros_like(lower)
    cv2.ellipse(lips, point(cx, ly + 1.13 * distance), (int(0.34 * distance * sx), int(0.14 * distance * sy)),
                0, 0, 360, 1, -1)
    lower[lips > 0] = 0
    cheeks = np.zeros_like(lower)
    radius = max(2, int(0.10 * distance * sx))
    for side in (-1, 1):
        cv2.circle(cheeks, point(cx + side * 0.54 * distance, ly + 0.55 * distance), radius, 1, -1)
    return lower > 0, cheeks > 0


def profile_beard_offset(profile: np.ndarray) -> np.ndarray:
    """Lab (L, a, b) of the lower face minus the cheeks in the aligned profile."""
    lower, cheeks = profile_regions(profile.shape[:2])
    lab = _lab(profile)
    return lab[lower].mean(axis=0) - lab[cheeks].mean(axis=0)


def face_regions(points: np.ndarray, shape: tuple[int, int], face_width: float) -> tuple[np.ndarray, np.ndarray]:
    """(lower face, cheeks) boolean masks from a 478-point mesh in image pixels."""
    height, width = shape
    lower = np.zeros((height, width), np.uint8)
    cv2.fillPoly(lower, [np.round(points[LOWER_FACE]).astype(np.int32)], 1)
    lips = np.zeros_like(lower)
    cv2.fillPoly(lips, [np.round(points[LIPS]).astype(np.int32)], 1)
    grow = int(max(3, face_width * 0.02)) | 1
    lips = cv2.dilate(lips, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    lower[lips > 0] = 0
    cheeks = np.zeros_like(lower)
    radius = max(2, int(face_width * 0.06))
    for index in CHEEKS:
        cv2.circle(cheeks, tuple(int(v) for v in np.round(points[index])), radius, 1, -1)
    return lower > 0, cheeks > 0


def match_beard(
    composite: np.ndarray,
    alpha: np.ndarray,
    points: np.ndarray,
    reference_offset: np.ndarray,
    face_width: float,
    strength: float = BEARD_STRENGTH
) -> np.ndarray:
    """Move the lower face's tone relative to the cheeks towards the person's.

    Only darkens (a swap that already has a darker beard than the person is
    left alone) and only inside the swap's matte. Returns uint8 BGR.
    """
    lower, cheeks = face_regions(points, composite.shape[:2], face_width)
    inside = alpha.astype(np.float32) / 255.0
    lower &= inside > 0.5
    cheeks &= inside > 0.5
    if lower.sum() < 30 or cheeks.sum() < 10 or strength <= 0:
        return composite
    lab = _lab(composite)
    current = lab[lower].mean(axis=0) - lab[cheeks].mean(axis=0)
    wanted = current + strength * (np.asarray(reference_offset, np.float32) - current)
    change = wanted - current
    change[1:] *= CHROMA_SHARE
    if change[0] > -MIN_CORRECTION and np.abs(change[1:]).max() < MIN_CORRECTION:
        return composite
    cheek_l = float(lab[cheeks][:, 0].mean())
    hair = np.clip((cheek_l - lab[:, :, 0]) / 40.0, 0.0, 1.0) ** 0.7   # darker than the cheek: hair-like
    soft = cv2.GaussianBlur(lower.astype(np.float32), (0, 0), max(1.5, face_width * 0.03)) * inside
    weight = soft * (0.25 + 0.75 * hair)
    mean_weight = float(weight[lower].mean())
    if mean_weight < 1e-3:
        return composite
    gain = change / mean_weight
    gain[0] = float(np.clip(gain[0], -MAX_DARKEN, 0.0))
    gain[1:] = np.clip(gain[1:], -MAX_CHROMA_SHIFT, MAX_CHROMA_SHIFT)
    corrected = lab + weight[:, :, None] * gain[None, None, :]
    converted = cv2.cvtColor(np.clip(corrected, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    # Lab round trips shift colours by a unit; touch only corrected pixels.
    out = np.clip(composite, 0, 255).astype(np.uint8).copy()
    touched = weight > 1e-3
    out[touched] = converted[touched]
    return out
