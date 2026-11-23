"""Conservative dark-halo correction for FaceFusion's thin face outline.

This is a deterministic cosmetic repair, not biometric verification. Edited
pixels stay inside the original FaceFusion alpha extent. A separate ArcFace
check must approve the candidate before it is returned.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def repair_dark_rim(
    target: np.ndarray, edited: np.ndarray, bounds: dict[str, Any]
) -> tuple[np.ndarray, dict[str, float]]:
    """Clean only a narrow darkened edge of the changed face; preserve alpha."""
    statistics = {"dark_line_pixels": 0.0, "corrected_pixels": 0.0, "dark_line_fraction": 0.0}
    if (
        target is None or edited is None or target.ndim != 3 or edited.ndim != 3
        or target.shape[:2] != edited.shape[:2] or target.shape[2] < 3
        or edited.shape[2] not in (3, 4)
    ):
        return edited, statistics
    try:
        x, y, width, height = (float(bounds[k]) for k in ("x", "y", "width", "height"))
    except (TypeError, ValueError, KeyError):
        return edited, statistics
    if not np.isfinite([x, y, width, height]).all() or min(width, height) < 40:
        return edited, statistics
    h, w = target.shape[:2]
    margin = max(15, int(round(0.24 * max(width, height))))
    left, top = max(0, int(x - margin)), max(0, int(y - margin))
    right, bottom = min(w, int(x + width + margin)), min(h, int(y + height + margin))
    if right - left < 40 or bottom - top < 40:
        return edited, statistics

    original = target[top:bottom, left:right, :3]
    result = edited[top:bottom, left:right, :3]
    alpha = (
        edited[top:bottom, left:right, 3]
        if edited.shape[2] == 4 else np.full(result.shape[:2], 255, np.uint8)
    )
    delta = cv2.absdiff(original, result).max(axis=2)
    changed = np.uint8(delta >= 14) * 255
    k = int(np.clip(round(width * 0.06), 7, 19)) | 1
    closed = cv2.morphologyEx(
        changed, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return edited, statistics
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < width * height * 0.12:
        return edited, statistics
    filled = np.zeros_like(changed)
    cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    ring_width = int(np.clip(round(width * 0.045), 7, 15)) | 1
    ring = cv2.morphologyEx(
        filled, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width, ring_width))
    ) > 0
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.int16)
    out_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.int16)
    new_dark = ring & ((orig_gray - out_gray) > 10) & (out_gray < 145) & (alpha > 0)
    count, edge_count = int(new_dark.sum()), int(ring.sum())
    fraction = count / max(1, edge_count)
    statistics["dark_line_pixels"] = float(count)
    statistics["dark_line_fraction"] = round(fraction, 5)
    if count < max(40, int(width * 0.14)) or fraction < 0.03:
        return edited, statistics

    support = cv2.dilate(
        new_dark.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ) > 0
    touched = ring & support & (alpha > 0)
    if not np.any(touched):
        return edited, statistics
    softened = cv2.medianBlur(result, 5)
    repaired = edited.copy()
    repaired_roi = repaired[top:bottom, left:right, :3]
    repaired_roi[touched] = softened[touched]
    statistics["corrected_pixels"] = float(np.count_nonzero(
        np.max(cv2.absdiff(repaired_roi, result), axis=2)[touched] >= 1
    ))
    return repaired, statistics


def repair_dark_rim_file(
    original_path: Path, composite_path: Path, repaired_path: Path,
    bounds: dict[str, Any]
) -> dict[str, float]:
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    result = cv2.imread(str(composite_path), cv2.IMREAD_UNCHANGED)
    if original is None or result is None:
        return {"dark_line_pixels": 0.0, "corrected_pixels": 0.0, "dark_line_fraction": 0.0}
    cleaned, report = repair_dark_rim(original, result, bounds)
    if report["corrected_pixels"] > 0 and not cv2.imwrite(str(repaired_path), cleaned):
        raise OSError("Yüz kontur onarımı yazılamadı.")
    return report
