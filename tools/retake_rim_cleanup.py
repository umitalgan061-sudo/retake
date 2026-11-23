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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clean only a narrow darkened edge of the changed face; preserve alpha."""
    statistics = {
        "dark_line_pixels": 0.0, "corrected_pixels": 0.0, "dark_line_fraction": 0.0,
        "dark_line_remaining": 0.0, "dark_line_reduction": 0.0, "stage": "not_examined"
    }
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
    # A 7-pixel contour search band misses the outer half of real 2–4px
    # matte outlines; the detector then "repairs" only part of the line.
    # The widened ring only locates candidates: pixels still need to be
    # darker than the unchanged original AND inside the face alpha.
    ring_width = int(np.clip(round(width * 0.085), 11, 27)) | 1
    # The broad ring locates the face contour, but its total area is too
    # large to score a one-pixel hairline/chin outline. In production v21 a
    # visible thin line was skipped: the numerator was a narrow trace while
    # the denominator was a 11–27px band. Assess the actual ink pixels
    # against a narrow 7–13px strip, not the broad search area.
    broad_ring = cv2.morphologyEx(
        filled, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width, ring_width))
    ) > 0
    narrow_width = int(np.clip(round(width * 0.035), 7, 13)) | 1
    narrow_ring = cv2.morphologyEx(
        filled, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (narrow_width, narrow_width))
    ) > 0
    # Score only the thin true contour; repair the entire broad halo band.
    # Restricting the repair itself to the thin band missed half of a 3px
    # synthetic outline and left a visible remnant (caught by CI).
    edge = broad_ring & (alpha > 0)
    scoring_edge = narrow_ring & edge
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.int16)
    out_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.int16)
    new_dark = edge & ((orig_gray - out_gray) > 10) & (out_gray < 145)
    count, edge_count = int(new_dark.sum()), int(scoring_edge.sum())
    fraction = int((new_dark & scoring_edge).sum()) / max(1, edge_count)
    statistics["dark_line_pixels"] = float(count)
    statistics["edge_pixels"] = float(edge_count)
    statistics["dark_line_fraction"] = round(fraction, 5)
    if count < max(24, int(width * 0.10)):
        statistics["stage"] = "few_dark_pixels"
        return edited, statistics
    if fraction < 0.015:
        statistics["stage"] = "low_edge_fraction"
        return edited, statistics
    statistics["stage"] = "candidate_detected"

    # Median filtering cannot reconstruct a continuous 2–4-pixel dark
    # outline. It can report thousands of edited pixels while leaving
    # a visible black chin/hairline edge. Reconstruct ONLY pixels that
    # became darker than the unedited target along the face boundary.
    paint_mask = new_dark.astype(np.uint8) * 255
    inpainted = cv2.inpaint(result, paint_mask, 3.0, cv2.INPAINT_TELEA)
    local_median = cv2.medianBlur(result, 5)
    candidates = (inpainted, local_median)
    baseline = int(np.count_nonzero(new_dark))
    best_candidate = None
    best_remaining = baseline
    for candidate in candidates:
        gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.int16)
        remaining = int(np.count_nonzero(
            new_dark & ((orig_gray - gray) > 10) & (gray < 145)
        ))
        if remaining < best_remaining:
            best_candidate = candidate
            best_remaining = remaining
    statistics["dark_line_remaining"] = float(best_remaining)
    statistics["dark_line_reduction"] = round((baseline - best_remaining) / max(1, baseline), 5)
    # Do not claim success because a blur changed arbitrary skin pixels:
    # require measurable reduction of the offending dark outline.
    if best_candidate is None or baseline - best_remaining < max(12, int(baseline * 0.25)):
        statistics["stage"] = "reduction_below_25_percent"
        return edited, statistics
    repaired = edited.copy()
    repaired_roi = repaired[top:bottom, left:right, :3]
    repaired_roi[new_dark] = best_candidate[new_dark]
    statistics["corrected_pixels"] = float(np.count_nonzero(
        np.max(cv2.absdiff(repaired_roi, result), axis=2)[new_dark] >= 1
    ))
    statistics["stage"] = "repaired"
    return repaired, statistics


def repair_dark_rim_file(
    original_path: Path, composite_path: Path, repaired_path: Path,
    bounds: dict[str, Any]
) -> dict[str, Any]:
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    result = cv2.imread(str(composite_path), cv2.IMREAD_UNCHANGED)
    if original is None or result is None:
        return {"dark_line_pixels": 0.0, "corrected_pixels": 0.0, "dark_line_fraction": 0.0}
    cleaned, report = repair_dark_rim(original, result, bounds)
    if report["corrected_pixels"] > 0 and not cv2.imwrite(str(repaired_path), cleaned):
        raise OSError("Yüz kontur onarımı yazılamadı.")
    return report


def inspect_dark_rim_file(
    original_path: Path, composite_path: Path, bounds: dict[str, Any]
) -> dict[str, Any]:
    """Measure the visible residual halo without changing an output file.

    Uses the *same* contour, changed-pixel, and target-relative darkness
    detector as repair_dark_rim. A low fraction is a visual-quality heuristic,
    not a biometric or calibrated photorealism probability.
    """
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    result = cv2.imread(str(composite_path), cv2.IMREAD_UNCHANGED)
    if original is None or result is None:
        return {"stage": "not_examined", "dark_line_fraction": 0.0}
    _, report = repair_dark_rim(original, result, bounds)
    return report
