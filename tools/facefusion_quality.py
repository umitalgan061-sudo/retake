from __future__ import annotations

PIXEL_BOOST_FALLBACKS: dict[str, tuple[str, ...]] = {
    "1024x1024": ("768x768", "512x512"),
    "768x768": ("512x512",),
    "512x512": ()
}

# Conservative no-op guard; this is not a perceptual-quality score.
NOOP_MAX_CHANGED_PIXEL_RATIO = 0.001
NOOP_MAX_MEAN_ABS_DELTA = 0.5
NOOP_MAX_P95_ABS_DELTA = 2.0


def is_near_noop(changed_pixel_ratio: float, mean_abs_delta: float, p95_abs_delta: float) -> bool:
    return (
        changed_pixel_ratio <= NOOP_MAX_CHANGED_PIXEL_RATIO
        and mean_abs_delta <= NOOP_MAX_MEAN_ABS_DELTA
        and p95_abs_delta <= NOOP_MAX_P95_ABS_DELTA
    )


def measure_final_face_edit(target_path, composite_path, bounds):
    """Measure face interior only, ignoring changed RGB outside delivered alpha.
    A heuristic no-op gate, not proof of perceptual improvement or identity."""
    import cv2
    import numpy as np
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    composite = cv2.imread(str(composite_path), cv2.IMREAD_UNCHANGED)
    if target is None or composite is None or composite.ndim != 3 or composite.shape[:2] != target.shape[:2] or composite.shape[2] not in (3, 4):
        raise ValueError("Yüz değişim ölçümü için görüntüler uyumsuz.")
    height, width = target.shape[:2]
    x, y, w, h = (float(bounds[k]) for k in ("x", "y", "width", "height"))
    if not all(np.isfinite([x, y, w, h])) or w <= 0 or h <= 0:
        raise ValueError("Yüz ölçüm alanı geçersiz.")
    yy, xx = np.ogrid[:height, :width]
    core = ((xx + .5 - x - w*.5)/(w*.36))**2 + ((yy + .5 - y - h*.5)/(h*.39))**2 <= 1
    if not np.any(core):
        raise ValueError("Yüz ölçüm alanı boş.")
    delta = np.max(cv2.absdiff(target, composite[:, :, :3]), axis=2)
    if composite.shape[2] == 4:
        delta = np.where(composite[:, :, 3] > 0, delta, 0)
    values = delta[core]
    changed_ratio = float(np.mean(values >= 8))
    mean_delta = float(np.mean(values))
    return {"changed_ratio": changed_ratio, "mean_delta": mean_delta,
            "meaningful": changed_ratio >= .03 and mean_delta >= 1.5}
