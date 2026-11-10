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
