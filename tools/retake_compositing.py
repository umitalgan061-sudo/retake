"""Deterministic (non-generative) compositing for the Retake face pipeline.

FaceFusion pastes the swapped face back with a narrow blurred mask. When the
swapped skin tone or lighting differs from the target photo, that narrow
transition reads as a "cut-out" line around the face. The functions here
re-blend the swap into the target photo with classic image processing only:

1. ``build_change_mask``      – where FaceFusion actually changed pixels.
2. ``membrane_correction``    – gradient-domain (Poisson-style) tone matching:
   the colour offset measured along the rim of the swapped region is diffused
   smoothly through the face and removed, so the new face takes on the skin
   tone and lighting of the surrounding head while keeping its own features.
3. ``seamless_face_composite`` – the above plus multi-band (Laplacian pyramid)
   blending, so remaining differences fade over a wide zone while fine detail
   keeps a tight edge, followed by sharpness/grain matching against the photo.
4. ``fit_head_geometry``      – a smooth "liquify" warp that lengthens,
   shortens, widens or narrows the whole head (face, hairline, jaw and the
   pixels right around it move together), so no seam can open at the face edge.

Everything works on float32 BGR arrays; nothing here calls a neural network.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

CHANGE_THRESHOLD = 2
# Share of the rim colour offset that is removed from the swapped face. The
# small remainder keeps a trace of the reference skin tone.
HARMONIZE_STRENGTH = 0.9
RIM_INSET_RATIO = 0.08
RIM_INSET_MIN = 6.0
RIM_INSET_MAX = 34.0
MEMBRANE_FACE_WIDTH = 32.0
INNER_FEATHER_RATIO = 0.06
INNER_FEATHER_MIN = 4.0
INNER_FEATHER_MAX = 28.0
MIN_PYRAMID_LEVELS = 3
MAX_PYRAMID_LEVELS = 6
MAX_SHARPEN_GAIN = 0.45
MAX_GRAIN_GAIN = 0.5
MAX_SOFTEN_SIGMA = 1.6
HEAD_SCALE_MIN = 0.90
HEAD_SCALE_MAX = 1.12
HEAD_INNER_RADIUS = 0.62
HEAD_OUTER_LIMIT = 2.4
HEAD_OUTER_MINIMUM = 1.25


def _smootherstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _validate_bounds(face_bounds: dict[str, float]) -> tuple[float, float, float, float]:
    try:
        x, y, width, height = (float(face_bounds[key]) for key in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Yüz sınırları x, y, width ve height içermeli.") from error
    if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
        raise ValueError("Yüz sınırları sonlu ve pozitif olmalı.")
    return x, y, width, height


def build_change_mask(target: np.ndarray, swapped: np.ndarray, face_width: float) -> np.ndarray:
    """Binary uint8 mask (0/255) of the region FaceFusion changed, holes filled."""
    if target.shape != swapped.shape or target.ndim != 3:
        raise ValueError("Hedef ve yüz çıktısı aynı boyutta renkli görüntüler olmalı.")
    delta = np.max(cv2.absdiff(swapped, target), axis=2)
    changed = np.where(delta >= CHANGE_THRESHOLD, 255, 0).astype(np.uint8)
    kernel_size = int(max(5, round(face_width * 0.03))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(changed)
    areas = [cv2.contourArea(contour) for contour in contours]
    largest = max(areas)
    filled = np.zeros_like(changed)
    # Drop isolated specks (compression noise) and fill interior holes where
    # the swap happened to equal the target.
    keep = [contour for contour, area in zip(contours, areas) if area >= largest * 0.02]
    cv2.drawContours(filled, keep, -1, 255, thickness=cv2.FILLED)
    return filled


def _pyramid_levels(face_width: float, shape: tuple[int, int]) -> int:
    by_face = int(round(math.log2(max(face_width, 1.0) / 24.0)))
    by_image = int(math.floor(math.log2(max(8, min(shape)) / 8.0)))
    return max(1, min(max(MIN_PYRAMID_LEVELS, min(MAX_PYRAMID_LEVELS, by_face)), by_image))


def _gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [image]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def multiband_difference_blend(
    difference: np.ndarray,
    mask: np.ndarray,
    levels: int
) -> np.ndarray:
    """Blend ``difference`` (swap − target) through ``mask`` band by band.

    Fine bands use the tight mask, coarse bands an automatically wider one, so
    a colour offset fades over a broad zone instead of forming an edge.
    """
    gaussian = _gaussian_pyramid(difference, levels)
    masks = _gaussian_pyramid(mask, levels)
    blended = gaussian[-1] * masks[-1][:, :, None]
    for level in range(levels - 1, -1, -1):
        size = (gaussian[level].shape[1], gaussian[level].shape[0])
        laplacian = gaussian[level] - cv2.pyrUp(gaussian[level + 1], dstsize=size)
        blended = cv2.pyrUp(blended, dstsize=size) + laplacian * masks[level][:, :, None]
    return blended


_CROSS_KERNEL = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], dtype=np.float32)


def _relax_membrane(values: np.ndarray, interior: np.ndarray, iterations: int) -> np.ndarray:
    """Jacobi iterations of Laplace's equation inside ``interior``."""
    fixed = values.copy()
    inside = interior[:, :, None]
    for _ in range(iterations):
        relaxed = cv2.filter2D(values, -1, _CROSS_KERNEL, borderType=cv2.BORDER_REPLICATE)
        values = np.where(inside, relaxed, fixed)
    return values


def rim_inset(face_width: float) -> float:
    return min(RIM_INSET_MAX, max(RIM_INSET_MIN, face_width * RIM_INSET_RATIO))


def membrane_correction(difference: np.ndarray, mask: np.ndarray, face_width: float) -> np.ndarray:
    """Smooth tone offset of the swap relative to the photo.

    The offset (swap − target) is sampled on a rim just inside the changed
    region, where the swap is fully opaque, and interpolated harmonically
    across the interior. Subtracting it is equivalent to Poisson blending:
    the face meets the surrounding skin without a tonal step, and interior
    contrast (features, stubble, shading detail) is preserved.
    """
    height, width = mask.shape
    scale = min(1.0, MEMBRANE_FACE_WIDTH / max(face_width, 1.0))
    coarse_size = (max(8, int(round(width * scale))), max(8, int(round(height * scale))))
    inset = rim_inset(face_width)

    full_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5) > inset
    if not full_inside.any():
        return np.zeros_like(difference)
    smooth = cv2.GaussianBlur(difference, (0, 0), max(1.0, 0.5 / scale))

    coarse_values = cv2.resize(smooth, coarse_size, interpolation=cv2.INTER_AREA)
    coarse_inside = cv2.resize(full_inside.astype(np.uint8), coarse_size, interpolation=cv2.INTER_NEAREST) > 0
    coarse_values = _relax_membrane(coarse_values, coarse_inside, 1500)

    fine_size = (min(width, coarse_size[0] * 2), min(height, coarse_size[1] * 2))
    fine_inside = cv2.resize(full_inside.astype(np.uint8), fine_size, interpolation=cv2.INTER_NEAREST) > 0
    fine_fixed = cv2.resize(smooth, fine_size, interpolation=cv2.INTER_AREA)
    fine_values = cv2.resize(coarse_values, fine_size, interpolation=cv2.INTER_LINEAR)
    fine_values = np.where(fine_inside[:, :, None], fine_values, fine_fixed)
    fine_values = _relax_membrane(fine_values, fine_inside, 250)

    membrane = cv2.resize(fine_values, (width, height), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(membrane, (0, 0), max(1.0, 0.6 / scale))


def _luma(image: np.ndarray) -> np.ndarray:
    return image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299


def _detail_energy(luma: np.ndarray, weight: np.ndarray, sigma: float = 1.2) -> float:
    high = luma - cv2.GaussianBlur(luma, (0, 0), sigma)
    total = float(weight.sum())
    if total < 16.0:
        return 0.0
    return math.sqrt(float((high * high * weight).sum()) / total)


def match_photo_texture(composite: np.ndarray, target: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Match the swapped face's sharpness and sensor grain to the target photo.

    A 256 px swapper upscaled into a sharp photo looks airbrushed; a crisp
    swap inside a soft, noisy photo looks pasted. Both are corrected inside
    ``core`` (0..1) only.
    """
    target_energy = _detail_energy(_luma(target), core)
    composite_energy = _detail_energy(_luma(composite), core)
    if target_energy < 1e-3 or composite_energy < 1e-3:
        return composite
    ratio = composite_energy / target_energy
    weight = core[:, :, None]

    if ratio < 0.88:
        # Softer than the photo: restore micro-contrast, then add the photo's
        # own finest residual (real sensor grain / pores) for what is missing.
        gain = min(MAX_SHARPEN_GAIN, (1.0 / ratio - 1.0) * 0.6)
        high = composite - cv2.GaussianBlur(composite, (0, 0), 1.2)
        composite = composite + np.clip(high, -22.0, 22.0) * gain * weight
        grain_gain = min(MAX_GRAIN_GAIN, max(0.0, 1.0 - ratio))
        grain = np.clip(target - cv2.GaussianBlur(target, (0, 0), 0.8), -9.0, 9.0)
        composite = composite + grain * grain_gain * weight
    elif ratio > 1.25:
        # Sharper than the photo: soften to the photo's level of focus.
        sigma = min(MAX_SOFTEN_SIGMA, 0.6 * math.sqrt(ratio - 1.0) + 0.3)
        softened = cv2.GaussianBlur(composite, (0, 0), sigma)
        composite = composite + (softened - composite) * weight
    return composite


def seamless_face_composite(
    target: np.ndarray,
    swapped: np.ndarray,
    face_bounds: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (composite BGR uint8, alpha uint8) for a swapped face.

    ``composite`` equals the target wherever ``alpha`` is 0.
    """
    _, _, face_width, _ = _validate_bounds(face_bounds)
    mask = build_change_mask(target, swapped, face_width)
    if not mask.any():
        return target.copy(), np.zeros(target.shape[:2], dtype=np.uint8)

    target_f = target.astype(np.float32)
    difference = swapped.astype(np.float32) - target_f
    difference = difference - membrane_correction(difference, mask, face_width) * HARMONIZE_STRENGTH

    feather = min(INNER_FEATHER_MAX, max(INNER_FEATHER_MIN, face_width * INNER_FEATHER_RATIO))
    inward = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Centre the re-blend on the rim where the tone offset was measured: there
    # the corrected difference is ~0 by construction, so no step can form.
    # FaceFusion's own semi-transparent border is discarded.
    blend_start = max(0.0, rim_inset(face_width) - feather / 2.0)
    inner_mask = _smootherstep((inward - blend_start) / feather).astype(np.float32)

    levels = _pyramid_levels(face_width, target.shape[:2])
    blended = multiband_difference_blend(difference, inner_mask, levels)
    composite = target_f + blended

    core = _smootherstep((inner_mask - 0.5) / 0.5).astype(np.float32)
    composite = match_photo_texture(composite, target_f, core)

    # The coarse bands spread a little beyond the changed region; the matte
    # must cover that support, then fade where composite and target agree.
    spread = float(2 ** levels)
    pad = int(round(spread * 1.5)) | 1
    support = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad)))
    alpha = cv2.GaussianBlur(support, (0, 0), max(1.5, spread * 0.35))

    composite_u8 = np.clip(np.rint(composite), 0, 255).astype(np.uint8)
    composite_u8[alpha == 0] = target[alpha == 0]
    return composite_u8, alpha


def head_fit_maps(
    shape: tuple[int, int],
    face_bounds: dict[str, float],
    scale_x: float,
    scale_y: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sampling maps for a smooth head reshape and the displacement magnitude.

    Inside the face ellipse the head is scaled by (scale_x, scale_y) about the
    face centre; the scale relaxes to exactly 1 before the crop border, so the
    surrounding photo joins without a step.
    """
    if not (HEAD_SCALE_MIN <= scale_x <= HEAD_SCALE_MAX and HEAD_SCALE_MIN <= scale_y <= HEAD_SCALE_MAX):
        raise ValueError("Kafa oranı düzeltmesi izin verilen aralığın dışında.")
    x, y, face_width, face_height = _validate_bounds(face_bounds)
    height, width = shape
    center_x = x + face_width / 2.0
    center_y = y + face_height / 2.0
    radius_x = max(1.0, face_width * HEAD_INNER_RADIUS)
    radius_y = max(1.0, face_height * HEAD_INNER_RADIUS)

    room = min(
        center_x / radius_x, (width - 1 - center_x) / radius_x,
        center_y / radius_y, (height - 1 - center_y) / radius_y
    )
    outer = min(HEAD_OUTER_LIMIT, room)
    # A face touching the crop border has no room to relax the warp: fade it out.
    strength = float(np.clip((outer - 1.0) / (HEAD_OUTER_MINIMUM - 1.0), 0.0, 1.0))
    outer = max(outer, HEAD_OUTER_MINIMUM)

    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = grid_x - center_x
    dy = grid_y - center_y
    radius = np.sqrt((dx / radius_x) ** 2 + (dy / radius_y) ** 2)
    falloff = (1.0 - _smootherstep((radius - 1.0) / (outer - 1.0))) * strength
    gain_x = 1.0 + (scale_x - 1.0) * falloff
    gain_y = 1.0 + (scale_y - 1.0) * falloff
    map_x = (center_x + dx / gain_x).astype(np.float32)
    map_y = (center_y + dy / gain_y).astype(np.float32)
    displacement = np.sqrt((map_x - grid_x) ** 2 + (map_y - grid_y) ** 2)
    return map_x, map_y, displacement


def fit_head_geometry(
    composite: np.ndarray,
    alpha: np.ndarray,
    face_bounds: dict[str, float],
    scale_x: float,
    scale_y: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape the whole head so the new face sits naturally on it."""
    if abs(scale_x - 1.0) < 1e-4 and abs(scale_y - 1.0) < 1e-4:
        return composite, alpha
    map_x, map_y, displacement = head_fit_maps(composite.shape[:2], face_bounds, scale_x, scale_y)
    warped = cv2.remap(composite, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    warped_alpha = cv2.remap(alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Every pixel that moved must be covered by the matte; sub-pixel motion at
    # the rim fades out, where warped and original pixels are indistinguishable.
    moved = _smootherstep((displacement - 0.08) / 0.5) * 255.0
    out_alpha = np.maximum(warped_alpha.astype(np.float32), moved)
    out_alpha = np.clip(np.rint(out_alpha), 0, 255).astype(np.uint8)
    untouched = out_alpha == 0
    warped[untouched] = composite[untouched]
    return warped, out_alpha

