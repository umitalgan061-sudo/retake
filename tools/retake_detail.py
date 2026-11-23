"""Real-photo face for the swap.

Swap models render the face at 256 px and upscale it, so nose, lips and skin
come out soft and synthetic. The person's own reference photos are real. The
browser picks the reference with the closest pose and expression and sends it
with its 478-point face mesh; here it is warped onto the target face triangle
by triangle and replaces the swap's face from the brows down to the chin.
The swap's own mask (hair, hands, glasses) is kept, the open mouth stays the
photo's, and the usual seamless composite then matches light, tone and grain.

Everything here is classic image processing (OpenCV/NumPy).
"""
from __future__ import annotations

import cv2
import numpy as np

MESH_POINTS = 478

# MediaPipe face-mesh contours.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
INNER_MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]


def _smootherstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def triangulate(points: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    """Delaunay triangles of `points` as index triplets."""
    height, width = shape
    margin = 4.0 * max(width, height)
    subdiv = cv2.Subdiv2D((int(-margin), int(-margin), int(width + 2 * margin), int(height + 2 * margin)))
    lookup: dict[tuple[float, float], int] = {}
    for index, (x, y) in enumerate(points):
        key = (round(float(x), 3), round(float(y), 3))
        if key in lookup:
            continue
        lookup[key] = index
        subdiv.insert((float(x), float(y)))
    triangles = []
    for x1, y1, x2, y2, x3, y3 in subdiv.getTriangleList():
        corners = [(round(float(x), 3), round(float(y), 3)) for x, y in ((x1, y1), (x2, y2), (x3, y3))]
        if all(corner in lookup for corner in corners):
            triangles.append(tuple(lookup[corner] for corner in corners))
    return triangles


def _mesh_size(points: np.ndarray) -> float:
    span = np.asarray(points, np.float64).max(axis=0) - np.asarray(points, np.float64).min(axis=0)
    return float(np.sqrt(max(span[0], 0.0) * max(span[1], 0.0)))


def warp_piecewise_affine(
    source: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Warp `source` so each mesh triangle lands on its target triangle.

    Returns the warped image (float32, target shape) and a 0..1 coverage mask.
    """
    height, width = shape
    channels = 1 if source.ndim == 2 else source.shape[2]
    warped = np.zeros((height, width, channels), np.float32)
    coverage = np.zeros((height, width), np.float32)
    # Shrinking with bilinear sampling skips pixels: hair, stubble and pores
    # alias into speckle and stripes. Bring the source to the target's scale
    # first with area averaging.
    source_points = np.asarray(source_points, np.float32)
    ratio = _mesh_size(target_points) / max(_mesh_size(source_points), 1e-6)
    if ratio < 0.9:
        size = (max(1, int(round(source.shape[1] * ratio))), max(1, int(round(source.shape[0] * ratio))))
        source = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
        source_points = source_points * np.float32(ratio)
    src = source.astype(np.float32).reshape(source.shape[0], source.shape[1], channels)
    for a, b, c in triangulate(target_points, shape):
        dst_tri = target_points[[a, b, c]].astype(np.float32)
        src_tri = source_points[[a, b, c]].astype(np.float32)
        u, v = dst_tri[1] - dst_tri[0], dst_tri[2] - dst_tri[0]
        area = abs(float(u[0] * v[1] - u[1] * v[0]))
        if area < 0.5:
            continue
        x, y, w, h = cv2.boundingRect(dst_tri)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        matrix = cv2.getAffineTransform(src_tri, dst_tri - np.float32([x0, y0]))
        patch = cv2.warpAffine(src, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        patch = patch.reshape(y1 - y0, x1 - x0, channels)
        tri_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(tri_mask, np.round(dst_tri - np.float32([x0, y0])).astype(np.int32), 1, lineType=cv2.LINE_8)
        region = tri_mask.astype(bool)
        warped[y0:y1, x0:x1][region] = patch[region]
        coverage[y0:y1, x0:x1][region] = 1.0
    # Rasterising triangle by triangle leaves 1-px seams between neighbours;
    # fill them from the surrounding pixels (normalised convolution).
    hull = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.round(target_points).astype(np.int32)), 1)
    gaps = (hull > 0) & (coverage == 0)
    if gaps.any():
        blurred = cv2.GaussianBlur(warped * coverage[:, :, None], (0, 0), 1.0).reshape(height, width, channels)
        norm = cv2.GaussianBlur(coverage, (0, 0), 1.0)
        filled = blurred / np.maximum(norm, 1e-4)[:, :, None]
        ok = gaps & (norm > 0.05)
        warped[ok] = filled[ok]
        coverage[ok] = 1.0
    return (warped[:, :, 0] if channels == 1 else warped), coverage


def validate_mesh(points, width: float, height: float, margin: float) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (MESH_POINTS, 2) or not np.isfinite(array).all():
        raise ValueError(f"Yüz ağı {MESH_POINTS} noktalık sayı çiftleri olmalı.")
    if (array[:, 0] < -margin).any() or (array[:, 0] > width + margin).any() \
            or (array[:, 1] < -margin).any() or (array[:, 1] > height + margin).any():
        raise ValueError("Yüz ağı noktaları görselin dışında.")
    return array.astype(np.float32)


def _brow_line(points: np.ndarray) -> tuple[float, float]:
    eye_distance = float(np.hypot(*(points[263] - points[33])))
    top = min(float(points[LEFT_BROW + RIGHT_BROW, 1].min()), float(points[[33, 263], 1].min()))
    return top - 0.28 * eye_distance, eye_distance


def reference_skin(warped: np.ndarray, points: np.ndarray, coverage: np.ndarray, face_width: float) -> np.ndarray:
    """0..1 likelihood that each warped-reference pixel is the person's skin.

    The skin colour is learnt from the reference's own cheeks and nose
    (between the eyes and the mouth); hair, brows-to-hairline shadows and
    glasses frames fall outside it. Used for the forehead, where the
    reference's fringe would otherwise be pasted onto the photo.
    """
    height, width = coverage.shape
    lab = cv2.cvtColor(np.clip(warped, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    eye_y = float(points[[33, 263], 1].mean())
    mouth_y = float(points[[13, 14], 1].mean())
    oval = np.zeros((height, width), np.uint8)
    cv2.fillPoly(oval, [np.round(points[FACE_OVAL]).astype(np.int32)], 1)
    yy = np.arange(height, dtype=np.float32)[:, None]
    sample = (oval > 0) & (coverage > 0.5) & (yy > eye_y + 0.15 * (mouth_y - eye_y)) & (yy < mouth_y - 0.25 * (mouth_y - eye_y))
    if int(sample.sum()) < 50:
        return np.ones((height, width), np.float32)
    pixels = lab[sample]
    centre = np.median(pixels, axis=0)
    spread = np.maximum(np.percentile(np.abs(pixels - centre), 75, axis=0) * 1.5, (9.0, 4.0, 4.0))
    # Lighter skin (a forehead in the light) is fine; darker or off-hue is not.
    delta = (lab - centre) / spread
    delta[:, :, 0] = np.minimum(delta[:, :, 0], 0.0) + np.maximum(delta[:, :, 0], 0.0) * 0.3
    distance = np.sqrt((delta * delta).sum(axis=2) / 3.0)
    skin = _smootherstep((2.6 - distance) / 1.2)
    return cv2.GaussianBlur(skin.astype(np.float32), (0, 0), max(1.0, face_width * 0.008))


MOUTH_OPEN_RATIO = 0.12   # inner-lip gap / mouth width


def mouth_is_open(points: np.ndarray) -> bool:
    """Teeth or a dark mouth show: the inner lips are apart."""
    width = float(np.hypot(*(points[308] - points[78])))
    gap = float(np.hypot(*(points[14] - points[13])))
    return width > 1e-6 and gap / width > MOUTH_OPEN_RATIO


def transplant_region(
    points: np.ndarray,
    shape: tuple[int, int],
    face_width: float,
    forehead_skin: np.ndarray | None = None,
    keep_mouth: bool = True
) -> np.ndarray:
    """0..1 region for the transplant: the face oval, minus the mouth opening
    when `keep_mouth` (an open mouth cannot be warped from a closed one).

    Above the brows (where hair often falls) the reference is only used where
    `forehead_skin` says it is skin; without it the forehead is left out.
    """
    height, width = shape
    oval = np.zeros((height, width), np.uint8)
    cv2.fillPoly(oval, [np.round(points[FACE_OVAL]).astype(np.int32)], 255)
    brow_top, eye_distance = _brow_line(points)
    yy = np.arange(height, dtype=np.float32)[:, None]
    fade = max(2.0, eye_distance * 0.12)
    below_brows = _smootherstep((yy - brow_top) / fade)
    if forehead_skin is not None:
        below_brows = below_brows + (1.0 - below_brows) * np.clip(forehead_skin, 0.0, 1.0)
    mouth = np.zeros_like(oval)
    if keep_mouth:
        cv2.fillPoly(mouth, [np.round(points[INNER_MOUTH]).astype(np.int32)], 255)
        grow = int(max(3, face_width * 0.01)) | 1
        mouth = cv2.dilate(mouth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    region = (oval > 0) & (mouth == 0)
    # A soft edge: the real beard and skin fade into the swap instead of
    # ending on the jaw contour.
    feather = max(1.5, face_width * 0.035)
    inward = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    return (_smootherstep(inward / feather) * below_brows).astype(np.float32)


def _changed_mask(swapped: np.ndarray, target: np.ndarray, face_width: float) -> np.ndarray:
    """Where the swap model replaced the photo (its mask keeps hair, hands, glasses)."""
    changed = np.max(cv2.absdiff(swapped, target), axis=2) >= 2
    grow = int(max(3, face_width * 0.02)) | 1
    changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    return cv2.GaussianBlur(changed.astype(np.float32), (0, 0), max(1.0, face_width * 0.01))


def _log_luma(image: np.ndarray) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0, 255)
    return np.log(image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299 + 4.0)


def _weighted_correlation(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    total = float(weight.sum())
    if total < 16:
        return 0.0
    a = a - float((a * weight).sum()) / total
    b = b - float((b * weight).sum()) / total
    denominator = np.sqrt(float((a * a * weight).sum()) * float((b * b * weight).sum()))
    return float((a * b * weight).sum()) / denominator if denominator > 1e-9 else 0.0


def candidate_fit(
    warped: np.ndarray,
    weight: np.ndarray,
    swapped: np.ndarray,
    target: np.ndarray,
    face_width: float
) -> dict[str, float]:
    """How naturally a warped reference sits on the face.

    structure: its dark features (eyes, brows, nostrils, lip line) land where
      the photo's own are, and where the swap (rendered in the photo's exact
      pose and expression) has them; glasses, a hand or another expression in
      the reference lower it. This decides the choice.
    light: its large-scale light and shadow resemble the photo's; a small
      tie-break, since the composite corrects the light afterwards anyway.
    """
    log_w = _log_luma(warped)
    low = max(0.8, face_width * 0.012)
    high = max(low * 2.0, face_width * 0.05)
    band = lambda image: cv2.GaussianBlur(image, (0, 0), low) - cv2.GaussianBlur(image, (0, 0), high)
    features = band(log_w)
    to_photo = _weighted_correlation(features, band(_log_luma(target)), weight)
    to_swap = _weighted_correlation(features, band(_log_luma(swapped)), weight)
    broad = max(2.0, face_width * 0.08)
    light = _weighted_correlation(
        cv2.GaussianBlur(log_w, (0, 0), broad), cv2.GaussianBlur(_log_luma(target), (0, 0), broad), weight
    )
    structure = to_photo + 0.5 * to_swap
    return {"structure": structure, "light": light, "score": structure + 0.2 * light}


LIGHT_MATCH_SIGMA_RATIO = 0.1
LIGHT_GAIN_RANGE = (0.5, 2.0)


def match_broad_light(warped: np.ndarray, swapped: np.ndarray, weight: np.ndarray, face_width: float) -> np.ndarray:
    """Give the real photo the scene's broad light and colour, keep its detail.

    The reference was taken in other light (indoor, other white balance).
    Per channel, its large-scale level is scaled to the swap's (which already
    sits in the photo's light) with a smooth gain field, so pores, hairs and
    creases stay the reference's own while the cheek, forehead and jaw edges
    meet the surrounding skin without a colour step.
    """
    mask = (weight > 0.05).astype(np.float32)
    if float(mask.sum()) < 16:
        return warped
    sigma = max(2.0, face_width * LIGHT_MATCH_SIGMA_RATIO)
    norm = np.maximum(cv2.GaussianBlur(mask, (0, 0), sigma), 1e-4)
    swap_f = swapped.astype(np.float32) + 1.0
    warp_f = np.clip(warped, 0, 255) + 1.0
    gains = []
    for channel in range(3):
        swap_level = cv2.GaussianBlur(swap_f[:, :, channel] * mask, (0, 0), sigma) / norm
        warp_level = cv2.GaussianBlur(warp_f[:, :, channel] * mask, (0, 0), sigma) / norm
        gains.append(np.clip(swap_level / np.maximum(warp_level, 1.0), *LIGHT_GAIN_RANGE))
    return np.clip(warp_f * np.dstack(gains) - 1.0, 0, 255).astype(np.float32)


def _prepare_candidate(swapped, target, candidate, target_points, face_width, allowed):
    image, reference_points = candidate
    shape = target.shape[:2]
    warped, coverage = warp_piecewise_affine(image, reference_points, target_points, shape)
    skin = reference_skin(warped, target_points, coverage, face_width)
    # Closed lips come from the real photo too; an open mouth (in either)
    # keeps the swap's mouth.
    keep_mouth = mouth_is_open(target_points) or mouth_is_open(np.asarray(reference_points, np.float32))
    weight = transplant_region(target_points, shape, face_width, skin, keep_mouth) * coverage * allowed
    return warped, coverage, weight


MIN_REAL_FACE_FIT = 0.3   # feature agreement below which a warp looks wrong


def choose_and_transplant(
    swapped: np.ndarray,
    target: np.ndarray,
    candidates: list[tuple[np.ndarray, np.ndarray]],
    target_points: np.ndarray,
    face_width: float,
    min_fit: float | None = MIN_REAL_FACE_FIT,
    complete_face: bool = False,
    strength: float = 1.0
) -> tuple[np.ndarray, int, list[dict[str, float]]]:
    """Warp each (image, mesh) candidate onto the face, keep the one that fits best.

    Candidates arrive best pose match first; a later one must fit clearly
    better to win. If even the best one's features do not line up with the
    photo (`min_fit`), ValueError: the AI swap is better than a wrong face.
    Returns (uint8 BGR result, chosen index, per-candidate fit).
    """
    if not candidates:
        raise ValueError("gerçek fotoğraf adayı yok")
    if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
        raise ValueError("Tam yüz referans aktarımı gücü geçersiz.")
    allowed = _changed_mask(swapped, target, face_width)
    if complete_face:
        # The parser-based FaceFusion result does not always change beard
        # or cheek-hair pixels. For a full-face *candidate*, reference detail
        # may also be carried through unchanged pixels INSIDE the mesh oval.
        # The transplant_region below still rejects outside-face pixels and
        # protects the mouth and the reference's forehead hairline.
        allowed = np.maximum(allowed, 0.78)
    best = None
    fits = []
    for index, candidate in enumerate(candidates):
        warped, coverage, weight = _prepare_candidate(swapped, target, candidate, target_points, face_width, allowed)
        # Outside the warped mesh use the swap, so the mesh border does not
        # count as a feature.
        filled = warped * coverage[:, :, None] + swapped.astype(np.float32) * (1.0 - coverage[:, :, None])
        fit = candidate_fit(filled, weight, swapped, target, face_width)
        fit["score"] -= 0.03 * index
        fits.append(fit)
        if best is None or fit["score"] > best[0]:
            best = (fit["score"], index, warped, weight)
    _, index, warped, weight = best
    if min_fit is not None and fits[index]["structure"] < min_fit:
        scores = ", ".join(f"{fit['structure']:.2f}" for fit in fits)
        raise ValueError(f"hiçbir aday yüze yeterince oturmadı (uyum {scores})")
    warped = match_broad_light(warped, swapped, weight, face_width)
    amount = np.clip(weight * strength, 0.0, 1.0)
    blended = swapped.astype(np.float32) * (1.0 - amount[:, :, None]) + warped * amount[:, :, None]
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8), index, fits


def transplant_reference_face(
    swapped: np.ndarray,
    target: np.ndarray,
    reference_image: np.ndarray,
    reference_points: np.ndarray,
    target_points: np.ndarray,
    face_width: float
) -> np.ndarray:
    """The swap with its face replaced by one real photo (warped). Returns uint8 BGR."""
    result, _, _ = choose_and_transplant(
        swapped, target, [(reference_image, reference_points)], target_points, face_width, min_fit=None
    )
    return result
