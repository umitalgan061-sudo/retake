"""Real-reference under-chin and throat transfer: beard neckline and throat skin.

The 478-point face mesh ends at the jaw line, so the real lower-face transfer
(lower_face_transfer.py) stops there and the beard that continues under the
chin and onto the throat, or the person's own throat skin, stayed the
photo's. Here the jaw contour is extended along each face's own downward
direction, in the reference and in the photo, into a band of triangles; the
reference band is warped triangle by triangle onto the photo's.

Only pixels that look like skin or facial hair in BOTH images (each face's own
lower-face colour statistics: chroma close, lightness within the face's
range) are taken, so collars, clothing, jewellery and background stay
untouched. Exposure and white balance come from the scene: one gain per
channel measured on bare mid-face skin (cheeks and nose) of the result,
which sits in the photo's light, against the reference's. A local light
match would lift a dark beard to the photo's bare-skin level; this keeps
the beard's own darkness, stubble, pores and edge. The band starts slightly inside the
jaw so it overlaps the lower-face transfer's feathered edge without a seam,
and fades out towards the neck and the sides.

Any doubt (throat hidden, other pose, too little skin) raises ValueError and
the caller keeps its image. RETAKE_THROAT_TRANSFER=off disables it.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# Jaw contour from the left jaw angle over the chin (152) to the right one.
THROAT_JAW = (172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397)
MOUTH_CORNERS = (291, 61)
LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
THROAT_DEPTH = 0.42      # band depth below the chin, of the eye-to-chin distance
THROAT_LIFT = 0.035      # row 0 starts this far inside the jaw (overlap, no seam)
THROAT_ROWS = 3
NECK_NARROWING = 0.15    # the outer row is this much narrower than the jaw
MIN_SKIN_SHARE = 0.35    # of the band that must look like skin/beard in each image
MAX_SHAPE_MISMATCH = 0.25  # jaw width/length ratio difference (pose) tolerated
SCENE_GAIN_RANGE = (0.6, 1.6)
DEFAULT_STRENGTH = 0.85


def throat_enabled() -> bool:
    return os.environ.get("RETAKE_THROAT_TRANSFER", "").strip().lower() not in {"0", "off", "false", "no"}


def _smootherstep(x):
    t = np.clip(x, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    eyes = (points[33] + points[263]) / 2.0
    chin = points[152].astype(np.float64)
    down = chin - eyes
    length = float(np.hypot(*down))
    if not np.isfinite(length) or length < 8.0:
        raise ValueError("Göz-çene mesafesi boğaz bandı için çok küçük.")
    down = down / length
    lateral = np.array([-down[1], down[0]])
    if float(np.dot(points[THROAT_JAW[-1]] - points[THROAT_JAW[0]], lateral)) < 0:
        lateral = -lateral
    return chin, down, lateral, length


def throat_grid(points: np.ndarray, depth: float = THROAT_DEPTH, rows: int = THROAT_ROWS,
                lift: float = THROAT_LIFT) -> np.ndarray:
    """(rows+1, len(THROAT_JAW), 2) band points: row 0 at the jaw, last row on the neck."""
    pts = np.asarray(points, np.float64)
    if pts.shape != (478, 2) or not np.isfinite(pts).all():
        raise ValueError("Boğaz bandı için 478 yüz işaret noktası gerekir.")
    chin, down, lateral, length = _frame(pts)
    jaw = pts[list(THROAT_JAW)]
    count = len(jaw)
    half = (count - 1) / 2.0
    grid = np.zeros((rows + 1, count, 2))
    for i, point in enumerate(jaw):
        along = 1.0 - abs(i - half) / half          # 0 at the jaw angles, 1 at the chin
        relative = point - chin
        side = float(relative @ lateral)
        drop = float(relative @ down)
        for r in range(rows + 1):
            t = r / rows
            extra = length * depth * t * (0.55 + 0.45 * along) - length * lift * (1.0 - t)
            grid[r, i] = chin + lateral * side * (1.0 - NECK_NARROWING * t) + down * (drop + extra)
    return grid


def _triangles(grid: np.ndarray):
    rows, count = grid.shape[:2]
    for r in range(rows - 1):
        for i in range(count - 1):
            a, b, c, d = (r, i), (r, i + 1), (r + 1, i + 1), (r + 1, i)
            yield a, b, c
            yield a, c, d


def warp_band(source: np.ndarray, source_grid: np.ndarray, target_grid: np.ndarray,
              shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Warp the source band onto the target band; float32 image and 0..1 coverage."""
    height, width = shape
    src = source.astype(np.float32)
    source_grid = np.asarray(source_grid, np.float64).copy()
    # Shrinking with bilinear sampling aliases stubble into speckle: bring the
    # reference to the photo's scale with area averaging first.
    scale = np.linalg.norm(target_grid[-1, len(THROAT_JAW) // 2] - target_grid[0, len(THROAT_JAW) // 2]) / max(
        np.linalg.norm(source_grid[-1, len(THROAT_JAW) // 2] - source_grid[0, len(THROAT_JAW) // 2]), 1e-6)
    if scale < 0.9:
        size = (max(1, int(round(src.shape[1] * scale))), max(1, int(round(src.shape[0] * scale))))
        src = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
        source_grid *= scale
    warped = np.zeros((height, width, 3), np.float32)
    coverage = np.zeros((height, width), np.float32)
    for tri in _triangles(target_grid):
        dst = np.float32([target_grid[r, i] for r, i in tri])
        srct = np.float32([source_grid[r, i] for r, i in tri])
        u, v = dst[1] - dst[0], dst[2] - dst[0]
        if abs(float(u[0] * v[1] - u[1] * v[0])) < 0.5:
            continue
        x, y, w, h = cv2.boundingRect(dst)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        matrix = cv2.getAffineTransform(srct, dst - np.float32([x0, y0]))
        patch = cv2.warpAffine(src, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst - np.float32([x0, y0])).astype(np.int32), 1, lineType=cv2.LINE_8)
        region = mask.astype(bool)
        warped[y0:y1, x0:x1][region] = patch.reshape(y1 - y0, x1 - x0, 3)[region]
        coverage[y0:y1, x0:x1][region] = 1.0
    # Close the 1-px gaps between rasterised triangles.
    gaps = (cv2.dilate(coverage, np.ones((3, 3), np.uint8)) > 0) & (coverage == 0)
    if gaps.any():
        blurred = cv2.GaussianBlur(warped * coverage[:, :, None], (0, 0), 1.0)
        norm = np.maximum(cv2.GaussianBlur(coverage, (0, 0), 1.0), 1e-4)
        warped[gaps] = (blurred / norm[:, :, None])[gaps]
        coverage[gaps] = 1.0
    return warped, coverage


def _lower_face_region(shape: tuple[int, int], points: np.ndarray, face_width: float) -> np.ndarray:
    height, width = shape
    pts = np.asarray(points, np.float32)
    region = np.zeros((height, width), np.uint8)
    cv2.fillPoly(region, [np.round(pts[list(THROAT_JAW) + list(MOUTH_CORNERS)]).astype(np.int32)], 255)
    lips = np.zeros_like(region)
    cv2.fillPoly(lips, [np.round(pts[list(LIPS)]).astype(np.int32)], 255)
    margin = int(max(3, face_width * 0.03)) | 1
    lips = cv2.dilate(lips, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin)))
    return (region > 0) & (lips == 0)


def _mid_face_region(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    """Bare skin in every face: cheeks and nose between the eye line and the mouth."""
    pts = np.asarray(points, np.float64)
    left_eye, right_eye = pts[33], pts[263]
    left_mouth, right_mouth = pts[61], pts[291]
    centre = (left_eye + right_eye + left_mouth + right_mouth) / 4.0

    def widen(point):
        return centre + (point - centre) * np.array([1.15, 1.0])

    top_left = widen(left_eye + 0.30 * (left_mouth - left_eye))
    top_right = widen(right_eye + 0.30 * (right_mouth - right_eye))
    bottom_right = widen(right_eye + 0.85 * (right_mouth - right_eye))
    bottom_left = widen(left_eye + 0.85 * (left_mouth - left_eye))
    region = np.zeros(shape, np.uint8)
    cv2.fillPoly(region, [np.round([top_left, top_right, bottom_right, bottom_left]).astype(np.int32)], 1)
    return region > 0


def scene_gain(image: np.ndarray, points: np.ndarray, reference: np.ndarray, reference_points: np.ndarray,
               face_width: float) -> np.ndarray:
    """Per-channel gain from the reference's light to the photo's.

    Measured on bare mid-face skin, never on the beard: the result's lower
    face may still hold a lighter swap beard, which would make a beard-based
    gain lift the reference's dark beard to skin level. Highlights-side
    pixels (above the 40th lightness percentile) skip nostrils and stubble.
    """
    del face_width

    def level(picture, mask):
        pixels = picture.astype(np.float32)[mask]
        if len(pixels) < 40:
            raise ValueError("Sahne ışığı orta yüz cildinden ölçülemedi.")
        lightness = pixels.mean(axis=1)
        return np.median(pixels[lightness >= np.percentile(lightness, 40)], axis=0) + 1.0

    photo = level(image, _mid_face_region(image.shape[:2], points))
    source = level(reference, _mid_face_region(reference.shape[:2], reference_points))
    return np.clip(photo / source, *SCENE_GAIN_RANGE).astype(np.float32)


def lower_face_colour(image: np.ndarray, points: np.ndarray, face_width: float):
    """Skin+beard colour statistics of a face's own lower face (lips excluded)."""
    sample = _lower_face_region(image.shape[:2], points, face_width)
    if int(sample.sum()) < 40:
        raise ValueError("Alt yüz rengi ölçülemedi.")
    lab = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[sample]
    centre = np.median(pixels[:, 1:], axis=0)
    spread = np.maximum(np.percentile(np.abs(pixels[:, 1:] - centre), 80, axis=0) * 1.8, 6.0)
    low = float(np.percentile(pixels[:, 0], 2)) - 14.0
    high = float(np.percentile(pixels[:, 0], 98)) + 14.0
    return centre, spread, low, high


def colour_likelihood(image: np.ndarray, model) -> np.ndarray:
    """0..1: how much each pixel looks like this face's skin or facial hair."""
    centre, spread, low, high = model
    lab = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    distance = np.sqrt((((lab[:, :, 1:] - centre) / spread) ** 2).sum(axis=2) / 2.0)
    chroma = _smootherstep((2.2 - distance) / 0.9)
    light = _smootherstep((lab[:, :, 0] - low) / 8.0) * _smootherstep((high - lab[:, :, 0]) / 8.0)
    return (chroma * light).astype(np.float32)


def band_fade(grid: np.ndarray, shape: tuple[int, int], face_width: float) -> np.ndarray:
    """1 near the jaw, fading to 0 at the neck row and at both sides."""
    height, width = shape
    outline = np.concatenate([grid[0], grid[-1][::-1]])
    band = np.zeros((height, width), np.uint8)
    cv2.fillPoly(band, [np.round(outline).astype(np.int32)], 1)

    def distance_to(polyline):
        canvas = np.full((height, width), 255, np.uint8)
        cv2.polylines(canvas, [np.round(polyline).astype(np.int32)], False, 0, 1)
        return cv2.distanceTransform(canvas, cv2.DIST_L2, 5)

    to_jaw = distance_to(grid[0])
    to_neck = distance_to(grid[-1])
    t = to_jaw / np.maximum(to_jaw + to_neck, 1e-6)
    fade = _smootherstep((1.0 - t) / 0.6)
    sides = np.minimum(distance_to(grid[:, 0]), distance_to(grid[:, -1]))
    side_fade = _smootherstep(sides / max(2.0, face_width * 0.08))
    return (band * fade * side_fade).astype(np.float32)


def _shape_ratio(points: np.ndarray) -> float:
    _, _, _, length = _frame(np.asarray(points, np.float64))
    jaw = np.asarray(points, np.float64)[[THROAT_JAW[0], THROAT_JAW[-1]]]
    return float(np.hypot(*(jaw[1] - jaw[0]))) / length


def transfer_throat(
    image: np.ndarray,
    target: np.ndarray,
    candidate: tuple[np.ndarray, np.ndarray],
    target_points: np.ndarray,
    face_width: float,
    strength: float = DEFAULT_STRENGTH,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return `image` with the reference's under-chin/throat detail, and a report.

    `image` is the current result (swap + lower-face transfer), `target` the
    untouched photo crop, `candidate` the (reference BGR, 478×2 mesh) the
    lower-face transfer selected.
    """
    if not 0.0 < strength <= 1.0:
        raise ValueError("Boğaz aktarımı gücü geçersiz.")
    if image.shape != target.shape or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Boğaz aktarımı için görseller aynı BGR boyutunda olmalı.")
    reference, reference_points = candidate
    source_points = np.asarray(reference_points, np.float64)
    points = np.asarray(target_points, np.float64)
    mismatch = abs(_shape_ratio(source_points) - _shape_ratio(points)) / max(_shape_ratio(points), 1e-6)
    if mismatch > MAX_SHAPE_MISMATCH:
        raise ValueError(f"Referans ile hedefin çene pozu farklı (%{mismatch * 100:.0f}).")
    target_grid = throat_grid(points)
    source_grid = throat_grid(source_points)
    shape = target.shape[:2]
    fade = band_fade(target_grid, shape, face_width)
    if float(fade.max()) <= 0.0:
        raise ValueError("Boğaz bandı görüntü dışında.")
    warped, coverage = warp_band(reference, source_grid, target_grid, shape)
    reference_like = colour_likelihood(warped, lower_face_colour(reference, source_points, face_width))
    target_like = colour_likelihood(target, lower_face_colour(target, points, face_width))
    area = fade > 0.05
    reference_share = float((reference_like * coverage)[area].mean()) if area.any() else 0.0
    target_share = float(target_like[area].mean()) if area.any() else 0.0
    if reference_share < MIN_SKIN_SHARE:
        raise ValueError(f"Referansta boğaz yeterince görünmüyor (%{reference_share * 100:.0f}).")
    if target_share < MIN_SKIN_SHARE:
        raise ValueError(f"Fotoğrafta boğaz yeterince görünmüyor (%{target_share * 100:.0f}).")
    weight = fade * coverage * reference_like * target_like
    smooth = max(1.0, face_width * 0.01)
    weight = cv2.GaussianBlur(weight, (0, 0), smooth) * strength
    gain = scene_gain(image, points, reference, source_points, face_width)
    lit = np.clip(warped * gain[None, None, :], 0, 255)
    merged = image.astype(np.float32) * (1.0 - weight[:, :, None]) + lit * weight[:, :, None]
    report = {
        "reference_share": round(reference_share, 3),
        "target_share": round(target_share, 3),
        "pixels": int(np.count_nonzero(weight > 0.1)),
        "gain": [round(float(value), 3) for value in gain],
    }
    return np.clip(np.rint(merged), 0, 255).astype(np.uint8), report
