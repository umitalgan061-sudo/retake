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




def transplant_region(points: np.ndarray, shape: tuple[int, int], face_width: float) -> np.ndarray:
    """0..1 region for the transplant: the face from just above the brows
    down to the chin (the forehead above is often covered by hair), minus
    the open mouth."""
    height, width = shape
    oval = np.zeros((height, width), np.uint8)
    cv2.fillPoly(oval, [np.round(points[FACE_OVAL]).astype(np.int32)], 255)
    eye_distance = float(np.hypot(*(points[263] - points[33])))
    brow_top = min(float(points[LEFT_BROW + RIGHT_BROW, 1].min()), float(points[[33, 263], 1].min())) - 0.28 * eye_distance
    yy = np.arange(height, dtype=np.float32)[:, None]
    fade = max(2.0, eye_distance * 0.12)
    below_brows = _smootherstep((yy - brow_top) / fade)
    mouth = np.zeros_like(oval)
    cv2.fillPoly(mouth, [np.round(points[INNER_MOUTH]).astype(np.int32)], 255)
    grow = int(max(3, face_width * 0.01)) | 1
    mouth = cv2.dilate(mouth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    region = (oval > 0) & (mouth == 0)
    # A soft edge: the real beard and skin fade into the swap instead of
    # ending on the jaw contour.
    feather = max(1.5, face_width * 0.035)
    inward = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    return (_smootherstep(inward / feather) * below_brows).astype(np.float32)


def transplant_reference_face(
    swapped: np.ndarray,
    target: np.ndarray,
    reference_image: np.ndarray,
    reference_points: np.ndarray,
    target_points: np.ndarray,
    face_width: float
) -> np.ndarray:
    """The swap with its face replaced by the person's real photo (warped).

    Only where the swap model itself changed the photo: its parser/occlusion
    mask keeps hair, hands and glasses of the original photo. Returns uint8 BGR.
    """
    shape = target.shape[:2]
    warped, coverage = warp_piecewise_affine(reference_image, reference_points, target_points, shape)
    region = transplant_region(target_points, shape, face_width) * coverage
    changed = np.max(cv2.absdiff(swapped, target), axis=2) >= 2
    grow = int(max(3, face_width * 0.02)) | 1
    changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    allowed = cv2.GaussianBlur(changed.astype(np.float32), (0, 0), max(1.0, face_width * 0.01))
    weight = (region * allowed)[:, :, None]
    blended = swapped.astype(np.float32) * (1.0 - weight) + warped * weight
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)
