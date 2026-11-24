"""Keep model padding separate from quality checks and transport final RGB losslessly.

A preblended PNG carries an edit extent, NOT a feather-opacity mask.
Canvas premultiplies transparent RGB when decoding. Alpha 1 cannot represent
arbitrary RGB, so copying unpremultiplied RGB at alpha 1 creates bright,
black and colored contour artifacts. Use 0/255 only on the wire.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def prepare_inference_target(
    target_path: Path, expected_size: dict[str, int], temp_dir: Path
) -> Path:
    """Keep the original crop untouched, pad a separate lossless model input."""
    image = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Hedef görüntü okunamadı.")
    height, width = image.shape[:2]
    if (width, height) != (expected_size["width"], expected_size["height"]):
        raise ValueError("Hedef görüntü boyutu metadata ile eşleşmiyor.")
    if width % 2 == 0 and height % 2 == 0:
        return target_path
    padded = cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
    path = temp_dir / "target-inference-even.png"
    if not cv2.imwrite(str(path), padded):
        raise OSError("FaceFusion için ayrı çift boyutlu hedef yazılamadı.")
    return path


def binary_preblended_rgba(target: np.ndarray, composite: np.ndarray) -> np.ndarray:
    """Retain the already feathered RGB exactly wherever the matte is nonzero.

    Ignore any RGB outside the authorized edit extent, even if a processing
    stage changed it. The browser copies original pixels at alpha zero.
    """
    if (
        target is None or composite is None
        or target.ndim != 3 or target.shape[2] != 3
        or composite.ndim != 3 or composite.shape[2] != 4
        or target.shape[:2] != composite.shape[:2]
        or target.dtype != np.uint8 or composite.dtype != np.uint8
    ):
        raise ValueError("Hedef ve kompozisyon aynı boyutta BGR/BGRA uint8 olmalı.")
    color = composite[:, :, :3]
    changed = np.any(color != target, axis=2)
    extent = (composite[:, :, 3] > 0) & changed
    result = np.empty_like(composite)
    result[:, :, :3] = target
    result[extent, :3] = color[extent]
    result[:, :, 3] = np.where(extent, 255, 0).astype(np.uint8)
    return result


def encode_preblended_result(target_path: Path, composite_path: Path) -> bytes:
    """Encode only binary alpha at the transport boundary, not during blending."""
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    composite = cv2.imread(str(composite_path), cv2.IMREAD_UNCHANGED)
    result = binary_preblended_rgba(target, composite)
    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise OSError("Kayıpsız kompozisyon PNG kodlanamadı.")
    return encoded.tobytes()
