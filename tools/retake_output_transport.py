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


def prepare_small_face_inference(model_path: Path, bounds: dict, variation: int, temp_dir: Path):
    """A bounded, native-size inference-only alternative for distant faces.

    Do not invent pores or facial hair. Recover mild local luminance contrast
    on a few existing variants, without geometry, upscaling, or changes to
    the original used for blending and verification. Other variants retain
    exactly the unmodified input.
    """
    if variation not in (3, 7, 11, 15, 19, 23):
        return model_path, "original"
    try:
        x,y,w,h = (float(bounds[k]) for k in ("x","y","width","height"))
    except (TypeError,ValueError,KeyError):
        return model_path,"invalid-bounds"
    if not np.isfinite([x,y,w,h]).all() or not 32<=min(w,h)<160:
        return model_path,"original"
    image=cv2.imread(str(model_path),cv2.IMREAD_COLOR)
    if image is None: return model_path,"unreadable"
    height,width=image.shape[:2]
    if x<0 or y<0 or x+w>width or y+h>height:
        return model_path,"invalid-bounds"
    left,top=int(np.floor(x)),int(np.floor(y))
    right,bottom=int(np.ceil(x+w)),int(np.ceil(y+h))
    crop=image[top:bottom,left:right]
    lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)
    luma=lab[:,:,0].astype(np.float32)
    # Skip flat, clipped, or already highly sharpened inputs.
    band=luma-cv2.GaussianBlur(luma,(0,0),1.0)
    energy=float(np.sqrt(np.mean(band**2)))
    if energy<1.0 or energy>12.0 or float(np.mean((luma<18)|(luma>242)))>.20:
        return model_path,"insufficient-evidence"
    yy,xx=np.mgrid[:crop.shape[0],:crop.shape[1]]
    distance=((xx-(crop.shape[1]-1)/2)/(crop.shape[1]*.48))**2+((yy-(crop.shape[0]-1)/2)/(crop.shape[0]*.48))**2
    weight=np.clip((1-distance)/.55,0,1).astype(np.float32)
    weight=weight*weight*(3-2*weight)
    gain=.30 if variation in (3,11,19) else .48
    # Dampen corrections near sharp edges, preserving hair and silhouette.
    change=np.clip(band*gain,-4,4)*weight*np.clip((28-np.abs(band))/16,0,1)
    new_luma=np.clip(np.rint(luma+change),0,255).astype(np.uint8)
    altered=new_luma!=lab[:,:,0]
    if np.count_nonzero(altered)<20: return model_path,"insufficient-evidence"
    lab[:,:,0]=new_luma
    enhanced=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    output=image.copy()
    target=output[top:bottom,left:right]
    target[altered]=enhanced[altered]
    path=temp_dir/"target-inference-detail.png"
    if not cv2.imwrite(str(path),output):
        return model_path,"write-failed"
    return path,"native-contrast"


def prepare_visible_identity_probe(target_path: Path, candidate_path: Path, temp_dir: Path) -> Path:
    """ArcFace must see the delivered color, not RGB hidden by alpha.

    Opaque model renders already carry their own visible RGB. For preblended
    BGRA candidates, restore untouched target pixels and score an opaque,
    content-addressed PNG local to this job. The name changes with pixels so
    FaceFusion's static image cache cannot return a stale overwritten image.
    """
    import hashlib
    candidate=cv2.imread(str(candidate_path),cv2.IMREAD_UNCHANGED)
    if candidate is None:
        raise ValueError("Kimlik adayı okunamadı.")
    if candidate.ndim==3 and candidate.shape[2]==3:
        return candidate_path
    target=cv2.imread(str(target_path),cv2.IMREAD_COLOR)
    visible=binary_preblended_rgba(target,candidate)[:,:,:3]
    ok,encoded=cv2.imencode(".png",visible)
    if not ok: raise OSError("Kimlik doğrulama görüntüsü yazılamadı.")
    payload=encoded.tobytes()
    path=temp_dir/("visible-identity-"+hashlib.sha256(payload).hexdigest()+".png")
    if not path.is_file():
        path.write_bytes(payload)
    return path
