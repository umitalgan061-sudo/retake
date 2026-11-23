from __future__ import annotations

import argparse
import hmac
import json
import math
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from email.parser import BytesParser
from email import policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from .facefusion_quality import PIXEL_BOOST_FALLBACKS, is_near_noop
    from .retake_cache import IdentityReportCache, PassCache, StageTimings, references_digest
    from .identity_selection import plan_identity_source_sets, select_texture_reference
    from .photo_realism_gate import check_render_file
    from .ultra_quality import check_photo_detail, choose_detail_safe_tie
    from .retake_rim_cleanup import repair_dark_rim_file
    from .ultra_mode import (
        ULTRA_VARIATIONS, ULTRA_MIN_SECONDS, ULTRA_SEARCH_BUDGET_SECONDS,
        prefer_ultra_candidate, score_from_headers, should_continue_ultra, with_variation
    )
except ImportError:
    from facefusion_quality import PIXEL_BOOST_FALLBACKS, is_near_noop
    from retake_cache import IdentityReportCache, PassCache, StageTimings, references_digest
    from identity_selection import plan_identity_source_sets, select_texture_reference
    from photo_realism_gate import check_render_file
    from ultra_quality import check_photo_detail, choose_detail_safe_tie
    from retake_rim_cleanup import repair_dark_rim_file
    from ultra_mode import (
        ULTRA_VARIATIONS, ULTRA_MIN_SECONDS, ULTRA_SEARCH_BUDGET_SECONDS,
        prefer_ultra_candidate, score_from_headers, should_continue_ultra, with_variation
    )

MAX_REFERENCES = 25
MIN_REFERENCES = MAX_REFERENCES
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_REQUEST_BYTES = 180 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
INFERENCE_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict[str, Any]] = {}
MAX_KEPT_JOBS = 3
JOB_STAGES = (
    "25 referans ArcFace ile karÅŸÄ±laÅŸtÄ±rÄ±lÄ±yor",
    "Kimlik geÃ§iÅŸi (AlphaFace)",
    "Kimlik pekiÅŸtirme (InSwapper)",
    "GÃ¶rÃ¼nÃ¼m ve sakal geÃ§iÅŸi (UniFace)",
    "DikiÅŸsiz birleÅŸtirme ve kafa oranÄ±",
    "FotoÄŸraf gerÃ§ekÃ§iliÄŸi (difÃ¼zyon)",
    "Kimlik benzerliÄŸi Ã¶lÃ§Ã¼lÃ¼yor",
)
REINFORCE_STAGE = 2
REALISM_STAGE = 5
SCORING_STAGE = 6
# A realism render may cost at most this much ArcFace similarity to be used.
REALISM_MAX_IDENTITY_LOSS = 0.004
REALISM_TIMEOUT_SECONDS = 900
# (tone harmonization, shading cap) per compositing variant; the first is the
# most seamless. Measured on the user's photos (ArcFace to the references):
# 0.9 -> 0.743, 0.6 -> 0.765, 0.35 -> 0.782, borders invisible in all three.
COMPOSITE_VARIANTS = ((0.9, None), (0.6, 0.25), (0.35, 0.15))
DEFAULT_SHADING_STRENGTH = 0.35
# A variant may show at most this much more border than the most seamless
# one (per 100 px of face width), and always this much.
BORDER_STEP_RATIO = 2.0
BORDER_STEP_FLOOR = 0.6
# Bumped when the cached ArcFace report gains fields (centroid for scoring).
IDENTITY_REPORT_VERSION = "centroid-v1"
PAIRING_HEADER = "X-Retake-Token"
IDENTITY_MODEL = "alphaface_256"
# A second identity pass over the first one's result. On the user's photos
# ArcFace similarity to the references went 0.803 (AlphaFace) -> 0.832
# (AlphaFace + InSwapper); AlphaFace twice gave 0.816. It is kept only when
# the measurement confirms it for the photo at hand.
REINFORCE_MODEL = "inswapper_128"
APPEARANCE_MODEL = "uniface_256"
HYBRID_PROFILE_VERSION = "retake_hybrid_25_reference_ultra_v21"
IDENTITY_VERIFIER_MODEL = "arcface_w600k_r50"
FACE_ENHANCER_MODEL = "gfpgan_1.4"
# FaceFusion renders a full-strength enhancer output; Retake applies the
# conservative 25% blend once after inference, preserving the existing alpha.
FACE_ENHANCER_BLEND = 25
FACE_ENHANCER_INTERNAL_BLEND = 100
FACE_ENHANCER_WEIGHT = 0.25
DETAIL_TRANSFER_STRENGTH = 0.20
DEFAULT_TARGET_WIDTH = 784
DEFAULT_TARGET_HEIGHT = 1024
MIN_TARGET_EDGE = 256
MAX_TARGET_WIDTH = 1600
MAX_TARGET_HEIGHT = 2048
HEAD_FIT_MIN = 0.85
HEAD_FIT_MAX = 1.18
# (swapper weight, lower-face detail strength, head-fit strength).
# Head-fit strength scales the head reshape measured by the browser from the
# 25 references: 1 = measured fit, 0 = keep the photo's own head proportions.
# The first three keep the swapper weight, so "regenerate" reuses both cached
# FaceFusion passes and only redoes blending and head fitting (seconds, not
# minutes). The fourth is a genuinely different, softer identity pass.
FACE_SWAP_VARIATIONS = (
    # v20: CLI supports exact 0.05 steps, not 0.94/0.96/0.99.
    # Cover 1.00 / 0.95 / 0.90 early with varied texture/shape settings.
    (1.00, 0.20, 1.00),
    (1.00, 0.24, 0.00),
    (1.00, 0.12, 1.50),
    (0.95, 0.30, 0.50),
    (1.00, 0.08, 0.00),
    (1.00, 0.32, 0.00),
    (0.90, 0.14, 0.65),
    (1.00, 0.28, 1.20),
    (0.95, 0.10, 0.00),
    (0.95, 0.28, 0.85),
    (0.95, 0.36, 0.00),
    (0.95, 0.10, 1.10),
    (0.95, 0.18, 0.00),
    (0.95, 0.30, 0.65),
    (0.95, 0.08, 0.00),
    (0.95, 0.24, 1.20),
    (0.90, 0.14, 0.70),
    (0.90, 0.32, 0.00),
    (0.95, 0.14, 0.00),
    (0.95, 0.24, 0.90),
    (0.90, 0.08, 0.00),
    (0.90, 0.24, 1.00),
    (0.90, 0.18, 0.00),
    (0.90, 0.30, 0.70),
)


class IdentityVerificationError(RuntimeError):
    pass


class IdentityConflictError(IdentityVerificationError):
    pass


class NoOpSwapError(RuntimeError):
    pass


FACE_OCCLUDER_MODEL = "xseg_1"
FACE_PARSER_MODEL = "bisenet_resnet_34"
FACE_MASK_TYPES = ("box", "occlusion", "region")
# Deliberately exclude "glasses" so target eyewear is preserved by the parser mask.
FACE_MASK_REGIONS = (
    "skin",
    "left-eyebrow",
    "right-eyebrow",
    "left-eye",
    "right-eye",
    "nose",
    "mouth",
    "upper-lip",
    "lower-lip"
)


def face_mask_cli_arguments() -> list[str]:
    """Return FaceFusion 3.9.2 parser/occlusion mask arguments."""
    return [
        "--face-occluder-model", FACE_OCCLUDER_MODEL,
        "--face-parser-model", FACE_PARSER_MODEL,
        "--face-mask-types", *FACE_MASK_TYPES,
        "--face-mask-regions", *FACE_MASK_REGIONS,
        "--face-mask-blur", "0.30"
    ]


def image_extension(content_type: str, payload: bytes) -> str:
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized == "image/png" and payload.startswith(PNG_SIGNATURE):
        return ".png"
    if normalized == "image/jpeg" and payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if normalized == "image/webp" and len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("YalnÄ±zca geÃ§erli PNG, JPEG veya WebP gÃ¶rselleri kabul edilir.")


MAX_JAW_CONTROLS = 32
MAX_JAW_ANCHORS = 16


def _point(value: Any, size: dict[str, int], margin: float) -> list[float]:
    if (
        not isinstance(value, list) or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value)
        or not -margin <= value[0] <= size["width"] + margin
        or not -margin <= value[1] <= size["height"] + margin
    ):
        raise ValueError("Ã‡ene hattÄ± noktalarÄ± hedef kÄ±rpÄ±mÄ±n iÃ§inde sayÄ± Ã§iftleri olmalÄ±.")
    return [float(value[0]), float(value[1])]


def _validate_jaw_warp(value: Any, size: dict[str, int], bounds: dict[str, float]) -> dict[str, Any] | None:
    """Optional jawline controls from the browser (target-crop pixel space)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Ã‡ene hattÄ± verisi nesne olmalÄ±.")
    controls, anchors = value.get("controls"), value.get("anchors", [])
    if not isinstance(controls, list) or not 1 <= len(controls) <= MAX_JAW_CONTROLS:
        raise ValueError(f"Ã‡ene hattÄ± 1 ile {MAX_JAW_CONTROLS} arasÄ±nda kontrol noktasÄ± iÃ§ermeli.")
    if not isinstance(anchors, list) or len(anchors) > MAX_JAW_ANCHORS:
        raise ValueError(f"Ã‡ene hattÄ± en fazla {MAX_JAW_ANCHORS} sabit nokta iÃ§erebilir.")
    margin = 0.25 * max(size["width"], size["height"])
    limit = bounds["width"] * 0.2
    checked = []
    for control in controls:
        if not isinstance(control, dict):
            raise ValueError("Ã‡ene hattÄ± kontrol noktasÄ± geÃ§ersiz.")
        start, end = _point(control.get("from"), size, margin), _point(control.get("to"), size, margin)
        if math.hypot(end[0] - start[0], end[1] - start[1]) > limit:
            raise ValueError("Ã‡ene hattÄ± kaydÄ±rmasÄ± izin verilen sÄ±nÄ±rÄ± aÅŸÄ±yor.")
        checked.append({"from": start, "to": end})
    return {"controls": checked, "anchors": [_point(anchor, size, margin) for anchor in anchors]}


MAX_REAL_FACE_CANDIDATES = 3


def _validate_real_face(value: Any, size: dict[str, int]) -> dict[str, Any] | None:
    """Optional real-photo face: the target crop's 478-point mesh and, for each
    candidate reference photo (best pose match first), its mesh."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("GerÃ§ek fotoÄŸraf verisi nesne olmalÄ±.")
    try:
        from .retake_detail import validate_mesh
    except ImportError:
        from retake_detail import validate_mesh
    margin = 0.25 * max(size["width"], size["height"])
    try:
        target_points = validate_mesh(value.get("targetLandmarks"), size["width"], size["height"], margin)
        references = value.get("references")
        if references is None and "referenceLandmarks" in value:  # single-candidate form
            references = [{"landmarks": value.get("referenceLandmarks")}]
        if not isinstance(references, list) or not 1 <= len(references) <= MAX_REAL_FACE_CANDIDATES:
            raise ValueError(f"1â€“{MAX_REAL_FACE_CANDIDATES} aday olmalÄ±.")
        meshes = [
            validate_mesh(item.get("landmarks") if isinstance(item, dict) else None, 4096, 4096, 0).tolist()
            for item in references
        ]
    except (ValueError, TypeError) as error:
        raise ValueError(f"GerÃ§ek fotoÄŸraf yÃ¼z aÄŸÄ± geÃ§ersiz: {error}") from error
    mode = value.get("mode", "full")
    if mode not in ("full", "lower", "both"):
        raise ValueError("GerÃ§ek fotoÄŸraf aktarÄ±m modu full, lower veya both olmalÄ±.")
    return {"mode": mode, "targetLandmarks": target_points.tolist(), "references": [{"landmarks": mesh} for mesh in meshes]}


def _validate_target_landmarks(value: Any, size: dict[str, int]) -> list[list[float]] | None:
    """Optional 478-point mesh of the face in the target crop (crop pixels)."""
    if value is None:
        return None
    try:
        from .retake_detail import validate_mesh
    except ImportError:
        from retake_detail import validate_mesh
    try:
        points = validate_mesh(value, size["width"], size["height"], 0.25 * max(size["width"], size["height"]))
    except (ValueError, TypeError) as error:
        raise ValueError(f"Hedef yÃ¼z aÄŸÄ± geÃ§ersiz: {error}") from error
    return points.tolist()


def _validate_metadata(metadata: Any, reference_count: int) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Profil metadata alanÄ± geÃ§erli JSON nesnesi olmalÄ±.")
    weights = metadata.get("referenceWeights")
    if not isinstance(weights, list) or len(weights) != reference_count:
        raise ValueError("Referans kalite aÄŸÄ±rlÄ±klarÄ±, gÃ¶nderilen referans sayÄ±sÄ±yla eÅŸleÅŸmiyor.")
    normalized_weights: list[int] = []
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or int(weight) != weight or not 1 <= int(weight) <= 3:
            raise ValueError("Her referans aÄŸÄ±rlÄ±ÄŸÄ± 1 ile 3 arasÄ±nda tam sayÄ± olmalÄ±.")
        normalized_weights.append(int(weight))

    face_bounds = metadata.get("faceBounds")
    if not isinstance(face_bounds, dict):
        raise ValueError("Hedef yÃ¼z sÄ±nÄ±rlarÄ± eksik.")
    required = ("x", "y", "width", "height")
    if any(isinstance(face_bounds.get(key), bool) or not isinstance(face_bounds.get(key), (int, float)) for key in required):
        raise ValueError("Hedef yÃ¼z sÄ±nÄ±rlarÄ± sayÄ±sal olmalÄ±.")
    bounds = {key: float(face_bounds[key]) for key in required}
    if bounds["width"] <= 0 or bounds["height"] <= 0:
        raise ValueError("Hedef yÃ¼z sÄ±nÄ±rlarÄ± pozitif geniÅŸlik ve yÃ¼ksekliÄŸe sahip olmalÄ±.")
    target_size = metadata.get("targetSize", {"width": DEFAULT_TARGET_WIDTH, "height": DEFAULT_TARGET_HEIGHT})
    if (
        not isinstance(target_size, dict)
        or any(
            isinstance(target_size.get(key), bool) or not isinstance(target_size.get(key), int)
            for key in ("width", "height")
        )
        or not MIN_TARGET_EDGE <= target_size["width"] <= MAX_TARGET_WIDTH
        or not MIN_TARGET_EDGE <= target_size["height"] <= MAX_TARGET_HEIGHT
    ):
        raise ValueError(
            f"Hedef kÄ±rpÄ±m boyutu {MIN_TARGET_EDGE} px ile "
            f"{MAX_TARGET_WIDTH}Ã—{MAX_TARGET_HEIGHT} px arasÄ±nda tam sayÄ± olmalÄ±."
        )
    if (
        bounds["x"] < 0 or bounds["y"] < 0
        or bounds["x"] + bounds["width"] > target_size["width"]
        or bounds["y"] + bounds["height"] > target_size["height"]
    ):
        raise ValueError("Hedef yÃ¼z sÄ±nÄ±rlarÄ±, gÃ¶nderilen portre kÄ±rpÄ±mÄ±nÄ±n dÄ±ÅŸÄ±nda.")
    head_fit = metadata.get("headFit", {"scaleX": 1.0, "scaleY": 1.0})
    if (
        not isinstance(head_fit, dict)
        or any(
            isinstance(head_fit.get(key), bool)
            or not isinstance(head_fit.get(key), (int, float))
            or not math.isfinite(head_fit[key])
            or not HEAD_FIT_MIN <= head_fit[key] <= HEAD_FIT_MAX
            for key in ("scaleX", "scaleY")
        )
    ):
        raise ValueError(f"Kafa oranÄ± Ã¶lÃ§ekleri {HEAD_FIT_MIN} ile {HEAD_FIT_MAX} arasÄ±nda sayÄ± olmalÄ±.")
    variation = metadata.get("variation", 0)
    if isinstance(variation, bool) or not isinstance(variation, int) or not 0 <= variation < len(FACE_SWAP_VARIATIONS):
        raise ValueError(f"GÃ¶rÃ¼nÃ¼m alternatifi 0 ile {len(FACE_SWAP_VARIATIONS) - 1} arasÄ±nda tam sayÄ± olmalÄ±.")
    jaw_warp = _validate_jaw_warp(metadata.get("jawWarp"), target_size, bounds)
    real_face = _validate_real_face(metadata.get("realFace"), target_size)
    target_landmarks = _validate_target_landmarks(metadata.get("targetLandmarks"), target_size)
    pose_distances = metadata.get("poseDistances")
    if pose_distances is not None:
        if (
            not isinstance(pose_distances, list) or len(pose_distances) != reference_count
            or any(
                value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 10
                )
                for value in pose_distances
            )
        ):
            raise ValueError("Referans poz mesafeleri 25 adet geÃ§erli sayÄ± veya null iÃ§ermeli.")
    return {
        "poseDistances": pose_distances,
        "targetLandmarks": target_landmarks,
        "jawWarp": jaw_warp,
        "realFace": real_face,
        "referenceWeights": normalized_weights,
        "faceBounds": bounds,
        "variation": variation,
        "targetSize": {"width": target_size["width"], "height": target_size["height"]},
        "headFit": {"scaleX": float(head_fit["scaleX"]), "scaleY": float(head_fit["scaleY"])}
    }


def validate_reference_count(reference_count: int) -> None:
    if reference_count != MAX_REFERENCES:
        raise ValueError(f"Retake hibrit profil iÅŸlemi iÃ§in tam olarak {MAX_REFERENCES} referans fotoÄŸrafÄ± gerekli.")


def score_identity(
    facefusion_dir: Path,
    centroid: list[float] | None,
    probe_paths: list[Path]
) -> list[float | None]:
    """ArcFace similarity of each probe image to the reference identity.

    Scoring is informative only: any failure returns Nones instead of failing
    the swap the user is waiting for.
    """
    empty: list[float | None] = [None] * len(probe_paths)
    worker_path = Path(__file__).with_name("facefusion_identity_worker.py")
    if not centroid or not worker_path.is_file():
        return empty
    centroid_file = probe_paths[0].with_name("retake-identity-centroid.json")
    centroid_file.write_text(json.dumps(centroid), encoding="utf-8")
    command = [
        sys.executable, str(worker_path),
        "--facefusion-dir", str(facefusion_dir),
        "--centroid-file", str(centroid_file),
        "--probe-paths", *[str(path) for path in probe_paths]
    ]
    try:
        completed = subprocess.run(
            command, cwd=str(facefusion_dir), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[ArcFace] Benzerlik Ã¶lÃ§Ã¼lemedi: {error}", flush=True)
        return empty
    marker = "RETAKE_IDENTITY_REPORT="
    lines = [line[len(marker):] for line in completed.stdout.splitlines() if line.startswith(marker)]
    try:
        report = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        report = {}
    values = report.get("similarities") if report.get("ok") else None
    if not isinstance(values, list) or len(values) != len(probe_paths):
        print("[ArcFace] Benzerlik Ã¶lÃ§Ã¼lemedi: " + tail(completed.stderr + "\n" + completed.stdout, 600), flush=True)
        return empty
    return [float(value) if isinstance(value, (int, float)) else None for value in values]


def verify_reference_identity(facefusion_dir: Path, reference_paths: list[Path]) -> dict[str, object]:
    worker_path = Path(__file__).with_name("facefusion_identity_worker.py")
    if not worker_path.is_file():
        raise IdentityVerificationError("ArcFace doÄŸrulama worker dosyasÄ± bulunamadÄ±.")
    command = [
        sys.executable, str(worker_path),
        "--facefusion-dir", str(facefusion_dir),
        "--reference-paths", *[str(path) for path in reference_paths]
    ]
    print(
        f"ArcFace Ã¶n kontrolÃ¼ baÅŸlÄ±yor: {len(reference_paths)} referans; "
        "yÃ¼z algÄ±lama ve ArcFace ONNX aÄŸÄ±rlÄ±klarÄ± kontrol edilecek.",
        flush=True
    )
    try:
        completed = subprocess.run(
            command, cwd=str(facefusion_dir), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=1800, check=False
        )
    except subprocess.TimeoutExpired as error:
        def decode_output(value: object) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value or "")

        detail = tail(
            decode_output(error.stdout) + "\n" + decode_output(error.stderr)
        )
        raise IdentityVerificationError(
            "ArcFace 25 referans Ã¶n kontrolÃ¼ 30 dakika iÃ§inde tamamlanamadÄ±. " + detail
        ) from error

    marker = "RETAKE_IDENTITY_REPORT="
    report_lines = [
        line[len(marker):]
        for line in completed.stdout.splitlines()
        if line.startswith(marker)
    ]
    if not report_lines:
        detail = tail(completed.stderr + "\n" + completed.stdout)
        raise IdentityVerificationError(
            f"ArcFace worker rapor Ã¼retemedi (Ã§Ä±kÄ±ÅŸ kodu {completed.returncode}). {detail}"
        )
    try:
        report = json.loads(report_lines[-1])
    except json.JSONDecodeError as error:
        raise IdentityVerificationError("ArcFace doÄŸrulama raporu geÃ§erli JSON deÄŸil.") from error

    if completed.returncode != 0 or not report.get("ok"):
        detail = str(report.get("error") or "ArcFace worker baÅŸarÄ±sÄ±z sonuÃ§ dÃ¶ndÃ¼rdÃ¼.")
        logs = tail(completed.stderr + "\n" + completed.stdout)
        raise IdentityVerificationError(
            f"{detail} (worker Ã§Ä±kÄ±ÅŸ kodu {completed.returncode}). {logs}"
        )
    if report.get("hard_conflict"):
        raise IdentityConflictError(
            "ArcFace, 25 referans arasÄ±nda ortak bir kimlik profili bulamadÄ± "
            f"(medyan cosine benzerliÄŸi: {report.get('median_similarity')}). "
            "FarklÄ± kiÅŸiye ait veya aÄŸÄ±r bozulmuÅŸ fotoÄŸraflarÄ± kontrol edip tekrar deneyin."
        )
    return report


def parse_multipart(
    content_type: str,
    body: bytes
) -> tuple[bytes, list[bytes], str, list[str], bytes, str, dict[str, Any]]:
    headers = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8")
    message = BytesParser(policy=policy.default).parsebytes(headers + body)
    if not message.is_multipart():
        raise ValueError("Ä°stek multipart/form-data biÃ§iminde olmalÄ±.")

    target_files: list[tuple[str, bytes]] = []
    reference_files: list[tuple[str, bytes]] = []
    appearance_files: list[tuple[str, bytes]] = []
    real_reference_files: list[tuple[str, bytes]] = []
    metadata_value: Any = None

    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        field_name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True) or b""
        if field_name == "metadata":
            try:
                metadata_value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("Profil metadata JSON olarak okunamadÄ±.") from error
            continue
        if field_name not in {"target", "reference", "appearance_profile", "real_reference"}:
            continue

        extension = image_extension(part.get_content_type(), payload)
        if not payload or len(payload) > MAX_FILE_BYTES:
            raise ValueError("Her gÃ¶rsel boÅŸ olmamalÄ± ve 25 MB sÄ±nÄ±rÄ±nÄ± aÅŸmamalÄ±.")
        item = (extension, payload)
        if field_name == "real_reference":
            real_reference_files.append(item)
        elif field_name == "target":
            target_files.append(item)
        elif field_name == "reference":
            reference_files.append(item)
        else:
            appearance_files.append(item)

    if len(target_files) != 1:
        raise ValueError("Tam olarak bir hedef fotoÄŸraf gÃ¶nderilmelidir.")
    if len(appearance_files) != 1:
        raise ValueError("25 referanstan Ã¼retilen tek bir gÃ¶rÃ¼nÃ¼m profili gÃ¶nderilmelidir.")
    validate_reference_count(len(reference_files))

    metadata = _validate_metadata(metadata_value, len(reference_files))
    if len(real_reference_files) > MAX_REAL_FACE_CANDIDATES:
        raise ValueError(f"En fazla {MAX_REAL_FACE_CANDIDATES} gerÃ§ek fotoÄŸraf adayÄ± gÃ¶nderilebilir.")
    real_face = metadata.get("realFace")
    if real_face and len(real_reference_files) < len(real_face["references"]):
        raise ValueError("GerÃ§ek fotoÄŸraf yÃ¼z aÄŸÄ± gÃ¶nderildi ama fotoÄŸrafÄ± eksik.")
    if real_face and len(real_reference_files) != len(real_face["references"]):
        raise ValueError("GerÃ§ek fotoÄŸraf sayÄ±sÄ± ile yÃ¼z aÄŸÄ± sayÄ±sÄ± eÅŸleÅŸmiyor.")
    # Kept with the metadata so the parsed tuple keeps its shape.
    metadata["realReferenceBytes"] = [payload for _, payload in real_reference_files] if real_face else None
    return (
        target_files[0][1],
        [payload for _, payload in reference_files],
        target_files[0][0],
        [ext for ext, _ in reference_files],
        appearance_files[0][1],
        appearance_files[0][0],
        metadata
    )


def tail(text: str, limit: int = 4500) -> str:
    value = (text or "").strip()
    return value[-limit:] if value else "FaceFusion ayrÄ±ntÄ±lÄ± hata kaydÄ± Ã¼retmedi."


def supported_facefusion_weight(value: float) -> float:
    """Normalize to FaceFusion's exact 0.05 CLI steps and share cache key."""
    try:
        amount = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("FaceFusion aÄŸÄ±rlÄ±ÄŸÄ± sayÄ± olmalÄ±.") from error
    if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise ValueError("FaceFusion aÄŸÄ±rlÄ±ÄŸÄ± 0 ile 1 arasÄ±nda olmalÄ±.")
    steps = min(20, max(0, int(math.floor(amount * 20.0 + 0.5))))
    return steps / 20.0


def run_facefusion(
    facefusion_dir: Path,
    target_path: Path,
    source_paths: list[Path],
    output_path: Path,
    model: str,
    pixel_boost: str,
    face_swapper_weight: float = 1.0
) -> tuple[bool, str]:
    face_swapper_weight = supported_facefusion_weight(face_swapper_weight)
    command = [
        sys.executable,
        str(facefusion_dir / "facefusion.py"),
        "headless-run",
        "--source-paths", *[str(path) for path in source_paths],
        "--target-path", str(target_path),
        "--output-path", str(output_path),
        "--processors", "face_swapper",
        "--face-swapper-model", model,
        "--face-swapper-pixel-boost", pixel_boost,
        "--face-swapper-weight", f"{face_swapper_weight:.2f}",
        "--face-selector-mode", "one",
        *face_mask_cli_arguments(),
        "--execution-providers", "cuda",
        "--execution-device-ids", "0",
        "--output-image-quality", "100",
        "--log-level", "info"
    ]
    completed = subprocess.run(
        command,
        cwd=str(facefusion_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False
    )
    if completed.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
        return True, completed.stdout[-1500:]
    return False, tail(completed.stderr + "\n" + completed.stdout)


def run_with_memory_fallback(
    facefusion_dir: Path,
    target_path: Path,
    source_paths: list[Path],
    output_path: Path,
    model: str,
    preferred_pixel_boost: str,
    face_swapper_weight: float = 1.0
) -> str:
    attempts = (preferred_pixel_boost, *PIXEL_BOOST_FALLBACKS.get(preferred_pixel_boost, ()))
    last_detail = ""
    for index, pixel_boost in enumerate(attempts):
        try:
            ok, detail = run_facefusion(
                facefusion_dir, target_path, source_paths, output_path, model, pixel_boost,
                face_swapper_weight
            )
        except subprocess.TimeoutExpired as error:
            detail = tail(str(error))
            ok = False
        if ok:
            return pixel_boost

        last_detail = detail
        lower_detail = detail.lower()
        is_memory_error = any(marker in lower_detail for marker in (
            "out of memory", "cuda error: out of memory", "failed to allocate",
            "cudnn_status_alloc_failed", "cublas_status_alloc_failed"
        ))
        has_fallback = index + 1 < len(attempts)
        if not is_memory_error or not has_fallback:
            raise RuntimeError(f"{model} iÅŸlemi baÅŸarÄ±sÄ±z ({pixel_boost}). {detail}")
        next_boost = attempts[index + 1]
        print(f"{model}: {pixel_boost} bellek hatasÄ±; {next_boost} ayarÄ±na dÃ¼ÅŸÃ¼lÃ¼yor.", flush=True)

    raise RuntimeError(f"{model} iÅŸlemi tamamlanamadÄ±. {last_detail}")


def run_face_enhancer(
    facefusion_dir: Path,
    target_path: Path,
    source_paths: list[Path],
    output_path: Path
) -> None:
    command = [
        sys.executable, str(facefusion_dir / "facefusion.py"), "headless-run",
        "--source-paths", *[str(path) for path in source_paths],
        "--target-path", str(target_path),
        "--output-path", str(output_path),
        "--processors", "face_enhancer",
        "--face-enhancer-model", FACE_ENHANCER_MODEL,
        "--face-enhancer-blend", str(FACE_ENHANCER_INTERNAL_BLEND),
        "--face-enhancer-weight", str(FACE_ENHANCER_WEIGHT),
        "--face-selector-mode", "one",
        *face_mask_cli_arguments(),
        "--execution-providers", "cuda",
        "--execution-device-ids", "0",
        "--output-image-quality", "100",
        "--log-level", "info"
    ]
    completed = subprocess.run(
        command, cwd=str(facefusion_dir), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900, check=False
    )
    if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(
            "GFPGAN isteÄŸe baÄŸlÄ± yÃ¼z iyileÅŸtirme aÅŸamasÄ± baÅŸarÄ±sÄ±z. "
            + tail(completed.stderr + "\n" + completed.stdout)
        )


def blend_enhanced_hybrid_output(hybrid_path: Path, enhanced_path: Path, output_path: Path) -> None:
    try:
        import cv2
        import numpy as np
    except Exception as error:
        raise RuntimeError(f"GFPGAN harmanlamasÄ± iÃ§in OpenCV/NumPy yÃ¼klenemedi: {error}") from error
    hybrid = cv2.imread(str(hybrid_path), cv2.IMREAD_UNCHANGED)
    enhanced = cv2.imread(str(enhanced_path), cv2.IMREAD_COLOR)
    if hybrid is None or enhanced is None or hybrid.ndim != 3 or hybrid.shape[2] != 4:
        raise RuntimeError("GFPGAN Ã§Ä±ktÄ±sÄ± ve alfa maskeli hibrit gÃ¶rÃ¼ntÃ¼ okunamadÄ±.")
    if hybrid.shape[:2] != enhanced.shape[:2]:
        raise RuntimeError("GFPGAN ve hibrit yÃ¼z Ã§Ä±ktÄ± boyutlarÄ± eÅŸleÅŸmiyor.")
    blend = FACE_ENHANCER_BLEND / 100.0
    mixed = (
        hybrid[:, :, :3].astype(np.float32) * (1.0 - blend)
        + enhanced.astype(np.float32) * blend
    )
    hybrid[:, :, :3] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    # Retain the exact pre-existing semantic/change matte.
    if not cv2.imwrite(str(output_path), hybrid):
        raise RuntimeError("GFPGAN sonrasÄ± alfa maskeli hibrit sonuÃ§ yazÄ±lamadÄ±.")


def pad_image_to_even(path: Path) -> tuple[int, int]:
    """FaceFusion rounds every output to even width/height (vision.normalize_resolution),
    rescaling odd-sized targets by a pixel. Pad odd targets by one replicated
    edge pixel instead, so outputs line up 1:1 with the target, and return the
    original size to crop back to."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Hedef kÄ±rpÄ±m okunamadÄ±.")
    height, width = image.shape[:2]
    pad_right, pad_bottom = width % 2, height % 2
    if pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
        if not cv2.imwrite(str(path), padded):
            raise RuntimeError("Hedef kÄ±rpÄ±m Ã§ift boyuta tamamlanamadÄ±.")
    return width, height


def crop_to_size(image, size: dict[str, int] | None):
    """Undo pad_image_to_even: trim at most one padded row/column."""
    if size is None or image is None:
        return image
    height, width = image.shape[:2]
    if 0 <= width - size["width"] <= 1 and 0 <= height - size["height"] <= 1:
        return image[: size["height"], : size["width"]]
    return image


def reshape_jawline(path: Path, jaw_warp: dict[str, Any], face_width: float, scale: float) -> None:
    """Move the jaw/chin contour towards the reference person's (see retake_compositing.warp_jawline)."""
    import cv2
    try:
        from .retake_compositing import warp_jawline
    except ImportError:
        from retake_compositing import warp_jawline
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise RuntimeError("Ã‡ene hattÄ± dÃ¼zeltmesi iÃ§in alfa maskeli PNG okunamadÄ±.")
    color, alpha = warp_jawline(
        image[:, :, :3].copy(), image[:, :, 3].copy(),
        jaw_warp["controls"], jaw_warp["anchors"], face_width, scale
    )
    if not cv2.imwrite(str(path), cv2.merge((*cv2.split(color), alpha))):
        raise RuntimeError("Ã‡ene hattÄ± dÃ¼zeltmesi uygulanmÄ±ÅŸ PNG yazÄ±lamadÄ±.")


def fit_head_to_face(
    input_path: Path,
    output_path: Path,
    face_bounds: dict[str, float],
    scale_x: float,
    scale_y: float
) -> None:
    """Smoothly lengthen/shorten/widen/narrow the head around the new face."""
    try:
        import cv2
        try:
            from .retake_compositing import fit_head_geometry
        except ImportError:
            from retake_compositing import fit_head_geometry
    except Exception as error:
        raise RuntimeError(f"Kafa oranÄ± dÃ¼zeltmesi iÃ§in OpenCV/NumPy yÃ¼klenemedi: {error}") from error

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise RuntimeError("Kafa oranÄ± dÃ¼zeltmesi iÃ§in alfa maskeli hibrit PNG okunamadÄ±.")
    color, alpha = fit_head_geometry(
        image[:, :, :3].copy(), image[:, :, 3].copy(), face_bounds, scale_x, scale_y
    )
    adjusted = cv2.merge((*cv2.split(color), alpha))
    if not cv2.imwrite(str(output_path), adjusted):
        raise RuntimeError("Kafa oranÄ± dÃ¼zeltmesi uygulanmÄ±ÅŸ PNG yazÄ±lamadÄ±.")


def person_beard(profile_path: Path, target_landmarks: list[list[float]] | None) -> dict[str, Any] | None:
    """The person's beard/stubble tone relation from the 25-photo profile, if usable."""
    if not target_landmarks:
        return None
    try:
        import cv2
        try:
            from .retake_appearance import profile_beard_offset
        except ImportError:
            from retake_appearance import profile_beard_offset
        profile = cv2.imread(str(profile_path), cv2.IMREAD_COLOR)
        if profile is None:
            return None
        offset = profile_beard_offset(profile)
    except Exception as error:  # informative only
        print(f"Sakal profili Ã¶lÃ§Ã¼lemedi: {error}", flush=True)
        return None
    print(f"KiÅŸinin sakal/yanak ton farkÄ± (Lab): {', '.join(f'{v:.1f}' for v in offset)}.", flush=True)
    return {"points": target_landmarks, "offset": offset.tolist()}


def choose_composite(
    facefusion_dir: Path,
    centroid: list[float] | None,
    blends: list[tuple[Path, float]],
    labels: list[str] | None = None
) -> Path:
    """The blend closest to the person among those with an invisible border.

    `blends` is (path, border step) with the most seamless one first; it is
    the fallback when nothing can be measured.
    """
    if len(blends) == 1 or not centroid:
        return blends[0][0]
    limit = max(BORDER_STEP_FLOOR, BORDER_STEP_RATIO * min(border for _, border in blends))
    names = {path: (labels[index] if labels and index < len(labels) else str(index + 1))
             for index, (path, _) in enumerate(blends)}
    allowed = [(path, border) for path, border in blends if border <= limit]
    scores = score_identity(facefusion_dir, centroid, [path for path, _ in allowed])
    scored = [(value, path, border) for (path, border), value in zip(allowed, scores) if value is not None]
    if not scored:
        return blends[0][0]
    value, path, border = max(scored, key=lambda item: item[0])
    # The global ArcFace embedding can barely notice beard and chin texture.
    # When a verified, seamless reference lower-face candidate is within
    # 0.002 cosine of the best globally-scored option, prefer genuine
    # reference details instead of a synthetic smoothed beard. Full-face
    # transplants do NOT receive this preference.
    matched_lower = [
        item for item in scored
        if names[item[1]].find("referans alt yÃ¼z") >= 0
        and item[0] >= value - 0.002
    ]
    # User-requested complete reconstruction: a true reference image may
    # win only after the same border threshold and within a strict ArcFace
    # near-tie. Full-face isn't forced on incompatible pose/light/expression.
    matched_full = [
        item for item in scored
        if "tam referans yÃ¼zÃ¼" in names[item[1]]
        and item[0] >= max(score for score, _, _ in scored) - 0.002
    ]
    if matched_full:
        value, path, border = max(matched_full, key=lambda item: item[0])
    elif matched_lower:
        value, path, border = max(matched_lower, key=lambda item: item[0])
    listed = ", ".join(
        f"{names[item[0]]}: {'â€”' if score is None else f'{score:.3f}'} (kenar {item[1]:.2f})"
        for item, score in zip(allowed, scores)
    )
    print(f"Kimlik Ã¶ncelikli birleÅŸtirme: {listed}; seÃ§ilen benzerlik {value:.3f}.", flush=True)
    return path


def run_realism(
    realism: dict[str, Any],
    final_output: Path,
    target_path: Path,
    face_bounds: dict[str, float],
    centroid: list[float],
    temp_dir: Path,
    landmarks: list[list[float]] | None = None
) -> list[Path]:
    """Render photographic versions of the face in the realism environment.

    Informative only: any failure returns [] and the FaceFusion result stays.
    """
    worker = Path(__file__).with_name("retake_realism_worker.py")
    identity_file = temp_dir / "retake-realism-identity.json"
    identity_file.write_text(json.dumps(centroid), encoding="utf-8")
    command = [
        str(realism["python"]), str(worker),
        "--models-dir", str(realism["models"]),
        "--swapped", str(final_output),
        "--target", str(target_path),
        "--bounds", json.dumps(face_bounds),
        "--identity-file", str(identity_file),
        "--out-dir", str(temp_dir / "realism"),
    ]
    if landmarks:
        landmarks_file = temp_dir / "retake-realism-landmarks.json"
        landmarks_file.write_text(json.dumps(landmarks), encoding="utf-8")
        command += ["--landmarks", str(landmarks_file)]
    try:
        completed = subprocess.run(
            command, cwd=str(worker.parent), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=REALISM_TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[GerÃ§ekÃ§ilik] Ã‡alÄ±ÅŸtÄ±rÄ±lamadÄ±: {error}", flush=True)
        return []
    marker = "RETAKE_REALISM_REPORT="
    report = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            try:
                report = json.loads(line[len(marker):])
            except json.JSONDecodeError:
                report = None
            break
    if completed.returncode != 0 or not report or not report.get("ok"):
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        print(f"[GerÃ§ekÃ§ilik] BaÅŸarÄ±sÄ±z (kod {completed.returncode}): {' | '.join(detail)}", flush=True)
        return []
    print(
        f"[GerÃ§ekÃ§ilik] {len(report['outputs'])} sÃ¼rÃ¼m Ã¼retildi ({report.get('device')}, "
        f"{'kiÅŸisel model' if report.get('personal') else 'genel model + kimlik'}, "
        f"yÃ¼kleme {report.get('load_seconds')} sn, iÅŸleme {report.get('render_seconds')} sn).",
        flush=True
    )
    return [Path(item["path"]) for item in report["outputs"] if Path(item["path"]).is_file()]


def head_fitted_landmarks(
    landmarks: list[list[float]] | None,
    face_bounds: dict[str, float],
    scale_x: float,
    scale_y: float
) -> list[list[float]] | None:
    """The face mesh after the head fit (a scale about the face centre there)."""
    if not landmarks:
        return None
    cx = face_bounds["x"] + face_bounds["width"] / 2.0
    cy = face_bounds["y"] + face_bounds["height"] / 2.0
    return [[cx + (x - cx) * scale_x, cy + (y - cy) * scale_y] for x, y in landmarks]


def choose_realism_render(
    base_similarity: float | None,
    renders: list[Path],
    similarities: list[float | None]
) -> tuple[Path, float] | None:
    """The render closest to the person, if it keeps (or improves) the identity.

    Without a score for the FaceFusion result nothing can be verified, so no
    render is used.
    """
    if base_similarity is None:
        return None
    scored = [(value, path) for path, value in zip(renders, similarities) if value is not None]
    if not scored:
        return None
    best_value, best_path = max(scored, key=lambda item: item[0])
    if best_value < base_similarity - REALISM_MAX_IDENTITY_LOSS:
        return None
    return best_path, best_value


def apply_realism_render(final_output: Path, render_path: Path) -> None:
    """Put the render's colour into the RGBA result; its matte covers every changed pixel."""
    import cv2
    import numpy as np
    final = cv2.imread(str(final_output), cv2.IMREAD_UNCHANGED)
    render = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
    if final is None or render is None or final.shape[:2] != render.shape[:2] or final.ndim != 3 or final.shape[2] != 4:
        raise RuntimeError("GerÃ§ekÃ§ilik sonucu hibrit PNG ile birleÅŸtirilemedi.")
    changed = (np.max(cv2.absdiff(render, final[:, :, :3]), axis=2) >= 2).astype(np.float32)
    changed = cv2.GaussianBlur(cv2.dilate(changed, np.ones((5, 5), np.uint8)), (0, 0), 2.0) * 255.0
    alpha = np.maximum(final[:, :, 3].astype(np.float32), changed)
    merged = cv2.merge((*cv2.split(render), np.clip(np.rint(alpha), 0, 255).astype(np.uint8)))
    if not cv2.imwrite(str(final_output), merged):
        raise RuntimeError("GerÃ§ekÃ§ilik sonucu yazÄ±lamadÄ±.")


def find_realism(python: Path | None, models: Path | None) -> dict[str, Any] | None:
    """The realism environment, if it is installed (tools/install_realism.ps1)."""
    if python is None or models is None:
        return None
    python, models = python.expanduser(), models.expanduser()
    if not python.is_file() or not models.is_dir():
        return None
    try:
        from .retake_realism_worker import model_paths
    except ImportError:
        try:
            from retake_realism_worker import model_paths
        except ImportError:  # its imports need numpy/cv2, which the engine has
            return None
    if not all(path.exists() for path in model_paths(models).values()):
        return None
    return {"python": python, "models": models}


def apply_real_face(result, target, real_face: dict[str, Any], face_width: float):
    """Replace the swap's face with the person's own photo, warped onto the target.

    Every candidate photo is warped and the one that fits the face best is
    used. Never fails the job: if no photo can be read or warped, the AI swap
    is kept. Returns (image, used).
    """
    import cv2
    import numpy as np
    try:
        from .retake_detail import choose_and_transplant
        from .lower_face_transfer import choose_and_transfer_lower_face
    except ImportError:
        from retake_detail import choose_and_transplant
        from lower_face_transfer import choose_and_transfer_lower_face
    try:
        candidates = []
        for payload, reference in zip(real_face["images"], real_face["references"]):
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            points = np.asarray(reference["landmarks"], np.float32)
            if image is None:
                continue
            height, width = image.shape[:2]
            if (points[:, 0] > width + 2).any() or (points[:, 1] > height + 2).any():
                continue
            candidates.append((image, points))
        if not candidates:
            raise ValueError("gerÃ§ek fotoÄŸraf okunamadÄ±")
        if real_face.get("mode") == "lower":
            transplanted, chosen, fits = choose_and_transfer_lower_face(
                result, target, candidates,
                np.asarray(real_face["targetLandmarks"], np.float32), face_width,
                # The rescue path can try a gentle real-beard transfer
                # over the stronger identity-only base, rather than losing
                # global facial identity to an all-or-nothing transplant.
                strength=0.94 * min(1.0, max(0.05, float(real_face.get("strength", 1.0))))
            )
        else:
            transplanted, chosen, fits = choose_and_transplant(
                result, target, candidates,
                np.asarray(real_face["targetLandmarks"], np.float32), face_width,
                complete_face=real_face.get("mode") == "full",
                strength=float(real_face.get("strength", 1.0))
            )
        scores = ", ".join(f"{fit['score']:.2f}" for fit in fits)
        print(
            f"GerÃ§ek fotoÄŸraf yÃ¼zÃ¼ uygulandÄ±: {len(candidates)} aday bÃ¼kÃ¼lÃ¼p karÅŸÄ±laÅŸtÄ±rÄ±ldÄ± "
            f"(uyum {scores}), {chosen + 1}. aday seÃ§ildi.",
            flush=True
        )
        return transplanted, True
    except Exception as error:  # informative only; keep the AI swap
        print(f"GerÃ§ek fotoÄŸraf yÃ¼zÃ¼ uygulanamadÄ±, yapay zekÃ¢ sonucu kullanÄ±lÄ±yor: {error}", flush=True)
        return result, False


def blend_lower_face(
    target_path: Path,
    identity_path: Path,
    appearance_path: Path,
    output_path: Path,
    face_bounds: dict[str, float],
    detail_transfer_strength: float = DETAIL_TRANSFER_STRENGTH,
    expected_size: dict[str, int] | None = None,
    shading_strength: float | None = None,
    real_face: dict[str, Any] | None = None,
    harmonize_strength: float | None = None,
    report: dict[str, Any] | None = None,
    beard: dict[str, Any] | None = None
) -> bool:
    try:
        import cv2
        import numpy as np
        try:
            from .retake_compositing import HARMONIZE_STRENGTH, matte_border_step, seamless_face_composite
        except ImportError:
            from retake_compositing import HARMONIZE_STRENGTH, matte_border_step, seamless_face_composite
    except Exception as error:
        raise RuntimeError(f"Alt yÃ¼z harmanlama iÃ§in OpenCV/NumPy yÃ¼klenemedi: {error}") from error

    target = crop_to_size(cv2.imread(str(target_path), cv2.IMREAD_COLOR), expected_size)
    identity = crop_to_size(cv2.imread(str(identity_path), cv2.IMREAD_COLOR), expected_size)
    appearance = crop_to_size(cv2.imread(str(appearance_path), cv2.IMREAD_COLOR), expected_size)
    if (
        target is None or identity is None or appearance is None
        or target.shape != identity.shape or target.shape != appearance.shape
    ):
        sizes = ", ".join(
            "yok" if image is None else f"{image.shape[1]}x{image.shape[0]}"
            for image in (target, identity, appearance)
        )
        raise RuntimeError(f"Hedef, kimlik ve gÃ¶rÃ¼nÃ¼m geÃ§iÅŸlerinin Ã§Ä±ktÄ± Ã¶lÃ§Ã¼leri eÅŸleÅŸmiyor ({sizes}).")

    height, width = identity.shape[:2]
    if expected_size is not None and (width, height) != (expected_size["width"], expected_size["height"]):
        raise ValueError("Hedef kÄ±rpÄ±mÄ±n gerÃ§ek boyutu bildirilen boyutla eÅŸleÅŸmiyor.")
    x = face_bounds["x"]
    y = face_bounds["y"]
    face_width = face_bounds["width"]
    face_height = face_bounds["height"]
    center = (
        int(round(x + face_width * 0.50)),
        int(round(y + face_height * 0.88))
    )
    axes = (
        max(1, int(round(face_width * 0.78))),
        max(1, int(round(face_height * 0.36)))
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    blur_sigma = max(5.0, face_width * 0.055)
    mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma).astype(np.float32) / 255.0
    # Keep AlphaFace's identity geometry; add only restrained high-frequency
    # UniFace details to the lower face instead of the previous 82% pixel blend.
    detail_sigma = max(1.0, face_width * 0.012)
    appearance_low = cv2.GaussianBlur(appearance, (0, 0), detail_sigma).astype(np.float32)
    appearance_detail = appearance.astype(np.float32) - appearance_low
    # Apply detail only where FaceFusion's identity pass already changed pixels.
    # This carries its parser/occlusion protection into the detail layer too.
    identity_changed = np.max(cv2.absdiff(identity, target), axis=2)
    identity_gate = (identity_changed >= 2).astype(np.float32)[:, :, None]
    detail_alpha = np.clip(mask * detail_transfer_strength, 0.0, detail_transfer_strength)[:, :, None]
    detail_alpha *= identity_gate
    blended = identity.astype(np.float32) + appearance_detail * detail_alpha
    result = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    # Reject only practically unchanged results rather than reporting a no-op as success.
    x0 = max(0, min(width - 1, int(face_bounds["x"])))
    y0 = max(0, min(height - 1, int(face_bounds["y"])))
    x1 = max(x0 + 1, min(width, int(face_bounds["x"] + face_width + 0.999)))
    y1 = max(y0 + 1, min(height, int(face_bounds["y"] + face_height + 0.999)))
    face_delta = np.max(cv2.absdiff(result, target), axis=2)[y0:y1, x0:x1]
    changed_ratio = float(np.count_nonzero(face_delta >= 3) / face_delta.size)
    mean_delta = float(np.mean(face_delta))
    p95_delta = float(np.percentile(face_delta, 95))
    if is_near_noop(changed_ratio, mean_delta, p95_delta):
        raise NoOpSwapError(
            "FaceFusion iki geÃ§iÅŸte de hedef yÃ¼ze anlamlÄ± bir deÄŸiÅŸiklik uygulamadÄ± "
            f"(deÄŸiÅŸen piksel oranÄ± %{changed_ratio * 100:.3f}, ortalama fark {mean_delta:.3f}). "
            "Ä°ÅŸlem baÅŸarÄ±lÄ± gibi gÃ¶sterilmedi; kaynak/hedef yÃ¼z algÄ±lamasÄ±nÄ± kontrol edin."
        )

    # FaceFusion has already applied its parser + occlusion masks while pasting
    # each pass back into the target crop. Re-blend that result into the photo
    # without a visible border: rim tone matching (gradient domain), multi-band
    # blending and sharpness/grain matching. The returned matte lets the
    # browser keep every other source pixel byte-for-byte.
    used_real_face = False
    if real_face is not None:
        # All compositing variants in one identity round start from the same
        # unprocessed BGR image. Real-photo triangulation is expensive:
        # compute each lower/full transfer once, then try the different
        # harmonisation strengths on that identical candidate.
        cached = real_face.get("_precomposite_cache")
        if cached is not None and cached.shape == result.shape:
            result = cached.copy()
            used_real_face = True
        else:
            result, used_real_face = apply_real_face(result, target, real_face, face_width)
            if used_real_face:
                real_face["_precomposite_cache"] = result.copy()
    options: dict[str, float] = {}
    if shading_strength is not None:
        options["shading_strength"] = shading_strength
    options["harmonize_strength"] = HARMONIZE_STRENGTH if harmonize_strength is None else harmonize_strength
    composite, alpha = seamless_face_composite(target, result, face_bounds, **options)
    if beard is not None:
        try:
            from .retake_appearance import match_beard
        except ImportError:
            from retake_appearance import match_beard
        composite = match_beard(
            composite, alpha, np.asarray(beard["points"], np.float32), np.asarray(beard["offset"], np.float32), face_width
        )
    if report is not None:
        report["border"] = matte_border_step(composite, target, alpha, face_width)
    rgba = cv2.cvtColor(composite, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    if not cv2.imwrite(str(output_path), rgba):
        raise RuntimeError("Hibrit yÃ¼z sonucu ve semantik alfa maskesi diske yazÄ±lamadÄ±.")
    return used_real_face


def process_swap(
    facefusion_dir: Path,
    pixel_boost: str,
    enhancer_enabled: bool,
    parsed: tuple,
    progress,
    caches: dict[str, Any] | None = None
) -> tuple[bytes, dict[str, str]]:
    """Run the full hybrid pipeline for one parsed request."""
    caches = caches or {}
    identity_cache: IdentityReportCache | None = caches.get("identity")
    pass_cache: PassCache | None = caches.get("passes")
    (
        target_bytes, reference_bytes, target_ext, reference_exts,
        appearance_bytes, appearance_ext, metadata
    ) = parsed
    with tempfile.TemporaryDirectory(prefix="retake-hybrid-profile-") as temp_name:
        temp_dir = Path(temp_name)
        target_path = temp_dir / ("target" + target_ext)
        target_path.write_bytes(target_bytes)
        pad_image_to_even(target_path)
        appearance_path = temp_dir / ("appearance-profile" + appearance_ext)
        appearance_path.write_bytes(appearance_bytes)

        reference_paths: list[Path] = []
        for index, (payload, extension) in enumerate(zip(reference_bytes, reference_exts), start=1):
            source_path = temp_dir / f"reference-{index:02d}{extension}"
            source_path.write_bytes(payload)
            reference_paths.append(source_path)

        identity_key = references_digest(reference_bytes, f"{IDENTITY_VERIFIER_MODEL}:{IDENTITY_REPORT_VERSION}")
        identity_report = identity_cache.get(identity_key) if identity_cache else None
        progress(0, cached=identity_report is not None)
        if identity_report is None:
            identity_report = verify_reference_identity(facefusion_dir, reference_paths)
            if identity_cache:
                identity_cache.put(identity_key, identity_report)
        else:
            print("ArcFace raporu Ã¶nbellekten alÄ±ndÄ± (referanslar deÄŸiÅŸmedi).", flush=True)
        outlier_indices = [int(index) for index in identity_report.get("outlier_indices", [])]
        weights: list[int] = list(metadata["referenceWeights"])
        for outlier_index in outlier_indices:
            if 0 <= outlier_index < len(weights):
                weights[outlier_index] = 1
        outlier_names = identity_report.get("outlier_files", [])
        print(
            "ArcFace 25 referans kontrolÃ¼: "
            f"medyan cosine={identity_report.get('median_similarity')}; "
            f"inceleme gerektiren fotoÄŸraf={outlier_names or 'yok'}. "
            "AykÄ±rÄ± referanslar Ã§Ä±karÄ±lmadÄ±; kimlik aÄŸÄ±rlÄ±klarÄ± 1 oya indirildi.",
            flush=True
        )
        # All 25 photographs must be checked, but averaging all of their
        # embeddings can wash out distinctive facial traits. Create bounded
        # alternative pools: full / most self-consistent / closest target pose.
        # Actual generated output is ArcFace-scored before selecting a pool.
        source_plans = plan_identity_source_sets(
            weights, identity_report, metadata.get("poseDistances")
        )
        weighted_identity_paths = [reference_paths[i] for i in source_plans[0][1]]
        identity_output = temp_dir / "retake-identity-alpha.png"
        appearance_output = temp_dir / "retake-appearance-uniface.png"
        hybrid_output = temp_dir / "retake-hybrid-profile.png"
        enhanced_output = temp_dir / "retake-gfpgan.png"
        final_output = temp_dir / "retake-final.png"
        preferred_boost = pixel_boost
        variation = metadata["variation"]
        face_swapper_weight, detail_transfer_strength, head_fit_strength = (
            FACE_SWAP_VARIATIONS[variation]
        )
        face_swapper_weight = supported_facefusion_weight(face_swapper_weight)
        head_fit = metadata["headFit"]
        jaw_warp = metadata.get("jawWarp")
        # Dedicated aggressive-jaw variants extend the *measured reference*
        # jaw transformation, never invent a generic chin shape. A hard cap
        # in retake_compositing prevents excessive pixel displacement.
        jaw_focused = variation in (2, 7, 11, 15, 19, 21, 23)
        jaw_scale = (
            min(1.75, max(0.0, head_fit_strength) * (1.40 if jaw_focused else 1.0))
            if jaw_warp else 0.0
        )
        # The jaw warp already reshapes the lower face; the global head fit then
        # adds three quarters of its proportion change, so length is not counted twice.
        if jaw_scale > 0:
            head_fit_strength *= 0.75
        face_width_scale = min(HEAD_FIT_MAX, max(
            HEAD_FIT_MIN, 1.0 + (head_fit["scaleX"] - 1.0) * head_fit_strength
        ))
        face_height_scale = min(HEAD_FIT_MAX, max(
            HEAD_FIT_MIN, 1.0 + (head_fit["scaleY"] - 1.0) * head_fit_strength
        ))

        print(
            f"Profil hazÄ±r: {len(reference_paths)} referans, aÄŸÄ±rlÄ±klÄ± kimlik oyu={len(weighted_identity_paths)}, "
            f"gÃ¶rÃ¼nÃ¼m profili={len(reference_paths)} hizalanmÄ±ÅŸ fotoÄŸraf.",
            flush=True
        )
        print(
            f"1/2 kimlik geÃ§iÅŸi: {IDENTITY_MODEL}; 2/2 gÃ¶rÃ¼nÃ¼m geÃ§iÅŸi: {APPEARANCE_MODEL}; "
            f"hedef=1 fotoÄŸraf; tercih edilen pixel boost={preferred_boost}.",
            flush=True
        )

        target_key_bytes = target_path.read_bytes()

        def cached_pass(step: int, sources: list[Path], output: Path, model: str, source_image: Path | None = None) -> str:
            image = source_image or target_path
            key = None
            if pass_cache is not None:
                key = PassCache.key(
                    target_key_bytes if source_image is None else image.read_bytes(),
                    [path.read_bytes() for path in sources], model,
                    face_swapper_weight, preferred_boost
                )
                hit = pass_cache.get(key)
                if hit is not None:
                    progress(step, cached=True)
                    output.write_bytes(hit[0])
                    print(f"{model}: aynÄ± hedef iÃ§in Ã¶nceki geÃ§iÅŸ yeniden kullanÄ±ldÄ±.", flush=True)
                    return hit[1]
            progress(step)
            used = run_with_memory_fallback(
                facefusion_dir, image, sources, output, model, preferred_boost, face_swapper_weight
            )
            if pass_cache is not None and key is not None:
                pass_cache.put(key, output.read_bytes(), used)
            return used

        identity_boost = cached_pass(1, weighted_identity_paths, identity_output, IDENTITY_MODEL)
        print(f"Kimlik profili geÃ§iÅŸi tamamlandÄ± ({identity_boost}).", flush=True)

        centroid = identity_report.get("centroid")
        identity_source_paths = weighted_identity_paths
        identity_source = identity_output
        identity_pool = "all-25"
        if centroid and len(source_plans) > 1:
            candidates: list[tuple[str, Path, list[Path], str]] = [
                ("all-25", identity_output, weighted_identity_paths, identity_boost)
            ]
            for name, pool in source_plans[1:]:
                source_subset = [reference_paths[i] for i in pool]
                alternative_output = temp_dir / f"retake-identity-{name}.png"
                try:
                    boost = cached_pass(1, source_subset, alternative_output, IDENTITY_MODEL)
                    candidates.append((name, alternative_output, source_subset, boost))
                except Exception as error:
                    # The original full-25 pass remains usable if an optional
                    # candidate runs out of VRAM or fails face detection.
                    print(f"Ek {name} kimlik adayÄ± atlandÄ±: {error}", flush=True)
            if len(candidates) > 1:
                candidate_scores = score_identity(
                    facefusion_dir, centroid, [path for _, path, _, _ in candidates]
                )
                best_index = 0
                best_value = candidate_scores[0]
                for i, value in enumerate(candidate_scores[1:], start=1):
                    # Require an observable improvement, not a tiny change
                    # from embedding / image-encoding measurement noise.
                    if value is not None and (
                        best_value is None or value > best_value + 0.006
                    ):
                        best_index, best_value = i, value
                identity_pool, identity_source, identity_source_paths, identity_boost = candidates[best_index]
                print(
                    "Kaynak havuzu karÅŸÄ±laÅŸtÄ±rmasÄ±: "
                    + ", ".join(
                        f"{entry[0]}={'â€”' if value is None else f'{value:.3f}'}"
                        for entry, value in zip(candidates, candidate_scores)
                    )
                    + f"; seÃ§ilen {identity_pool}.",
                    flush=True
                )

        # Identity reinforcement: a second swapper over the selected output.
        reinforced = False
        if centroid and face_swapper_weight >= 1.0:
            reinforced_output = temp_dir / "retake-identity-reinforced.png"
            try:
                cached_pass(REINFORCE_STAGE, identity_source_paths, reinforced_output, REINFORCE_MODEL, identity_source)
                single, double = score_identity(facefusion_dir, centroid, [identity_source, reinforced_output])
                if single is not None and double is not None and double > single + 0.008:
                    identity_source = reinforced_output
                    reinforced = True
                print(
                    "Kimlik pekiÅŸtirme: "
                    + ("uygulandÄ±" if reinforced else "kullanÄ±lmadÄ±")
                    + (f" (benzerlik {single:.3f} â†’ {double:.3f})" if single is not None and double is not None else ""),
                    flush=True
                )
            except Exception as error:  # informative only; the first pass stays
                print(f"Kimlik pekiÅŸtirme atlandÄ±: {error}", flush=True)

        appearance_boost = cached_pass(3, [appearance_path], appearance_output, APPEARANCE_MODEL)
        print(f"25 fotoÄŸraflÄ± gÃ¶rÃ¼nÃ¼m profili geÃ§iÅŸi tamamlandÄ± ({appearance_boost}).", flush=True)

        # The 25-image pixel mean can look waxy: aligned eyes and facial hair
        # rarely land on identical pixels. Try a crisp real reference as an
        # optional UniFace *detail* source, not as a replacement for identity.
        texture_reference_index = select_texture_reference(
            weights, identity_report, metadata.get("poseDistances")
        )
        texture_output: Path | None = None
        if texture_reference_index is not None:
            alternative = temp_dir / "retake-appearance-single.png"
            try:
                cached_pass(3, [reference_paths[texture_reference_index]], alternative, APPEARANCE_MODEL)
                texture_output = alternative
            except Exception as error:
                print(f"Tek fotoÄŸraftan doku adayÄ± atlandÄ±: {error}", flush=True)

        progress(4)
        # Identity-first compositing: a few blends from seamless (photo's skin
        # tone pulled in) to faithful (the swap's own skin kept). Those whose
        # border stays invisible are scored with ArcFace; the one closest to
        # the person is used.
        real_face = (
            {**metadata["realFace"], "images": metadata["realReferenceBytes"]}
            if metadata.get("realFace") and metadata.get("realReferenceBytes") else None
        )
        user_shading = caches.get("shading")
        beard = person_beard(appearance_path, metadata.get("targetLandmarks"))
        blends: list[tuple[Path, float]] = []
        labels: list[str] = []
        transplanted: dict[Path, str] = {}
        # In selected rounds compare the complete reference face, its
        # beard/jaw-only detail, and model-only identity. Every option starts
        # from the original target; only ArcFace-verified, seamless results
        # are eligible to win. Restrict extra warps to budgeted rounds.
        lower_reference_rounds = (0, 2, 7, 11, 15, 19, 21, 23)
        full_reference_rounds = (0, 4, 11, 19)
        transplant_options: list[dict[str, Any] | None] = [None]
        if real_face:
            mode = real_face.get("mode", "full")
            if mode in ("lower", "both") and variation in lower_reference_rounds:
                transplant_options.append({**real_face, "mode": "lower"})
            if mode in ("full", "both") and (mode == "full" or variation in full_reference_rounds):
                transplant_options.append({**real_face, "mode": "full"})
        for index, (harmonize, shading_cap) in enumerate(COMPOSITE_VARIANTS):
            shading = user_shading if user_shading is not None else None
            if index > 0:
                base_shading = user_shading if user_shading is not None else DEFAULT_SHADING_STRENGTH
                shading = min(base_shading, shading_cap)
            for transplant in transplant_options:
                report: dict[str, Any] = {}
                number = len(blends)
                path = hybrid_output if number == 0 else temp_dir / f"retake-hybrid-{number}.png"
                if blend_lower_face(
                    target_path, identity_source, appearance_output, path, metadata["faceBounds"],
                    detail_transfer_strength, metadata["targetSize"], shading, transplant, harmonize, report, beard
                ):
                    transplanted[path] = transplant.get("mode", "full") if transplant else "full"
                blends.append((path, float(report.get("border", 0.0))))
                label = f"ton {harmonize:g}"
                if path in transplanted:
                    label += " + referans alt yÃ¼z" if transplanted[path] == "lower" else " + tam referans yÃ¼zÃ¼"
                labels.append(label)
        texture_blends: set[Path] = set()
        if texture_output is not None:
            for harmonize in (0.60, 0.35):
                report = {}
                path = temp_dir / f"retake-texture-{len(texture_blends)}.png"
                try:
                    blend_lower_face(
                        target_path, identity_source, texture_output, path,
                        metadata["faceBounds"], detail_transfer_strength,
                        metadata["targetSize"], min(DEFAULT_SHADING_STRENGTH, 0.20),
                        None, harmonize, report, beard
                    )
                except Exception as error:
                    print(f"GerÃ§ek referans dokusu adayÄ± atlandÄ±: {error}", flush=True)
                    continue
                texture_blends.add(path)
                blends.append((path, float(report.get("border", 0.0))))
                labels.append(f"gerÃ§ek referans dokusu {harmonize:g}")
        chosen_blend = choose_composite(facefusion_dir, centroid, blends, labels)
        used_real_face = chosen_blend in transplanted
        used_lower_face = transplanted.get(chosen_blend) == "lower"
        used_full_face = transplanted.get(chosen_blend) == "full"
        chosen_texture = (
            f"reference-{texture_reference_index + 1:02d}"
            if chosen_blend in texture_blends and texture_reference_index is not None
            else "average"
        )
        if chosen_blend != hybrid_output:
            shutil.copyfile(chosen_blend, hybrid_output)
        used_enhancer = bool(enhancer_enabled)
        if used_enhancer:
            print(
                f"Ä°steÄŸe baÄŸlÄ± son aÅŸama: {FACE_ENHANCER_MODEL}, blend={FACE_ENHANCER_BLEND}%.",
                flush=True
            )
            run_face_enhancer(facefusion_dir, hybrid_output, reference_paths, enhanced_output)
            blend_enhanced_hybrid_output(hybrid_output, enhanced_output, final_output)
        else:
            final_output = hybrid_output

        # Preserve the unwarped face. Head/jaw geometry adjustment is meant
        # to improve resemblance, but must not silently erase a good identity.
        pre_geometry_output = temp_dir / "retake-pre-geometry.png"
        shutil.copyfile(final_output, pre_geometry_output)
        if jaw_scale > 0:
            reshape_jawline(final_output, jaw_warp, metadata["faceBounds"]["width"], jaw_scale)
            print(f"Ã‡ene hattÄ± referans kiÅŸiye gÃ¶re dÃ¼zeltildi (gÃ¼Ã§ {jaw_scale:.2f}).", flush=True)
        fit_head_to_face(
            final_output,
            final_output,
            metadata["faceBounds"],
            face_width_scale,
            face_height_scale
        )
        # An identity-first fallback without the appearance/beard layer or
        # head reshaping. This is a separately masked, harmonized PNG, never
        # an unmasked raw FaceFusion output. It can rescue cases where
        # beautifying and geometric edits have pulled the person off-identity.
        identity_fallback = temp_dir / "retake-identity-only.png"
        identity_border_report: dict[str, Any] = {}
        try:
            blend_lower_face(
                target_path, identity_source, appearance_output, identity_fallback,
                metadata["faceBounds"], 0.0, metadata["targetSize"],
                shading_strength=0.0, harmonize_strength=0.35,
                report=identity_border_report
            )
        except Exception as error:
            print(f"Yedek saf kimlik bileÅŸimi oluÅŸturulamadÄ±: {error}", flush=True)
            identity_fallback = None

        # v19's biggest practical failure: the pure-identity fallback won
        # ArcFace in every real round (0.833), deleting the reference beard
        # and all jaw changes. Give that strong ID base its OWN local,
        # feathered beard and chin geometry alternatives. These are rescue
        # candidates, not blind pasted faces. The next ArcFace check must
        # approve them, and a significantly worse matte border disqualifies
        # them before candidate selection.
        identity_regional_candidates: list[tuple[str, Path]] = []
        if identity_fallback is not None:
            original_border = float(identity_border_report.get("border", 0.0))
            max_regional_border = max(BORDER_STEP_FLOOR, original_border * 1.15)
            if real_face and real_face.get("mode") in ("lower", "both") and variation in lower_reference_rounds:
                for transfer_strength in (0.35, 0.65):
                    path = temp_dir / f"retake-identity-beard-{int(transfer_strength * 100)}.png"
                    report: dict[str, Any] = {}
                    try:
                        used = blend_lower_face(
                            target_path, identity_source, appearance_output, path,
                            metadata["faceBounds"], 0.0, metadata["targetSize"],
                            shading_strength=0.0,
                            real_face={**real_face, "mode": "lower", "strength": transfer_strength},
                            harmonize_strength=0.35, report=report
                        )
                        border = float(report.get("border", float("inf")))
                        if used and math.isfinite(border) and border <= max_regional_border:
                            identity_regional_candidates.append(("identity-beard", path))
                        else:
                            print(
                                f"[Retake] GÃ¼venli sakal adayÄ± atlandÄ± "
                                f"(kenar {border:.2f}, sÄ±nÄ±r {max_regional_border:.2f}).", flush=True
                            )
                    except Exception as error:
                        print(f"[Retake] Kimlik Ã¼stÃ¼ sakal nakli atlandÄ±: {error}", flush=True)
            if real_face and real_face.get("mode") in ("full", "both") and variation in full_reference_rounds:
                for transfer_strength in (0.20, 0.40):
                    path = temp_dir / f"retake-identity-full-{int(transfer_strength * 100)}.png"
                    report: dict[str, Any] = {}
                    try:
                        used = blend_lower_face(
                            target_path, identity_source, appearance_output, path,
                            metadata["faceBounds"], 0.0, metadata["targetSize"],
                            shading_strength=0.0,
                            real_face={**real_face, "mode": "full", "strength": transfer_strength},
                            harmonize_strength=0.35, report=report
                        )
                        border = float(report.get("border", float("inf")))
                        if used and math.isfinite(border) and border <= max_regional_border:
                            identity_regional_candidates.append(("identity-full", path))
                        else:
                            print(
                                f"[Retake] GÃ¼venli tam yÃ¼z adayÄ± atlandÄ± "
                                f"(kenar {border:.2f}, sÄ±nÄ±r {max_regional_border:.2f}).", flush=True
                            )
                    except Exception as error:
                        print(f"[Retake] Kimlik Ã¼stÃ¼ tam referans dokusu atlandÄ±: {error}", flush=True)
            if jaw_warp and jaw_scale > 0:
                jaw_candidate = temp_dir / "retake-identity-jaw.png"
                try:
                    shutil.copyfile(identity_fallback, jaw_candidate)
                    reshape_jawline(
                        jaw_candidate, jaw_warp, metadata["faceBounds"]["width"],
                        min(1.25, jaw_scale)
                    )
                    identity_regional_candidates.append(("identity-jaw", jaw_candidate))
                except Exception as error:
                    print(f"[Retake] Kimlik Ã¼stÃ¼ Ã§ene uyarlamasÄ± atlandÄ±: {error}", flush=True)

        # Photographic realism (optional): re-render the face with a diffusion
        # model steered by the person's ArcFace identity.
        realism = caches.get("realism")
        renders: list[Path] = []
        if realism and centroid:
            progress(REALISM_STAGE)
            renders = run_realism(
                realism, final_output, target_path, metadata["faceBounds"], centroid, temp_dir,
                head_fitted_landmarks(metadata.get("targetLandmarks"), metadata["faceBounds"],
                                      face_width_scale, face_height_scale)
            )
        # Measure the REAL post-processed result, a pre-geometry candidate,
        # an identity-only candidate, and optional diffusion renders against
        # the same ArcFace reference centroid. A nice looking render that no
        # longer resembles the source person is not an acceptable output.
        progress(SCORING_STAGE)
        rescue_candidates = [("head-fit", final_output), ("no-head-warp", pre_geometry_output)]
        if identity_fallback is not None:
            rescue_candidates.append(("identity-only", identity_fallback))
        rescue_candidates.extend(identity_regional_candidates)
        probes = [path for _, path in rescue_candidates] + [*renders, target_path]
        scores = score_identity(facefusion_dir, centroid, probes)
        before_similarity = scores[-1]
        candidate_scores = scores[:len(rescue_candidates)]
        final_score = candidate_scores[0]
        # A rescue must improve ArcFace similarity meaningfully, not merely
        # win on measurement noise. Keep the smoother main result on ties.
        chosen_index = 0
        best_score = final_score if final_score is not None else -1.0
        for index, value in enumerate(candidate_scores[1:], start=1):
            if value is not None and value > best_score + 0.006:
                chosen_index, best_score = index, value
        # When ArcFace cannot meaningfully distinguish finalists, prefer the
        # one that retains the target photo's regional detail. This is a
        # tie-break only: detail can never override a clearly stronger
        # identity result.
        candidate_detail_scores: list[float | None] = []
        for _, candidate_path in rescue_candidates:
            try:
                detail, _ = check_photo_detail(target_path, candidate_path, metadata["faceBounds"])
            except Exception:
                detail = None
            candidate_detail_scores.append(detail)
        detail_choice = choose_detail_safe_tie(
            candidate_scores, candidate_detail_scores, chosen_index
        )
        if detail_choice != chosen_index:
            previous = chosen_index
            chosen_index = detail_choice
            best_score = candidate_scores[chosen_index] or best_score
            print(
                f"[Ultra] ArcFace yakÄ±n eÅŸitliÄŸinde daha doÄŸal doku seÃ§ildi: "
                f"{rescue_candidates[previous][0]} â†’ {rescue_candidates[chosen_index][0]} "
                f"(detay {candidate_detail_scores[previous]:.3f} â†’ "
                f"{candidate_detail_scores[chosen_index]:.3f}).",
                flush=True
            )
        selected_identity = rescue_candidates[chosen_index][0]
        # ArcFace can undervalue beard edges. ONLY when pure identity won
        # and a measured regional alternative is within 0.003 absolute
        # cosine (not "3%" similarity), allow the strongest verified
        # beard/jaw candidate to replace the unmodified beard.
        if selected_identity == "identity-only":
            regional_ties = [
                (index, value)
                for index, ((label, _), value) in enumerate(zip(rescue_candidates, candidate_scores))
                if label in ("identity-beard", "identity-full", "identity-jaw")
                and value is not None and value >= best_score - 0.003
            ]
            if regional_ties:
                chosen_index, best_score = max(regional_ties, key=lambda item: (
                    item[1],
                    {"identity-full": 2, "identity-beard": 1, "identity-jaw": 0}
                    [rescue_candidates[item[0]][0]]
                ))
                selected_identity = rescue_candidates[chosen_index][0]
                print(
                    f"[Retake] Saf kimlik korundu; son aÅŸamada {selected_identity} "
                    f"referans ayrÄ±ntÄ±sÄ± kullanÄ±ldÄ± (ArcFace={best_score:.3f}).", flush=True
                )
        if selected_identity in ("identity-only", "identity-jaw"):
            used_lower_face = False
            used_full_face = False
            used_real_face = False
        elif selected_identity == "identity-beard":
            used_lower_face = True
            used_full_face = False
            used_real_face = True
        elif selected_identity == "identity-full":
            used_lower_face = False
            used_full_face = True
            used_real_face = True
        if final_score is None and best_score < -0.99:
            raise IdentityVerificationError(
                "Ãœretilen yÃ¼z ArcFace ile doÄŸrulanamadÄ±. Kimlik kontrolÃ¼ olmadan gÃ¼venilir sonuÃ§ gÃ¶sterilmiyor."
            )
        if chosen_index > 0:
            print(
                f"Kimlik koruma: {selected_identity} seÃ§ildi "
                f"({final_score if final_score is not None else 'â€”'} â†’ {best_score:.3f}).",
                flush=True
            )
            shutil.copyfile(rescue_candidates[chosen_index][1], final_output)
        result_similarity = best_score
        used_realism = False
        render_scores = scores[len(rescue_candidates):-1]
        # ArcFace alone can reward an uncanny, waxy, over-sharpened or clipped
        # diffusion face. Retain only candidates that preserve natural cheek
        # microtexture relative to the original FaceFusion composite.
        acceptable_renders: list[Path] = []
        acceptable_scores: list[float | None] = []
        for render_path, render_score in zip(renders, render_scores):
            try:
                accepted, reason = check_render_file(
                    final_output, render_path, metadata["faceBounds"]
                )
            except Exception as error:
                accepted, reason = False, f"kontrol hatasÄ±: {error}"
            if accepted:
                acceptable_renders.append(render_path)
                acceptable_scores.append(render_score)
            else:
                print(f"GerÃ§ekÃ§ilik adayÄ± elendi ({render_path.name}): {reason}.", flush=True)
        chosen = choose_realism_render(result_similarity, acceptable_renders, acceptable_scores)
        if chosen is not None:
            render_path, render_similarity = chosen
            apply_realism_render(final_output, render_path)
            print(
                f"FotoÄŸraf gerÃ§ekÃ§iliÄŸi uygulandÄ±: kimlik benzerliÄŸi {result_similarity:.3f} â†’ {render_similarity:.3f}.",
                flush=True
            )
            result_similarity = render_similarity
            used_realism = True
        elif renders:
            listed = ", ".join("â€”" if value is None else f"{value:.3f}" for value in render_scores)
            print(
                "FotoÄŸraf gerÃ§ekÃ§iliÄŸi kullanÄ±lmadÄ±: kimlik benzerliÄŸi seÃ§ilen sonucun altÄ±nda kaldÄ± "
                f"({listed}; FaceFusion {result_similarity:.3f}).",
                flush=True
            )
        # An ink-like one-pixel face outline can survive the global ArcFace
        # gate. Attempt a narrow, local cleanup on the existing face matte,
        # then keep it only if ArcFace still verifies the same identity.
        rim_cleaned = False
        rim_candidate = temp_dir / "retake-rim-clean.png"
        try:
            rim_stats = repair_dark_rim_file(
                target_path, final_output, rim_candidate, metadata["faceBounds"]
            )
            if rim_stats.get("corrected_pixels", 0) > 0 and centroid:
                corrected_score = score_identity(facefusion_dir, centroid, [rim_candidate])[0]
                if corrected_score is not None and corrected_score >= result_similarity - 0.004:
                    shutil.copyfile(rim_candidate, final_output)
                    result_similarity = corrected_score
                    rim_cleaned = True
                    print(
                        f"[Retake] Koyu yÃ¼z konturu temizlendi "
                        f"({rim_stats['corrected_pixels']:.0f} piksel; "
                        f"koyu sÄ±nÄ±r %{rim_stats.get('dark_line_reduction', 0.0) * 100:.1f} azaldÄ±; "
                        f"ArcFace={corrected_score:.3f}).", flush=True
                    )
                else:
                    print("[Retake] Kontur adayÄ± kimliÄŸi koruyamadÄ±; atlandÄ±.", flush=True)
        except Exception as error:
            print(f"[Retake] Kontur dÃ¼zeltmesi atlandÄ±: {error}", flush=True)

        # An unverifiable or measurably worse identity is a failed edit, not
        # a successful-looking photo. This relative guard avoids claiming a
        # universal fixed cosine threshold for every angle and lighting.
        if before_similarity is not None and result_similarity + 0.02 < before_similarity:
            raise IdentityConflictError(
                "Ã‡Ä±ktÄ±nÄ±n referans kimlik benzerliÄŸi orijinal fotoÄŸraftan dÃ¼ÅŸÃ¼k "
                f"({before_similarity:.3f} â†’ {result_similarity:.3f}). "
                "BaÅŸka bir hedef fotoÄŸraf/poz veya daha net referans deneyin."
            )
        print(
            f"Kimlik benzerliÄŸi: sonuÃ§ {result_similarity:.3f}"
            + (f", orijinal {before_similarity:.3f}" if before_similarity is not None else ""),
            flush=True
        )
        try:
            detail_score, detail_regions = check_photo_detail(
                target_path, final_output, metadata["faceBounds"]
            )
        except Exception as error:
            detail_score, detail_regions = None, {}
            print(f"[Ultra] BÃ¶lgesel doku Ã¶lÃ§Ã¼mÃ¼ atlandÄ±: {error}", flush=True)
        if detail_score is not None:
            print(
                f"[Ultra] GÃ¶z/burun/aÄŸÄ±z/alt yÃ¼z detay korunumu: {detail_score:.3f}; "
                f"bÃ¶lgesel oranlar={detail_regions}.",
                flush=True
            )
        result = final_output.read_bytes()
        headers = {
            "X-Retake-Detail-Score": "" if detail_score is None else f"{detail_score:.4f}",
            "X-Retake-Rim-Cleanup": "on" if rim_cleaned else "off",
            "X-Retake-Similarity": "" if result_similarity is None else f"{result_similarity:.4f}",
            "X-Retake-Similarity-Before": "" if before_similarity is None else f"{before_similarity:.4f}",
            "X-Retake-Identity-Median-Similarity": str(identity_report.get("median_similarity", "")),
            "X-Retake-Identity-Outliers": ",".join(str(index + 1) for index in outlier_indices),
            "X-Retake-Pixel-Boost": f"{identity_boost},{appearance_boost}",
            "X-Retake-Face-Enhancer": FACE_ENHANCER_MODEL if used_enhancer else "off",
            "X-Retake-Identity-Weighted-Votes": str(len(weighted_identity_paths)),
            "X-Retake-Variation": str(variation),
            "X-Retake-Real-Face": "on" if used_full_face else "off",
            "X-Retake-Lower-Face-Transfer": "on" if used_lower_face else "off",
            "X-Retake-Jaw-Focused": "on" if jaw_focused and jaw_scale > 0 and selected_identity in ("head-fit", "identity-jaw") else "off",
            "X-Retake-Complete-Identity": "on" if used_full_face else "off",
            "X-Retake-Identity-Reinforced": "on" if reinforced else "off",
            "X-Retake-Identity-Selection": selected_identity,
            "X-Retake-Identity-Source": identity_pool,
            "X-Retake-Texture-Source": chosen_texture,
            "X-Retake-Compositing": "preblended",
            "X-Retake-Realism": "on" if used_realism else ("off" if realism else "unavailable")
        }
        print(
            "Ä°ÅŸlem tamamlandÄ±: AlphaFace kimlik geÃ§iÅŸi + UniFace yÃ¼ksek frekanslÄ± alt yÃ¼z detay "
            f"katmanÄ±; dikiÅŸsiz ton/Ã§ok bantlÄ± birleÅŸtirme; "
            f"kafa oranÄ± x{face_width_scale:.3f}/y{face_height_scale:.3f}; "
            f"pixel boost={identity_boost}/{appearance_boost}; "
            f"GFPGAN={'aÃ§Ä±k' if used_enhancer else 'kapalÄ±'}; tek alfa maskeli PNG dÃ¶ndÃ¼rÃ¼ldÃ¼.",
            flush=True
        )
        return result, headers


def _swap_error(error: Exception) -> tuple[int, str]:
    if isinstance(error, IdentityConflictError):
        print(f"[Retake ArcFace] Referans kimlik tutarsÄ±zlÄ±ÄŸÄ±: {error}", flush=True)
        return 422, str(error)
    if isinstance(error, IdentityVerificationError):
        print(f"[Retake ArcFace] Ã–n kontrol baÅŸarÄ±sÄ±z: {error}", flush=True)
        return 503, f"ArcFace kimlik kontrolÃ¼ yapÄ±lamadÄ±: {error}"
    if isinstance(error, NoOpSwapError):
        return 422, str(error)
    if isinstance(error, ValueError):
        return 400, str(error)
    return 500, f"Yerel Retake hibrit profil iÅŸlemi baÅŸarÄ±sÄ±z: {error}"


def run_ultra_search(
    facefusion_dir: Path, pixel_boost: str, enhancer_enabled: bool,
    parsed: tuple, progress, caches: dict[str, Any] | None = None,
    clock=None, sleeper=None, on_hold=None, on_finalist=None,
    min_duration_seconds: float | None = None
) -> tuple[bytes, dict[str, str]]:
    """Quality-first iterations, each on the *original* photo and 25 references.

    Intermediate results never reach the browser. Failed later candidates do
    not destroy an already verified image. No blind recursive re-swapping.
    """
    clock = clock or time.monotonic
    sleeper = sleeper or time.sleep
    minimum = ULTRA_MIN_SECONDS if min_duration_seconds is None else max(0.0, min_duration_seconds)
    started = clock()
    best_result: bytes | None = None
    best_headers: dict[str, str] | None = None
    best_round = 0
    completed = 0
    errors: list[Exception] = []
    # Diffusion is expensive and stochastic. Running 2â€“3 diffusion renders
    # in every one of 24 variants exhausted the seven-minute search window
    # before Retake could actually compare different identities. Compare
    # all feasible identity candidates first, then invest the heavy GPU
    # realism pass in the single best validated finalist.
    has_realism = bool(caches and caches.get("realism"))
    search_caches = {**caches, "realism": None} if has_realism else caches

    for index, variation in enumerate(ULTRA_VARIATIONS):
        if not should_continue_ultra(completed, clock() - started):
            break
        round_number = index + 1
        print(
            f"[Ultra] Tur {round_number}/{len(ULTRA_VARIATIONS)}: "
            f"orijinal hedef + 25 referans; varyasyon {variation}.",
            flush=True
        )

        def report(step: int, cached: bool = False) -> None:
            progress(step + index * len(JOB_STAGES), cached=cached)

        try:
            image, headers = process_swap(
                facefusion_dir, pixel_boost, enhancer_enabled,
                with_variation(parsed, variation), report, search_caches
            )
        except Exception as error:
            errors.append(error)
            print(f"[Ultra] Aday atlandÄ±: {error}", flush=True)
            # The best already verified result remains usable.
            continue
        completed += 1
        if prefer_ultra_candidate(headers, best_headers):
            best_result, best_headers, best_round = image, headers, round_number
            print(
                f"[Ultra] En iyi aday gÃ¼ncellendi: tur {round_number}, "
                f"ArcFace={score_from_headers(headers)}.",
                flush=True
            )

    if best_result is None or best_headers is None:
        if errors:
            # Preserve the actionable HTTP status and diagnosis: no-op,
            # invalid parameters and identity-conflict errors are distinct.
            raise errors[0]
        raise IdentityVerificationError("Ultra Mod hiÃ§bir adayÄ±n kimliÄŸini doÄŸrulayamadÄ±.")

    # One GPU realism investment on the best identity, not 24. This
    # evaluation always uses the untouched original target and reference
    # sources; it never feeds a generated face back as identity input.
    if has_realism:
        try:
            if on_finalist is not None:
                on_finalist()
            upgraded_image, upgraded_headers = process_swap(
                facefusion_dir, pixel_boost, enhancer_enabled,
                with_variation(parsed, ULTRA_VARIATIONS[best_round - 1]),
                lambda *_args, **_kwargs: None, caches
            )
            if prefer_ultra_candidate(upgraded_headers, best_headers):
                best_result, best_headers = upgraded_image, upgraded_headers
                print("[Ultra] Finalist gerÃ§ekÃ§ilik adayÄ± kimlik ve doku testlerini geÃ§ti.", flush=True)
            else:
                print("[Ultra] Finalist difÃ¼zyon adayÄ± elendi; gÃ¼Ã§lÃ¼ kimlik sonucu korundu.", flush=True)
        except Exception as error:
            # Realism is optional: preserve a previously verified result.
            print(f"[Ultra] Finalist gerÃ§ekÃ§ilik atlandÄ±: {error}", flush=True)

    # Spend the available interval trying distinct variants; when the GPU
    # finishes them faster, honor the explicit seven-minute minimum without
    # pointlessly running the identical model again. Failed jobs return early.
    remaining = minimum - (clock() - started)
    while remaining > 0:
        if on_hold is not None:
            on_hold(remaining)
        sleeper(min(5.0, remaining))
        remaining = minimum - (clock() - started)
    if on_hold is not None:
        on_hold(0.0)

    final = {
        **best_headers,
        "X-Retake-Ultra": "on",
        "X-Retake-Ultra-Rounds": str(completed),
        "X-Retake-Ultra-Winner": str(best_round),
        "X-Retake-Ultra-Elapsed": str(round(clock() - started, 1)),
        "X-Retake-Ultra-Realism-Policy": "finalist-only" if has_realism else "not-installed",
    }
    print(
        f"[Ultra] SonuÃ§: {completed} doÄŸrulanmÄ±ÅŸ tur, kazanan {best_round}. tur; "
        f"benzerlik={score_from_headers(best_headers)}; sÃ¼re {final['X-Retake-Ultra-Elapsed']} sn.",
        flush=True
    )
    return best_result, final


def run_job(job: dict[str, Any], server: Any, parsed: tuple) -> None:
    """Worker thread: long swaps must not depend on one open HTTP response
    (HTTPS tunnels cut responses that take longer than ~100 seconds)."""
    timings: StageTimings | None = getattr(server, "stage_timings", None)

    def hold(remaining: float) -> None:
        job["stage"] = "Ultra Â· kalite taramasÄ± tamamlandÄ±, 7 dakika sÃ¼resi doluyor"
        job["minimum_remaining_seconds"] = round(remaining, 1)
        # Actual inference has finished; mark 99% until requested minimum.
        job["step"] = job["total"] - 1
        job["stage_started"] = None  # do not log final waiting as GPU inference

    def finalist() -> None:
        # No false ETA: the diffusion worker's duration is GPU-dependent
        # and not covered by the bounded 420-second candidate search.
        job["stage"] = "Ultra Â· en iyi yÃ¼z iÃ§in son fotogerÃ§ekÃ§ilik geÃ§iÅŸi"
        job["step"] = job["total"] - 1
        job["stage_started"] = None

    def close_stage(now: float) -> None:
        current = job.get("step", 0)
        if timings is not None and current not in job["skipped"] and job.get("stage_started"):
            timings.record(JOB_STAGES[current % len(JOB_STAGES)], now - job["stage_started"])

    def progress(step: int, cached: bool = False) -> None:
        now = time.time()
        if step != job.get("step") and job.get("stage_started"):
            close_stage(now)
        # Stages jumped over (e.g. realism when it is not installed) were skipped.
        for skipped in range(int(job.get("step") or 0) + 1, step):
            job["skipped"].add(skipped)
        if cached:
            job["skipped"].add(step)
        job["step"] = step
        job["round"] = step // len(JOB_STAGES) + 1
        job["stage"] = (
            f"Ultra {job['round']}/{len(ULTRA_VARIATIONS)} Â· "
            + JOB_STAGES[step % len(JOB_STAGES)]
            + (" (Ã¶nbellekten)" if cached else "")
        )
        job["stage_started"] = now

    try:
        job["status"] = "running"
        result, headers = run_ultra_search(
            server.facefusion_dir, server.face_swapper_pixel_boost,
            server.face_enhancer_enabled, parsed, progress,
            {
                "identity": getattr(server, "identity_cache", None),
                "passes": getattr(server, "pass_cache", None),
                "shading": getattr(server, "shading_strength", None),
                "realism": getattr(server, "realism", None)
            },
            on_hold=hold, on_finalist=finalist
        )
        close_stage(time.time())
        job["result"] = result
        job["headers"] = headers
        job["step"] = job["total"]
        job["minimum_remaining_seconds"] = 0.0
        job["stage"] = "Ultra Â· en iyi sonuÃ§ seÃ§ildi"
        job["status"] = "done"
    except Exception as error:  # reported to the client through the job
        job["http_status"], job["error"] = _swap_error(error)
        job["status"] = "error"
    finally:
        job["finished"] = time.time()
        INFERENCE_LOCK.release()


def estimate_remaining(job: dict[str, Any], timings: StageTimings | None) -> float | None:
    if job["status"] not in {"queued", "running"}:
        return None
    minimum_remaining = (
        max(0.0, ULTRA_MIN_SECONDS - (time.time() - job["created"]))
        if "minimum_remaining_seconds" in job and "created" in job else 0.0
    )
    if timings is None or not isinstance(job.get("stage_started"), (int, float)) and "created" in job:
        return minimum_remaining if minimum_remaining > 0 else None
    total = job.get("total", len(JOB_STAGES))
    step = min(job.get("step", 0), total - 1)
    pending = [
        JOB_STAGES[index % len(JOB_STAGES)]
        for index in range(step, total) if index not in job["skipped"]
    ]
    if not pending:
        return minimum_remaining
    started = job.get("stage_started")
    in_current = max(0.0, time.time() - started) if step not in job["skipped"] and isinstance(started, (int, float)) else 0.0
    estimate = timings.remaining(pending, 0, in_current)
    if estimate is None:
        return minimum_remaining if minimum_remaining > 0 else None
    if "created" in job:
        # The v16 ETA projected ALL 24 potential rounds and claimed 73min
        # even though new rounds are forbidden after 420 seconds. Count only
        # the remaining search window plus the currently running CUDA stage.
        elapsed = max(0.0, time.time() - job["created"])
        remaining_window = max(0.0, ULTRA_SEARCH_BUDGET_SECONDS - elapsed)
        current_stage = JOB_STAGES[step % len(JOB_STAGES)]
        current_eta = timings.remaining([current_stage], 0, in_current)
        estimate = min(estimate, remaining_window + max(0.0, current_eta or 0.0))
    return max(minimum_remaining, estimate)


def create_job(realism: bool = False) -> dict[str, Any]:
    job: dict[str, Any] = {
        "id": secrets.token_urlsafe(16), "status": "queued", "step": 0,
        "stage": f"Ultra 1/{len(ULTRA_VARIATIONS)} Â· " + JOB_STAGES[0], "created": time.time(),
        "minimum_remaining_seconds": ULTRA_MIN_SECONDS,
        "total": len(JOB_STAGES) * len(ULTRA_VARIATIONS),
        "round": 1,
        # Without realism, each round skips that step. It must never appear
        # in the ETA nor stall the progress display.
        "skipped": set() if realism else {
            REALISM_STAGE + index * len(JOB_STAGES)
            for index in range(len(ULTRA_VARIATIONS))
        }
    }
    with JOBS_LOCK:
        JOBS[job["id"]] = job
        for stale in sorted(JOBS.values(), key=lambda item: item["created"])[:-MAX_KEPT_JOBS]:
            JOBS.pop(stale["id"], None)
    return job


class RetakeFaceFusionHandler(BaseHTTPRequestHandler):
    server_version = "RetakeFaceFusion/2.0"

    def log_message(self, fmt: str, *args: object) -> None:
        print("[Retake FaceFusion] " + (fmt % args), flush=True)

    @property
    def allowed_origins(self) -> set[str]:
        return self.server.allowed_origins  # type: ignore[attr-defined]

    @property
    def facefusion_dir(self) -> Path:
        return self.server.facefusion_dir  # type: ignore[attr-defined]

    def _is_loopback_host(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        port = self.server.server_port
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _host_allowed(self) -> bool:
        # Without pairing the service only answers requests addressed to
        # loopback. With pairing (phone / PWA through an HTTPS tunnel that
        # connects to loopback) any Host is accepted, but every request must
        # then carry the pairing code â€” see _paired().
        return self._is_loopback_host() or bool(self.server.pairing_token)  # type: ignore[attr-defined]

    def _paired(self) -> bool:
        token = self.server.pairing_token  # type: ignore[attr-defined]
        if not token:
            return True
        supplied = self.headers.get(PAIRING_HEADER) or ""
        if hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
            return True
        time.sleep(0.5)  # slow down guessing
        return False

    def _origin_allowed(self, required: bool = False) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return not required
        return origin in self.allowed_origins

    def _cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin and origin in self.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", f"Content-Type, {PAIRING_HEADER}")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header(
            "Access-Control-Expose-Headers",
            "X-Retake-Identity-Median-Similarity, X-Retake-Identity-Outliers, "
            "X-Retake-Pixel-Boost, X-Retake-Face-Enhancer, X-Retake-Identity-Weighted-Votes, "
            "X-Retake-Variation, X-Retake-Similarity, X-Retake-Similarity-Before, X-Retake-Real-Face, "
            "X-Retake-Realism, X-Retake-Compositing, X-Retake-Identity-Selection, "
            "X-Retake-Identity-Source, X-Retake-Texture-Source, "
            "X-Retake-Ultra, X-Retake-Ultra-Rounds, X-Retake-Ultra-Winner, X-Retake-Ultra-Elapsed, "
            "X-Retake-Detail-Score, X-Retake-Rim-Cleanup, "
            "X-Retake-Lower-Face-Transfer, X-Retake-Jaw-Focused, X-Retake-Complete-Identity, "
            "X-Retake-Ultra-Realism-Policy"
        )
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        if not self._host_allowed() or not self._origin_allowed(required=True):
            self._send_json(403, {"ok": False, "error": "Origin veya Host izinli deÄŸil."})
            return
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if not self._host_allowed() or not self._origin_allowed():
            self._send_json(403, {"ok": False, "error": "Origin veya Host izinli deÄŸil."})
            return
        if not self._paired():
            self._send_json(401, {
                "ok": False, "pairing_required": True,
                "error": "EÅŸleÅŸtirme kodu gerekli veya hatalÄ±."
            })
            return
        if self.path.startswith("/api/jobs/"):
            self._send_job(self.path[len("/api/jobs/"):])
            return
        if self.path != "/health":
            self._send_json(404, {"ok": False, "error": "UÃ§ nokta bulunamadÄ±."})
            return

        script = self.facefusion_dir / "facefusion.py"
        providers: list[str] = []
        provider_error = None
        try:
            import onnxruntime as ort  # type: ignore
            providers = list(ort.get_available_providers())
        except Exception as error:
            provider_error = str(error)

        self._send_json(200, {
            "ok": script.is_file(),
            "engine": "facefusion",
            "version": "3.9.2",
            "model": HYBRID_PROFILE_VERSION,
            "identity_model": IDENTITY_MODEL,
            "identity_verifier": IDENTITY_VERIFIER_MODEL,
            "identity_verification": "local-arcface-medoid-screen",
            "appearance_model": APPEARANCE_MODEL,
            "pixel_boost": self.server.face_swapper_pixel_boost,  # type: ignore[attr-defined]
            "face_enhancer_enabled": self.server.face_enhancer_enabled,  # type: ignore[attr-defined]
            "face_enhancer_model": FACE_ENHANCER_MODEL if self.server.face_enhancer_enabled else None,  # type: ignore[attr-defined]
            "face_enhancer_blend": FACE_ENHANCER_BLEND if self.server.face_enhancer_enabled else 0,
            "face_mask_types": list(FACE_MASK_TYPES),
            "face_parser_model": FACE_PARSER_MODEL,
            "face_occluder_model": FACE_OCCLUDER_MODEL,
            "face_mask_regions": list(FACE_MASK_REGIONS),
            "profile_references": MAX_REFERENCES,
            "ultra_mode": True,
            "ultra_rounds": len(ULTRA_VARIATIONS),
            "ultra_min_seconds": ULTRA_MIN_SECONDS,
            "ultra_search_budget_seconds": ULTRA_SEARCH_BUDGET_SECONDS,
            "async_jobs": True,
            "caches": True,
            "paired": bool(self.server.pairing_token),  # type: ignore[attr-defined]
            "providers": providers,
            "cuda": "CUDAExecutionProvider" in providers,
            "realism": bool(getattr(self.server, "realism", None)),
            "personal_model": bool(
                getattr(self.server, "realism", None)
                and (self.server.realism["models"] / "personal-lora" / "pytorch_lora_weights.safetensors").is_file()
                and (self.server.realism["models"] / "personal-lora" / "person.json").is_file()
            ),
            "provider_error": provider_error,
            "error": None if script.is_file() else f"FaceFusion dosyasÄ± bulunamadÄ±: {script}"
        })

    def _send_job(self, tail_path: str) -> None:
        job_id, _, action = tail_path.partition("/")
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None or action not in {"", "result"}:
            self._send_json(404, {"ok": False, "error": "Ä°ÅŸ bulunamadÄ± veya sÃ¼resi doldu."})
            return
        if action == "":
            self._send_json(200, {
                "ok": job["status"] != "error",
                "status": job["status"],
                "step": job["step"],
                "total": job.get("total", len(JOB_STAGES)),
                "round": job.get("round", 1),
                "ultra": True,
                "minimum_remaining_seconds": (
                    max(0.0, round(ULTRA_MIN_SECONDS - (time.time() - job["created"]), 1))
                    if job["status"] in {"running", "queued"}
                    else 0.0
                ),
                "stage": job["stage"],
                "elapsed": round((job.get("finished") or time.time()) - job["created"], 1),
                "error": job.get("error"),
                "http_status": job.get("http_status"),
                "eta_seconds": (
                    None if (eta := estimate_remaining(job, getattr(self.server, "stage_timings", None))) is None
                    else round(eta)
                )
            })
            return
        if job["status"] != "done":
            self._send_json(409, {"ok": False, "error": "SonuÃ§ henÃ¼z hazÄ±r deÄŸil."})
            return
        result = job["result"]
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(result)))
        for name, value in job["headers"].items():
            self.send_header(name, value)
        self._cors_headers()
        self.end_headers()
        self.wfile.write(result)

    def do_POST(self) -> None:
        if not self._host_allowed() or not self._origin_allowed(required=True):
            self._send_json(403, {"ok": False, "error": "Bu web kaynaÄŸÄ±ndan gelen isteÄŸe izin verilmiyor."})
            return
        if not self._paired():
            self._send_json(401, {
                "ok": False, "pairing_required": True,
                "error": "EÅŸleÅŸtirme kodu gerekli veya hatalÄ±."
            })
            return
        if self.path != "/api/face-swap":
            self._send_json(404, {"ok": False, "error": "UÃ§ nokta bulunamadÄ±."})
            return
        if not INFERENCE_LOCK.acquire(blocking=False):
            self._send_json(409, {"ok": False, "error": "Retake ÅŸu anda baÅŸka bir fotoÄŸrafÄ± iÅŸliyor."})
            return

        started = False
        try:
            length_header = self.headers.get("Content-Length")
            if not length_header or not length_header.isdigit():
                self._send_json(411, {"ok": False, "error": "Content-Length gerekli."})
                return
            content_length = int(length_header)
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                self._send_json(413, {"ok": False, "error": "Ä°stek boyutu izin verilen sÄ±nÄ±rÄ± aÅŸÄ±yor."})
                return
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("multipart/form-data;"):
                self._send_json(415, {"ok": False, "error": "multipart/form-data gerekli."})
                return

            parsed = parse_multipart(content_type, self.rfile.read(content_length))
            if not (self.facefusion_dir / "facefusion.py").is_file():
                self._send_json(503, {"ok": False, "error": "FaceFusion kurulumu bulunamadÄ±. Kurulum betiÄŸini Ã§alÄ±ÅŸtÄ±r."})
                return

            job = create_job(bool(getattr(self.server, "realism", None)))
            threading.Thread(
                target=run_job, args=(job, self.server, parsed), daemon=True,
                name=f"retake-job-{job['id'][:6]}"
            ).start()
            started = True  # the worker thread now owns INFERENCE_LOCK
            self._send_json(202, {"ok": True, "job_id": job["id"], "total": job["total"], "ultra": True})
        except ValueError as error:
            self._send_json(400, {"ok": False, "error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            print("Ä°stemci baÄŸlantÄ±yÄ± kapattÄ±.", flush=True)
        except Exception as error:
            self._send_json(500, {"ok": False, "error": f"Yerel Retake isteÄŸi iÅŸlenemedi: {error}"})
        finally:
            if not started:
                INFERENCE_LOCK.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retake iÃ§in yerel 25 referanslÄ± yÃ¼z profili kÃ¶prÃ¼sÃ¼.")
    parser.add_argument("--facefusion-dir", required=True, type=Path, help="FaceFusion kaynak klasÃ¶rÃ¼")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--face-swapper-pixel-boost",
        choices=("512x512", "768x768", "1024x1024"),
        default="1024x1024",
        help="Ä°lk deneme 1024x1024; CUDA bellek hatasÄ±nda sÄ±rasÄ±yla 768x768 ve 512x512 denenir."
    )
    parser.add_argument(
        "--shading-strength", type=float, default=None,
        help="AsÄ±l fotoÄŸrafÄ±n Ä±ÅŸÄ±k-gÃ¶lge kabartmasÄ±nÄ±n yeni yÃ¼ze aktarÄ±lma gÃ¼cÃ¼ (0 kapalÄ±, varsayÄ±lan 0.35, en Ã§ok 1.2)."
    )
    parser.add_argument(
        "--enable-face-enhancer",
        action="store_true",
        help="Ä°steÄŸe baÄŸlÄ± GFPGAN 1.4 son aÅŸamasÄ±nÄ± %25 karÄ±ÅŸÄ±mla etkinleÅŸtir."
    )
    parser.add_argument(
        "--enable-pairing",
        action="store_true",
        help="Telefon/PWA eriÅŸimi iÃ§in eÅŸleÅŸtirme kodunu zorunlu kÄ±l (HTTPS tÃ¼nel ile kullanÄ±lÄ±r)."
    )
    parser.add_argument(
        "--pairing-token-file", type=Path, default=Path.home() / ".retake" / "pairing-token",
        help="EÅŸleÅŸtirme kodunun saklandÄ±ÄŸÄ± dosya; yoksa oluÅŸturulur."
    )
    parser.add_argument("--origin", action="append", default=[], help="Ä°zin verilen web origin'i; birden Ã§ok kez verilebilir.")
    parser.add_argument(
        "--realism-python", type=Path, default=None,
        help="FotoÄŸraf gerÃ§ekÃ§iliÄŸi ortamÄ±nÄ±n Python'u (tools/install_realism.ps1 kurar)."
    )
    parser.add_argument(
        "--realism-models", type=Path, default=None,
        help="FotoÄŸraf gerÃ§ekÃ§iliÄŸi modellerinin klasÃ¶rÃ¼ (tools/install_realism.ps1 indirir)."
    )
    return parser.parse_args()


def load_pairing_token(path: Path) -> str:
    """Read the pairing code, creating a random one on first use."""
    path = path.expanduser()
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if len(token) >= 16:
            return token
    token = secrets.token_urlsafe(18)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return token


def main() -> int:
    args = parse_args()
    facefusion_dir = args.facefusion_dir.expanduser().resolve()
    if not (facefusion_dir / "facefusion.py").is_file():
        print(f"FaceFusion kurulumu bulunamadÄ±: {facefusion_dir / 'facefusion.py'}", file=sys.stderr)
        return 2
    if args.host not in {"127.0.0.1", "localhost"}:
        print("GÃ¼venlik nedeniyle servis yalnÄ±zca loopback adresinde Ã§alÄ±ÅŸabilir.", file=sys.stderr)
        return 2

    origins = set(args.origin) | {
        "https://retake06-ua.web.app",
        "https://retake06-ua.firebaseapp.com",
        "https://retake06.web.app",
        "http://localhost:4173",
        "http://127.0.0.1:4173"
    }
    server = ThreadingHTTPServer((args.host, args.port), RetakeFaceFusionHandler)
    server.allowed_origins = origins  # type: ignore[attr-defined]
    server.facefusion_dir = facefusion_dir  # type: ignore[attr-defined]
    server.face_swapper_pixel_boost = args.face_swapper_pixel_boost  # type: ignore[attr-defined]
    server.face_enhancer_enabled = args.enable_face_enhancer  # type: ignore[attr-defined]
    server.pairing_token = load_pairing_token(args.pairing_token_file) if args.enable_pairing else ""  # type: ignore[attr-defined]
    server.identity_cache = IdentityReportCache()  # type: ignore[attr-defined]
    server.pass_cache = PassCache()  # type: ignore[attr-defined]
    server.stage_timings = StageTimings()  # type: ignore[attr-defined]
    if args.shading_strength is not None and not 0.0 <= args.shading_strength <= 1.2:
        print("--shading-strength 0 ile 1.2 arasÄ±nda olmalÄ±.", file=sys.stderr)
        return 2
    server.shading_strength = args.shading_strength  # type: ignore[attr-defined]
    server.realism = find_realism(args.realism_python, args.realism_models)  # type: ignore[attr-defined]
    if server.realism:  # type: ignore[attr-defined]
        print(f"FotoÄŸraf gerÃ§ekÃ§iliÄŸi AÃ‡IK (difÃ¼zyon + kimlik yÃ¶nlendirmesi): {args.realism_models}", flush=True)
    elif args.realism_python or args.realism_models:
        print(
            "FotoÄŸraf gerÃ§ekÃ§iliÄŸi kullanÄ±lamÄ±yor: ortam veya modeller eksik. "
            "Kurmak iÃ§in: .\\tools\\install_realism.ps1",
            flush=True
        )
    print(
        f"Retake kiÅŸisel yÃ¼z profili servisi: http://{args.host}:{args.port} "
        f"(profile={HYBRID_PROFILE_VERSION}, pixel boost={args.face_swapper_pixel_boost}, "
        f"GFPGAN={'on' if args.enable_face_enhancer else 'off'})",
        flush=True
    )
    print("Servis yalnÄ±zca loopback adresini dinler.", flush=True)
    if server.pairing_token:  # type: ignore[attr-defined]
        print(
            "Telefon/PWA eÅŸleÅŸtirmesi AÃ‡IK. Bu bilgisayara bir HTTPS tÃ¼nel aÃ§ "
            f"(Ã¶r. cloudflared tunnel --url http://127.0.0.1:{args.port}) ve Retake'te "
            "'Motor baÄŸlantÄ±sÄ±' bÃ¶lÃ¼mÃ¼ne tÃ¼nel adresi ile ÅŸu kodu gir:",
            flush=True
        )
        print(f"    EÅLEÅTÄ°RME KODU: {server.pairing_token}", flush=True)  # type: ignore[attr-defined]
        print("Kodu bilen herkes bu motoru kullanabilir; paylaÅŸma.", flush=True)
    print("Ä°zinli web origin'leri: " + ", ".join(sorted(origins)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nRetake yerel yÃ¼z profili servisi kapatÄ±lÄ±yor.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

