from __future__ import annotations

import argparse
import gc
import os
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
    from .nim_photo_edit import configured_editors, candidate_from_editor
    from .ultra_calibration import CalibrationStore, bank_key, schedule_signature
    from .nvidia_judge import (
        MAX_CANDIDATES as JUDGE_MAX_CANDIDATES, build_judge, face_crop_jpeg, judge_headers,
        judge_status, judged_winner
    )
    from .facefusion_quality import PIXEL_BOOST_FALLBACKS, is_near_noop, measure_final_face_edit
    from .retake_cache import IdentityReportCache, PassCache, StageTimings, references_digest
    from .identity_scorer import PersistentIdentityScorer
    from .identity_selection import plan_identity_source_sets, select_texture_reference, choose_identity_pool, strict_self_test_pools
    from .photo_realism_gate import check_render_file
    from .ultra_quality import check_photo_detail, self_fidelity, regional_edit_statistics, choose_detail_safe_tie, select_verified_regional_candidate, check_detail_gate, balance_with_original, check_reinforcement_detail, select_sufficient_realism, reinforcement_wanted
    from .retake_output_transport import prepare_inference_target, encode_preblended_result, prepare_small_face_inference
    from .retake_rim_cleanup import inspect_dark_rim_file, repair_dark_rim_file
    from .neck_harmonization import neck_candidate_file, neck_delivery_evidence
    from .portrait_evidence import reference_evidence_file, format_evidence, regional_tie_indices, choose_regional_evidence
    from .fine_features import refine_fine_features_file, accept_fine_refinement, delivered_fine_report, describe_report as describe_fine_report
    from .ultra_mode import (
        ULTRA_VARIATIONS, ULTRA_MIN_SECONDS, ULTRA_SEARCH_BUDGET_SECONDS, ULTRA_REALISM_TIE_LOSS,
        prefer_ultra_candidate, score_from_headers, should_continue_ultra, with_variation,
        identity_acceptable, identity_sufficiency, is_identity_sufficient, realism_order, sufficiency_from_level,
        identity_aspiration, identity_gain, needs_extra_finalist, choose_extra_variations, defer_flux_retry,
        ULTRA_EXTRA_START_BUDGET_SECONDS, ULTRA_DEPTHS, ULTRA_DEFAULT_DEPTH, ultra_depth, depth_summary,
        RefinePlanner, with_refinement, normalize_refinement, finalist_specs, REALISM_SEED_BASE,
        REALISM_SEED_STRIDE, format_ultra_timing, reference_appearance_from_headers, fine_quality_from_headers
    )
except ImportError:
    from nim_photo_edit import configured_editors, candidate_from_editor
    from ultra_calibration import CalibrationStore, bank_key, schedule_signature
    from nvidia_judge import (
        MAX_CANDIDATES as JUDGE_MAX_CANDIDATES, build_judge, face_crop_jpeg, judge_headers,
        judge_status, judged_winner
    )
    from facefusion_quality import PIXEL_BOOST_FALLBACKS, is_near_noop, measure_final_face_edit
    from retake_cache import IdentityReportCache, PassCache, StageTimings, references_digest
    from identity_scorer import PersistentIdentityScorer
    from identity_selection import plan_identity_source_sets, select_texture_reference, choose_identity_pool, strict_self_test_pools
    from photo_realism_gate import check_render_file
    from ultra_quality import check_photo_detail, self_fidelity, regional_edit_statistics, choose_detail_safe_tie, select_verified_regional_candidate, check_detail_gate, balance_with_original, check_reinforcement_detail, select_sufficient_realism, reinforcement_wanted
    from retake_output_transport import prepare_inference_target, encode_preblended_result, prepare_small_face_inference
    from retake_rim_cleanup import inspect_dark_rim_file, repair_dark_rim_file
    from neck_harmonization import neck_candidate_file, neck_delivery_evidence
    from portrait_evidence import reference_evidence_file, format_evidence, regional_tie_indices, choose_regional_evidence
    from fine_features import refine_fine_features_file, accept_fine_refinement, delivered_fine_report, describe_report as describe_fine_report
    from ultra_mode import (
        ULTRA_VARIATIONS, ULTRA_MIN_SECONDS, ULTRA_SEARCH_BUDGET_SECONDS, ULTRA_REALISM_TIE_LOSS,
        prefer_ultra_candidate, score_from_headers, should_continue_ultra, with_variation,
        identity_acceptable, identity_sufficiency, is_identity_sufficient, realism_order, sufficiency_from_level,
        identity_aspiration, identity_gain, needs_extra_finalist, choose_extra_variations, defer_flux_retry,
        ULTRA_EXTRA_START_BUDGET_SECONDS, ULTRA_DEPTHS, ULTRA_DEFAULT_DEPTH, ultra_depth, depth_summary,
        RefinePlanner, with_refinement, normalize_refinement, finalist_specs, REALISM_SEED_BASE,
        REALISM_SEED_STRIDE, format_ultra_timing, reference_appearance_from_headers, fine_quality_from_headers
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
JOB_RESULT_TTL_SECONDS = 30 * 60


def subprocess_options() -> dict[str, Any]:
    # A Pythonw-hosted Windows bridge must not flash console windows for
    # each of the FaceFusion/ArcFace/diffusion child processes.
    if sys.platform == 'win32' and os.environ.get('RETAKE_BACKGROUND') == '1':
        return {'creationflags': subprocess.CREATE_NO_WINDOW}
    return {}
JOB_STAGES = (
    "25 referans ArcFace ile karşılaştırılıyor",
    "Kimlik geçişi (AlphaFace)",
    "Kimlik pekiştirme (InSwapper)",
    "Görünüm ve sakal geçişi (UniFace)",
    "Dikişsiz birleştirme ve kafa oranı",
    "Fotoğraf gerçekçiliği (difüzyon)",
    "Kimlik benzerliği ölçülüyor",
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
IDENTITY_REPORT_VERSION = "centroid-v2"
PAIRING_HEADER = "X-Retake-Token"
IDENTITY_MODEL = "alphaface_256"
# A second identity pass over the first one's result. On the user's photos
# ArcFace similarity to the references went 0.803 (AlphaFace) -> 0.832
# (AlphaFace + InSwapper); AlphaFace twice gave 0.816. It is kept only when
# the measurement confirms it for the photo at hand.
REINFORCE_MODEL = "inswapper_128"
APPEARANCE_MODEL = "uniface_256"
HYBRID_PROFILE_VERSION = "retake_hybrid_25_reference_ultra_v22"
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
    raise ValueError("Yalnızca geçerli PNG, JPEG veya WebP görselleri kabul edilir.")


MAX_JAW_CONTROLS = 32
MAX_JAW_ANCHORS = 16


def _point(value: Any, size: dict[str, int], margin: float) -> list[float]:
    if (
        not isinstance(value, list) or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value)
        or not -margin <= value[0] <= size["width"] + margin
        or not -margin <= value[1] <= size["height"] + margin
    ):
        raise ValueError("Çene hattı noktaları hedef kırpımın içinde sayı çiftleri olmalı.")
    return [float(value[0]), float(value[1])]


def _validate_jaw_warp(value: Any, size: dict[str, int], bounds: dict[str, float]) -> dict[str, Any] | None:
    """Optional jawline controls from the browser (target-crop pixel space)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Çene hattı verisi nesne olmalı.")
    controls, anchors = value.get("controls"), value.get("anchors", [])
    if not isinstance(controls, list) or not 1 <= len(controls) <= MAX_JAW_CONTROLS:
        raise ValueError(f"Çene hattı 1 ile {MAX_JAW_CONTROLS} arasında kontrol noktası içermeli.")
    if not isinstance(anchors, list) or len(anchors) > MAX_JAW_ANCHORS:
        raise ValueError(f"Çene hattı en fazla {MAX_JAW_ANCHORS} sabit nokta içerebilir.")
    margin = 0.25 * max(size["width"], size["height"])
    limit = bounds["width"] * 0.2
    checked = []
    for control in controls:
        if not isinstance(control, dict):
            raise ValueError("Çene hattı kontrol noktası geçersiz.")
        start, end = _point(control.get("from"), size, margin), _point(control.get("to"), size, margin)
        if math.hypot(end[0] - start[0], end[1] - start[1]) > limit:
            raise ValueError("Çene hattı kaydırması izin verilen sınırı aşıyor.")
        checked.append({"from": start, "to": end})
    return {"controls": checked, "anchors": [_point(anchor, size, margin) for anchor in anchors]}


MAX_REAL_FACE_CANDIDATES = 3


def _validate_real_face(value: Any, size: dict[str, int]) -> dict[str, Any] | None:
    """Optional real-photo face: the target crop's 478-point mesh and, for each
    candidate reference photo (best pose match first), its mesh."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Gerçek fotoğraf verisi nesne olmalı.")
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
            raise ValueError(f"1–{MAX_REAL_FACE_CANDIDATES} aday olmalı.")
        meshes = [
            validate_mesh(item.get("landmarks") if isinstance(item, dict) else None, 4096, 4096, 0).tolist()
            for item in references
        ]
    except (ValueError, TypeError) as error:
        raise ValueError(f"Gerçek fotoğraf yüz ağı geçersiz: {error}") from error
    mode = value.get("mode", "full")
    if mode not in ("full", "lower", "both"):
        raise ValueError("Gerçek fotoğraf aktarım modu full, lower veya both olmalı.")
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
        raise ValueError(f"Hedef yüz ağı geçersiz: {error}") from error
    return points.tolist()


def _validate_metadata(metadata: Any, reference_count: int) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Profil metadata alanı geçerli JSON nesnesi olmalı.")
    weights = metadata.get("referenceWeights")
    if not isinstance(weights, list) or len(weights) != reference_count:
        raise ValueError("Referans kalite ağırlıkları, gönderilen referans sayısıyla eşleşmiyor.")
    normalized_weights: list[int] = []
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or int(weight) != weight or not 1 <= int(weight) <= 3:
            raise ValueError("Her referans ağırlığı 1 ile 3 arasında tam sayı olmalı.")
        normalized_weights.append(int(weight))

    face_bounds = metadata.get("faceBounds")
    if not isinstance(face_bounds, dict):
        raise ValueError("Hedef yüz sınırları eksik.")
    required = ("x", "y", "width", "height")
    if any(isinstance(face_bounds.get(key), bool) or not isinstance(face_bounds.get(key), (int, float)) for key in required):
        raise ValueError("Hedef yüz sınırları sayısal olmalı.")
    bounds = {key: float(face_bounds[key]) for key in required}
    if bounds["width"] <= 0 or bounds["height"] <= 0:
        raise ValueError("Hedef yüz sınırları pozitif genişlik ve yüksekliğe sahip olmalı.")
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
            f"Hedef kırpım boyutu {MIN_TARGET_EDGE} px ile "
            f"{MAX_TARGET_WIDTH}×{MAX_TARGET_HEIGHT} px arasında tam sayı olmalı."
        )
    if (
        bounds["x"] < 0 or bounds["y"] < 0
        or bounds["x"] + bounds["width"] > target_size["width"]
        or bounds["y"] + bounds["height"] > target_size["height"]
    ):
        raise ValueError("Hedef yüz sınırları, gönderilen portre kırpımının dışında.")
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
        raise ValueError(f"Kafa oranı ölçekleri {HEAD_FIT_MIN} ile {HEAD_FIT_MAX} arasında sayı olmalı.")
    variation = metadata.get("variation", 0)
    if isinstance(variation, bool) or not isinstance(variation, int) or not 0 <= variation < len(FACE_SWAP_VARIATIONS):
        raise ValueError(f"Görünüm alternatifi 0 ile {len(FACE_SWAP_VARIATIONS) - 1} arasında tam sayı olmalı.")
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
            raise ValueError("Referans poz mesafeleri 25 adet geçerli sayı veya null içermeli.")
    exact_index = metadata.get('exactReferenceIndex')
    if exact_index is not None and (type(exact_index) is not int or not 0 <= exact_index < reference_count):
        raise ValueError('Birebir referans indeksi geçersiz.')
    # Benchmark self-test: strict leave-one-out (needs the exact reference).
    self_test = metadata.get('selfTest', False)
    if type(self_test) is not bool or (self_test and exact_index is None):
        raise ValueError('Öz-test yalnız referans bankasındaki bir hedefle ve true/false olarak istenebilir.')
    # Per-job opt-in: finalist face crops may go to NVIDIA only via Hafize.
    nvidia_judge = metadata.get('nvidiaJudge', False)
    if type(nvidia_judge) is not bool:
        raise ValueError('NVIDIA jüri onayı true/false olmalı.')
    neck_harmonize = metadata.get('neckHarmonize', False)
    if type(neck_harmonize) is not bool:
        raise ValueError('Boyun uyumu seçeneği true/false olmalı.')
    # How long Ultra may search. Old browsers send none: seven minutes.
    quality_depth = metadata.get('qualityDepth', ULTRA_DEFAULT_DEPTH)
    if not isinstance(quality_depth, str) or quality_depth not in ULTRA_DEPTHS:
        raise ValueError('Kalite derinliği standard, deep veya max olmalı.')
    return {
        'qualityDepth': quality_depth,
        'selfTest': self_test,
        'neckHarmonize': neck_harmonize,
        'nvidiaJudge': nvidia_judge,
        'exactReferenceIndex': exact_index,
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
        raise ValueError(f"Retake hibrit profil işlemi için tam olarak {MAX_REFERENCES} referans fotoğrafı gerekli.")


# v25: one ArcFace worker with loaded models per job (see identity_scorer).
# None (tests, RETAKE_PERSISTENT_ARCFACE=off) means a process per measurement.
IDENTITY_SCORER: "PersistentIdentityScorer | None" = None


def release_identity_scorer() -> None:
    """Free the persistent scorer's GPU memory (job end, before diffusion)."""
    scorer = IDENTITY_SCORER
    if scorer is not None:
        try:
            scorer.close()
        except Exception as error:
            print(f"[ArcFace] Kalıcı ölçüm sunucusu kapatılamadı: {error}", flush=True)


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
    scorer = IDENTITY_SCORER
    if scorer is not None and probe_paths and Path(scorer.facefusion_dir) == Path(facefusion_dir):
        values = scorer.score(centroid, probe_paths)
        if values is not None:
            return values
        # Any persistent-path problem: measure this once the original way.
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
            encoding="utf-8", errors="replace", timeout=600, check=False, **subprocess_options()
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[ArcFace] Benzerlik ölçülemedi: {error}", flush=True)
        return empty
    marker = "RETAKE_IDENTITY_REPORT="
    lines = [line[len(marker):] for line in completed.stdout.splitlines() if line.startswith(marker)]
    try:
        report = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        report = {}
    values = report.get("similarities") if report.get("ok") else None
    if not isinstance(values, list) or len(values) != len(probe_paths):
        print("[ArcFace] Benzerlik ölçülemedi: " + tail(completed.stderr + "\n" + completed.stdout, 600), flush=True)
        return empty
    return [float(value) if type(value) in (int, float) and math.isfinite(value) and -1 <= value <= 1 else None for value in values]


def verify_reference_identity(facefusion_dir: Path, reference_paths: list[Path]) -> dict[str, object]:
    worker_path = Path(__file__).with_name("facefusion_identity_worker.py")
    if not worker_path.is_file():
        raise IdentityVerificationError("ArcFace doğrulama worker dosyası bulunamadı.")
    command = [
        sys.executable, str(worker_path),
        "--facefusion-dir", str(facefusion_dir),
        "--reference-paths", *[str(path) for path in reference_paths]
    ]
    print(
        f"ArcFace ön kontrolü başlıyor: {len(reference_paths)} referans; "
        "yüz algılama ve ArcFace ONNX ağırlıkları kontrol edilecek.",
        flush=True
    )
    try:
        completed = subprocess.run(
            command, cwd=str(facefusion_dir), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=1800, check=False, **subprocess_options()
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
            "ArcFace 25 referans ön kontrolü 30 dakika içinde tamamlanamadı. " + detail
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
            f"ArcFace worker rapor üretemedi (çıkış kodu {completed.returncode}). {detail}"
        )
    try:
        report = json.loads(report_lines[-1])
    except json.JSONDecodeError as error:
        raise IdentityVerificationError("ArcFace doğrulama raporu geçerli JSON değil.") from error

    if completed.returncode != 0 or not report.get("ok"):
        detail = str(report.get("error") or "ArcFace worker başarısız sonuç döndürdü.")
        logs = tail(completed.stderr + "\n" + completed.stdout)
        raise IdentityVerificationError(
            f"{detail} (worker çıkış kodu {completed.returncode}). {logs}"
        )
    if report.get("hard_conflict"):
        raise IdentityConflictError(
            "ArcFace, 25 referans arasında ortak bir kimlik profili bulamadı "
            f"(medyan cosine benzerliği: {report.get('median_similarity')}). "
            "Farklı kişiye ait veya ağır bozulmuş fotoğrafları kontrol edip tekrar deneyin."
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
        raise ValueError("İstek multipart/form-data biçiminde olmalı.")

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
                raise ValueError("Profil metadata JSON olarak okunamadı.") from error
            continue
        if field_name not in {"target", "reference", "appearance_profile", "real_reference"}:
            continue

        extension = image_extension(part.get_content_type(), payload)
        if not payload or len(payload) > MAX_FILE_BYTES:
            raise ValueError("Her görsel boş olmamalı ve 25 MB sınırını aşmamalı.")
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
        raise ValueError("Tam olarak bir hedef fotoğraf gönderilmelidir.")
    if len(appearance_files) != 1:
        raise ValueError("25 referanstan üretilen tek bir görünüm profili gönderilmelidir.")
    validate_reference_count(len(reference_files))

    metadata = _validate_metadata(metadata_value, len(reference_files))
    if len(real_reference_files) > MAX_REAL_FACE_CANDIDATES:
        raise ValueError(f"En fazla {MAX_REAL_FACE_CANDIDATES} gerçek fotoğraf adayı gönderilebilir.")
    real_face = metadata.get("realFace")
    if real_face and len(real_reference_files) < len(real_face["references"]):
        raise ValueError("Gerçek fotoğraf yüz ağı gönderildi ama fotoğrafı eksik.")
    if real_face and len(real_reference_files) != len(real_face["references"]):
        raise ValueError("Gerçek fotoğraf sayısı ile yüz ağı sayısı eşleşmiyor.")
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
    return value[-limit:] if value else "FaceFusion ayrıntılı hata kaydı üretmedi."


def supported_facefusion_weight(value: float) -> float:
    """Normalize to FaceFusion's exact 0.05 CLI steps and share cache key."""
    try:
        amount = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("FaceFusion ağırlığı sayı olmalı.") from error
    if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise ValueError("FaceFusion ağırlığı 0 ile 1 arasında olmalı.")
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
        check=False, **subprocess_options()
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
    released_scorer = False
    index = 0
    while index < len(attempts):
        pixel_boost = attempts[index]
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
        scorer = IDENTITY_SCORER
        if is_memory_error and not released_scorer and scorer is not None and scorer.describe().get("running"):
            # Free the ArcFace server's GPU memory before giving up quality.
            released_scorer = True
            release_identity_scorer()
            print(f"{model}: {pixel_boost} bellek hatası; ArcFace sunucusu kapatılıp aynı ayarla yeniden deneniyor.", flush=True)
            continue
        has_fallback = index + 1 < len(attempts)
        if not is_memory_error or not has_fallback:
            raise RuntimeError(f"{model} işlemi başarısız ({pixel_boost}). {detail}")
        next_boost = attempts[index + 1]
        print(f"{model}: {pixel_boost} bellek hatası; {next_boost} ayarına düşülüyor.", flush=True)
        index += 1

    raise RuntimeError(f"{model} işlemi tamamlanamadı. {last_detail}")


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
        encoding="utf-8", errors="replace", timeout=900, check=False, **subprocess_options()
    )
    if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(
            "GFPGAN isteğe bağlı yüz iyileştirme aşaması başarısız. "
            + tail(completed.stderr + "\n" + completed.stdout)
        )


def blend_enhanced_hybrid_output(hybrid_path: Path, enhanced_path: Path, output_path: Path) -> None:
    try:
        import cv2
        import numpy as np
    except Exception as error:
        raise RuntimeError(f"GFPGAN harmanlaması için OpenCV/NumPy yüklenemedi: {error}") from error
    hybrid = cv2.imread(str(hybrid_path), cv2.IMREAD_UNCHANGED)
    enhanced = cv2.imread(str(enhanced_path), cv2.IMREAD_COLOR)
    if hybrid is None or enhanced is None or hybrid.ndim != 3 or hybrid.shape[2] != 4:
        raise RuntimeError("GFPGAN çıktısı ve alfa maskeli hibrit görüntü okunamadı.")
    enhanced = crop_to_size(enhanced, {"width": hybrid.shape[1], "height": hybrid.shape[0]})
    if hybrid.shape[:2] != enhanced.shape[:2]:
        raise RuntimeError("GFPGAN ve hibrit yüz çıktı boyutları eşleşmiyor.")
    blend = FACE_ENHANCER_BLEND / 100.0
    mixed = (
        hybrid[:, :, :3].astype(np.float32) * (1.0 - blend)
        + enhanced.astype(np.float32) * blend
    )
    hybrid[:, :, :3] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    # Retain the exact pre-existing semantic/change matte.
    if not cv2.imwrite(str(output_path), hybrid):
        raise RuntimeError("GFPGAN sonrası alfa maskeli hibrit sonuç yazılamadı.")


def pad_image_to_even(path: Path) -> tuple[int, int]:
    """FaceFusion rounds every output to even width/height (vision.normalize_resolution),
    rescaling odd-sized targets by a pixel. Pad odd targets by one replicated
    edge pixel instead, so outputs line up 1:1 with the target, and return the
    original size to crop back to."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Hedef kırpım okunamadı.")
    height, width = image.shape[:2]
    pad_right, pad_bottom = width % 2, height % 2
    if pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
        if not cv2.imwrite(str(path), padded):
            raise RuntimeError("Hedef kırpım çift boyuta tamamlanamadı.")
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
        raise RuntimeError("Çene hattı düzeltmesi için alfa maskeli PNG okunamadı.")
    color, alpha = warp_jawline(
        image[:, :, :3].copy(), image[:, :, 3].copy(),
        jaw_warp["controls"], jaw_warp["anchors"], face_width, scale
    )
    if not cv2.imwrite(str(path), cv2.merge((*cv2.split(color), alpha))):
        raise RuntimeError("Çene hattı düzeltmesi uygulanmış PNG yazılamadı.")


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
        raise RuntimeError(f"Kafa oranı düzeltmesi için OpenCV/NumPy yüklenemedi: {error}") from error

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise RuntimeError("Kafa oranı düzeltmesi için alfa maskeli hibrit PNG okunamadı.")
    color, alpha = fit_head_geometry(
        image[:, :, :3].copy(), image[:, :, 3].copy(), face_bounds, scale_x, scale_y
    )
    adjusted = cv2.merge((*cv2.split(color), alpha))
    if not cv2.imwrite(str(output_path), adjusted):
        raise RuntimeError("Kafa oranı düzeltmesi uygulanmış PNG yazılamadı.")


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
        print(f"Sakal profili ölçülemedi: {error}", flush=True)
        return None
    print(f"Kişinin sakal/yanak ton farkı (Lab): {', '.join(f'{v:.1f}' for v in offset)}.", flush=True)
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
        if names[item[1]].find("referans alt yüz") >= 0
        and item[0] >= value - 0.002
    ]
    # User-requested complete reconstruction: a true reference image may
    # win only after the same border threshold and within a strict ArcFace
    # near-tie. Full-face isn't forced on incompatible pose/light/expression.
    matched_full = [
        item for item in scored
        if "tam referans yüzü" in names[item[1]]
        and item[0] >= max(score for score, _, _ in scored) - 0.002
    ]
    if matched_full:
        value, path, border = max(matched_full, key=lambda item: item[0])
    elif matched_lower:
        value, path, border = max(matched_lower, key=lambda item: item[0])
    listed = ", ".join(
        f"{names[item[0]]}: {'—' if score is None else f'{score:.3f}'} (kenar {item[1]:.2f})"
        for item, score in zip(allowed, scores)
    )
    print(f"Kimlik öncelikli birleştirme: {listed}; seçilen benzerlik {value:.3f}.", flush=True)
    return path


def run_realism(
    realism: dict[str, Any],
    final_output: Path,
    target_path: Path,
    face_bounds: dict[str, float],
    centroid: list[float],
    temp_dir: Path,
    landmarks: list[list[float]] | None = None,
    seed_base: int | None = None
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
    if isinstance(seed_base, int) and not isinstance(seed_base, bool) and seed_base != REALISM_SEED_BASE:
        # Another finalist: different diffusion seeds, same plans and gates.
        command += ["--seed-base", str(seed_base)]
    # Diffusion needs the GPU memory the ArcFace server holds; it restarts
    # on the next measurement.
    release_identity_scorer()
    try:
        completed = subprocess.run(
            command, cwd=str(worker.parent), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=REALISM_TIMEOUT_SECONDS, check=False, **subprocess_options()
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[Gerçekçilik] Çalıştırılamadı: {error}", flush=True)
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
        print(f"[Gerçekçilik] Başarısız (kod {completed.returncode}): {' | '.join(detail)}", flush=True)
        return []
    print(
        f"[Gerçekçilik] {len(report['outputs'])} sürüm üretildi ({report.get('device')}, "
        f"{'kişisel model' if report.get('personal') else 'genel model + kimlik'}, "
        f"yükleme {report.get('load_seconds')} sn, işleme {report.get('render_seconds')} sn).",
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
        raise RuntimeError("Gerçekçilik sonucu hibrit PNG ile birleştirilemedi.")
    changed = (np.max(cv2.absdiff(render, final[:, :, :3]), axis=2) >= 2).astype(np.float32)
    changed = cv2.GaussianBlur(cv2.dilate(changed, np.ones((5, 5), np.uint8)), (0, 0), 2.0) * 255.0
    alpha = np.maximum(final[:, :, 3].astype(np.float32), changed)
    merged = cv2.merge((*cv2.split(render), np.clip(np.rint(alpha), 0, 255).astype(np.uint8)))
    if not cv2.imwrite(str(final_output), merged):
        raise RuntimeError("Gerçekçilik sonucu yazılamadı.")


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
            raise ValueError("gerçek fotoğraf okunamadı")
        if real_face.get("mode") == "lower":
            transplanted, chosen, fits = choose_and_transfer_lower_face(
                result, target, candidates,
                np.asarray(real_face["targetLandmarks"], np.float32), face_width,
                # The rescue path can try a gentle real-beard transfer
                # over the stronger identity-only base, rather than losing
                # global facial identity to an all-or-nothing transplant.
                strength=0.94 * min(1.0, max(0.05, float(real_face.get("strength", 1.0)))),
                moustache_scale=real_face.get("moustacheScale",1.),
                jaw_scale=real_face.get("jawScale",1.),
                alternate=real_face.get("alternateLower",False)
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
            f"Gerçek fotoğraf yüzü uygulandı: {len(candidates)} aday bükülüp karşılaştırıldı "
            f"(uyum {scores}), {chosen + 1}. aday seçildi.",
            flush=True
        )
        return transplanted, True
    except Exception as error:  # informative only; keep the AI swap
        print(f"Gerçek fotoğraf yüzü uygulanamadı, yapay zekâ sonucu kullanılıyor: {error}", flush=True)
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
        raise RuntimeError(f"Alt yüz harmanlama için OpenCV/NumPy yüklenemedi: {error}") from error

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
        raise RuntimeError(f"Hedef, kimlik ve görünüm geçişlerinin çıktı ölçüleri eşleşmiyor ({sizes}).")

    height, width = identity.shape[:2]
    if expected_size is not None and (width, height) != (expected_size["width"], expected_size["height"]):
        raise ValueError("Hedef kırpımın gerçek boyutu bildirilen boyutla eşleşmiyor.")
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
    if real_face is None and is_near_noop(changed_ratio, mean_delta, p95_delta):
        raise NoOpSwapError(
            "FaceFusion iki geçişte de hedef yüze anlamlı bir değişiklik uygulamadı "
            f"(değişen piksel oranı %{changed_ratio * 100:.3f}, ortalama fark {mean_delta:.3f}). "
            "İşlem başarılı gibi gösterilmedi; kaynak/hedef yüz algılamasını kontrol edin."
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
        raise RuntimeError("Hibrit yüz sonucu ve semantik alfa maskesi diske yazılamadı.")
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
    import cv2

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
        # Only the model input is padded. Every quality gate and final RGB
        # composite must use the native-size, untouched target.
        inference_target = prepare_inference_target(target_path, metadata["targetSize"], temp_dir)
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
            print("ArcFace raporu önbellekten alındı (referanslar değişmedi).", flush=True)
        outlier_indices = [int(index) for index in identity_report.get("outlier_indices", [])]
        weights: list[int] = list(metadata["referenceWeights"])
        for outlier_index in outlier_indices:
            if 0 <= outlier_index < len(weights):
                weights[outlier_index] = 1
        outlier_names = identity_report.get("outlier_files", [])
        print(
            "ArcFace 25 referans kontrolü: "
            f"medyan cosine={identity_report.get('median_similarity')}; "
            f"inceleme gerektiren fotoğraf={outlier_names or 'yok'}. "
            "Aykırı referanslar çıkarılmadı; kimlik ağırlıkları 1 oya indirildi.",
            flush=True
        )
        # All 25 photographs must be checked, but averaging all of their
        # embeddings can wash out distinctive facial traits. Create bounded
        # alternative pools: full / most self-consistent / closest target pose.
        # Actual generated output is ArcFace-scored before selecting a pool.
        source_plans = plan_identity_source_sets(
            weights, identity_report, metadata.get("poseDistances")
        )
        exact_index = metadata.get('exactReferenceIndex')
        if type(exact_index) is int and 0 <= exact_index < len(reference_paths):
            # All 25 still build the verifier. Add an independent 24-photo
            # alternative so a duplicate target does not dominate the search.
            independent = tuple(i for i in source_plans[0][1] if i != exact_index)
            source_plans.insert(1, ('independent-24', independent))
            if metadata.get('selfTest') is True:
                # Benchmark: the answer must not be among the sources.
                source_plans = strict_self_test_pools(source_plans, exact_index)
        weighted_identity_paths = [reference_paths[i] for i in source_plans[0][1]]
        identity_output = temp_dir / "retake-identity-alpha.png"
        appearance_output = temp_dir / "retake-appearance-uniface.png"
        hybrid_output = temp_dir / "retake-hybrid-profile.png"
        enhanced_output = temp_dir / "retake-gfpgan.png"
        final_output = temp_dir / "retake-final.png"
        preferred_boost = pixel_boost
        variation = metadata["variation"]
        inference_target, input_preparation = prepare_small_face_inference(inference_target,metadata["faceBounds"],variation,temp_dir)
        # Ultra refinement may step the winner's continuous settings; only
        # the engine sets this key (client metadata is whitelisted).
        refined_params = normalize_refinement(metadata.get("variationParams"))
        face_swapper_weight, detail_transfer_strength, head_fit_strength = (
            refined_params or FACE_SWAP_VARIATIONS[variation]
        )
        applied_params = (face_swapper_weight, detail_transfer_strength, head_fit_strength)
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
            f"Profil hazır: {len(reference_paths)} referans, ağırlıklı kimlik oyu={len(weighted_identity_paths)}, "
            f"görünüm profili={len(reference_paths)} hizalanmış fotoğraf.",
            flush=True
        )
        print(
            f"1/2 kimlik geçişi: {IDENTITY_MODEL}; 2/2 görünüm geçişi: {APPEARANCE_MODEL}; "
            f"hedef=1 fotoğraf; tercih edilen pixel boost={preferred_boost}.",
            flush=True
        )

        target_key_bytes = inference_target.read_bytes()

        def cached_pass(step: int, sources: list[Path], output: Path, model: str, source_image: Path | None = None) -> str:
            image = source_image or inference_target
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
                    print(f"{model}: aynı hedef için önceki geçiş yeniden kullanıldı.", flush=True)
                    return hit[1]
            progress(step)
            used = run_with_memory_fallback(
                facefusion_dir, image, sources, output, model, preferred_boost, face_swapper_weight
            )
            if pass_cache is not None and key is not None:
                pass_cache.put(key, output.read_bytes(), used)
            return used

        identity_boost = cached_pass(1, weighted_identity_paths, identity_output, IDENTITY_MODEL)
        print(f"Kimlik profili geçişi tamamlandı ({identity_boost}).", flush=True)

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
                    print(f"Ek {name} kimlik adayı atlandı: {error}", flush=True)
            if len(candidates) > 1:
                candidate_scores = score_identity(
                    facefusion_dir, centroid, [path for _, path, _, _ in candidates]
                )
                best_index = choose_identity_pool(candidate_scores, variation)
                identity_pool, identity_source, identity_source_paths, identity_boost = candidates[best_index]
                print(
                    "Kaynak havuzu karşılaştırması: "
                    + ", ".join(
                        f"{entry[0]}={'—' if value is None else f'{value:.3f}'}"
                        for entry, value in zip(candidates, candidate_scores)
                    )
                    + f"; seçilen {identity_pool}.",
                    flush=True
                )

        # Identity reinforcement: a second swapper over the selected output.
        reinforced = False
        reinforce_guard = "not-attempted"
        if centroid and face_swapper_weight >= 1.0:
            reinforced_output = temp_dir / "retake-identity-reinforced.png"
            try:
                cached_pass(REINFORCE_STAGE, identity_source_paths, reinforced_output, REINFORCE_MODEL, identity_source)
                single, double = score_identity(facefusion_dir, centroid, [identity_source, reinforced_output])
                wanted, why = reinforcement_wanted(
                    single, double, sufficiency_from_level(identity_report.get("photo_level_similarity"))
                )
                if wanted:
                    # A higher ArcFace from the 128 px swapper is not worth
                    # blurred, ghosted eyes or lips on a large face.
                    keeps_detail, reinforce_guard, reinforce_ratios = check_reinforcement_detail(
                        identity_source, reinforced_output, metadata["faceBounds"]
                    )
                    if keeps_detail:
                        identity_source = reinforced_output
                        reinforced = True
                    else:
                        print(
                            f"Kimlik pekiştirme reddedildi: 128 px sonuç ayrıntı kaybettiriyor ({reinforce_guard}; oranlar={reinforce_ratios}).",
                            flush=True
                        )
                else:
                    # Already at the person's photo level, or no real gain.
                    reinforce_guard = why
                print(
                    "Kimlik pekiştirme: "
                    + ("uygulandı" if reinforced else "kullanılmadı")
                    + (f" (benzerlik {single:.3f} → {double:.3f})" if single is not None and double is not None else ""),
                    flush=True
                )
            except Exception as error:  # informative only; the first pass stays
                reinforce_guard = "error"
                print(f"Kimlik pekiştirme atlandı: {error}", flush=True)

        appearance_boost = cached_pass(3, [appearance_path], appearance_output, APPEARANCE_MODEL)
        print(f"25 fotoğraflı görünüm profili geçişi tamamlandı ({appearance_boost}).", flush=True)

        # The 25-image pixel mean can look waxy: aligned eyes and facial hair
        # rarely land on identical pixels. Try a crisp real reference as an
        # optional UniFace *detail* source, not as a replacement for identity.
        texture_reference_index = select_texture_reference(
            weights, identity_report, metadata.get('poseDistances'), excluded_index=exact_index, variation=variation
        )
        texture_output: Path | None = None
        if texture_reference_index is not None:
            alternative = temp_dir / "retake-appearance-single.png"
            try:
                cached_pass(3, [reference_paths[texture_reference_index]], alternative, APPEARANCE_MODEL)
                texture_output = alternative
            except Exception as error:
                print(f"Tek fotoğraftan doku adayı atlandı: {error}", flush=True)

        progress(4)
        # Identity-first compositing: a few blends from seamless (photo's skin
        # tone pulled in) to faithful (the swap's own skin kept). Those whose
        # border stays invisible are scored with ArcFace; the one closest to
        # the person is used.
        real_face = (
            {**metadata["realFace"], "images": metadata["realReferenceBytes"]}
            if metadata.get("realFace") and metadata.get("realReferenceBytes") else None
        )
        if real_face:
            # Different lower-face hypotheses across existing rounds, not
            # extra expensive full-face model calls.
            if variation in (3,11,19):
                real_face.update(moustacheScale=1.25,jawScale=.8,alternateLower=True)
            elif variation in (7,15,21,23):
                real_face.update(moustacheScale=.8,jawScale=1.25,alternateLower=True)
        user_shading = caches.get("shading")
        beard = person_beard(appearance_path, metadata.get("targetLandmarks"))
        blends: list[tuple[Path, float]] = []
        labels: list[str] = []
        transplanted: dict[Path, str] = {}
        # In selected rounds compare the complete reference face, its
        # beard/jaw-only detail, and model-only identity. Every option starts
        # from the original target; only ArcFace-verified, seamless results
        # are eligible to win. Restrict extra warps to budgeted rounds.
        lower_reference_rounds = (0, 2, 3, 6, 7, 11, 15, 19, 21, 23)
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
                try:
                    used = blend_lower_face(
                        target_path, identity_source, appearance_output, path, metadata["faceBounds"],
                        detail_transfer_strength, metadata["targetSize"], shading, transplant, harmonize, report, beard
                    )
                    if not measure_final_face_edit(target_path, path, metadata["faceBounds"])["meaningful"]:
                        continue
                except NoOpSwapError:
                    # An unchanged neural pass must not prevent the independent
                    # real-reference transplant from being attempted.
                    continue
                if used:
                    transplanted[path] = transplant.get("mode", "full") if transplant else "full"
                blends.append((path, float(report.get("border", 0.0))))
                label = f"ton {harmonize:g}"
                if path in transplanted:
                    label += " + referans alt yüz" if transplanted[path] == "lower" else " + tam referans yüzü"
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
                    print(f"Gerçek referans dokusu adayı atlandı: {error}", flush=True)
                    continue
                if not measure_final_face_edit(target_path, path, metadata["faceBounds"])["meaningful"]:
                    continue
                texture_blends.add(path)
                blends.append((path, float(report.get("border", 0.0))))
                labels.append(f"gerçek referans dokusu {harmonize:g}")
        if not blends:
            raise NoOpSwapError('Yapay zekâ ve gerçek referans adaylarının hiçbirinde yeterli yüz içi değişim oluşmadı.')
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
                f"İsteğe bağlı son aşama: {FACE_ENHANCER_MODEL}, blend={FACE_ENHANCER_BLEND}%.",
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
            print(f"Çene hattı referans kişiye göre düzeltildi (güç {jaw_scale:.2f}).", flush=True)
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
            print(f"Yedek saf kimlik bileşimi oluşturulamadı: {error}", flush=True)
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
                for transfer_strength in (0.12, 0.22, 0.35, 0.65):
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
                                f"[Retake] Güvenli sakal adayı atlandı "
                                f"(kenar {border:.2f}, sınır {max_regional_border:.2f}).", flush=True
                            )
                    except Exception as error:
                        print(f"[Retake] Kimlik üstü sakal nakli atlandı: {error}", flush=True)
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
                                f"[Retake] Güvenli tam yüz adayı atlandı "
                                f"(kenar {border:.2f}, sınır {max_regional_border:.2f}).", flush=True
                            )
                    except Exception as error:
                        print(f"[Retake] Kimlik üstü tam referans dokusu atlandı: {error}", flush=True)
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
                    print(f"[Retake] Kimlik üstü çene uyarlaması atlandı: {error}", flush=True)

        # Photographic realism (optional): re-render the face with a diffusion
        # model steered by the person's ArcFace identity.
        realism = caches.get("realism")
        renders: list[Path] = []
        if realism and centroid:
            progress(REALISM_STAGE)
            renders = run_realism(
                realism, final_output, target_path, metadata["faceBounds"], centroid, temp_dir,
                head_fitted_landmarks(metadata.get("targetLandmarks"), metadata["faceBounds"],
                                      face_width_scale, face_height_scale),
                # Only another finalist passes a seed; the first keeps the defaults.
                **({"seed_base": metadata["realismSeedBase"]} if metadata.get("realismSeedBase") is not None else {})
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
        # Add bounded recovery alternatives only when the base composite
        # destroys native detail. Score each actual new image independently.
        recovery_candidates = []
        for label, candidate_path in rescue_candidates:
            if label not in ("head-fit", "identity-only"):
                continue
            try:
                if check_detail_gate(target_path, candidate_path, metadata["faceBounds"])[0]:
                    continue
                for strength in (.35, .60):
                    repaired = temp_dir / f"photo-balance-{label}-{int(strength * 100)}.png"
                    balance_with_original(target_path, candidate_path, repaired, strength)
                    if check_detail_gate(target_path, repaired, metadata["faceBounds"])[0]:
                        recovery_candidates.append(("photo-balance", repaired))
            except (ValueError, OSError, cv2.error):
                continue
        rescue_candidates.extend(recovery_candidates)
        probes = [path for _, path in rescue_candidates] + [*renders, target_path]
        scores = score_identity(facefusion_dir, centroid, probes)
        # Filter unchanged interiors before selection, not only before blending.
        # This prevents a high ArcFace score on an almost original face winning.
        had_verified_identity = any(value is not None for value in scores[:len(rescue_candidates)])
        if not had_verified_identity:
            raise IdentityVerificationError('Üretilen yüz ArcFace ile doğrulanamadı.')
        for idx, (_, candidate_path) in enumerate(rescue_candidates):
            try:
                if not measure_final_face_edit(target_path, candidate_path, metadata['faceBounds'])['meaningful']:
                    scores[idx] = None
                elif not check_detail_gate(target_path, candidate_path, metadata["faceBounds"])[0]:
                    scores[idx] = None
            except (ValueError, OSError, cv2.error):
                scores[idx] = None
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
        # Only an ArcFace near-tie can be settled by regional photographic
        # detail. Measure the actual delivered pixels, not a metadata flag.
        candidate_detail_scores: list[float | None] = []
        for _, candidate_path in rescue_candidates:
            try:
                detail, _ = check_photo_detail(target_path, candidate_path, metadata["faceBounds"])
            except (ValueError, OSError, cv2.error):
                detail = None
            candidate_detail_scores.append(detail)
        # Preserve the actual highest verified score, not a chain of near-ties.
        previous = chosen_index
        # Existing full/lower reference composites already carry real texture.
        # Do not replace one with a weaker regional rescue merely for its label.
        effective_labels = [
            ("identity-full" if used_full_face else "identity-beard" if used_lower_face else label)
            if label in ("head-fit", "no-head-warp") else label
            for label, _ in rescue_candidates
        ]
        chosen_index = select_verified_regional_candidate(
            candidate_scores, candidate_detail_scores, effective_labels, chosen_index
        )
        # At the person's photo level, keep the candidate with the most real
        # photographic detail rather than the highest cosine.
        chosen_index = select_sufficient_realism(
            candidate_scores, candidate_detail_scores, chosen_index,
            sufficiency_from_level(identity_report.get("photo_level_similarity"))
        )
        # Compare regional appearance for only a few already measured close
        # candidates. No extra GPU passes and no preference for missing data.
        if real_face:
            level = sufficiency_from_level(identity_report.get('photo_level_similarity'))
            indices = regional_tie_indices(candidate_scores,candidate_detail_scores,chosen_index,level)
            if len(indices)>1:
                evidence_scores = {}
                for index in indices:
                    try:
                        _, value = reference_evidence_file(target_path,rescue_candidates[index][1],metadata['faceBounds'],metadata.get('targetLandmarks'),real_face)
                        evidence_scores[index] = value
                    except (ValueError,TypeError,OSError,cv2.error):
                        continue
                chosen_index = choose_regional_evidence(candidate_scores,candidate_detail_scores,chosen_index,evidence_scores,level)
        if candidate_scores[chosen_index] is not None:
            best_score = candidate_scores[chosen_index]
        selected_identity = rescue_candidates[chosen_index][0]
        if chosen_index != previous:
            print(f"[Retake] Verified regional/identity finalist: {selected_identity} (ArcFace={best_score:.4f}).", flush=True)
        if selected_identity not in ('head-fit','no-head-warp'):
            chosen_texture = 'average'
        if selected_identity in ("identity-only", "identity-jaw", "photo-balance"):
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
        if all(value is None for value in candidate_scores):
            raise NoOpSwapError('Hiçbir aday kimlik, anlamlı değişim ve bölgesel fotoğraf ayrıntısı kontrollerinin tamamını geçemedi.')
        if final_score is None and best_score < -0.99:
            raise IdentityVerificationError(
                "Üretilen yüz ArcFace ile doğrulanamadı. Kimlik kontrolü olmadan güvenilir sonuç gösterilmiyor."
            )
        if chosen_index > 0:
            print(
                f"Kimlik koruma: {selected_identity} seçildi "
                f"({final_score if final_score is not None else '—'} → {best_score:.3f}).",
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
                accepted, reason = False, f"kontrol hatası: {error}"
            if accepted:
                accepted, reason, _, _ = check_detail_gate(target_path, render_path, metadata["faceBounds"])
            if accepted:
                acceptable_renders.append(render_path)
                acceptable_scores.append(render_score)
            else:
                print(f"Gerçekçilik adayı elendi ({render_path.name}): {reason}.", flush=True)
        chosen = choose_realism_render(result_similarity, acceptable_renders, acceptable_scores)
        if chosen is not None:
            render_path, render_similarity = chosen
            apply_realism_render(final_output, render_path)
            print(
                f"Fotoğraf gerçekçiliği uygulandı: kimlik benzerliği {result_similarity:.3f} → {render_similarity:.3f}.",
                flush=True
            )
            result_similarity = render_similarity
            used_realism = True
        elif renders:
            listed = ", ".join("—" if value is None else f"{value:.3f}" for value in render_scores)
            print(
                "Fotoğraf gerçekçiliği kullanılmadı: kimlik benzerliği seçilen sonucun altında kaldı "
                f"({listed}; FaceFusion {result_similarity:.3f}).",
                flush=True
            )
        # Optional local NIM finalist candidates. Actual selected pixels, never
        # a model's own claims, must pass identity, texture and edit checks.
        nim_used = ""
        nim_editors = caches.get("nim_editors") or {}
        flux_model = "retake-local/flux2-klein-4b"
        flux_status = "not-attempted" if flux_model in nim_editors else "not-configured"
        flux_delta = None
        flux_detail_delta = None
        flux_strength = None
        if nim_editors and centroid:
            indices = sorted((i for i in range(len(weights)) if i != exact_index and i not in outlier_indices), key=lambda i: (-weights[i], i))
            nim_refs = [reference_paths[i] for i in indices]
            nim_candidates = []
            try:
                base_detail, _ = check_photo_detail(target_path, final_output, metadata['faceBounds'])
            except Exception:
                base_detail = None
            for index, (model, port) in enumerate(nim_editors.items()):
                try:
                    path = candidate_from_editor(model, port, final_output, nim_refs, metadata["faceBounds"], temp_dir / f"nim-{index}.png",
                                                 **({"extra_strengths": (.025, .05, .10, .20, .50)} if model == flux_model else {}))
                    alternatives = [(path, .35)]
                    if model == flux_model:
                        alternatives.extend((path.with_name(f"{path.stem}-{round(strength*100)}{path.suffix}"), strength) for strength in (.025, .05, .10, .20, .50))
                        flux_status = "quality-rejected"
                    measured = []
                    for candidate, strength in alternatives:
                        acceptable, reason = check_render_file(final_output, candidate, metadata["faceBounds"])
                        if not acceptable or not check_detail_gate(target_path, candidate, metadata["faceBounds"])[0] or not measure_final_face_edit(target_path, candidate, metadata["faceBounds"])["meaningful"]:
                            continue
                        detail, _ = check_photo_detail(target_path, candidate, metadata["faceBounds"])
                        measured.append((candidate, strength, detail))
                    if not measured:
                        print(f"[NIM] {model}: kalite kontrolünden geçmedi.", flush=True)
                        continue
                    values = score_identity(facefusion_dir, centroid, [row[0] for row in measured])
                    for (candidate, strength, detail), value in zip(measured, values):
                        if value is None:
                            continue
                        delta = value - result_similarity
                        detail_delta = detail - base_detail if detail is not None and base_detail is not None else None
                        if model == flux_model and (flux_delta is None or delta > flux_delta):
                            flux_delta, flux_detail_delta, flux_strength = delta, detail_delta, strength
                            flux_status = "identity-loss" if delta < -ULTRA_REALISM_TIE_LOSS else "detail-gain-insufficient"
                        if delta >= -ULTRA_REALISM_TIE_LOSS and (
                            delta > .001 or (detail_delta is not None and detail_delta > .06)
                        ):
                            nim_candidates.append((value, detail or 0., model, candidate, strength, delta, detail_delta))
                    if model == flux_model and flux_delta is None:
                        flux_status = "identity-unmeasured"
                except Exception as error:
                    if model == flux_model:
                        flux_status = "worker-error"
                    print(f"[NIM] {model}: yerel aday kullanılamadı ({type(error).__name__}).", flush=True)
            if nim_candidates:
                result_similarity, _, nim_used, path, strength, delta, detail_delta = max(nim_candidates, key=lambda row: (row[0], row[1]))
                shutil.copyfile(path, final_output)
                if nim_used == flux_model:
                    flux_status = "accepted"
                    flux_delta, flux_detail_delta, flux_strength = delta, detail_delta, strength
                elif any(row[2] == flux_model for row in nim_candidates):
                    flux_status = "candidate-not-selected"
                print(f"[NIM] Ölçümlerle kabul edilen yerel aday: {nim_used}", flush=True)
        # Finest features (v24): eyes and lashes, brows, lips, nose tip and
        # teeth are measured one by one on the target mesh against the photo.
        # A soft feature gets its own detail back, a ringing one is softened
        # and a lost catchlight returns from the photo's light. Kept only if
        # ArcFace and the regional detail gate verify the delivered pixels.
        # Do not allow a chain of optional finishing steps to each spend an
        # independent identity loss allowance.
        finish_anchor = result_similarity
        finish_level = sufficiency_from_level(identity_report.get("photo_level_similarity"))
        finish_floor = max(finish_anchor-.004,finish_level) if finish_level is not None and finish_anchor>=finish_level else finish_anchor-.004
        fine_report: dict[str, Any] = {"status": "no-mesh"}
        fine_decision = "no-mesh"
        if metadata.get("targetLandmarks") and centroid:
            fine_path = temp_dir / "retake-fine-features.png"
            try:
                fine_report = refine_fine_features_file(
                    target_path, final_output, fine_path, metadata["faceBounds"], metadata["targetLandmarks"]
                )
                fine_score = None
                gate_passed = False
                if fine_report.get("status") == "applied":
                    gate_passed = check_detail_gate(target_path, fine_path, metadata["faceBounds"])[0]
                    if gate_passed:
                        fine_score = score_identity(facefusion_dir, centroid, [fine_path])[0]
                accepted_fine, fine_decision = accept_fine_refinement(
                    fine_report, fine_score, result_similarity, gate_passed
                )
                if accepted_fine and fine_score < finish_floor:
                    accepted_fine, fine_decision = False, "identity-loss"
                if accepted_fine:
                    shutil.copyfile(fine_path, final_output)
                    result_similarity = fine_score
                    fine_report["delivered_quality"] = fine_report.get("quality_after")
                elif fine_report.get("quality_before") is not None:
                    fine_report["delivered_quality"] = fine_report.get("quality_before")
                fine_report = delivered_fine_report(fine_report, accepted_fine)
                print(
                    f"[Retake] İnce bölgeler: karar={fine_decision}; "
                    f"oranlar={describe_fine_report(fine_report) or '—'}; "
                    f"göz ışığı={fine_report.get('catchlights', {})}; "
                    f"göz akı={fine_report.get('sclera', {})}; "
                    f"kalite {fine_report.get('quality_before', '—')} → {fine_report.get('quality_after', '—')}"
                    + (f"; ArcFace={fine_score:.3f}" if fine_score is not None else "") + ".",
                    flush=True
                )
            except Exception as error:  # finishing is optional: keep the verified result
                fine_report, fine_decision = {"status": "error"}, "error"
                print(f"[Retake] İnce bölge iyileştirmesi atlandı: {error}", flush=True)

        # An ink-like one-pixel face outline can survive the global ArcFace
        # gate. Attempt a narrow, local cleanup on the existing face matte,
        # then keep it only if ArcFace still verifies the same identity.
        rim_cleaned = False
        rim_stats: dict[str, Any] | None = None
        rim_candidate = temp_dir / "retake-rim-clean.png"
        try:
            rim_stats = repair_dark_rim_file(
                target_path, final_output, rim_candidate, metadata["faceBounds"]
            )
            # Production v21 returned the old visible outline with no
            # explanation. Always report detected ink pixels and the actual
            # decision (even when the conservative repair does not activate).
            print(
                "[Retake] Kontur taraması: "
                f"koyu piksel={rim_stats.get('dark_line_pixels', 0):.0f}; "
                f"kenar payı=%{rim_stats.get('dark_line_fraction', 0) * 100:.2f}; "
                f"karar={rim_stats.get('stage', 'unknown')}; "
                f"azalma=%{rim_stats.get('dark_line_reduction', 0) * 100:.1f}.",
                flush=True
            )
            if rim_stats.get("corrected_pixels", 0) > 0 and centroid:
                corrected_score = score_identity(facefusion_dir, centroid, [rim_candidate])[0]
                if corrected_score is not None and corrected_score >= max(result_similarity - 0.004,finish_floor) and check_detail_gate(target_path, rim_candidate, metadata["faceBounds"])[0]:
                    shutil.copyfile(rim_candidate, final_output)
                    result_similarity = corrected_score
                    rim_cleaned = True
                    print(
                        f"[Retake] Koyu yüz konturu temizlendi "
                        f"({rim_stats['corrected_pixels']:.0f} piksel; "
                        f"koyu sınır %{rim_stats.get('dark_line_reduction', 0.0) * 100:.1f} azaldı; "
                        f"ArcFace={corrected_score:.3f}).", flush=True
                    )
                else:
                    print("[Retake] Kontur adayı kimliği koruyamadı; atlandı.", flush=True)
        except Exception as error:
            print(f"[Retake] Kontur düzeltmesi atlandı: {error}", flush=True)

        # Neck is a separate optional native-pixel candidate. Require a mesh,
        # connected skin evidence, and a fresh identity/detail verification.
        neck_status, neck_pixels = 'off', 0
        neck_evidence = ''
        if metadata.get('neckHarmonize') is True and centroid:
            neck_path = temp_dir / 'retake-neck-harmonized.png'
            try:
                neck_status, neck_pixels = neck_candidate_file(target_path, final_output, neck_path, metadata['faceBounds'], metadata.get('targetLandmarks'))
                if neck_status == 'applied':
                    neck_score = score_identity(facefusion_dir, centroid, [neck_path])[0]
                    if neck_score is not None and neck_score >= max(result_similarity - .001,finish_floor) and check_detail_gate(target_path, neck_path, metadata['faceBounds'])[0]:
                        neck_evidence = neck_delivery_evidence(target_path,final_output,neck_path,metadata["faceBounds"])
                        shutil.copyfile(neck_path, final_output)
                        result_similarity = neck_score
                    else:
                        neck_status, neck_pixels = 'quality-rejected', 0
            except Exception as error:
                neck_status, neck_pixels = 'error', 0
                print(f'[Retake] Neck candidate skipped: {type(error).__name__}', flush=True)

        # Measure the FINAL delivered pixels, not the pre-repair proposal.
        # The ArcFace guard may reject a successful-looking cleanup; never
        # reward a candidate for a repair that wasn't actually committed.
        rim_fraction: float | None = None
        try:
            residual = (
                inspect_dark_rim_file(target_path, final_output, metadata["faceBounds"])
                if rim_cleaned else rim_stats
            )
            if isinstance(residual, dict) and residual.get("stage") != "not_examined":
                measure = float(residual["dark_line_fraction"])
                if math.isfinite(measure) and 0 <= measure <= 1:
                    rim_fraction = measure
                    print(f"[Ultra] Sonuç yüz konturu kararma payı: %{measure * 100:.2f}.", flush=True)
        except (TypeError, ValueError, KeyError, OSError) as error:
            print(f"[Ultra] Yüz konturu ölçümü doğrulanamadı: {error}", flush=True)

        # An unverifiable or measurably worse identity is a failed edit, not
        # a successful-looking photo. This relative guard avoids claiming a
        # universal fixed cosine threshold for every angle and lighting.
        if before_similarity is not None and result_similarity + 0.02 < before_similarity:
            raise IdentityConflictError(
                "Çıktının referans kimlik benzerliği orijinal fotoğraftan düşük "
                f"({before_similarity:.3f} → {result_similarity:.3f}). "
                "Başka bir hedef fotoğraf/poz veya daha net referans deneyin."
            )
        print(
            f"Kimlik benzerliği: sonuç {result_similarity:.3f}"
            + (f", orijinal {before_similarity:.3f}" if before_similarity is not None else ""),
            flush=True
        )
        try:
            detail_score, detail_regions = check_photo_detail(
                target_path, final_output, metadata["faceBounds"]
            )
        except Exception as error:
            detail_score, detail_regions = None, {}
            print(f"[Ultra] Bölgesel doku ölçümü atlandı: {error}", flush=True)
        if detail_score is not None:
            print(
                f"[Ultra] Göz/burun/ağız/alt yüz detay korunumu: {detail_score:.3f}; "
                f"bölgesel oranlar={detail_regions}.",
                flush=True
            )
        # The RGB has already been feathered and blended. Soft alpha is an
        # internal matte, not display opacity: browser canvas premultiplication
        # loses RGB precision at tiny alpha, causing colored contour lines.
        edit = measure_final_face_edit(target_path, final_output, metadata['faceBounds'])
        if not edit['meaningful']:
            raise NoOpSwapError('Son kompozisyonda anlamlı yüz içi değişim kalmadı; sonuç reddedildi.')
        region_edits = regional_edit_statistics(target_path, final_output, metadata["faceBounds"])
        try:
            portrait_evidence, reference_appearance = reference_evidence_file(target_path, final_output, metadata['faceBounds'], metadata.get('targetLandmarks'), real_face)
        except (ValueError, TypeError, OSError, cv2.error):
            portrait_evidence, reference_appearance = {}, None
        accepted, reason, detail_score, detail_regions = check_detail_gate(target_path, final_output, metadata["faceBounds"])
        if not accepted:
            raise NoOpSwapError("Son fotoğraf bölgesel ayrıntı kontrolünü geçemedi: " + reason)
        # Self-test: the target is one of the person's own references, so the
        # original face is ground truth. Measured, never used for selection.
        self_test: dict[str, float] = {}
        if type(exact_index) is int:
            try:
                self_test = self_fidelity(target_path, final_output, metadata["faceBounds"])
            except (ValueError, OSError, cv2.error):
                self_test = {}
            if self_test:
                print(f"[Öz-test] Orijinale yapısal benzerlik {self_test['ssim']:.3f}, renk farkı ΔE {self_test['delta_e']:.2f}.", flush=True)
        result = encode_preblended_result(target_path, final_output)
        headers = {
            'X-Retake-Self-Fidelity': (f"ssim:{self_test['ssim']:.4f},deltaE:{self_test['delta_e']:.3f}" if self_test else ""),
            'X-Retake-Neck-Status': neck_status,
            'X-Retake-Neck-Pixels': str(neck_pixels),
            'X-Retake-Neck-Evidence': neck_evidence,
            'X-Retake-NIM-Editor': nim_used,
            'X-Retake-FLUX-Status': flux_status,
            'X-Retake-FLUX-Identity-Delta': '' if flux_delta is None else f"{flux_delta:.5f}",
            'X-Retake-FLUX-Detail-Delta': '' if flux_detail_delta is None else f"{flux_detail_delta:.5f}",
            'X-Retake-FLUX-Strength': '' if flux_strength is None else f"{flux_strength:.3f}",
            'X-Retake-Photo-Balance': 'on' if selected_identity == 'photo-balance' else 'off',
            'X-Retake-Detail-Gate': 'passed',
            'X-Retake-Face-Changed-Ratio': f"{edit['changed_ratio']:.5f}",
            'X-Retake-Face-Mean-Delta': f"{edit['mean_delta']:.4f}",
            "X-Retake-Detail-Score": "" if detail_score is None else f"{detail_score:.4f}",
            "X-Retake-Detail-Regions": ",".join(f"{name}:{detail_regions[name]:.4f}" for name in ("eyes","nose","mouth","lower_face") if name in detail_regions),
            "X-Retake-Regional-Edit": ",".join(f"{name}:{value:.3f}" for name,value in region_edits.items()),
            "X-Retake-Input-Preparation": input_preparation,
            "X-Retake-Lower-Texture": format_evidence(portrait_evidence),
            "X-Retake-Reference-Appearance": "" if reference_appearance is None else f"{reference_appearance:.4f}",
            "X-Retake-Rim-Cleanup": "on" if rim_cleaned else "off",
            "X-Retake-Fine-Features": fine_decision,
            # The delivered pixels' finest-feature naturalness (0..1), for Ultra ties.
            "X-Retake-Fine-Quality": (
                f"{fine_report['delivered_quality']:.4f}"
                if isinstance(fine_report.get("delivered_quality"), (int, float)) else ""
            ),
            "X-Retake-Fine-Detail": describe_fine_report(fine_report),
            "X-Retake-Catchlights": (
                f"{fine_report.get('catchlights_restored', 0)}/{len(fine_report.get('catchlights') or {})}"
                if isinstance(fine_report.get("catchlights"), dict) else ""
            ),
            "X-Retake-Sclera": (
                f"{fine_report.get('sclera_corrected', 0)}/{len(fine_report.get('sclera') or {})}"
                if isinstance(fine_report.get("sclera"), dict) else ""
            ),
            "X-Retake-Rim-Fraction": "" if rim_fraction is None else f"{rim_fraction:.5f}",
            "X-Retake-Similarity": "" if result_similarity is None else f"{result_similarity:.4f}",
            "X-Retake-Similarity-Before": "" if before_similarity is None else f"{before_similarity:.4f}",
            "X-Retake-Identity-Median-Similarity": str(identity_report.get("median_similarity", "")),
            "X-Retake-Identity-Photo-Level": str(identity_report.get("photo_level_similarity", "")),
            "X-Retake-Identity-Outliers": ",".join(str(index + 1) for index in outlier_indices),
            "X-Retake-Pixel-Boost": f"{identity_boost},{appearance_boost}",
            "X-Retake-Face-Enhancer": FACE_ENHANCER_MODEL if used_enhancer else "off",
            "X-Retake-Identity-Weighted-Votes": str(len(weighted_identity_paths)),
            "X-Retake-Variation": str(variation),
            "X-Retake-Variation-Params": ",".join(f"{value:.3f}" for value in applied_params),
            "X-Retake-Variation-Refined": "on" if refined_params else "off",
            "X-Retake-Real-Face": "on" if used_full_face else "off",
            "X-Retake-Lower-Face-Transfer": "on" if used_lower_face else "off",
            "X-Retake-Jaw-Focused": "on" if jaw_focused and jaw_scale > 0 and selected_identity in ("head-fit", "identity-jaw") else "off",
            "X-Retake-Complete-Identity": "on" if used_full_face else "off",
            "X-Retake-Identity-Reinforced": "on" if reinforced else "off",
            "X-Retake-Reinforce-Guard": reinforce_guard,
            "X-Retake-Identity-Selection": selected_identity,
            "X-Retake-Identity-Source": identity_pool,
            "X-Retake-Texture-Source": chosen_texture,
            "X-Retake-Compositing": "preblended",
            "X-Retake-Alpha-Transport": "binary",
            "X-Retake-Realism": "on" if used_realism else ("off" if realism else "unavailable")
        }
        print(
            "İşlem tamamlandı: AlphaFace kimlik geçişi + UniFace yüksek frekanslı alt yüz detay "
            f"katmanı; dikişsiz ton/çok bantlı birleştirme; "
            f"kafa oranı x{face_width_scale:.3f}/y{face_height_scale:.3f}; "
            f"pixel boost={identity_boost}/{appearance_boost}; "
            f"GFPGAN={'açık' if used_enhancer else 'kapalı'}; tek alfa maskeli PNG döndürüldü.",
            flush=True
        )
        return result, headers


def _swap_error(error: Exception) -> tuple[int, str]:
    if isinstance(error, IdentityConflictError):
        print(f"[Retake ArcFace] Referans kimlik tutarsızlığı: {error}", flush=True)
        return 422, str(error)
    if isinstance(error, IdentityVerificationError):
        print(f"[Retake ArcFace] Ön kontrol başarısız: {error}", flush=True)
        return 503, f"ArcFace kimlik kontrolü yapılamadı: {error}"
    if isinstance(error, NoOpSwapError):
        return 422, str(error)
    if isinstance(error, ValueError):
        return 400, str(error)
    return 500, f"Yerel Retake hibrit profil işlemi başarısız: {error}"


# Near-ties kept in memory for the optional jury (PNG bytes, bounded).
JUDGE_FINALIST_BUFFER = 6


def diverse_finalists(rows, limit):
    """Keep the top local candidate plus verified appearance/fine-detail peaks.

    Only measured, already-eligible candidates reach this function. Missing
    metrics never create a winner and duplicate image bytes never take a slot.
    """
    selected = []
    if limit <= 0:
        return selected
    def add(row):
        if len(selected) < limit and not any(row[0] == prev[0] for prev in selected):
            selected.append(row)
    if rows:
        add(rows[0])
    for measure in (reference_appearance_from_headers, fine_quality_from_headers, score_from_headers):
        measured = [(measure(row[1]), row) for row in rows]
        measured = [(value,row) for value,row in measured if value is not None]
        if len(measured) >= 2 and max(value for value,_ in measured)-min(value for value,_ in measured) > .001:
            add(max(measured, key=lambda item:item[0])[1])
    for row in rows:
        add(row)
    return selected


def retain_judge_finalists(
    finalists: list[tuple[bytes, dict[str, str], int]],
    candidate: tuple[bytes, dict[str, str], int],
    peak_identity: float,
) -> list[tuple[bytes, dict[str, str], int]]:
    """Keep only verified candidates inside the current ArcFace tie band."""
    kept = [
        item for item in [*finalists, candidate]
        if identity_acceptable(item[1], peak_identity)
        and item[1].get("X-Retake-Detail-Gate") == "passed"
    ]
    def rank(item):
        sufficient = is_identity_sufficient(item[1])
        return (sufficient, realism_order(item[1]) if sufficient else (score_from_headers(item[1]) or -1.0,), item[2] == candidate[2])
    kept.sort(key=rank, reverse=True)
    return diverse_finalists(kept, JUDGE_FINALIST_BUFFER)


def judge_pool(
    best_result: bytes, best_headers: dict[str, str], best_round: int,
    finalists: list[tuple[bytes, dict[str, str], int]], peak_identity: float,
) -> list[tuple[bytes, dict[str, str], int]]:
    """Local winner first, then the strongest distinct near-ties (max 4)."""
    pool = [(best_result, best_headers, best_round)]
    eligible = [item for item in finalists if item[0] != best_result
                and identity_acceptable(item[1], peak_identity)
                and (not is_identity_sufficient(best_headers) or is_identity_sufficient(item[1]))
                and item[1].get("X-Retake-Detail-Gate") == "passed"]
    return diverse_finalists(pool + eligible, JUDGE_MAX_CANDIDATES)


def engine_stats(pass_cache: Any, cache_start: dict[str, Any] | None,
                 scorer_start: dict[str, Any] | None) -> str:
    """This job's pass-cache reuse and ArcFace measurements, e.g.
    ``cache=84/31;arcface=96x0.42;arcface_starts=2;arcface_failures=0``."""
    parts = []
    try:
        if cache_start is not None and hasattr(pass_cache, "describe"):
            now = pass_cache.describe()
            parts.append(f"cache={now['hits'] - cache_start['hits']}/{now['misses'] - cache_start['misses']}")
    except Exception:
        pass
    scorer = IDENTITY_SCORER
    try:
        if scorer is not None and scorer_start is not None:
            calls = scorer.stats["calls"] - scorer_start["calls"]
            seconds = scorer.stats["seconds"] - scorer_start["seconds"]
            parts.append(f"arcface={calls}x{(seconds / calls if calls else 0.0):.2f}")
            parts.append(f"arcface_starts={scorer.stats['starts'] - scorer_start['starts']}")
            parts.append(f"arcface_failures={scorer.stats['failures'] - scorer_start['failures']}")
        elif scorer is None:
            parts.append("arcface=one-shot")
    except Exception:
        pass
    return ";".join(parts)


def run_ultra_search(
    facefusion_dir: Path, pixel_boost: str, enhancer_enabled: bool,
    parsed: tuple, progress, caches: dict[str, Any] | None = None,
    clock=None, sleeper=None, on_hold=None, on_finalist=None,
    min_duration_seconds: float | None = None, on_judge=None, on_phase=None
) -> tuple[bytes, dict[str, str]]:
    """Quality-first iterations, each on the *original* photo and 25 references.

    Intermediate results never reach the browser. Failed later candidates do
    not destroy an already verified image. No blind recursive re-swapping.
    The requested quality depth only buys more measured candidates: a longer
    variant search, a compass-search refinement around the winner and more
    realism finalists. Gates and selection rules are identical at every depth.
    """
    clock = clock or time.monotonic
    sleeper = sleeper or time.sleep
    minimum = ULTRA_MIN_SECONDS if min_duration_seconds is None else max(0.0, min_duration_seconds)
    started = clock()
    # Where the time goes (v26): per-phase durations for the timing header.
    timing: dict[str, list[float]] = {"search": [], "refine": [], "realism": [], "extra": []}
    judge_seconds = 0.0
    hold_seconds = 0.0
    pass_cache = (caches or {}).get("passes")
    cache_start = pass_cache.describe() if hasattr(pass_cache, "describe") else None
    scorer_start = dict(IDENTITY_SCORER.stats) if IDENTITY_SCORER is not None else None
    best_result: bytes | None = None
    best_headers: dict[str, str] | None = None
    best_round = 0
    # A candidate is identified by its spec: (grid variant, refined settings or None).
    best_spec: tuple[int, Any] = (ULTRA_VARIATIONS[0], None)
    strongest: tuple[bytes, dict[str, str], int, tuple[int, Any]] | None = None
    peak_identity = -math.inf
    attempted = 0
    completed = 0
    errors: list[Exception] = []
    records: list[tuple[tuple[int, Any], dict[str, str]]] = []
    # Diffusion is expensive and stochastic. Running 2–3 diffusion renders
    # in every one of 24 variants exhausted the seven-minute search window
    # before Retake could actually compare different identities. Compare
    # all feasible identity candidates first, then invest the heavy GPU
    # realism pass in the best validated finalists.
    has_realism = bool(caches and caches.get("realism"))
    has_nim = bool(caches and caches.get('nim_editors'))
    search_caches = {**caches, 'realism': None, 'nim_editors': None} if has_realism or has_nim else caches
    # Optional NVIDIA jury (via Hafize): only with a configured judge AND the
    # user's per-job consent. It may only choose among identity near-ties.
    judge = (caches or {}).get("nvidia_judge")
    judge_consent = bool(isinstance(parsed, tuple) and len(parsed) == 7 and isinstance(parsed[6], dict) and parsed[6].get("nvidiaJudge") is True)
    collect_finalists = judge is not None and judge_consent
    finalists: list[tuple[bytes, dict[str, str], int]] = []
    round_scores: dict[int, float] = {}

    # Learned Ultra: only a few GPU rounds fit in the window, so run the
    # variants that measured most natural for this person first (UCB keeps
    # exploring untried ones). Gates and selection rules are unchanged.
    calibration = (caches or {}).get("calibration")
    metadata = parsed[6] if isinstance(parsed, tuple) and len(parsed) == 7 and isinstance(parsed[6], dict) else {}
    depth = ultra_depth(metadata.get("qualityDepth"))
    bank = ""
    schedule, schedule_mode, schedule_samples = list(ULTRA_VARIATIONS), "default", 0
    if calibration is not None:
        try:
            bank = bank_key(parsed[1], metadata.get("referenceWeights"))
            schedule, schedule_mode, schedule_samples = calibration.schedule(bank, ULTRA_VARIATIONS)
        except Exception as error:  # learning is advisory
            schedule, schedule_mode, schedule_samples = list(ULTRA_VARIATIONS), "default", 0
            print(f"[Ultra] Öğrenilmiş sıra kullanılamadı: {error}", flush=True)
    if schedule_mode != "default":
        print(f"[Ultra] Öğrenilmiş sıra ({schedule_mode}, {schedule_samples} ölçüm): {schedule[:6]}…", flush=True)
    if depth.name != ULTRA_DEFAULT_DEPTH:
        print(
            f"[Ultra] Kalite derinliği: {depth.name} — arama {int(depth.search_seconds)} sn, "
            f"{depth.refinements} ince ayar, {depth.realism_finalists} gerçekçilik finalisti.",
            flush=True
        )
    observations: list[tuple[int, dict[str, str]]] = []

    def learn() -> None:
        if calibration is None or not bank or not observations:
            return
        try:
            calibration.record(bank, observations, self_test=type(metadata.get("exactReferenceIndex")) is int)
        except Exception as error:
            print(f"[Ultra] Öğrenme kaydı atlandı: {error}", flush=True)

    def phase(kind: str, index: int, total: int) -> None:
        if on_phase is not None:
            try:
                on_phase(kind, index, total)
            except Exception:  # progress text must never fail a job
                pass

    def round_of(spec: tuple[int, Any]) -> int:
        return ULTRA_VARIATIONS.index(spec[0]) + 1

    def keep_identity_anchor() -> None:
        # Pairwise near-ties must not drift downward across many candidates.
        nonlocal best_result, best_headers, best_round, best_spec
        if (strongest is not None and best_headers is not None and not is_identity_sufficient(best_headers)
                and score_from_headers(best_headers) < peak_identity - ULTRA_REALISM_TIE_LOSS):
            best_result, best_headers, best_round, best_spec = strongest

    for index, variation in enumerate(schedule):
        if not should_continue_ultra(completed, clock() - started, budget_seconds=depth.search_seconds):
            break
        # A round is identified by its variant (variant + 1), whatever the order.
        round_number = ULTRA_VARIATIONS.index(variation) + 1
        attempted += 1
        print(
            f"[Ultra] Tur {index + 1}/{len(schedule)}: "
            f"orijinal hedef + 25 referans; varyasyon {variation}.",
            flush=True
        )

        def report(step: int, cached: bool = False) -> None:
            progress(step + index * len(JOB_STAGES), cached=cached)

        round_started = clock()
        try:
            image, headers = process_swap(
                facefusion_dir, pixel_boost, enhancer_enabled,
                with_variation(parsed, variation), report, search_caches
            )
        except Exception as error:
            timing["search"].append(clock() - round_started)
            errors.append(error)
            print(f"[Ultra] Aday atlandı: {error}", flush=True)
            # The best already verified result remains usable.
            continue
        timing["search"].append(clock() - round_started)
        identity = score_from_headers(headers)
        if identity is None:
            print('[Ultra] Adayın kimlik ölçümü geçersiz; doğrulanmış tur sayılmadı.', flush=True)
            continue
        completed += 1
        round_scores[variation] = identity
        observations.append((variation, headers))
        records.append(((variation, None), headers))
        if identity > peak_identity:
            peak_identity = identity
            strongest = (image, headers, round_number, (variation, None))
        # Keep a global identity anchor. Retain only two full images in memory.
        keep_identity_anchor()
        if identity_acceptable(headers, peak_identity) and prefer_ultra_candidate(headers, best_headers):
            best_result, best_headers, best_round, best_spec = image, headers, round_number, (variation, None)
            print(
                f"[Ultra] En iyi aday güncellendi: tur {round_number}, "
                f"ArcFace={score_from_headers(headers)}.",
                flush=True
            )
        if collect_finalists:
            finalists = retain_judge_finalists(finalists, (image, headers, round_number), peak_identity)

    if best_result is None or best_headers is None:
        learn()
        if errors:
            # Preserve the actionable HTTP status and diagnosis: no-op,
            # invalid parameters and identity-conflict errors are distinct.
            raise errors[0]
        raise IdentityVerificationError("Ultra Mod hiçbir adayın kimliğini doğrulayamadı.")

    # Refinement (deep/max): the 24 variants are a coarse grid. Step the
    # winner's continuous settings with the same swapper weight, so cached
    # FaceFusion passes are reused; keep a step only when the normal rules
    # prefer it. Always from the original photo, never a generated face.
    refine_attempts = 0
    refine_wins = 0
    if depth.refinements > 0:
        base_variation = best_spec[0]
        planner = RefinePlanner(
            best_spec[1] or FACE_SWAP_VARIATIONS[base_variation],
            tried=[FACE_SWAP_VARIATIONS[variation] for variation in round_scores],
            budget=depth.refinements
        )
        refine_started = clock()
        while clock() - refine_started < depth.refine_seconds:
            params = planner.next()
            if params is None:
                break
            refine_attempts += 1
            phase("refine", refine_attempts, depth.refinements)
            spec = (base_variation, params)
            step_started = clock()
            try:
                image, headers = process_swap(
                    facefusion_dir, pixel_boost, enhancer_enabled,
                    with_refinement(parsed, base_variation, params),
                    lambda *_args, **_kwargs: None, search_caches
                )
            except Exception as error:
                timing["refine"].append(clock() - step_started)
                planner.feedback(params, False)
                print(f"[Ultra] İnce ayar atlandı: {error}", flush=True)
                continue
            timing["refine"].append(clock() - step_started)
            identity = score_from_headers(headers)
            if identity is None:
                planner.feedback(params, False)
                continue
            records.append((spec, headers))
            if identity > peak_identity:
                peak_identity = identity
                strongest = (image, headers, round_of(spec), spec)
            keep_identity_anchor()
            if identity_acceptable(headers, peak_identity) and prefer_ultra_candidate(headers, best_headers):
                best_result, best_headers, best_round, best_spec = image, headers, round_of(spec), spec
            # A step wins when it is now the selected result (also via the
            # identity anchor when it clearly raised the peak).
            improved = best_spec == spec
            if improved:
                refine_wins += 1
                print(
                    f"[Ultra] İnce ayar kazandı: {params} (varyasyon {base_variation}), "
                    f"ArcFace={score_from_headers(headers)}.",
                    flush=True
                )
            planner.feedback(params, improved)
            if collect_finalists:
                finalists = retain_judge_finalists(finalists, (image, headers, round_of(spec)), peak_identity)

    # The GPU realism investment goes to the best identities only, not to
    # every round. It always uses the untouched original target and reference
    # sources; it never feeds a generated face back as identity input.
    flux_outcome = "not-attempted" if ((caches or {}).get("nim_editors") or {}).get("retake-local/flux2-klein-4b") is not None else "not-configured"
    flux_diagnostics = {}
    finalist_attempts = 0
    realism_finalists = 0
    if has_realism or has_nim:
        specs = finalist_specs(records, best_spec, peak_identity, max(1, depth.realism_finalists))
        for position, spec in enumerate(specs):
            finalist_attempts += 1
            realism_finalists += 1
            # Another finalist also gets other diffusion seeds.
            seed = None if position == 0 else REALISM_SEED_BASE + REALISM_SEED_STRIDE * position
            step_started = clock()
            try:
                if on_finalist is not None:
                    on_finalist()
                phase("realism", position + 1, len(specs))
                try:
                    upgraded_image, upgraded_headers = process_swap(
                        facefusion_dir, pixel_boost, enhancer_enabled,
                        with_refinement(parsed, spec[0], spec[1], realism_seed=seed),
                        lambda *_args, **_kwargs: None, caches
                    )
                finally:
                    timing["realism"].append(clock() - step_started)
                outcome = upgraded_headers.get("X-Retake-FLUX-Status", flux_outcome)
                diagnostics = {key: value for key, value in upgraded_headers.items() if key.startswith("X-Retake-FLUX-") and key != "X-Retake-FLUX-Status"}
                upgraded_identity = score_from_headers(upgraded_headers)
                if upgraded_identity is not None and upgraded_identity > peak_identity:
                    peak_identity = upgraded_identity
                    strongest = (upgraded_image, upgraded_headers, round_of(spec), spec)
                if collect_finalists and upgraded_identity is not None:
                    finalists = retain_judge_finalists(finalists, (upgraded_image, upgraded_headers, round_of(spec)), max(peak_identity, upgraded_identity))
                if (upgraded_identity is not None
                        and identity_acceptable(upgraded_headers, peak_identity)
                        and prefer_ultra_candidate(upgraded_headers, best_headers)):
                    best_result, best_headers, best_round, best_spec = upgraded_image, upgraded_headers, round_of(spec), spec
                    peak_identity = max(peak_identity, upgraded_identity)
                    flux_outcome, flux_diagnostics = outcome, diagnostics
                    print(f"[Ultra] Finalist {position + 1} gerçekçilik adayı kimlik ve doku testlerini geçti.", flush=True)
                else:
                    if position == 0:
                        flux_outcome = "finalist-not-selected" if outcome == "accepted" else outcome
                        flux_diagnostics = diagnostics
                    print(f"[Ultra] Finalist {position + 1} difüzyon adayı elendi; güçlü kimlik sonucu korundu.", flush=True)
                keep_identity_anchor()
            except Exception as error:
                # Realism is optional: preserve a previously verified result.
                if flux_outcome == "not-attempted":
                    flux_outcome = "finalist-error"
                print(f"[Ultra] Finalist gerçekçilik atlandı: {error}", flush=True)

    # An under-target result deserves a bounded second search, not lower
    # quality gates. Only local editors are used on extra finalists; avoid
    # repeating the expensive multi-render diffusion pipeline.
    extra_started = clock()
    skip_flux_once = defer_flux_retry(flux_outcome, flux_diagnostics)
    extra_attempts = 0
    if needs_extra_finalist(best_headers):
        untried = [v for v in schedule if v not in round_scores]
        if has_nim:
            extra_variations = choose_extra_variations(
                round_scores, best_round, count=depth.extra_finalists, fallback=untried
            )
        else:
            # Without a local editor an extra pass of a measured variant is
            # the same deterministic image again: spend it on new variants.
            extra_variations = untried[:depth.extra_finalists]
        for variation in extra_variations:
            if not needs_extra_finalist(best_headers) or clock() - extra_started >= depth.extra_start_seconds:
                break
            finalist_attempts += 1
            extra_attempts += 1
            local_only = skip_flux_once
            skip_flux_once = False
            extra_caches = {**(caches or {}), "realism": None}
            if local_only:
                extra_caches["nim_editors"] = None
                print("[Ultra] Poor diffusion result: first extra finalist is identity-only.", flush=True)
            try:
                if on_finalist is not None:
                    on_finalist()
                phase("extra", extra_attempts, depth.extra_finalists)
                step_started = clock()
                try:
                    candidate_image, candidate_headers = process_swap(
                        facefusion_dir, pixel_boost, enhancer_enabled,
                        with_variation(parsed, variation), lambda *_a, **_k: None,
                        extra_caches
                    )
                finally:
                    timing["extra"].append(clock() - step_started)
                candidate_score = score_from_headers(candidate_headers)
                if candidate_score is None or candidate_headers.get("X-Retake-Detail-Gate") != "passed":
                    continue
                observations.append((variation, candidate_headers))
                if candidate_score > peak_identity:
                    peak_identity = candidate_score
                    strongest = (candidate_image, candidate_headers, round_of((variation,None)), (variation,None))
                if identity_acceptable(candidate_headers, peak_identity) and prefer_ultra_candidate(candidate_headers, best_headers):
                    best_result, best_headers = candidate_image, candidate_headers
                    best_round = ULTRA_VARIATIONS.index(variation) + 1
                    best_spec = (variation, None)
                    peak_identity = max(peak_identity, candidate_score)
                    if not local_only:
                        flux_outcome = candidate_headers.get("X-Retake-FLUX-Status", flux_outcome)
                        flux_diagnostics = {key: value for key, value in candidate_headers.items() if key.startswith("X-Retake-FLUX-") and key != "X-Retake-FLUX-Status"}
                elif flux_outcome != "accepted" and not local_only:
                    flux_outcome = candidate_headers.get("X-Retake-FLUX-Status", flux_outcome)
                    if flux_outcome == "accepted":
                        flux_outcome = "finalist-not-selected"
                    flux_diagnostics = {key: value for key, value in candidate_headers.items() if key.startswith("X-Retake-FLUX-") and key != "X-Retake-FLUX-Status"}
                keep_identity_anchor()
                if collect_finalists:
                    finalists = retain_judge_finalists(finalists, (candidate_image, candidate_headers, ULTRA_VARIATIONS.index(variation) + 1), peak_identity)
            except Exception as error:
                print(f"[Ultra] Ek finalist atlandı: {type(error).__name__}", flush=True)
    judge_result = judge_headers("off" if judge is None else "not-consented")
    if collect_finalists:
        pool = judge_pool(best_result, best_headers, best_round, finalists, peak_identity)
        if len(pool) < 2:
            judge_result = judge_headers("single-finalist", len(pool))
        else:
            if on_judge is not None:
                on_judge()
            judge_started = clock()
            try:
                bounds = parsed[6]["faceBounds"]
                if getattr(judge, "accepts_images", False):
                    # v28: review sheets (face + eye/mouth close-ups) from the delivered pixels.
                    verdict = judge([item[0] for item in pool], bounds, metadata.get("targetLandmarks"))
                else:
                    verdict = judge([face_crop_jpeg(item[0], bounds) for item in pool])
                judge_seconds = clock() - judge_started
                pick = judged_winner(verdict, len(pool))
                if pick is None:
                    outcome = "abstain"
                elif pick == 0:
                    outcome = "kept"
                else:
                    outcome = "switched"
                    best_result, best_headers, best_round = pool[pick]
                    if best_headers.get("X-Retake-NIM-Editor") == "retake-local/flux2-klein-4b":
                        flux_outcome = "accepted"
                        flux_diagnostics = {key: value for key, value in best_headers.items() if key.startswith("X-Retake-FLUX-") and key != "X-Retake-FLUX-Status"}
                    elif flux_outcome == "accepted":
                        flux_outcome = "finalist-not-selected"
                judge_result = judge_headers(outcome, len(pool), verdict)
                print(
                    f"[NVIDIA jüri] {len(pool)} finalist; karar={verdict.get('decision')}, "
                    f"oy={verdict.get('votes')}, uzlaşı={verdict.get('agreement')}; sonuç={outcome}.",
                    flush=True
                )
            except Exception as error:
                # The jury is advisory: never lose a verified local result.
                judge_seconds = clock() - judge_started
                judge_result = judge_headers("error", len(pool), error=error)
                print(f"[NVIDIA jüri] Atlandı, yerel seçim korundu: {error}", flush=True)
    finalists = []

    # Spend the available interval trying distinct variants; when the GPU
    # finishes them faster, honor the explicit seven-minute minimum without
    # pointlessly running the identical model again. Failed jobs return early.
    remaining = minimum - (clock() - started)
    hold_started = clock()
    while remaining > 0:
        if on_hold is not None:
            on_hold(remaining)
        sleeper(min(5.0, remaining))
        remaining = minimum - (clock() - started)
    hold_seconds = clock() - hold_started
    if on_hold is not None:
        on_hold(0.0)

    target = identity_aspiration(best_headers)
    gain = identity_gain(best_headers)
    final = {
        **best_headers,
        **flux_diagnostics,
        "X-Retake-Finalist-Attempts": str(finalist_attempts),
        "X-Retake-Identity-Aspiration": "" if target is None else f"{target:.4f}",
        "X-Retake-Identity-Gain": "" if gain is None else f"{gain:.4f}",
        "X-Retake-Identity-Sufficient": (
            "unknown" if identity_sufficiency(best_headers) is None
            else "yes" if is_identity_sufficient(best_headers) else "no"
        ),
        **judge_result,
        "X-Retake-FLUX-Status": flux_outcome,
        "X-Retake-Ultra": "on",
        "X-Retake-Ultra-Rounds": str(completed),
        "X-Retake-Ultra-Attempted": str(attempted),
        "X-Retake-Ultra-Peak-Identity": str(max(peak_identity, score_from_headers(best_headers))),
        "X-Retake-Ultra-Winner": str(best_round),
        "X-Retake-Ultra-Elapsed": str(round(clock() - started, 1)),
        "X-Retake-Ultra-Realism-Policy": "finalist-only" if has_realism else "not-installed",
        "X-Retake-Ultra-Schedule": schedule_mode,
        "X-Retake-Ultra-Order": ",".join(str(v) for v in schedule[:8]),
        "X-Retake-Calibration-Samples": str(schedule_samples),
        "X-Retake-Ultra-Depth": depth.name,
        "X-Retake-Ultra-Search-Budget": str(int(depth.search_seconds)),
        "X-Retake-Ultra-Refinements": f"{refine_wins}/{refine_attempts}",
        "X-Retake-Ultra-Realism-Finalists": str(realism_finalists),
        "X-Retake-Ultra-Timing": format_ultra_timing(timing, judge_seconds, hold_seconds, clock() - started),
        "X-Retake-Engine-Stats": engine_stats(pass_cache, cache_start, scorer_start),
    }
    learn()
    print(
        f"[Ultra] Sonuç: {completed} doğrulanmış tur, kazanan {best_round}. tur; "
        f"benzerlik={score_from_headers(best_headers)}; süre {final['X-Retake-Ultra-Elapsed']} sn.",
        flush=True
    )
    return best_result, final


def prune_jobs(now: float | None = None) -> int:
    """Retain a bounded result delivery window; never evict active jobs."""
    current = time.time() if now is None else now
    with JOBS_LOCK:
        terminal = sorted((item for item in JOBS.values()
                           if item.get("status") in {"done", "error"} and item.get("finished") is not None),
                          key=lambda item: item["finished"], reverse=True)
        expired = [item for index, item in enumerate(terminal)
                   if index >= MAX_KEPT_JOBS or current - item["finished"] >= JOB_RESULT_TTL_SECONDS]
        for item in expired:
            JOBS.pop(item["id"], None)
        return len(expired)


def cleanup_after_job(server: Any) -> dict[str, Any]:
    """Run while inference lock is held. Never touch other processes or files."""
    stats = {"pass_entries_cleared": 0, "objects_collected": 0, "ok": True}
    try:
        cache = getattr(server, "pass_cache", None)
        if cache is not None:
            stats["pass_entries_cleared"] = cache.clear()
    except Exception:
        stats["ok"] = False
    try:
        release_identity_scorer()
    except Exception:
        stats["ok"] = False
    try:
        stats["objects_collected"] = gc.collect()
    except Exception:
        stats["ok"] = False
    stats["completed_at"] = time.time()
    server.last_job_cleanup = stats
    return stats


def run_job(job: dict[str, Any], server: Any, parsed: tuple) -> None:
    """Worker thread: long swaps must not depend on one open HTTP response
    (HTTPS tunnels cut responses that take longer than ~100 seconds)."""
    timings: StageTimings | None = getattr(server, "stage_timings", None)

    def hold(remaining: float) -> None:
        job["stage"] = "Ultra · kalite taraması tamamlandı, 7 dakika süresi doluyor"
        job["minimum_remaining_seconds"] = round(remaining, 1)
        # Actual inference has finished; mark 99% until requested minimum.
        job["step"] = job["total"] - 1
        job["stage_started"] = None  # do not log final waiting as GPU inference

    def finalist() -> None:
        # No false ETA: the diffusion worker's duration is GPU-dependent
        # and not covered by the bounded 420-second candidate search.
        job["stage"] = "Ultra · en iyi yüz için son fotogerçekçilik geçişi"
        job["step"] = job["total"] - 1
        job["stage_started"] = None

    def judging() -> None:
        job["stage"] = "Ultra · NVIDIA görsel jürisi finalistleri karşılaştırıyor (Hafize)"
        job["step"] = job["total"] - 1
        job["stage_started"] = None

    def phase(kind: str, index: int, total: int) -> None:
        # Long depths: say which finishing step runs, without a false ETA.
        labels = {
            "refine": "ince ayar — kazanan ayarın çevresi ölçülüyor",
            "realism": "fotogerçekçilik finalisti",
            "extra": "ek kimlik finalisti",
        }
        job["stage"] = f"Ultra · {labels.get(kind, kind)} {index}/{max(index, total)}"
        job["phase"] = kind
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
            f"Ultra {job['round']}/{len(ULTRA_VARIATIONS)} · "
            + JOB_STAGES[step % len(JOB_STAGES)]
            + (" (önbellekten)" if cached else "")
        )
        job["stage_started"] = now

    terminal_status = "error"
    try:
        job["status"] = "running"
        result, headers = run_ultra_search(
            server.facefusion_dir, server.face_swapper_pixel_boost,
            server.face_enhancer_enabled, parsed, progress,
            {
                "identity": getattr(server, "identity_cache", None),
                "passes": getattr(server, "pass_cache", None),
                "shading": getattr(server, "shading_strength", None),
                "realism": getattr(server, "realism", None),
                "nim_editors": configured_editors() if getattr(server, "dynamic_editors", False) else getattr(server, "nim_editors", None),
                "nvidia_judge": getattr(server, "judge_factory", build_judge)(),
                "calibration": getattr(server, "calibration", None)
            },
            on_hold=hold, on_finalist=finalist, on_judge=judging, on_phase=phase
        )
        close_stage(time.time())
        job["result"] = result
        job["headers"] = headers
        job["step"] = job["total"]
        job["minimum_remaining_seconds"] = 0.0
        job["stage"] = "Ultra · en iyi sonuç seçildi"
        terminal_status = "done"
    except Exception as error:  # reported to the client through the job
        job["http_status"], job["error"] = _swap_error(error)
    finally:
        # Drop uploaded crop buffers before collection. Child GPU processes
        # are already terminated by subprocess.run; their CUDA contexts exit.
        parsed = None
        try:
            job["stage"] = "Ultra · işlem sonrası kaynaklar temizleniyor"
            cleanup_after_job(server)
        finally:
            job["finished"] = time.time()
            job["status"] = terminal_status
            if terminal_status == "done":
                job["stage"] = "Ultra · en iyi sonuç seçildi"
            INFERENCE_LOCK.release()
        prune_jobs()


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
        remaining_window = max(0.0, float(job.get("search_budget_seconds", ULTRA_SEARCH_BUDGET_SECONDS)) - elapsed)
        current_stage = JOB_STAGES[step % len(JOB_STAGES)]
        current_eta = timings.remaining([current_stage], 0, in_current)
        estimate = min(estimate, remaining_window + max(0.0, current_eta or 0.0))
    return max(minimum_remaining, estimate)


def create_job(realism: bool = False, depth: str = ULTRA_DEFAULT_DEPTH) -> dict[str, Any]:
    settings = ultra_depth(depth)
    job: dict[str, Any] = {
        "id": secrets.token_urlsafe(16), "status": "queued", "step": 0,
        "depth": settings.name, "search_budget_seconds": settings.search_seconds,
        "stage": f"Ultra 1/{len(ULTRA_VARIATIONS)} · " + JOB_STAGES[0], "created": time.time(),
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
    prune_jobs()
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
        # then carry the pairing code — see _paired().
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
            "X-Retake-Identity-Photo-Level, X-Retake-Identity-Sufficient, X-Retake-Self-Fidelity, "
            "X-Retake-Ultra-Schedule, X-Retake-Ultra-Order, X-Retake-Calibration-Samples, "
            "X-Retake-Pixel-Boost, X-Retake-Face-Enhancer, X-Retake-Identity-Weighted-Votes, "
            "X-Retake-Variation, X-Retake-Similarity, X-Retake-Similarity-Before, X-Retake-Real-Face, "
            "X-Retake-Realism, X-Retake-Compositing, X-Retake-Alpha-Transport, X-Retake-Identity-Selection, "
            "X-Retake-Identity-Source, X-Retake-Texture-Source, "
            "X-Retake-Ultra, X-Retake-Ultra-Rounds, X-Retake-Ultra-Attempted, X-Retake-Ultra-Peak-Identity, X-Retake-Ultra-Winner, X-Retake-Ultra-Elapsed, "
            "X-Retake-Detail-Score, X-Retake-Detail-Regions, X-Retake-Regional-Edit, X-Retake-Input-Preparation, X-Retake-Lower-Texture, X-Retake-Reference-Appearance, X-Retake-Neck-Status, X-Retake-Neck-Pixels, X-Retake-Neck-Evidence, X-Retake-Rim-Cleanup, X-Retake-Rim-Fraction, "
            "X-Retake-Lower-Face-Transfer, X-Retake-Jaw-Focused, X-Retake-Complete-Identity, "
            "X-Retake-Ultra-Realism-Policy, X-Retake-Face-Changed-Ratio, X-Retake-Face-Mean-Delta, X-Retake-NIM-Editor, "
            "X-Retake-FLUX-Status, X-Retake-FLUX-Identity-Delta, X-Retake-FLUX-Detail-Delta, X-Retake-FLUX-Strength, "
            "X-Retake-Finalist-Attempts, X-Retake-Identity-Aspiration, X-Retake-Identity-Gain, X-Retake-Photo-Balance, X-Retake-Detail-Gate, "
            "X-Retake-NVIDIA-Judge, X-Retake-NVIDIA-Judge-Finalists, X-Retake-NVIDIA-Judge-Votes, "
            "X-Retake-NVIDIA-Judge-Agreement, X-Retake-NVIDIA-Judge-Models, X-Retake-NVIDIA-Judge-Error, X-Retake-NVIDIA-Judge-Reason, "
            "X-Retake-Identity-Reinforced, X-Retake-Reinforce-Guard, "
            "X-Retake-Ultra-Depth, X-Retake-Ultra-Search-Budget, X-Retake-Ultra-Refinements, "
            "X-Retake-Ultra-Realism-Finalists, X-Retake-Variation-Params, X-Retake-Variation-Refined, "
            "X-Retake-Fine-Features, X-Retake-Fine-Detail, X-Retake-Catchlights, X-Retake-Sclera, "
            "X-Retake-Fine-Quality, X-Retake-Ultra-Timing, X-Retake-Engine-Stats, X-Retake-NVIDIA-Judge-Layout"
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
            self._send_json(403, {"ok": False, "error": "Origin veya Host izinli değil."})
            return
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if not self._host_allowed() or not self._origin_allowed():
            self._send_json(403, {"ok": False, "error": "Origin veya Host izinli değil."})
            return
        if not self._paired():
            self._send_json(401, {
                "ok": False, "pairing_required": True,
                "error": "Eşleştirme kodu gerekli veya hatalı."
            })
            return
        if self.path.startswith("/api/jobs/"):
            self._send_job(self.path[len("/api/jobs/"):])
            return
        if self.path == "/api/calibration":
            store = getattr(self.server, "calibration", None)
            # Numbers only: per-variant counts and means, no images.
            self._send_json(200, {"ok": True, **(store.summary() if store else {"enabled": False, "banks": {}})})
            return
        if self.path != "/health":
            self._send_json(404, {"ok": False, "error": "Uç nokta bulunamadı."})
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
            "lossless_preblended_transport": True,
            "final_face_edit_proof": True,
            "regional_detail_gate": True,
            "ultra_learning": bool(getattr(getattr(self.server, "calibration", None), "enabled", False)),
            "self_test_benchmark": True,
            "neck_harmonization": True,
            "regional_edit_evidence": True,
            "lower_face_texture_evidence": True,
            "native_input_recovery": True,
            "reference_appearance_tiebreak": True,
            "adaptive_finalists": True,
            "nvidia_judge": getattr(self.server, "judge_status", judge_status)(),
            "nim_editors_configured": list(configured_editors() if getattr(self.server, "dynamic_editors", False) else getattr(self.server, 'nim_editors', {}) or {}),
            "managed_flux2": (configured_editors() if getattr(self.server, "dynamic_editors", False) else getattr(self.server, 'nim_editors', {}) or {}).get("retake-local/flux2-klein-4b") == 0,
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
            # v23: selectable quality depth (longer search, refinement,
            # several realism finalists) and a job-sized pass cache.
            "quality_depths": True,
            # v24: mesh-guided finishing of eyes, brows, lips, nose tip,
            # teeth, catchlights and eye whites (verified per candidate).
            "fine_features": True,
            # v25: models loaded once per job for ArcFace measurements.
            "identity_scorer": IDENTITY_SCORER.describe() if IDENTITY_SCORER is not None else {"enabled": False},
            "ultra_depths": depth_summary(),
            "ultra_default_depth": ULTRA_DEFAULT_DEPTH,
            "pass_cache": (
                self.server.pass_cache.describe()  # type: ignore[attr-defined]
                if hasattr(getattr(self.server, "pass_cache", None), "describe") else None
            ),
            "async_jobs": True,
            "caches": True,
            "background_mode": os.environ.get('RETAKE_BACKGROUND') == '1',
            "busy": INFERENCE_LOCK.locked(),
            "cleanup_after_job": True,
            "last_job_cleanup": getattr(self.server, 'last_job_cleanup', None),
            "job_result_ttl_seconds": JOB_RESULT_TTL_SECONDS,
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
            "error": None if script.is_file() else f"FaceFusion dosyası bulunamadı: {script}"
        })

    def _send_job(self, tail_path: str) -> None:
        job_id, _, action = tail_path.partition("/")
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None or action not in {"", "result"}:
            self._send_json(404, {"ok": False, "error": "İş bulunamadı veya süresi doldu."})
            return
        if action == "":
            self._send_json(200, {
                "ok": job["status"] != "error",
                "status": job["status"],
                "step": job["step"],
                "total": job.get("total", len(JOB_STAGES)),
                "round": job.get("round", 1),
                "ultra": True,
                "depth": job.get("depth", ULTRA_DEFAULT_DEPTH),
                "phase": job.get("phase", "search"),
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
            self._send_json(409, {"ok": False, "error": "Sonuç henüz hazır değil."})
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
            self._send_json(403, {"ok": False, "error": "Bu web kaynağından gelen isteğe izin verilmiyor."})
            return
        if not self._paired():
            self._send_json(401, {
                "ok": False, "pairing_required": True,
                "error": "Eşleştirme kodu gerekli veya hatalı."
            })
            return
        if self.path != "/api/face-swap":
            self._send_json(404, {"ok": False, "error": "Uç nokta bulunamadı."})
            return
        if not INFERENCE_LOCK.acquire(blocking=False):
            self._send_json(409, {"ok": False, "error": "Retake şu anda başka bir fotoğrafı işliyor."})
            return

        started = False
        try:
            length_header = self.headers.get("Content-Length")
            if not length_header or not length_header.isdigit():
                self._send_json(411, {"ok": False, "error": "Content-Length gerekli."})
                return
            content_length = int(length_header)
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                self._send_json(413, {"ok": False, "error": "İstek boyutu izin verilen sınırı aşıyor."})
                return
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("multipart/form-data;"):
                self._send_json(415, {"ok": False, "error": "multipart/form-data gerekli."})
                return

            parsed = parse_multipart(content_type, self.rfile.read(content_length))
            if not (self.facefusion_dir / "facefusion.py").is_file():
                self._send_json(503, {"ok": False, "error": "FaceFusion kurulumu bulunamadı. Kurulum betiğini çalıştır."})
                return

            depth = parsed[6].get("qualityDepth") if isinstance(parsed, tuple) and len(parsed) == 7 and isinstance(parsed[6], dict) else None
            job = create_job(bool(getattr(self.server, "realism", None)), depth or ULTRA_DEFAULT_DEPTH)
            threading.Thread(
                target=run_job, args=(job, self.server, parsed), daemon=True,
                name=f"retake-job-{job['id'][:6]}"
            ).start()
            started = True  # the worker thread now owns INFERENCE_LOCK
            self._send_json(202, {"ok": True, "job_id": job["id"], "total": job["total"], "ultra": True})
        except ValueError as error:
            self._send_json(400, {"ok": False, "error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            print("İstemci bağlantıyı kapattı.", flush=True)
        except Exception as error:
            self._send_json(500, {"ok": False, "error": f"Yerel Retake isteği işlenemedi: {error}"})
        finally:
            if not started:
                INFERENCE_LOCK.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retake için yerel 25 referanslı yüz profili köprüsü.")
    parser.add_argument("--facefusion-dir", required=True, type=Path, help="FaceFusion kaynak klasörü")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--face-swapper-pixel-boost",
        choices=("512x512", "768x768", "1024x1024"),
        default="1024x1024",
        help="İlk deneme 1024x1024; CUDA bellek hatasında sırasıyla 768x768 ve 512x512 denenir."
    )
    parser.add_argument(
        "--shading-strength", type=float, default=None,
        help="Asıl fotoğrafın ışık-gölge kabartmasının yeni yüze aktarılma gücü (0 kapalı, varsayılan 0.35, en çok 1.2)."
    )
    parser.add_argument(
        "--enable-face-enhancer",
        action="store_true",
        help="İsteğe bağlı GFPGAN 1.4 son aşamasını %25 karışımla etkinleştir."
    )
    parser.add_argument(
        "--enable-pairing",
        action="store_true",
        help="Telefon/PWA erişimi için eşleştirme kodunu zorunlu kıl (HTTPS tünel ile kullanılır)."
    )
    parser.add_argument(
        "--pairing-token-file", type=Path, default=Path.home() / ".retake" / "pairing-token",
        help="Eşleştirme kodunun saklandığı dosya; yoksa oluşturulur."
    )
    parser.add_argument("--origin", action="append", default=[], help="İzin verilen web origin'i; birden çok kez verilebilir.")
    parser.add_argument(
        "--realism-python", type=Path, default=None,
        help="Fotoğraf gerçekçiliği ortamının Python'u (tools/install_realism.ps1 kurar)."
    )
    parser.add_argument(
        "--realism-models", type=Path, default=None,
        help="Fotoğraf gerçekçiliği modellerinin klasörü (tools/install_realism.ps1 indirir)."
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


class RetakeHTTPServer(ThreadingHTTPServer):
    def service_actions(self) -> None:
        prune_jobs()


def main() -> int:
    args = parse_args()
    facefusion_dir = args.facefusion_dir.expanduser().resolve()
    if not (facefusion_dir / "facefusion.py").is_file():
        print(f"FaceFusion kurulumu bulunamadı: {facefusion_dir / 'facefusion.py'}", file=sys.stderr)
        return 2
    if args.host not in {"127.0.0.1", "localhost"}:
        print("Güvenlik nedeniyle servis yalnızca loopback adresinde çalışabilir.", file=sys.stderr)
        return 2

    origins = set(args.origin) | {
        "https://retake06-ua.web.app",
        "https://retake06-ua.firebaseapp.com",
        "https://retake06.web.app",
        "http://localhost:4173",
        "http://127.0.0.1:4173"
    }
    server = RetakeHTTPServer((args.host, args.port), RetakeFaceFusionHandler)
    server.allowed_origins = origins  # type: ignore[attr-defined]
    server.facefusion_dir = facefusion_dir  # type: ignore[attr-defined]
    server.face_swapper_pixel_boost = args.face_swapper_pixel_boost  # type: ignore[attr-defined]
    server.face_enhancer_enabled = args.enable_face_enhancer  # type: ignore[attr-defined]
    server.pairing_token = load_pairing_token(args.pairing_token_file) if args.enable_pairing else ""  # type: ignore[attr-defined]
    server.identity_cache = IdentityReportCache()  # type: ignore[attr-defined]
    server.pass_cache = PassCache.from_environment()  # type: ignore[attr-defined]
    global IDENTITY_SCORER
    if os.environ.get("RETAKE_PERSISTENT_ARCFACE", "").strip().lower() not in {"0", "off", "false", "no"}:
        IDENTITY_SCORER = PersistentIdentityScorer(facefusion_dir, popen_options=subprocess_options())
    server.stage_timings = StageTimings()  # type: ignore[attr-defined]
    server.calibration = CalibrationStore(  # type: ignore[attr-defined]
        signature=schedule_signature(HYBRID_PROFILE_VERSION, ULTRA_VARIATIONS),
        enabled=os.environ.get("RETAKE_ULTRA_LEARNING", "").strip().lower() not in {"0", "off", "false", "no"},
    )
    if args.shading_strength is not None and not 0.0 <= args.shading_strength <= 1.2:
        print("--shading-strength 0 ile 1.2 arasında olmalı.", file=sys.stderr)
        return 2
    server.shading_strength = args.shading_strength  # type: ignore[attr-defined]
    try:
        server.nim_editors = configured_editors()
        server.dynamic_editors = True
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    server.realism = find_realism(args.realism_python, args.realism_models)  # type: ignore[attr-defined]
    if server.realism:  # type: ignore[attr-defined]
        print(f"Fotoğraf gerçekçiliği AÇIK (difüzyon + kimlik yönlendirmesi): {args.realism_models}", flush=True)
    elif args.realism_python or args.realism_models:
        print(
            "Fotoğraf gerçekçiliği kullanılamıyor: ortam veya modeller eksik. "
            "Kurmak için: .\\tools\\install_realism.ps1",
            flush=True
        )
    print(
        f"Retake kişisel yüz profili servisi: http://{args.host}:{args.port} "
        f"(profile={HYBRID_PROFILE_VERSION}, pixel boost={args.face_swapper_pixel_boost}, "
        f"GFPGAN={'on' if args.enable_face_enhancer else 'off'})",
        flush=True
    )
    print("Servis yalnızca loopback adresini dinler.", flush=True)
    if server.pairing_token:  # type: ignore[attr-defined]
        print(
            "Telefon/PWA eşleştirmesi AÇIK. Bu bilgisayara bir HTTPS tünel aç "
            f"(ör. cloudflared tunnel --url http://127.0.0.1:{args.port}) ve Retake'te "
            "'Motor bağlantısı' bölümüne tünel adresi ile şu kodu gir:",
            flush=True
        )
        print(f"    EŞLEŞTİRME KODU: {server.pairing_token}", flush=True)  # type: ignore[attr-defined]
        print("Kodu bilen herkes bu motoru kullanabilir; paylaşma.", flush=True)
    print("İzinli web origin'leri: " + ", ".join(sorted(origins)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nRetake yerel yüz profili servisi kapatılıyor.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
