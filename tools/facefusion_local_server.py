from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
from email.parser import BytesParser
from email import policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from .facefusion_quality import PIXEL_BOOST_FALLBACKS, is_near_noop
except ImportError:
    from facefusion_quality import PIXEL_BOOST_FALLBACKS, is_near_noop

MAX_REFERENCES = 25
MIN_REFERENCES = MAX_REFERENCES
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_REQUEST_BYTES = 180 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
INFERENCE_LOCK = threading.Lock()
IDENTITY_MODEL = "alphaface_256"
APPEARANCE_MODEL = "uniface_256"
HYBRID_PROFILE_VERSION = "retake_hybrid_25_reference_v4"
IDENTITY_VERIFIER_MODEL = "arcface_w600k_r50"
FACE_ENHANCER_MODEL = "gfpgan_1.4"
# FaceFusion renders a full-strength enhancer output; Retake applies the
# conservative 25% blend once after inference, preserving the existing alpha.
FACE_ENHANCER_BLEND = 25
FACE_ENHANCER_INTERNAL_BLEND = 100
FACE_ENHANCER_WEIGHT = 0.25
DETAIL_TRANSFER_STRENGTH = 0.20


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
    if bounds["x"] < 0 or bounds["y"] < 0 or bounds["x"] + bounds["width"] > 784 or bounds["y"] + bounds["height"] > 1024:
        raise ValueError("Hedef yüz sınırları, gönderilen portre kırpımının dışında.")
    return {"referenceWeights": normalized_weights, "faceBounds": bounds}


def validate_reference_count(reference_count: int) -> None:
    if reference_count != MAX_REFERENCES:
        raise ValueError(f"Retake hibrit profil işlemi için tam olarak {MAX_REFERENCES} referans fotoğrafı gerekli.")


def verify_reference_identity(facefusion_dir: Path, reference_paths: list[Path]) -> dict[str, object]:
    worker_path = Path(__file__).with_name("facefusion_identity_worker.py")
    if not worker_path.is_file():
        raise IdentityVerificationError("ArcFace doğrulama worker dosyası bulunamadı.")
    command = [
        sys.executable, str(worker_path),
        "--facefusion-dir", str(facefusion_dir),
        "--reference-paths", *[str(path) for path in reference_paths]
    ]
    try:
        completed = subprocess.run(
            command, cwd=str(facefusion_dir), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600, check=False
        )
    except subprocess.TimeoutExpired as error:
        raise IdentityVerificationError("ArcFace 25 referans ön kontrolü 10 dakika içinde tamamlanamadı.") from error

    marker = "RETAKE_IDENTITY_REPORT="
    report_lines = [line[len(marker):] for line in completed.stdout.splitlines() if line.startswith(marker)]
    if not report_lines:
        detail = tail(completed.stderr + "\n" + completed.stdout)
        raise IdentityVerificationError("ArcFace doğrulama raporu alınamadı. " + detail)
    try:
        report = json.loads(report_lines[-1])
    except json.JSONDecodeError as error:
        raise IdentityVerificationError("ArcFace doğrulama raporu geçerli JSON değil.") from error

    if completed.returncode != 0 or not report.get("ok"):
        raise IdentityVerificationError(str(report.get("error") or tail(completed.stderr + "\n" + completed.stdout)))
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
        if field_name not in {"target", "reference", "appearance_profile"}:
            continue

        extension = image_extension(part.get_content_type(), payload)
        if not payload or len(payload) > MAX_FILE_BYTES:
            raise ValueError("Her görsel boş olmamalı ve 25 MB sınırını aşmamalı.")
        item = (extension, payload)
        if field_name == "target":
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


def run_facefusion(
    facefusion_dir: Path,
    target_path: Path,
    source_paths: list[Path],
    output_path: Path,
    model: str,
    pixel_boost: str
) -> tuple[bool, str]:
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
        "--face-swapper-weight", "1.0",
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
    preferred_pixel_boost: str
) -> str:
    attempts = (preferred_pixel_boost, *PIXEL_BOOST_FALLBACKS.get(preferred_pixel_boost, ()))
    last_detail = ""
    for index, pixel_boost in enumerate(attempts):
        try:
            ok, detail = run_facefusion(
                facefusion_dir, target_path, source_paths, output_path, model, pixel_boost
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
            raise RuntimeError(f"{model} işlemi başarısız ({pixel_boost}). {detail}")
        next_boost = attempts[index + 1]
        print(f"{model}: {pixel_boost} bellek hatası; {next_boost} ayarına düşülüyor.", flush=True)

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
        encoding="utf-8", errors="replace", timeout=900, check=False
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


def blend_lower_face(
    target_path: Path,
    identity_path: Path,
    appearance_path: Path,
    output_path: Path,
    face_bounds: dict[str, float]
) -> None:
    try:
        import cv2
        import numpy as np
    except Exception as error:
        raise RuntimeError(f"Alt yüz harmanlama için OpenCV/NumPy yüklenemedi: {error}") from error

    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    identity = cv2.imread(str(identity_path), cv2.IMREAD_COLOR)
    appearance = cv2.imread(str(appearance_path), cv2.IMREAD_COLOR)
    if (
        target is None or identity is None or appearance is None
        or target.shape != identity.shape or target.shape != appearance.shape
    ):
        raise RuntimeError("Hedef, kimlik ve görünüm geçişlerinin çıktı ölçüleri eşleşmiyor.")

    height, width = identity.shape[:2]
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
    detail_alpha = np.clip(mask * DETAIL_TRANSFER_STRENGTH, 0.0, DETAIL_TRANSFER_STRENGTH)[:, :, None]
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
            "FaceFusion iki geçişte de hedef yüze anlamlı bir değişiklik uygulamadı "
            f"(değişen piksel oranı %{changed_ratio * 100:.3f}, ortalama fark {mean_delta:.3f}). "
            "İşlem başarılı gibi gösterilmedi; kaynak/hedef yüz algılamasını kontrol edin."
        )

    # FaceFusion has already applied its parser + occlusion masks while pasting
    # each pass back into the target crop. Derive a final alpha matte from the
    # actual changed pixels so the browser can preserve every unchanged source
    # pixel byte-for-byte instead of applying a large landmark ellipse.
    pixel_delta = np.max(cv2.absdiff(result, target), axis=2)
    changed = np.where(pixel_delta >= 2, 255, 0).astype(np.uint8)
    matte_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, matte_kernel)
    changed = cv2.dilate(changed, matte_kernel, iterations=1)
    alpha = cv2.GaussianBlur(changed, (0, 0), 1.8)
    rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    if not cv2.imwrite(str(output_path), rgba):
        raise RuntimeError("Hibrit yüz sonucu ve semantik alfa maskesi diske yazılamadı.")


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

    def _host_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        port = self.server.server_port
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

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
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header(
            "Access-Control-Expose-Headers",
            "X-Retake-Identity-Median-Similarity, X-Retake-Identity-Outliers, "
            "X-Retake-Pixel-Boost, X-Retake-Face-Enhancer, X-Retake-Identity-Weighted-Votes"
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
            "providers": providers,
            "cuda": "CUDAExecutionProvider" in providers,
            "provider_error": provider_error,
            "error": None if script.is_file() else f"FaceFusion dosyası bulunamadı: {script}"
        })

    def do_POST(self) -> None:
        if not self._host_allowed() or not self._origin_allowed(required=True):
            self._send_json(403, {"ok": False, "error": "Bu web kaynağından gelen isteğe izin verilmiyor."})
            return
        if self.path != "/api/face-swap":
            self._send_json(404, {"ok": False, "error": "Uç nokta bulunamadı."})
            return
        if not INFERENCE_LOCK.acquire(blocking=False):
            self._send_json(409, {"ok": False, "error": "Retake şu anda başka bir fotoğrafı işliyor."})
            return

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

            body = self.rfile.read(content_length)
            (
                target_bytes,
                reference_bytes,
                target_ext,
                reference_exts,
                appearance_bytes,
                appearance_ext,
                metadata
            ) = parse_multipart(content_type, body)
            if not (self.facefusion_dir / "facefusion.py").is_file():
                self._send_json(503, {"ok": False, "error": "FaceFusion kurulumu bulunamadı. Kurulum betiğini çalıştır."})
                return

            with tempfile.TemporaryDirectory(prefix="retake-hybrid-profile-") as temp_name:
                temp_dir = Path(temp_name)
                target_path = temp_dir / ("target" + target_ext)
                target_path.write_bytes(target_bytes)
                appearance_path = temp_dir / ("appearance-profile" + appearance_ext)
                appearance_path.write_bytes(appearance_bytes)

                reference_paths: list[Path] = []
                for index, (payload, extension) in enumerate(zip(reference_bytes, reference_exts), start=1):
                    source_path = temp_dir / f"reference-{index:02d}{extension}"
                    source_path.write_bytes(payload)
                    reference_paths.append(source_path)

                identity_report = verify_reference_identity(self.facefusion_dir, reference_paths)
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
                # Repeating paths makes FaceFusion's source-face embedding mean
                # quality-weighted. Every reference remains in the profile and
                # every accepted reference receives at least one identity vote.
                weighted_identity_paths = [
                    path
                    for path, weight in zip(reference_paths, weights)
                    for _ in range(weight)
                ]
                identity_output = temp_dir / "retake-identity-alpha.png"
                appearance_output = temp_dir / "retake-appearance-uniface.png"
                hybrid_output = temp_dir / "retake-hybrid-profile.png"
                enhanced_output = temp_dir / "retake-gfpgan.png"
                final_output = temp_dir / "retake-final.png"
                preferred_boost = self.server.face_swapper_pixel_boost  # type: ignore[attr-defined]

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

                identity_boost = run_with_memory_fallback(
                    self.facefusion_dir,
                    target_path,
                    weighted_identity_paths,
                    identity_output,
                    IDENTITY_MODEL,
                    preferred_boost
                )
                print(f"Kimlik profili geçişi tamamlandı ({identity_boost}).", flush=True)

                appearance_boost = run_with_memory_fallback(
                    self.facefusion_dir,
                    target_path,
                    [appearance_path],
                    appearance_output,
                    APPEARANCE_MODEL,
                    preferred_boost
                )
                print(f"25 fotoğraflı görünüm profili geçişi tamamlandı ({appearance_boost}).", flush=True)

                blend_lower_face(
                    target_path,
                    identity_output,
                    appearance_output,
                    hybrid_output,
                    metadata["faceBounds"]
                )
                used_enhancer = bool(self.server.face_enhancer_enabled)  # type: ignore[attr-defined]
                if used_enhancer:
                    print(
                        f"İsteğe bağlı son aşama: {FACE_ENHANCER_MODEL}, blend={FACE_ENHANCER_BLEND}%.",
                        flush=True
                    )
                    run_face_enhancer(self.facefusion_dir, hybrid_output, reference_paths, enhanced_output)
                    blend_enhanced_hybrid_output(hybrid_output, enhanced_output, final_output)
                else:
                    final_output = hybrid_output

                result = final_output.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(result)))
                self.send_header(
                    "X-Retake-Identity-Median-Similarity",
                    str(identity_report.get("median_similarity", ""))
                )
                self.send_header(
                    "X-Retake-Identity-Outliers",
                    ",".join(str(index + 1) for index in outlier_indices)
                )
                self.send_header("X-Retake-Pixel-Boost", f"{identity_boost},{appearance_boost}")
                self.send_header("X-Retake-Face-Enhancer", FACE_ENHANCER_MODEL if used_enhancer else "off")
                self.send_header("X-Retake-Identity-Weighted-Votes", str(len(weighted_identity_paths)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(result)
                print(
                    "İşlem tamamlandı: AlphaFace kimlik geçişi + UniFace yüksek frekanslı alt yüz detay "
                    f"katmanı; pixel boost={identity_boost}/{appearance_boost}; "
                    f"GFPGAN={'açık' if used_enhancer else 'kapalı'}; tek alfa maskeli PNG döndürüldü.",
                    flush=True
                )
        except IdentityConflictError as error:
            self._send_json(422, {"ok": False, "error": str(error)})
        except IdentityVerificationError as error:
            self._send_json(503, {"ok": False, "error": f"ArcFace kimlik kontrolü yapılamadı: {error}"})
        except NoOpSwapError as error:
            self._send_json(422, {"ok": False, "error": str(error)})
        except ValueError as error:
            self._send_json(400, {"ok": False, "error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            print("İstemci bağlantıyı kapattı.", flush=True)
        except Exception as error:
            self._send_json(500, {"ok": False, "error": f"Yerel Retake hibrit profil işlemi başarısız: {error}"})
        finally:
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
        "--enable-face-enhancer",
        action="store_true",
        help="İsteğe bağlı GFPGAN 1.4 son aşamasını %25 karışımla etkinleştir."
    )
    parser.add_argument("--origin", action="append", default=[], help="İzin verilen web origin'i; birden çok kez verilebilir.")
    return parser.parse_args()


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
        "https://retake06.web.app",
        "http://localhost:4173",
        "http://127.0.0.1:4173"
    }
    server = ThreadingHTTPServer((args.host, args.port), RetakeFaceFusionHandler)
    server.allowed_origins = origins  # type: ignore[attr-defined]
    server.facefusion_dir = facefusion_dir  # type: ignore[attr-defined]
    server.face_swapper_pixel_boost = args.face_swapper_pixel_boost  # type: ignore[attr-defined]
    server.face_enhancer_enabled = args.enable_face_enhancer  # type: ignore[attr-defined]
    print(
        f"Retake kişisel yüz profili servisi: http://{args.host}:{args.port} "
        f"(profile={HYBRID_PROFILE_VERSION}, pixel boost={args.face_swapper_pixel_boost}, "
        f"GFPGAN={'on' if args.enable_face_enhancer else 'off'})",
        flush=True
    )
    print("Yalnızca loopback; dış ağ erişimi kapalı.", flush=True)
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
