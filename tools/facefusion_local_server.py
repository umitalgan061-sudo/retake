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

MIN_REFERENCES = 10
MAX_REFERENCES = 25
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_REQUEST_BYTES = 180 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
INFERENCE_LOCK = threading.Lock()
IDENTITY_MODEL = "alphaface_256"
APPEARANCE_MODEL = "uniface_256"
HYBRID_PROFILE_VERSION = "retake_hybrid_25_reference_v2"
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
    if not MIN_REFERENCES <= len(reference_files) <= MAX_REFERENCES:
        raise ValueError(f"Kimlik profili için {MIN_REFERENCES}–{MAX_REFERENCES} referans fotoğrafı gerekli.")

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
    try:
        ok, detail = run_facefusion(
            facefusion_dir, target_path, source_paths, output_path, model, preferred_pixel_boost
        )
    except subprocess.TimeoutExpired as error:
        detail = tail(str(error))
        ok = False
    if ok:
        return preferred_pixel_boost

    lower_detail = detail.lower()
    is_memory_error = any(marker in lower_detail for marker in (
        "out of memory", "cuda error: out of memory", "failed to allocate", "cudnn_status_alloc_failed"
    ))
    if not is_memory_error or preferred_pixel_boost == "512x512":
        raise RuntimeError(f"{model} işlemi başarısız ({preferred_pixel_boost}). {detail}")

    print(f"{model}: {preferred_pixel_boost} bellek hatası; 512x512 güvenli ayarına düşülüyor.", flush=True)
    try:
        ok, retry_detail = run_facefusion(
            facefusion_dir, target_path, source_paths, output_path, model, "512x512"
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{model} 512x512 yeniden denemesi zaman aşımına uğradı. {tail(str(error))}") from error
    if not ok:
        raise RuntimeError(f"{model} işlemi 512x512 ayarında da başarısız. {retry_detail}")
    return "512x512"


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
    # The identity pass stays dominant; the appearance pass contributes beard,
    # moustache, lower-cheek texture and chin detail inside a soft lower-face region.
    alpha = np.clip(mask * 0.82, 0.0, 0.82)[:, :, None]
    blended = identity.astype(np.float32) * (1.0 - alpha) + appearance.astype(np.float32) * alpha
    result = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

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
            "appearance_model": APPEARANCE_MODEL,
            "pixel_boost": self.server.face_swapper_pixel_boost,  # type: ignore[attr-defined]
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

                weights: list[int] = metadata["referenceWeights"]
                # Repeating paths makes FaceFusion's source-face embedding mean
                # quality-weighted while guaranteeing every accepted reference votes.
                weighted_identity_paths = [
                    path
                    for path, weight in zip(reference_paths, weights)
                    for _ in range(weight)
                ]
                identity_output = temp_dir / "retake-identity-alpha.png"
                appearance_output = temp_dir / "retake-appearance-uniface.png"
                hybrid_output = temp_dir / "retake-hybrid-profile.png"
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
                result = hybrid_output.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(result)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(result)
                print(
                    "İşlem tamamlandı: AlphaFace ağırlıklı kimlik + UniFace 25 fotoğraflı görünüm profili "
                    "alt yüzde harmanlandı; tek PNG çıktı döndürüldü.",
                    flush=True
                )
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
        default="768x768",
        help="İki geçiş için tercih edilen çözünürlük; CUDA bellek hatasında otomatik 512x512 denenir."
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
    print(
        f"Retake kişisel yüz profili servisi: http://{args.host}:{args.port} "
        f"(profile={HYBRID_PROFILE_VERSION}, pixel boost={args.face_swapper_pixel_boost})",
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
