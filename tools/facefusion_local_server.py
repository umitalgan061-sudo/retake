from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from email.parser import BytesParser
from email import policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

MIN_REFERENCES = 10
MAX_REFERENCES = 25
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_REQUEST_BYTES = 180 * 1024 * 1024
ALLOWED_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
INFERENCE_LOCK = threading.Lock()


def image_extension(content_type: str, payload: bytes) -> str:
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized == "image/png" and payload.startswith(PNG_SIGNATURE):
        return ".png"
    if normalized == "image/jpeg" and payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if normalized == "image/webp" and len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("Yalnızca geçerli PNG, JPEG veya WebP görselleri kabul edilir.")


def parse_multipart(content_type: str, body: bytes) -> tuple[bytes, list[bytes]]:
    headers = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8")
    message = BytesParser(policy=policy.default).parsebytes(headers + body)
    if not message.is_multipart():
        raise ValueError("İstek multipart/form-data biçiminde olmalı.")

    target_files: list[tuple[str, bytes]] = []
    reference_files: list[tuple[str, bytes]] = []
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        field_name = part.get_param("name", header="content-disposition")
        filename = part.get_filename() or "upload"
        payload = part.get_payload(decode=True) or b""
        content_type = part.get_content_type()
        if field_name not in {"target", "reference"}:
            continue
        extension = image_extension(content_type, payload)
        if not payload or len(payload) > MAX_FILE_BYTES:
            raise ValueError("Her görsel boş olmamalı ve 25 MB sınırını aşmamalı.")
        item = (extension, payload)
        if field_name == "target":
            target_files.append(item)
        else:
            reference_files.append(item)

    if len(target_files) != 1:
        raise ValueError("Tam olarak bir hedef fotoğraf gönderilmelidir.")
    if not MIN_REFERENCES <= len(reference_files) <= MAX_REFERENCES:
        raise ValueError(f"Kimlik profili için {MIN_REFERENCES}–{MAX_REFERENCES} referans fotoğrafı gerekli.")
    return target_files[0][1], [payload for _, payload in reference_files], target_files[0][0], [ext for ext, _ in reference_files]


def tail(text: str, limit: int = 4500) -> str:
    value = (text or "").strip()
    return value[-limit:] if value else "FaceFusion ayrıntılı hata kaydı üretmedi."


class RetakeFaceFusionHandler(BaseHTTPRequestHandler):
    server_version = "RetakeFaceFusion/1.0"

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
            "model": "alphaface_256",
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
            self._send_json(409, {"ok": False, "error": "FaceFusion şu anda başka bir fotoğrafı işliyor."})
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
            target_bytes, reference_bytes, target_ext, reference_exts = parse_multipart(content_type, body)
            if not (self.facefusion_dir / "facefusion.py").is_file():
                self._send_json(503, {"ok": False, "error": "FaceFusion kurulumu bulunamadı. Kurulum betiğini çalıştır."})
                return

            with tempfile.TemporaryDirectory(prefix="retake-facefusion-") as temp_name:
                temp_dir = Path(temp_name)
                target_path = temp_dir / ("target" + target_ext)
                target_path.write_bytes(target_bytes)
                source_paths: list[Path] = []
                for index, (payload, extension) in enumerate(zip(reference_bytes, reference_exts), start=1):
                    source_path = temp_dir / f"reference-{index:02d}{extension}"
                    source_path.write_bytes(payload)
                    source_paths.append(source_path)
                output_path = temp_dir / "retake-facefusion-result.png"

                command = [
                    sys.executable,
                    str(self.facefusion_dir / "facefusion.py"),
                    "headless-run",
                    "--source-paths", *[str(path) for path in source_paths],
                    "--target-path", str(target_path),
                    "--output-path", str(output_path),
                    "--processors", "face_swapper",
                    "--face-swapper-model", "alphaface_256",
                    "--face-swapper-pixel-boost", "512x512",
                    "--face-swapper-weight", "1.0",
                    "--face-selector-mode", "one",
                    "--face-mask-types", "box",
                    "--face-mask-blur", "0.30",
                    "--execution-providers", "cuda",
                    "--execution-device-ids", "0",
                    "--output-image-quality", "100",
                    "--log-level", "info"
                ]

                print(f"İş başlatıldı: {len(source_paths)} kimlik referansı, hedef 1 fotoğraf.", flush=True)
                try:
                    completed = subprocess.run(
                        command,
                        cwd=str(self.facefusion_dir),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=900,
                        check=False
                    )
                except subprocess.TimeoutExpired as error:
                    detail = tail((error.stdout or "") + "\n" + (error.stderr or ""))
                    self._send_json(504, {"ok": False, "error": "FaceFusion işlemi 15 dakika sınırını aştı. " + detail})
                    return

                if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
                    detail = tail(completed.stderr + "\n" + completed.stdout)
                    self._send_json(500, {"ok": False, "error": "FaceFusion yüz değiştirme işlemi başarısız.\n" + detail})
                    return

                result = output_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(result)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(result)
                print("İşlem tamamlandı: tek PNG çıktı döndürüldü.", flush=True)
        except ValueError as error:
            self._send_json(400, {"ok": False, "error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            print("İstemci bağlantıyı kapattı.", flush=True)
        except Exception as error:
            self._send_json(500, {"ok": False, "error": f"Yerel FaceFusion sunucu hatası: {error}"})
        finally:
            INFERENCE_LOCK.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retake için yalnızca loopback üzerinde FaceFusion köprüsü.")
    parser.add_argument("--facefusion-dir", required=True, type=Path, help="FaceFusion kaynak klasörü")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", default=8765, type=int)
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
    print(f"Retake FaceFusion servisi: http://{args.host}:{args.port}", flush=True)
    print("Yalnızca loopback; dış ağ erişimi kapalı.", flush=True)
    print("İzinli web origin'leri: " + ", ".join(sorted(origins)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFaceFusion servisi kapatılıyor.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
