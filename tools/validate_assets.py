#!/usr/bin/env python3
"""Validate Retake's local binary assets before attempting offline inference.

No third-party packages are required. This is a sanity check, not a substitute
for loading an ONNX/Core ML model with its runtime on a real iPhone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_FILES = [
    "33F4BCA9-78DC-4382-B4E7-F7FD897722B3.jpeg",
    "37D27514-F417-42AC-B9ED-4C5C3D98EA3C.jpeg",
    "6074c30a-147b-481e-9fe5-36d02c892d48.jpeg",
    "769BDFF4-5555-471B-AEC8-768C274912D0.jpeg",
    "90F8BF41-96FF-44C3-AD23-0E14C2D6A1E4.jpeg",
    "97574DEE-ED27-45BC-834A-4CCEC2E7A8F9.png",
    "BAAC2698-A3BE-4C24-87FF-1345627644A7.jpeg",
    "IMG_0706.jpeg",
    "IMG_0892.jpeg",
    "IMG_1699.jpeg",
    "IMG_2181.jpeg",
    "IMG_2266.jpeg",
    "IMG_3031.jpeg",
    "IMG_3331.jpeg",
    "IMG_3511.jpeg",
    "IMG_3882.jpeg",
    "IMG_4435.jpeg",
    "IMG_6609.jpeg",
    "IMG_9038.jpeg",
    "IMG_9438.jpeg",
]

MODEL_FILES = {
    "models/det_10g.onnx": "ONNX",
    "models/w600k_r50.onnx": "ONNX",
    "models/face_landmarker.task": "MediaPipe task",
}
MIN_MODEL_BYTES = 64 * 1024

JPEG = b"\xff\xd8\xff"
PNG = b"\x89PNG\r\n\x1a\n"
RIFF = b"RIFF"


def is_text_pointer(data: bytes) -> bool:
    sample = data[:512].decode("utf-8", errors="ignore").strip().lower()
    return (
        sample.startswith("/api/")
        or sample.startswith("version https://git-lfs.github.com/spec/v1")
        or "api/documents/" in sample
        or "api/images/" in sample
    )


def check_image(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "dosya bulunamadı"
    data = path.read_bytes()
    if is_text_pointer(data):
        return False, "gerçek görüntü yerine API/asset referans metni"
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"} and data.startswith(JPEG):
        return True, f"JPEG ({len(data):,} bayt)"
    if suffix == ".png" and data.startswith(PNG):
        return True, f"PNG ({len(data):,} bayt)"
    if suffix == ".webp" and data.startswith(RIFF) and data[8:12] == b"WEBP":
        return True, f"WebP ({len(data):,} bayt)"
    return False, f"geçerli {suffix or 'görüntü'} imzası yok ({len(data):,} bayt)"


def check_model(relative_path: str, model_type: str) -> tuple[bool, str]:
    path = ROOT / relative_path
    if not path.is_file():
        return False, "dosya bulunamadı"
    data = path.read_bytes()
    if is_text_pointer(data):
        return False, "gerçek model ağırlığı yerine API/asset referans metni"
    if len(data) < MIN_MODEL_BYTES:
        return False, f"şüpheli derecede küçük: {len(data):,} bayt (en az {MIN_MODEL_BYTES:,} bekleniyor)"
    if data and all(byte in b"\\t\\n\\r" or 32 <= byte <= 126 for byte in data[: min(len(data), 4096)]):
        return False, "metin dosyasına benziyor; ikili model ağırlığı değil"
    return True, f"{model_type} ikili dosyası için temel kontrol geçti ({len(data):,} bayt)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Retake yerel model ve referans fotoğraf doğrulaması")
    parser.add_argument("--json", action="store_true", help="makinece okunabilir JSON raporu üret")
    args = parser.parse_args()

    results: list[dict[str, str | bool]] = []
    for filename in REFERENCE_FILES:
        ok, detail = check_image(ROOT / "reference-faces" / filename)
        results.append({"kind": "reference", "file": f"reference-faces/{filename}", "ok": ok, "detail": detail})

    for filename, model_type in MODEL_FILES.items():
        ok, detail = check_model(filename, model_type)
        results.append({"kind": "model", "file": filename, "ok": ok, "detail": detail})

    index_path = ROOT / "index.html"
    if index_path.is_file():
        source = index_path.read_text(encoding="utf-8", errors="replace")
        remote_domains = [
            domain for domain in (
                "cdn.jsdelivr.net",
                "storage.googleapis.com",
                "unpkg.com",
                "cdnjs.cloudflare.com",
            ) if domain in source
        ]
        results.append({
            "kind": "offline",
            "file": "index.html",
            "ok": not remote_domains,
            "detail": "harici CDN bağımlılığı yok" if not remote_domains
                     else "çevrimdışı çalışmayı engelleyen uzak bağımlılıklar: " + ", ".join(remote_domains),
        })

    has_xcode_project = any(ROOT.glob("*.xcodeproj")) or any(ROOT.glob("ios/**/*.xcodeproj"))
    has_swift_package = (ROOT / "Package.swift").is_file() or any(ROOT.glob("ios/**/Package.swift"))
    results.append({
        "kind": "ios",
        "file": "iOS project",
        "ok": has_xcode_project or has_swift_package,
        "detail": "Xcode/Swift Package yapısı bulundu" if (has_xcode_project or has_swift_package)
                  else "Xcode projesi veya Swift Package bulunamadı",
    })

    failures = [item for item in results if not item["ok"]]
    if args.json:
        print(json.dumps({"ok": not failures, "passed": len(results) - len(failures),
                          "failed": len(failures), "checks": results}, ensure_ascii=False, indent=2))
    else:
        print("RETAKE — OFFLINE ASSET CHECK")
        print("=" * 72)
        for item in results:
            state = "PASS" if item["ok"] else "FAIL"
            print(f"[{state}] {item['file']}: {item['detail']}")
        print("=" * 72)
        print(f"Sonuç: {len(results) - len(failures)} başarılı, {len(failures)} başarısız kontrol.")
        if failures:
            print("Not: Bu kontrol dosya imzalarını/boyutlarını doğrular; modelin gerçek inference "
                  "başarısı ayrıca iPhone üzerinde test edilmelidir.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
