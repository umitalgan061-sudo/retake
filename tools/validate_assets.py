#!/usr/bin/env python3
"""Validate Retake's checked-in web/iOS structure and prevent stale demo regressions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "index.html", "styles.css", "app.js", "src/core.js", "src/face-only.js",
    "tests/core.test.js", "tests/face-only.test.js", "tests/no-camera.test.js", "docs/FACE_ONLY_PIPELINE.md",
    "package.json", "manifest.webmanifest", "sw.js", "icons/retake.svg",
    "ios/Retake.xcodeproj/project.pbxproj",
)
FORBIDDEN_WEB_MARKERS = (
    "drawWarpedFace", "createMPImage", "lastSwappedImage",
    "REFERENCE_FACE_FILES", "face-swap-",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retake repository quality checks")
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args()
    checks: list[dict[str, str | bool]] = []

    for name in REQUIRED_FILES:
        exists = (ROOT / name).is_file()
        checks.append({
            "check": "required-file", "file": name, "ok": exists,
            "detail": "bulundu" if exists else "dosya eksik",
        })

    index_path = ROOT / "index.html"
    app_path = ROOT / "app.js"
    if index_path.is_file():
        source = index_path.read_text(encoding="utf-8", errors="replace")
        markers = ('lang="tr"', 'id="targetInput"', 'id="referenceInput"',
                   'id="swapButton"', "styles.css", "app.js", "manifest.webmanifest")
        missing = [marker for marker in markers if marker not in source]
        checks.append({
            "check": "web-entry", "file": "index.html", "ok": not missing,
            "detail": "temel erişilebilir arayüz işaretleri mevcut" if not missing
                      else "eksik işaretler: " + ", ".join(missing),
        })

    if app_path.is_file():
        source = app_path.read_text(encoding="utf-8", errors="replace")
        forbidden = [marker for marker in FORBIDDEN_WEB_MARKERS if marker in source]
        pinned = "@mediapipe/tasks-vision@1.2.2" in source
        passed = not forbidden and pinned
        checks.append({
            "check": "web-runtime", "file": "app.js", "ok": passed,
            "detail": "eski geometrik swap kodu yok; MediaPipe sürümü sabit" if passed
                      else "eski kod veya sabitlenmemiş bağımlılık: "
                           + ", ".join(forbidden or ["MediaPipe sürümü sabit değil"]),
        })


    source_files = [ROOT / "index.html", ROOT / "app.js"]
    source_files.extend((ROOT / "ios" / "RetakeApp").rglob("*.swift"))
    camera_markers = (
        "getUserMedia", "getDisplayMedia", "navigator.mediaDevices", "MediaStream",
        "AVCaptureSession", "AVCaptureDeviceInput", "AVCaptureVideoDataOutput",
        "AVCapturePhotoOutput", "UIImagePickerController", "cameraFlipBtn", "startCamera",
    )
    camera_hits: list[str] = []
    for path in source_files:
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        for marker in camera_markers:
            if marker in source:
                camera_hits.append(f"{path.relative_to(ROOT)}:{marker}")
    checks.append({
        "check": "no-camera",
        "file": "web/iOS source",
        "ok": not camera_hits,
        "detail": "kamera ve canlı video API'si yok" if not camera_hits
                  else "yasak kamera API'si bulundu: " + ", ".join(camera_hits),
    })

    workflow_path = ROOT / ".github/workflows/web-quality.yml"
    checks.append({
        "check": "web-ci", "file": ".github/workflows/web-quality.yml",
        "ok": workflow_path.is_file(),
        "detail": "web kalite iş akışı mevcut" if workflow_path.is_file() else "web CI eksik",
    })

    failures = [item for item in checks if not item["ok"]]
    report = {"ok": not failures, "passed": len(checks) - len(failures),
              "failed": len(failures), "checks": checks}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("RETAKE — REPOSITORY QUALITY CHECK")
        print("=" * 72)
        for item in checks:
            state = "PASS" if item["ok"] else "FAIL"
            print(f"[{state}] {item['file']}: {item['detail']}")
        print("=" * 72)
        print(f"Sonuç: {report['passed']} başarılı, {report['failed']} başarısız kontrol.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())