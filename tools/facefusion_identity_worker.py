from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from identity_consistency import analyze_identity_embeddings

REPORT_PREFIX = "RETAKE_IDENTITY_REPORT="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retake local ArcFace reference consistency worker.")
    parser.add_argument("--facefusion-dir", required=True, type=Path)
    parser.add_argument("--reference-paths", required=True, nargs="+", type=Path)
    return parser.parse_args()


def run(facefusion_dir: Path, reference_paths: list[Path]) -> dict[str, object]:
    if len(reference_paths) != 25:
        raise ValueError(f"ArcFace doğrulaması için tam 25 referans gerekli; gelen: {len(reference_paths)}.")

    sys.path.insert(0, str(facefusion_dir))
    from facefusion import face_detector, face_recognizer, state_manager, vision

    # This worker is a separate process so its CUDA sessions are released
    # before the two high-resolution FaceFusion swap passes start.
    state_manager.init_item("execution_device_ids", [0])
    state_manager.init_item("execution_providers", ["cuda", "cpu"])
    state_manager.init_item("face_detector_model", "yolo_face")
    state_manager.init_item("face_detector_size", "640x640")
    state_manager.init_item("face_detector_margin", [0, 0, 0, 0])
    state_manager.init_item("face_detector_score", 0.35)

    if not face_detector.pre_check():
        raise RuntimeError("FaceFusion yüz algılayıcı ağırlıkları hazırlanamadı.")
    if not face_recognizer.pre_check():
        raise RuntimeError("FaceFusion ArcFace ONNX ağırlığı hazırlanamadı.")

    embeddings: list[list[float]] = []
    names: list[str] = []
    for reference_path in reference_paths:
        if not reference_path.is_file():
            raise RuntimeError(f"Referans dosyası bulunamadı: {reference_path.name}")
        image = vision.read_static_image(str(reference_path))
        if image is None:
            raise RuntimeError(f"Referans görseli okunamadı: {reference_path.name}")
        boxes, scores, landmarks = face_detector.detect_faces(image)
        if not boxes:
            raise RuntimeError(f"ArcFace ön kontrolü yüz bulamadı: {reference_path.name}")

        def detection_rank(index: int) -> tuple[float, float]:
            box = boxes[index]
            area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            return area, float(scores[index])

        selected = max(range(len(boxes)), key=detection_rank)
        _, normalized_embedding = face_recognizer.calculate_face_embedding(image, landmarks[selected])
        embeddings.append([float(value) for value in normalized_embedding.ravel()])
        names.append(reference_path.name)

    report = analyze_identity_embeddings(embeddings, names)
    report["ok"] = True
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run(args.facefusion_dir.expanduser().resolve(), args.reference_paths)
        print(REPORT_PREFIX + json.dumps(report, ensure_ascii=True), flush=True)
        return 0
    except Exception as error:
        print(REPORT_PREFIX + json.dumps({"ok": False, "error": str(error)}, ensure_ascii=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
