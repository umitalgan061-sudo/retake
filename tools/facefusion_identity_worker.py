from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from identity_consistency import analyze_identity_embeddings

REPORT_PREFIX = "RETAKE_IDENTITY_REPORT="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retake local ArcFace reference consistency worker.")
    parser.add_argument("--facefusion-dir", required=True, type=Path)
    parser.add_argument("--reference-paths", required=True, nargs="+", type=Path)
    return parser.parse_args()


def _fail_reference(reference_path: Path, message: str) -> RuntimeError:
    return RuntimeError(f"{reference_path.name}: {message}")


def run(facefusion_dir: Path, reference_paths: list[Path]) -> dict[str, object]:
    if len(reference_paths) != 25:
        raise ValueError(f"ArcFace doğrulaması için tam 25 referans gerekli; gelen: {len(reference_paths)}.")

    sys.path.insert(0, str(facefusion_dir))
    from facefusion import choices, face_detector, face_recognizer, state_manager, vision

    # Keep the verifier isolated from the later high-resolution swap passes.
    # FaceFusion's resolve_download_url() iterates download_providers while
    # building detector/recognizer model metadata. CLI normally initializes
    # this value; a standalone worker must initialize it explicitly.
    state_manager.init_item("download_providers", list(choices.download_providers))
    state_manager.init_item("execution_device_ids", [0])
    state_manager.init_item("execution_providers", ["cuda", "cpu"])
    state_manager.init_item("face_detector_model", "yolo_face")
    state_manager.init_item("face_detector_size", "640x640")
    state_manager.init_item("face_detector_margin", [0, 0, 0, 0])
    state_manager.init_item("face_detector_score", 0.35)

    if not face_detector.pre_check():
        raise RuntimeError("FaceFusion YOLO yüz algılayıcı ağırlıkları hazırlanamadı.")
    if not face_recognizer.pre_check():
        raise RuntimeError("FaceFusion ArcFace ONNX ağırlığı hazırlanamadı.")

    embeddings: list[list[float]] = []
    names: list[str] = []
    for number, reference_path in enumerate(reference_paths, start=1):
        if not reference_path.is_file():
            raise _fail_reference(reference_path, "referans dosyası bulunamadı.")

        image = vision.read_static_image(str(reference_path))
        if image is None:
            raise _fail_reference(reference_path, "FaceFusion görsel okuyucusu None döndürdü.")

        detections = face_detector.detect_faces(image)
        if not isinstance(detections, (tuple, list)) or len(detections) != 3:
            raise _fail_reference(
                reference_path,
                "FaceFusion detect_faces beklenen (boxes, scores, landmarks) üçlüsünü döndürmedi."
            )
        boxes, scores, landmarks = detections
        if boxes is None or scores is None or landmarks is None:
            raise _fail_reference(
                reference_path,
                "FaceFusion algılayıcısı boxes/scores/landmarks alanlarından birini None döndürdü."
            )
        if len(boxes) == 0:
            raise _fail_reference(reference_path, "ArcFace ön kontrolü yüz bulamadı.")
        if len(scores) != len(boxes) or len(landmarks) != len(boxes):
            raise _fail_reference(
                reference_path,
                f"Algılama alanlarının sayısı eşleşmiyor (boxes={len(boxes)}, "
                f"scores={len(scores)}, landmarks={len(landmarks)})."
            )

        def detection_rank(index: int) -> tuple[float, float]:
            box = boxes[index]
            if box is None or len(box) < 4:
                return 0.0, 0.0
            area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            score = float(scores[index]) if scores[index] is not None else 0.0
            return area, score

        selected = max(range(len(boxes)), key=detection_rank)
        selected_landmarks = landmarks[selected]
        if selected_landmarks is None:
            raise _fail_reference(reference_path, "Seçilen yüz için 5 noktalı landmark None döndü.")

        # FaceFusion 3.9.2 returns (raw_embedding, normalized_embedding).
        embedding_result: Any = face_recognizer.calculate_face_embedding(image, selected_landmarks)
        if not isinstance(embedding_result, (tuple, list)) or len(embedding_result) != 2:
            raise _fail_reference(
                reference_path,
                "ArcFace embedding çağrısı beklenen iki vektörlü sonucu döndürmedi."
            )
        raw_embedding, normalized_embedding = embedding_result
        if raw_embedding is None or normalized_embedding is None:
            raise _fail_reference(reference_path, "ArcFace ONNX modeli embedding vektörü döndürmedi.")

        vector = normalized_embedding.ravel()
        if getattr(vector, "size", 0) == 0:
            raise _fail_reference(reference_path, "ArcFace embedding boş.")
        embeddings.append([float(value) for value in vector])
        names.append(reference_path.name)
        print(f"[ArcFace] Referans {number}/25 doğrulandı: {reference_path.name}", flush=True)

    report = analyze_identity_embeddings(embeddings, names)
    report["ok"] = True
    return report


def main() -> int:
    args = parse_args()
    stage = "FaceFusion ArcFace worker başlatılıyor"
    try:
        stage = "FaceFusion dizini doğrulanıyor"
        facefusion_dir = args.facefusion_dir.expanduser().resolve()
        if not (facefusion_dir / "facefusion.py").is_file():
            raise FileNotFoundError(f"FaceFusion facefusion.py bulunamadı: {facefusion_dir}")

        stage = "FaceFusion yüz algılayıcı ve ArcFace modülleri yükleniyor"
        # run() logs each successful reference. The traceback below preserves
        # the exact failing FaceFusion call instead of returning only its text.
        report = run(facefusion_dir, args.reference_paths)
        print(REPORT_PREFIX + json.dumps(report, ensure_ascii=True), flush=True)
        return 0
    except Exception as error:
        detail = (
            f"{type(error).__name__}: {error}\n"
            f"Traceback (son çağrı en altta):\n{traceback.format_exc()}"
        )
        print(REPORT_PREFIX + json.dumps({
            "ok": False,
            "stage": stage,
            "error": detail
        }, ensure_ascii=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
