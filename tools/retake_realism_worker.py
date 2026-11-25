"""Photographic-realism worker, run by the Retake engine in its own Python
environment (torch + diffusers; see tools/install_realism.ps1).

    python retake_realism_worker.py --models-dir DIR --swapped final.png
        --target target.png --bounds '{"x":..,"y":..,"width":..,"height":..}'
        --identity-file centroid.json --out-dir DIR

Writes one PNG per plan (realism-1.png, ...) and prints a single line
RETAKE_REALISM_REPORT={...} with the written paths, the device and timings.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from retake_realism import (  # noqa: E402
    PERSONAL_PLANS, PERSONAL_WEIGHTS, PLANS, Renderer, lower_face_hybrid, refine_face, validated_identity
)

CHECKPOINT = "Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors"
VAE_DIR = "sd-vae-ft-mse"
FACEID_DIR = "ip-adapter-faceid"
PERSONAL_DIR = "personal-lora"


def use_system_certificates() -> None:
    """Trust what Windows trusts. Behind a company proxy or antivirus that
    inspects HTTPS, Python's own certificate list rejects Hugging Face (model
    configs are fetched on first use); the system store has their root."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:  # optional: plain networks work without it
        pass


def personal_model(models_dir: Path) -> Path | None:
    """The person's trained LoRA folder, if tools/train_personal_model.ps1 made one."""
    folder = models_dir / PERSONAL_DIR
    return folder if (folder / PERSONAL_WEIGHTS).is_file() and (folder / "person.json").is_file() else None


def model_paths(models_dir: Path) -> dict[str, Path]:
    return {
        "checkpoint": models_dir / CHECKPOINT,
        "vae": models_dir / VAE_DIR,
        "faceid": models_dir / FACEID_DIR,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--swapped", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--bounds", required=True)
    parser.add_argument("--identity-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--landmarks", type=Path, default=None, help="478-point mesh of the face (JSON, image px)")
    parser.add_argument("--seed-base", type=int, default=7, help="first diffusion seed (another finalist uses another base)")
    parser.add_argument("--check", action="store_true", help="only load the models")
    args = parser.parse_args()
    use_system_certificates()

    started = time.time()
    paths = model_paths(args.models_dir)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Gerçekçilik modelleri eksik: {', '.join(missing)} ({args.models_dir})")
    # Validate image geometry and identity before loading gigabytes of weights.
    # In check-only mode, loading the installed models is the sole purpose.
    if not args.check:
        swapped = cv2.imread(str(args.swapped), cv2.IMREAD_COLOR)
        target = cv2.imread(str(args.target), cv2.IMREAD_COLOR)
        if swapped is None or target is None:
            raise SystemExit("Gerçekçilik için sonuç veya hedef görseli okunamadı.")
        if swapped.shape != target.shape:
            raise SystemExit(f"Gerçekçilik boyut uyuşmazlığı: sonuç={swapped.shape}, hedef={target.shape}.")
        bounds = json.loads(args.bounds)
        identity = validated_identity(np.asarray(json.loads(args.identity_file.read_text(encoding="utf-8")), np.float32))
    personal = personal_model(args.models_dir)
    renderer = Renderer(
        str(paths["checkpoint"]), vae=str(paths["vae"]), faceid_dir=str(paths["faceid"]),
        personal_dir=str(personal) if personal else None
    )
    loaded = time.time()
    if args.check:
        print("RETAKE_REALISM_REPORT=" + json.dumps({
            "ok": True, "device": renderer.device, "personal": renderer.personal,
            "faceid_embeddings": renderer.identity
        }), flush=True)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = None
    if args.landmarks is not None and args.landmarks.is_file():
        points = np.asarray(json.loads(args.landmarks.read_text(encoding="utf-8")), np.float32)
        if points.shape != (478, 2) or not np.isfinite(points).all():
            points = None
    outputs = []
    failures = []
    for index, (identity_scale, strength) in enumerate(PERSONAL_PLANS if renderer.personal else PLANS, start=1):
        try:
            rendered = refine_face(
                renderer, swapped, target, bounds, strength,
                seed=max(0, args.seed_base) + index, identity=identity, identity_scale=identity_scale
            )
            if rendered.shape != swapped.shape or rendered.dtype != np.uint8:
                raise ValueError("Gerçekçilik adayı boyutu/türü geçersiz.")
            if np.array_equal(rendered, swapped):
                continue
            path = args.out_dir / f"realism-{index}.png"
            if not cv2.imwrite(str(path), rendered):
                raise OSError("Gerçekçilik sonucu yazılamadı.")
            outputs.append({"path": str(path), "identity_scale": identity_scale, "strength": strength})
            # Reuse the same render: no additional diffusion runs. Conservative
            # blends also help the generic model when the full render drifts.
            if points is not None:
                for blend in ((0.35, 1.0) if renderer.personal else (0.20, 0.35, 0.50)):
                    hybrid = args.out_dir / f"realism-{index}-lower-{int(blend * 100)}.png"
                    candidate = lower_face_hybrid(swapped, rendered, points, strength=blend)
                    if np.array_equal(candidate, swapped):
                        continue
                    if not cv2.imwrite(str(hybrid), candidate):
                        raise OSError("Alt yüz gerçekçilik sonucu yazılamadı.")
                    outputs.append({"path": str(hybrid), "identity_scale": identity_scale,
                                    "strength": strength, "lower_face_only": True, "blend": blend})
        except Exception as error:
            failures.append({"plan": index, "type": type(error).__name__, "message": str(error)[:300]})
            print(f"[Gerçekçilik] Aday {index} atlandı: {error}", file=sys.stderr, flush=True)
    if not outputs:
        print("RETAKE_REALISM_REPORT=" + json.dumps({"ok": False, "outputs": [], "failures": failures}), flush=True)
        return 1
    print("RETAKE_REALISM_REPORT=" + json.dumps({
        "ok": True,
        "device": renderer.device,
        "personal": renderer.personal,
        "outputs": outputs,
        "failures": failures,
        "load_seconds": round(loaded - started, 1),
        "render_seconds": round(time.time() - loaded, 1),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
