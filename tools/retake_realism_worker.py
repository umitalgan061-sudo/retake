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
    PERSONAL_PLANS, PERSONAL_WEIGHTS, PLANS, Renderer, lower_face_hybrid, refine_face
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


def prepare_windows_huggingface_cache() -> None:
    """Probe cache symlink permissions BEFORE Diffusers' parallel downloads.

    huggingface_hub==0.26.2 lazily populates its symlink-support cache. If
    snapshot_download() starts several worker threads simultaneously, a
    Windows non-admin process can race and attempt os.symlink even though
    symlinks require Developer Mode or elevated privileges (WinError 1314).

    Calling the official probe once on this thread makes the unsupported
    result available to all download threads. Existing HF cache files stay
    intact; the library falls back to ordinary files without admin rights.
    """
    if sys.platform != "win32":
        return
    from huggingface_hub.file_download import are_symlinks_supported

    if not are_symlinks_supported():
        print(
            "Hugging Face cache: Windows symlink izni yok; normal dosya "
            "kopyalari kullanilacak (WinError 1314 onlemi).",
            file=sys.stderr,
            flush=True,
        )


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
    parser.add_argument("--check", action="store_true", help="only load the models")
    args = parser.parse_args()
    use_system_certificates()
    prepare_windows_huggingface_cache()

    started = time.time()
    paths = model_paths(args.models_dir)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Gerçekçilik modelleri eksik: {', '.join(missing)} ({args.models_dir})")
    personal = personal_model(args.models_dir)
    renderer = Renderer(
        str(paths["checkpoint"]), vae=str(paths["vae"]), faceid_dir=str(paths["faceid"]),
        personal_dir=str(personal) if personal else None
    )
    loaded = time.time()
    if args.check:
        print("RETAKE_REALISM_REPORT=" + json.dumps({
            "ok": True, "device": renderer.device, "personal": renderer.personal
        }), flush=True)
        return 0

    swapped = cv2.imread(str(args.swapped), cv2.IMREAD_COLOR)
    target = cv2.imread(str(args.target), cv2.IMREAD_COLOR)
    if swapped is None or target is None or swapped.shape != target.shape:
        raise SystemExit("Gerçekçilik için sonuç ve hedef görseli okunamadı ya da boyutları farklı.")
    bounds = json.loads(args.bounds)
    identity = np.asarray(json.loads(args.identity_file.read_text(encoding="utf-8")), np.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = None
    if args.landmarks is not None and args.landmarks.is_file():
        points = np.asarray(json.loads(args.landmarks.read_text(encoding="utf-8")), np.float32)
        if points.shape != (478, 2):
            points = None
    outputs = []
    for index, (identity_scale, strength) in enumerate(PERSONAL_PLANS if renderer.personal else PLANS, start=1):
        rendered = refine_face(
            renderer, swapped, target, bounds, strength,
            seed=7 + index, identity=identity, identity_scale=identity_scale
        )
        path = args.out_dir / f"realism-{index}.png"
        if not cv2.imwrite(str(path), rendered):
            raise SystemExit(f"Gerçekçilik sonucu yazılamadı: {path}")
        outputs.append({"path": str(path), "identity_scale": identity_scale, "strength": strength})
        if renderer.personal and points is not None:
            hybrid = args.out_dir / f"realism-{index}-lower.png"
            if not cv2.imwrite(str(hybrid), lower_face_hybrid(swapped, rendered, points)):
                raise SystemExit(f"Gerçekçilik sonucu yazılamadı: {hybrid}")
            outputs.append({"path": str(hybrid), "identity_scale": identity_scale, "strength": strength,
                            "lower_face_only": True})
    print("RETAKE_REALISM_REPORT=" + json.dumps({
        "ok": True,
        "device": renderer.device,
        "personal": renderer.personal,
        "outputs": outputs,
        "load_seconds": round(loaded - started, 1),
        "render_seconds": round(time.time() - loaded, 1),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
