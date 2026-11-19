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

from retake_realism import PLANS, Renderer, refine_face  # noqa: E402

CHECKPOINT = "Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors"
VAE_DIR = "sd-vae-ft-mse"
FACEID_DIR = "ip-adapter-faceid"


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
    parser.add_argument("--check", action="store_true", help="only load the models")
    args = parser.parse_args()

    started = time.time()
    paths = model_paths(args.models_dir)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Gerçekçilik modelleri eksik: {', '.join(missing)} ({args.models_dir})")
    renderer = Renderer(str(paths["checkpoint"]), vae=str(paths["vae"]), faceid_dir=str(paths["faceid"]))
    loaded = time.time()
    if args.check:
        print("RETAKE_REALISM_REPORT=" + json.dumps({"ok": True, "device": renderer.device}), flush=True)
        return 0

    swapped = cv2.imread(str(args.swapped), cv2.IMREAD_COLOR)
    target = cv2.imread(str(args.target), cv2.IMREAD_COLOR)
    if swapped is None or target is None or swapped.shape != target.shape:
        raise SystemExit("Gerçekçilik için sonuç ve hedef görseli okunamadı ya da boyutları farklı.")
    bounds = json.loads(args.bounds)
    identity = np.asarray(json.loads(args.identity_file.read_text(encoding="utf-8")), np.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, (identity_scale, strength) in enumerate(PLANS, start=1):
        rendered = refine_face(
            renderer, swapped, target, bounds, strength,
            seed=7 + index, identity=identity, identity_scale=identity_scale
        )
        path = args.out_dir / f"realism-{index}.png"
        if not cv2.imwrite(str(path), rendered):
            raise SystemExit(f"Gerçekçilik sonucu yazılamadı: {path}")
        outputs.append({"path": str(path), "identity_scale": identity_scale, "strength": strength})
    print("RETAKE_REALISM_REPORT=" + json.dumps({
        "ok": True,
        "device": renderer.device,
        "outputs": outputs,
        "load_seconds": round(loaded - started, 1),
        "render_seconds": round(time.time() - loaded, 1),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
