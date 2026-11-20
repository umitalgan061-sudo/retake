"""Train the person's own model: a LoRA for the realism layer's diffusion model.

IP-Adapter FaceID steers the realism layer with one identity vector; a model
fine-tuned on the person's 25 photos also knows their beard (how far it runs
up the cheeks), skin, brows and lips. The realism layer uses it automatically
once it exists (tools/retake_realism_worker.py), and every render is still
checked with ArcFace before it is used.

Runs in the realism environment (tools/install_realism.ps1) on an NVIDIA GPU:

    python retake_personal_model.py --models-dir C:\\AI\\RetakeRealism\\models
        --references C:\\AI\\retake\\reference-faces

Writes <models-dir>/personal-lora/pytorch_lora_weights.safetensors and
person.json. Training settings follow the GPU memory (8 GB is enough).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_CHECKPOINT = "Realistic_Vision_V6.0_NV_B1_fp16.safetensors"
PERSONAL_DIR = "personal-lora"
WEIGHTS_NAME = "pytorch_lora_weights.safetensors"
TRIGGER = "rtkperson"
INSTANCE_PROMPT = f"a RAW photo of {TRIGGER} person, face, natural skin texture"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FACE_CROP_SCALE = 2.0      # crop side ÷ detected face side: face, beard, hairline
LORA_RANK = 16
LEARNING_RATE = 1e-4
STEPS_PER_IMAGE = 40       # 25 photos -> 1000 steps
MAX_STEPS = 1500


def use_system_certificates() -> None:
    """Trust what Windows trusts. Behind a company proxy or antivirus that
    inspects HTTPS, Python's own certificate list rejects Hugging Face (model
    configs are fetched on first use); the system store has their root."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:  # optional: plain networks work without it
        pass


def face_crops(paths: list[Path], resolution: int) -> list[np.ndarray]:
    """Square crops around the largest face of each photo (RGB, resolution²)."""
    if not hasattr(cv2, "CascadeClassifier"):  # OpenCV 5 moved it out of the main package
        raise SystemExit(
            "Bu OpenCV sürümünde yüz algılayıcı yok. Şunu çalıştır: "
            "python -m pip install \"opencv-python-headless<5\" (tools\\train_personal_model.ps1 bunu kendisi yapar)."
        )
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    crops = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = min(1.0, 1600.0 / max(image.shape[:2]))
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        faces = detector.detectMultiScale(gray, 1.1, 6, minSize=(64, 64))
        if len(faces) == 0:
            print(f"Yüz bulunamadı, atlandı: {path.name}", flush=True)
            continue
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        # Sunglasses would be learnt as part of the face: keep photos whose
        # eyes are visible.
        upper = gray[y:y + h // 2, x:x + w]
        if len(eyes.detectMultiScale(upper, 1.05, 2, minSize=(max(6, w // 14), max(6, w // 14)))) == 0:
            print(f"Gözler görünmüyor (gözlük?), atlandı: {path.name}", flush=True)
            continue
        cx, cy = (x + w / 2) / scale, (y + h / 2 + 0.08 * h) / scale
        side = max(w, h) / scale * FACE_CROP_SCALE
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        size = int(round(side))
        pad = max(0, -x0, -y0, x0 + size - image.shape[1], y0 + size - image.shape[0])
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101) if pad else image
        crop = padded[y0 + pad:y0 + pad + size, x0 + pad:x0 + pad + size]
        crop = cv2.resize(crop, (resolution, resolution), interpolation=cv2.INTER_AREA)
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return crops


def plan(device: str, memory_gb: float | None, image_count: int, steps: int | None) -> dict:
    """Training settings for the hardware at hand."""
    total = steps or min(MAX_STEPS, max(400, STEPS_PER_IMAGE * image_count))
    if device != "cuda":
        return {"resolution": 256, "steps": steps or 10, "checkpointing": False, "fp16": False}
    return {
        "resolution": 512,
        "steps": total,
        # Under ~10 GB, trade speed for memory.
        "checkpointing": memory_gb is None or memory_gb < 10,
        "fp16": True,
    }


def train(
    models_dir: Path,
    references: Path,
    steps: int | None = None,
    seed: int = 7,
    resolution: int | None = None
) -> Path:
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    base = models_dir / BASE_CHECKPOINT
    if not base.is_file():
        raise SystemExit(f"Temel model bulunamadı: {base}")
    photos = sorted(path for path in references.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(photos) < 5:
        raise SystemExit(f"En az 5 fotoğraf gerekli ({references}).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if device == "cuda" else None
    settings = plan(device, memory_gb, len(photos), steps)
    if resolution:
        settings["resolution"] = int(resolution)
    print(f"Kişisel model eğitimi: {device}"
          + (f" ({torch.cuda.get_device_name(0)}, {memory_gb:.1f} GB)" if memory_gb else "")
          + f", {settings['steps']} adım, {settings['resolution']} px.", flush=True)

    crops = face_crops(photos, settings["resolution"])
    if len(crops) < 5:
        raise SystemExit("Yeterli yüz bulunamadı (en az 5).")
    random.seed(seed)
    torch.manual_seed(seed)

    weight_dtype = torch.float16 if settings["fp16"] else torch.float32
    pipe = StableDiffusionPipeline.from_single_file(str(base), torch_dtype=torch.float32)
    vae_dir = models_dir / "sd-vae-ft-mse"
    if vae_dir.is_dir():
        from diffusers import AutoencoderKL
        pipe.vae = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=torch.float32)
    unet, vae, text_encoder, tokenizer = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    for module in (unet, vae, text_encoder):
        module.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    # Everything frozen is computed once: latents (with flips) and the prompt.
    with torch.no_grad():
        ids = tokenizer(INSTANCE_PROMPT, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        prompt_embeds = text_encoder(ids)[0].to(torch.float32)
        latents = []
        for crop in crops:
            for image in (crop, crop[:, ::-1]):
                tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 127.5 - 1.0
                encoded = vae.encode(tensor[None].to(device, dtype=weight_dtype)).latent_dist.sample()
                latents.append((encoded * vae.config.scaling_factor).to(torch.float32).cpu())
    vae.to("cpu")
    text_encoder.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    unet.add_adapter(LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    unet.to(device)
    if settings["checkpointing"]:
        unet.enable_gradient_checkpointing()
    parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=1e-2)
    total = settings["steps"]
    warmup = max(1, total // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total))
    )
    autocast = torch.autocast("cuda", dtype=torch.float16) if settings["fp16"] else torch.autocast("cpu", enabled=False)
    scaler = torch.amp.GradScaler("cuda") if settings["fp16"] else None
    unet.train()
    started = time.time()
    running = None
    for step in range(total):
        latent = random.choice(latents).to(device)
        noise = torch.randn_like(latent)
        timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy = scheduler.add_noise(latent, noise, timestep)
        with autocast:
            prediction = unet(noisy, timestep, prompt_embeds).sample
        target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latent, noise, timestep)
        loss = F.mse_loss(prediction.float(), target.float())
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        schedule.step()
        running = loss.item() if running is None else running * 0.95 + loss.item() * 0.05
        if step % max(1, total // 20) == 0 or step == total - 1:
            elapsed = time.time() - started
            print(f"  adım {step + 1}/{total} · kayıp {running:.4f} · {elapsed:.0f} sn", flush=True)

    out_dir = models_dir / PERSONAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    unet = unet.to(torch.float32)
    layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(out_dir, unet_lora_layers=layers, weight_name=WEIGHTS_NAME,
                                              safe_serialization=True)
    (out_dir / "person.json").write_text(json.dumps({
        "trigger": TRIGGER,
        "photos": len(crops),
        "steps": total,
        "resolution": settings["resolution"],
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kişisel model kaydedildi: {out_dir / WEIGHTS_NAME}", flush=True)
    return out_dir / WEIGHTS_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description="Retake kişisel model (LoRA) eğitimi")
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None, choices=(256, 384, 512, 640))
    args = parser.parse_args()
    use_system_certificates()
    train(args.models_dir, args.references, args.steps, resolution=args.resolution)
    return 0


if __name__ == "__main__":
    sys.exit(main())
