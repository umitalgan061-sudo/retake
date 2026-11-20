"""Photographic realism for the swapped face (optional, needs a GPU).

Swap models paint the face at 256 px: skin turns waxy, lips flat and glossy,
eyes smudged. Here a diffusion model trained on real photographs (Realistic
Vision, Stable Diffusion 1.5 inpainting) re-renders only the face. The
FaceFusion result is its starting point, so pose, expression and light stay,
and IP-Adapter FaceID steers it with the person's own ArcFace identity (the
centroid of the 25 references), so realism does not pull the face towards a
generic person. What the model adds is what a camera records: pores, fine
hairs, lip creases, eyelids, catchlights, sensor grain.

The broad light and colour are then taken back from the swap. The engine
renders a few plans, measures each with ArcFace and keeps the one closest to
the person; if none beats the swap itself, the swap is kept.

Run as a separate process (tools/retake_realism_worker.py) so the torch
installation never touches the FaceFusion environment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

MIN_RENDER_SIZE = 512         # SD 1.5 native resolution
MAX_RENDER_SIZE = 768         # larger crops are rendered here and keep the swap's broad detail
CROP_SCALE = 1.55             # crop side ÷ larger face side
# (identity guidance, denoising strength), each with its own seed. Measured
# on real photos: 0.45-0.55 with guidance 1.2-1.3 is photographic and keeps
# (usually raises) ArcFace similarity; at 0.65 the score can still rise while
# the nose and cheeks start to look swollen, so the engine never goes there.
PLANS = ((1.2, 0.45), (1.3, 0.55), (1.3, 0.55))
# With the person's own model the render knows their face, beard and skin,
# so it can go a little further; ArcFace still decides per photo.
PERSONAL_PLANS = ((1.0, 0.45), (1.0, 0.55), (1.1, 0.62))
STEPS = 30
GUIDANCE = 4.5
LIGHT_SIGMA_RATIO = 0.045     # broader than this comes back from the swap (light, colour, shading)

FACEID_WEIGHTS = "ip-adapter-faceid_sd15.bin"
IDENTITY_SCALE = 1.3

PROMPT = (
    "RAW photo, close-up of a real person's face, natural skin texture with visible pores, "
    "fine vellus hair, natural lips with fine creases, detailed eyes with natural catchlights, "
    "natural light, sharp focus, shot on a DSLR, subtle film grain"
)
PERSONAL_WEIGHTS = "pytorch_lora_weights.safetensors"
PERSONAL_LORA_SCALE = 0.8


def personal_prompt(trigger: str) -> str:
    """The realism prompt naming the person the LoRA was trained on."""
    return PROMPT.replace("a real person's face", f"{trigger} person's face")


NEGATIVE_PROMPT = (
    "cgi, 3d render, painting, illustration, cartoon, anime, airbrushed, smooth plastic skin, "
    "waxy skin, beauty filter, makeup, lipstick, glossy lips, blurry, out of focus, deformed, "
    "asymmetric eyes, extra teeth, watermark, text"
)


def render_size(side: int) -> int:
    """Render at the crop's own resolution (multiple of 64) within SD 1.5's range."""
    return int(min(MAX_RENDER_SIZE, max(MIN_RENDER_SIZE, round(side / 64.0) * 64)))


@dataclass
class FaceCrop:
    x: int
    y: int
    side: int

    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.side, self.y + self.side


def face_crop(shape: tuple[int, int], face_bounds: dict[str, float]) -> FaceCrop:
    """Square crop around the face (may extend past the image; padded on read)."""
    side = int(math.ceil(max(face_bounds["width"], face_bounds["height"]) * CROP_SCALE))
    cx = face_bounds["x"] + face_bounds["width"] / 2.0
    cy = face_bounds["y"] + face_bounds["height"] / 2.0
    return FaceCrop(int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), max(16, side))


def read_crop(image: np.ndarray, crop: FaceCrop) -> np.ndarray:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = crop.box()
    pad = (max(0, -y0), max(0, y1 - height), max(0, -x0), max(0, x1 - width))
    region = image[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
    if any(pad):
        region = cv2.copyMakeBorder(region, *pad, cv2.BORDER_REFLECT_101)
    return region


def write_crop(image: np.ndarray, crop: FaceCrop, patch: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Blend `patch` (crop.side²) into `image` with a 0..1 `weight`."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = crop.box()
    ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return image
    sub = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
    out = image.astype(np.float32)
    w = weight[sub][:, :, None]
    out[iy0:iy1, ix0:ix1] = out[iy0:iy1, ix0:ix1] * (1.0 - w) + patch[sub].astype(np.float32) * w
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def face_mask(shape: tuple[int, int], changed: np.ndarray, face_bounds: dict[str, float], crop: FaceCrop) -> np.ndarray:
    """0..1 area to re-render, in crop space: where the swap changed the face
    (its mask already keeps hair, hands and glasses out), inside a face ellipse."""
    side = crop.side
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    cx = face_bounds["x"] + face_bounds["width"] / 2.0 - crop.x
    cy = face_bounds["y"] + face_bounds["height"] * 0.52 - crop.y
    rx = face_bounds["width"] * 0.56
    ry = face_bounds["height"] * 0.60
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    oval = np.clip((1.0 - ellipse) / 0.18, 0.0, 1.0)
    changed_crop = read_crop(changed.astype(np.float32), crop)
    feather = max(1.5, face_bounds["width"] * 0.025)
    changed_crop = cv2.GaussianBlur(changed_crop, (0, 0), feather)
    return np.clip(oval * changed_crop, 0.0, 1.0).astype(np.float32)


def restore_broad_light(rendered: np.ndarray, swapped: np.ndarray, face_width: float) -> np.ndarray:
    """Keep the render's fine detail, the swap's light and colour (in crop space)."""
    sigma = max(1.5, face_width * LIGHT_SIGMA_RATIO)
    r = rendered.astype(np.float32)
    s = swapped.astype(np.float32)
    return np.clip(r - cv2.GaussianBlur(r, (0, 0), sigma) + cv2.GaussianBlur(s, (0, 0), sigma), 0, 255)


FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOSE = [1, 2, 98, 327, 64, 294, 4, 5, 195, 197, 6, 168]


def lower_face_weight(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """0..1 weight of the cheeks, jaw, chin, moustache and lips (not the eyes,
    brows or nose), from a 478-point mesh in image pixels.

    With the person's own model, this part of a render carries what a swap
    cannot: where their beard grows, its density, their lips. The eyes, brows
    and nose, which carry most of the ArcFace identity, stay the swap's.
    Measured on the user's photos: a full personal render scored 0.69, the
    swap 0.79, the swap with the render's lower face 0.79 with the person's
    full beard.
    """
    height, width = shape
    points = np.asarray(points, np.float32)
    oval = np.zeros((height, width), np.uint8)
    cv2.fillPoly(oval, [np.round(points[FACE_OVAL]).astype(np.int32)], 1)
    nose = np.zeros_like(oval)
    cv2.fillConvexPoly(nose, cv2.convexHull(np.round(points[NOSE]).astype(np.int32)), 1)
    rows = np.arange(height, dtype=np.float32)[:, None]
    region = (oval > 0) & (rows > float(points[1, 1])) & (nose == 0)
    face_width = float(np.ptp(points[:, 0]))
    return cv2.GaussianBlur(region.astype(np.float32), (0, 0), max(1.5, face_width * 0.04))


def lower_face_hybrid(swapped: np.ndarray, rendered: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The swap with the render's lower face (uint8 BGR)."""
    weight = lower_face_weight(points, swapped.shape[:2])[:, :, None]
    blended = swapped.astype(np.float32) * (1.0 - weight) + rendered.astype(np.float32) * weight
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def changed_mask(swapped: np.ndarray, target: np.ndarray) -> np.ndarray:
    changed = (np.max(cv2.absdiff(swapped, target), axis=2) >= 3).astype(np.uint8)
    return cv2.morphologyEx(changed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


class Renderer:
    """Loads the inpainting model once; `render` re-renders a face crop."""

    def __init__(
        self,
        checkpoint: str,
        vae: str | None = None,
        device: str | None = None,
        faceid_dir: str | None = None,
        identity_scale: float = IDENTITY_SCALE,
        personal_dir: str | None = None
    ):
        import torch
        from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_single_file(checkpoint, torch_dtype=dtype)
        if vae:  # folder with config.json + diffusion_pytorch_model.safetensors (sd-vae-ft-mse)
            pipe.vae = AutoencoderKL.from_pretrained(vae, torch_dtype=dtype)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        )
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        pipe.set_progress_bar_config(disable=True)
        self.identity = False
        if faceid_dir:
            # IP-Adapter FaceID: the person's ArcFace identity steers the render,
            # so realism does not pull the face towards a generic person.
            pipe.load_ip_adapter(faceid_dir, subfolder="", weight_name=FACEID_WEIGHTS, image_encoder_folder=None)
            pipe.set_ip_adapter_scale(identity_scale)
            self.identity = True
        self.prompt = PROMPT
        self.personal = False
        if personal_dir:
            # The person's own LoRA (tools/retake_personal_model.py), trained on
            # the base model; its attention layers fit the inpainting UNet too.
            import json
            from pathlib import Path
            info = json.loads((Path(personal_dir) / "person.json").read_text(encoding="utf-8"))
            pipe.load_lora_weights(personal_dir, weight_name=PERSONAL_WEIGHTS, adapter_name="person")
            pipe.set_adapters(["person"], adapter_weights=[PERSONAL_LORA_SCALE])
            self.prompt = personal_prompt(info.get("trigger", "rtkperson"))
            self.personal = True
        self.dtype = dtype
        self.pipe = pipe.to(self.device)
        if self.device == "cuda":
            try:
                self.pipe.enable_attention_slicing()
            except Exception:  # memory saving is optional
                pass

    def render(
        self,
        crop_bgr: np.ndarray,
        mask: np.ndarray,
        strength: float,
        seed: int,
        identity: np.ndarray | None = None,
        identity_scale: float | None = None
    ) -> np.ndarray:
        from PIL import Image

        if self.identity and identity_scale is not None:
            self.pipe.set_ip_adapter_scale(float(identity_scale))
        size = render_size(crop_bgr.shape[0])
        image = Image.fromarray(cv2.cvtColor(cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_AREA
                                                        if crop_bgr.shape[0] > size else cv2.INTER_CUBIC),
                                             cv2.COLOR_BGR2RGB))
        mask_image = Image.fromarray(np.clip(cv2.resize(mask, (size, size)) * 255, 0, 255).astype(np.uint8))
        generator = self.torch.Generator(device="cpu").manual_seed(int(seed))
        extra = {}
        if self.identity:
            if identity is None:
                raise ValueError("kimlik vektörü gerekli")
            embeds = np.asarray(identity, np.float32).reshape(-1, 512)
            embeds = embeds / np.maximum(np.linalg.norm(embeds, axis=1, keepdims=True), 1e-6)
            positive = self.torch.from_numpy(embeds).unsqueeze(0)          # (1, n, 512)
            stacked = self.torch.cat([self.torch.zeros_like(positive), positive])
            extra["ip_adapter_image_embeds"] = [stacked.to(device=self.device, dtype=self.dtype)]
        result = self.pipe(
            **extra,
            prompt=self.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=image,
            mask_image=mask_image,
            strength=float(strength),
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=generator,
            height=size,
            width=size,
        ).images[0]
        rendered = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
        side = crop_bgr.shape[0]
        return cv2.resize(rendered, (side, side), interpolation=cv2.INTER_LANCZOS4 if side > size else cv2.INTER_AREA)


def refine_face(
    renderer: Renderer,
    swapped: np.ndarray,
    target: np.ndarray,
    face_bounds: dict[str, float],
    strength: float,
    seed: int = 7,
    identity: np.ndarray | None = None,
    identity_scale: float | None = None
) -> np.ndarray:
    """The swap with its face re-rendered photographically (uint8 BGR, same size)."""
    crop = face_crop(swapped.shape[:2], face_bounds)
    crop_img = read_crop(swapped, crop)
    mask = face_mask(swapped.shape[:2], changed_mask(swapped, target), face_bounds, crop)
    if float(mask.max()) < 0.5:
        return swapped
    rendered = renderer.render(crop_img, mask, strength, seed, identity, identity_scale)
    rendered = restore_broad_light(rendered, crop_img, face_bounds["width"])
    return write_crop(swapped, crop, rendered, mask)
