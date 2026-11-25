"""NVIDIA vision-model photo judge for Retake Ultra finalists (optional).

Retake's own checks (ArcFace identity, photo_naturalness) measure numbers;
a large vision-language model can *look* at a result the way a retoucher
does: waxy skin, a pasted forehead edge, dead eyes, a tone that does not
belong to the neck. This module asks one of NVIDIA's hosted vision models
(build.nvidia.com free endpoint, OpenAI-compatible chat API) to compare a
few near-identical Ultra finalists against the untouched original crop.

What is sent, and only when NVIDIA_API_KEY is available on this computer
(environment, or a git-ignored .env file, see key_source) and
RETAKE_NVIDIA_JUDGE is not "0": small JPEG face crops of the original photo
and of at most JUDGE_MAX_CANDIDATES results. The 25 reference photos, the
full photo, embeddings and file names are never sent. The key is read from
there and never logged or returned.

The judge only re-ranks candidates the identity rules already consider a
tie (see ultra_mode.judge_pick); it cannot trade likeness for looks. It is
asked twice with the candidate order reversed, and a switch needs both
answers to agree, which removes the position bias these models have.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# Hosted vision models of the free catalogue, best first; the next one is
# tried when a model is unavailable, overloaded or answers without JSON.
MODELS = (
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "mistralai/mistral-large-4-1t-a52b",
    "z-ai/glm-5.3-flash",
    "stepfun-ai/step-5-preview",
)
JUDGE_MAX_CANDIDATES = 3
CROP_SIDE = 512                # px, longest side of each face crop
CROP_MARGIN = 0.28             # of the face box, around it (hairline, jaw, neck edge)
JPEG_QUALITY = 90
TIMEOUT_SECONDS = 75
MAX_RESPONSE_BYTES = 512 * 1024
ISSUES = ("seam", "plastic_skin", "tone", "lighting", "eyes", "mouth",
          "blur", "oversharp", "warp", "hair", "noise")
ISSUE_WORDS = {
    "seam": "kenar dikişi", "plastic_skin": "plastik cilt", "tone": "ten rengi",
    "lighting": "ışık", "eyes": "gözler", "mouth": "ağız/diş", "blur": "bulanıklık",
    "oversharp": "aşırı keskinlik", "warp": "şekil bozulması", "hair": "saç çizgisi",
    "noise": "gren/gürültü",
}

SYSTEM_PROMPT = (
    "You are a meticulous senior photo retoucher reviewing face-replacement edits. "
    "You judge only photographic realism and consistency with the original photo. "
    "You never identify people and never judge attractiveness. "
    "Answer with one JSON object and nothing else."
)


def _user_prompt(count: int, sheet: bool = False) -> str:
    labels = ", ".join(f"{'panel' if sheet else 'image'} {index}" for index in range(2, count + 2))
    layout = (f"The picture is one row of {count + 1} panels, numbered in their top-left corner. "
              "Panel 1 is" if sheet else "Image 1 is")
    return (
        f"{layout} an untouched crop of the ORIGINAL photograph; use it as the reference for camera, "
        "lighting direction, color, sharpness and grain. "
        f"{labels} are edited versions of the same crop in which the face was replaced. "
        "For each edited image, judge how convincingly it looks like an untouched photograph taken with "
        "the same camera at the same moment. Inspect: skin texture (pores vs. plastic/waxy), blend edges at "
        "the forehead, hairline, cheeks and jaw, skin tone versus neck and ears, light and shadow direction, "
        "eyes (catchlights, symmetry, gaze), lips and teeth, blur or over-sharpening relative to the rest of "
        "the photo, geometric warping. Score realism from 0 (obviously edited) to 10 (indistinguishable "
        "from a real photo); use decimals and spread the scores when the images differ. "
        f"List visible problems using only these words: {', '.join(ISSUES)}. "
        'Reply exactly as: {"candidates": [{"image": 2, "realism": 7.5, "issues": ["seam"]}, ...], '
        '"best": 2}' + (' where "image" is the panel number.' if sheet else "")
    )


ROOT = Path(__file__).resolve().parents[1]
# Git-ignored local secret files, in the order they are consulted after the
# process environment. RETAKE_ENV_FILE may name one more file explicitly.
ENV_FILES = (".env", "functions/.env", ".dev.vars")
KEY_NAME = "NVIDIA_API_KEY"


def _key_from_file(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    # PowerShell 5 writes UTF-16 by default (Out-File, >); editors write UTF-8.
    encoding = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    try:
        lines = raw.decode(encoding).splitlines()
    except UnicodeDecodeError:
        return None
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, sep, value = line.partition("=")
        if not sep or name.strip() != KEY_NAME:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        if value:
            return value
    return None


def key_source() -> tuple[str | None, str]:
    """(key, where it came from) — the source is a name, never the value."""
    if os.environ.get("RETAKE_NVIDIA_JUDGE", "").strip() == "0":
        return None, "RETAKE_NVIDIA_JUDGE=0"
    key = os.environ.get(KEY_NAME, "").strip()
    if key:
        return key, "ortam değişkeni"
    candidates = [ROOT / name for name in ENV_FILES]
    explicit = os.environ.get("RETAKE_ENV_FILE", "").strip()
    if explicit:
        candidates.insert(0, Path(explicit).expanduser())
    for path in candidates:
        key = _key_from_file(path)
        if key:
            return key, path.name if path.parent == ROOT else str(path)
    return None, "bulunamadı"


def api_key() -> str | None:
    return key_source()[0]


def enabled() -> bool:
    return api_key() is not None


def _decode(data: bytes) -> np.ndarray | None:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def flatten(result: bytes, target: bytes) -> np.ndarray | None:
    """The result as the user sees it: an alpha result goes over the target."""
    image = _decode(result)
    if image is None:
        return None
    if image.shape[2] == 4:
        base = _decode(target)
        if base is None or base.shape[:2] != image.shape[:2]:
            return image[:, :, :3]
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(image[:, :, :3] * alpha + base[:, :, :3] * (1 - alpha), 0, 255).astype(np.uint8)
    return image[:, :, :3]


def face_crop(image: np.ndarray, bounds: dict | None) -> bytes:
    """JPEG of the face and its surroundings, longest side CROP_SIDE."""
    height, width = image.shape[:2]
    if bounds:
        x, y = float(bounds.get("x", 0)), float(bounds.get("y", 0))
        w, h = float(bounds.get("width", width)), float(bounds.get("height", height))
        left = int(max(0, x - CROP_MARGIN * w))
        top = int(max(0, y - CROP_MARGIN * h))
        right = int(min(width, x + w * (1 + CROP_MARGIN)))
        bottom = int(min(height, y + h * (1 + CROP_MARGIN)))
        if right - left >= 32 and bottom - top >= 32:
            image = image[top:bottom, left:right]
    scale = CROP_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("Yüz kırpımı kodlanamadı.")
    return encoded.tobytes()


SHEET_PANEL_HEIGHT = 360


def contact_sheet(images: list[bytes]) -> bytes:
    """One JPEG with the crops side by side, numbered 1..N (for one-image models)."""
    panels = []
    for number, data in enumerate(images, 1):
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Yüz kırpımı okunamadı.")
        scale = SHEET_PANEL_HEIGHT / image.shape[0]
        image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), SHEET_PANEL_HEIGHT),
                           interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (44, 40), (255, 255, 255), -1)
        cv2.putText(image, str(number), (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        panels.append(image)
        panels.append(np.full((SHEET_PANEL_HEIGHT, 8, 3), 255, np.uint8))
    ok, encoded = cv2.imencode(".jpg", np.hstack(panels[:-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("Karşılaştırma görseli kodlanamadı.")
    return encoded.tobytes()


def _content(images: list[bytes], sheet: bool = False) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": _user_prompt(len(images) - 1, sheet)}]
    for data in [contact_sheet(images)] if sheet else images:
        parts.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}})
    return parts


def _ssl_context() -> ssl.SSLContext:
    # Corporate HTTPS inspection: trust the Windows certificate store when
    # the truststore package is available (as the realism installer does).
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        return ssl.create_default_context()


def post_json(payload: dict[str, Any], key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS, context=_ssl_context()) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise JudgeError(describe_status(error.code), error.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise JudgeError(f"NVIDIA'ya bağlanılamadı ({type(error).__name__}).") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise JudgeError("NVIDIA yanıtı çok büyük.")
    return json.loads(raw.decode("utf-8"))


class JudgeError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def describe_status(status: int) -> str:
    if status in (401, 403):
        return "NVIDIA API anahtarı geçersiz veya bu modele erişimi yok."
    if status == 404:
        return "Model bu hesapta etkin değil."
    if status == 413:
        return "Görseller NVIDIA'nın sınırını aştı."
    if status == 429:
        return "NVIDIA istek sınırı doldu."
    if status >= 500:
        return "NVIDIA servisi geçici olarak hata verdi."
    return f"NVIDIA isteği reddetti (HTTP {status})."


def parse_verdict(text: str, count: int) -> dict[int, dict[str, Any]] | None:
    """{candidate index 0..count-1: {"realism": 0..10, "issues": [...]}} or None."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    starts = [match.start() for match in re.finditer(r"\{", text)]
    for start in starts:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start:end + 1])
                    except ValueError:
                        break
                    verdict = _read_candidates(data, count)
                    if verdict is not None:
                        return verdict
                    break
    return None


def _read_candidates(data: Any, count: int) -> dict[int, dict[str, Any]] | None:
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        return None
    verdict: dict[int, dict[str, Any]] = {}
    for row in data["candidates"]:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("image")) - 2
            realism = float(row.get("realism"))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < count or not math.isfinite(realism) or index in verdict:
            continue
        issues = [issue for issue in row.get("issues", []) if isinstance(issue, str) and issue in ISSUES] \
            if isinstance(row.get("issues"), list) else []
        verdict[index] = {"realism": min(10.0, max(0.0, realism)), "issues": issues}
    return verdict if len(verdict) == count else None


def ask(images: list[bytes], key: str, poster: Callable[[dict, str], dict] = post_json,
        models: tuple[str, ...] = MODELS) -> tuple[str, dict[int, dict[str, Any]]]:
    """One judgement of len(images)-1 candidates. Returns (model, verdict)."""
    last: Exception | None = None
    for model in models:
        # Separate images first; a model that takes only one image per
        # request gets the same crops as one numbered contact sheet.
        for sheet in (False, True):
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": _content(images, sheet)}],
                "temperature": 0.1, "top_p": 0.7, "max_tokens": 1200, "stream": False,
            }
            try:
                response = poster(payload, key)
                message = (response.get("choices") or [{}])[0].get("message") or {}
                verdict = parse_verdict(message.get("content") or "", len(images) - 1)
            except JudgeError as error:
                last = error
                if error.status in (400, 413, 422) and not sheet:
                    continue
                break
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                last = error
                break
            if verdict is not None:
                return model, verdict
            last = JudgeError(f"{model} okunabilir puan döndürmedi.")
            break
    raise last or JudgeError("NVIDIA hakemi yanıt vermedi.")


def judge(original: bytes, candidates: list[bytes], key: str,
          poster: Callable[[dict, str], dict] = post_json) -> dict[str, Any]:
    """Two passes (given order, then reversed) over face-crop JPEGs.

    Returns {"model", "scores": [mean realism per candidate],
    "passes": [[realism per candidate] per pass], "issues": [[...] per candidate]}.
    """
    if not 1 <= len(candidates) <= JUDGE_MAX_CANDIDATES:
        raise ValueError("Hakem 1-3 aday karşılaştırır.")
    model, first = ask([original, *candidates], key, poster)
    passes = [[first[index]["realism"] for index in range(len(candidates))]]
    issues = [set(first[index]["issues"]) for index in range(len(candidates))]
    if len(candidates) > 1:
        reversed_candidates = list(reversed(candidates))
        try:
            _, second = ask([original, *reversed_candidates], key, poster, models=(model,))
        except (JudgeError, ValueError):
            second = None
        if second is not None:
            count = len(candidates)
            passes.append([second[count - 1 - index]["realism"] for index in range(count)])
            for index in range(count):
                issues[index] &= set(second[count - 1 - index]["issues"])
    scores = [round(sum(row[index] for row in passes) / len(passes), 2) for index in range(len(candidates))]
    return {"model": model, "scores": scores, "passes": passes,
            "issues": [sorted(found) for found in issues]}


def describe(score: float | None, issues: list[str]) -> str:
    if score is None:
        return ""
    text = f"NVIDIA hakem {score:.1f}/10"
    if issues:
        text += " · " + ", ".join(ISSUE_WORDS.get(issue, issue) for issue in issues[:3])
    return text


def _selftest() -> int:
    """Checks the key and the model on a synthetic (non-personal) image."""
    key, source = key_source()
    if key is None:
        print(f"NVIDIA_API_KEY bulunamadı ({source}). Aranan: ortam değişkeni, "
              + ", ".join(ENV_FILES) + " (Retake klasörü), RETAKE_ENV_FILE.")
        return 2
    print(f"NVIDIA_API_KEY kaynağı: {source}")
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:384, 0:320].astype(np.float32)
    scene = np.stack([90 + xx * 0.1, 110 + yy * 0.05, 140 + xx * 0.05], 2)
    cv2.ellipse(scene, (160, 190), (90, 120), 0, 0, 360, (130, 160, 210), -1)
    scene = np.clip(scene + rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)
    pasted = scene.copy()
    cv2.ellipse(pasted, (160, 190), (70, 95), 0, 0, 360, (90, 170, 240), -1)
    smooth = cv2.GaussianBlur(scene, (0, 0), 3)
    try:
        report = judge(face_crop(scene, None), [face_crop(pasted, None), face_crop(smooth, None)], key)
    except Exception as error:
        print(f"NVIDIA hakem testi başarısız: {error}")
        return 1
    print(f"NVIDIA hakem çalışıyor: model={report['model']} puanlar={report['scores']} "
          f"geçişler={report['passes']} sorunlar={report['issues']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
