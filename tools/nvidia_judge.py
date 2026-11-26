"""Optional NVIDIA visual jury for Retake Ultra finalists, through Hafize's backend.

The local engine still generates and verifies every face. This module only asks
free hosted NVIDIA vision models (via Hafize's project-AI gateway, which holds the
NVIDIA key) which of 2–4 *already verified* Ultra finalists looks most like an
untouched camera photo: seams, tone/lighting mismatch, waxy skin, warped eyes or
teeth. It never generates pixels and never overrides identity: every finalist is
inside the local ArcFace tie band and passed the regional detail gate.

Data rule: only downscaled face crops of finalists leave the computer, only when
the user opted in for that job, only to an allow-listed Hafize address. Any
error, timeout or split vote keeps the local winner.

v28: each finalist is sent as one "review sheet" — the face with its
surroundings plus enlarged eye/brow and mouth close-ups of the same edit —
because vision models downscale large images and miss lashes, catchlights,
brow hair and lip lines. An older Hafize that does not know the layout gets
the plain face crops instead (one retry, same data rule).
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

# Hafize's live backend is the Render service (hafize-ua.web.app serves no /api).
DEFAULT_URL = "https://hafize-free.onrender.com/api/project-ai/retake/vision-judge"
JUDGE_PATH = "/api/project-ai/retake/vision-judge"
ALLOWED_HOSTS = frozenset({"hafize-ua.web.app", "hafize-ua.firebaseapp.com", "hafize-free.onrender.com"})
ALLOWED_HOST_SUFFIXES = (".run.app",)
TOKEN_FILE = Path.home() / ".retake" / "hafize-project-token"
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43,128}$")
TASK = "face-realism-rank"
MIN_CANDIDATES = 2
MAX_CANDIDATES = 4
CROP_EDGE = 512
JPEG_QUALITIES = (90, 82, 74)
MAX_CROP_BYTES = 600 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
# Render Free sleeps when idle: allow a ~60 s cold start plus the jury itself.
TIMEOUT_SECONDS = 190.0
LAYOUT_SHEET = "review-sheet"
LAYOUT_FACE = "face"
SHEET_HEIGHT = 512
SHEET_CLOSEUP_WIDTH = 384
SHEET_GAP = 8
SHEET_BACKGROUND = 118
MAX_CLOSEUP_ZOOM = 3.0
MIN_VOTES = 2
MIN_AGREEMENT = 2 / 3
OUTCOMES = ("off", "not-consented", "single-finalist", "kept", "switched", "abstain", "error")


class JudgeError(RuntimeError):
    """Jury unavailable; the caller keeps its local decision."""

    def __init__(self, message, code="unavailable", upstream_code=""):
        super().__init__(message)
        self.code = code
        self.upstream_code = upstream_code


@dataclass(frozen=True)
class JudgeConfig:
    url: str
    token: str


def validate_judge_url(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if (
        parts.scheme != "https" or parts.username or parts.password or parts.query or parts.fragment
        or parts.port not in (None, 443) or parts.path != JUDGE_PATH
        or not (host in ALLOWED_HOSTS or any(host.endswith(suffix) and len(host) > len(suffix) for suffix in ALLOWED_HOST_SUFFIXES))
    ):
        raise ValueError(
            "RETAKE_HAFIZE_JUDGE_URL yalnız Hafize'nin https://…" + JUDGE_PATH
            + " adresi olabilir (hafize-ua.web.app, hafize-free.onrender.com veya *.run.app)."
        )
    return f"https://{host}{JUDGE_PATH}"


def judge_config(env: Mapping[str, str] | None = None, token_path: Path | None = None) -> JudgeConfig | None:
    """None when not set up or explicitly off; ValueError when set up wrongly."""
    env = os.environ if env is None else env
    if env.get("RETAKE_NVIDIA_JUDGE", "").strip().lower() in {"0", "off", "false", "no"}:
        return None
    token = env.get("RETAKE_HAFIZE_PROJECT_TOKEN", "").strip()
    if not token:
        path = token_path or TOKEN_FILE
        try:
            if path.is_file() and path.stat().st_size <= 512:
                token = path.read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
    if not token:
        return None
    if not TOKEN_PATTERN.fullmatch(token):
        raise ValueError("Hafize proje jetonu biçimi geçersiz (43–128 karakter, A-Z a-z 0-9 _ -).")
    url = validate_judge_url(env.get("RETAKE_HAFIZE_JUDGE_URL", "").strip() or DEFAULT_URL)
    return JudgeConfig(url=url, token=token)


def judge_status(env: Mapping[str, str] | None = None, token_path: Path | None = None) -> str:
    try:
        return "configured" if judge_config(env, token_path) else "not-configured"
    except ValueError:
        return "invalid"


def _decode_rgb(png_bytes: bytes):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Jüri için aday görseli okunamadı.")
    return image[:, :, :3]


def _face_bounds(bounds: Mapping[str, Any]) -> tuple[float, float, float, float]:
    try:
        x, y = float(bounds["x"]), float(bounds["y"])
        fw, fh = float(bounds["width"]), float(bounds["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Jüri için yüz sınırları geçersiz.") from error
    if not all(math.isfinite(value) for value in (x, y, fw, fh)) or min(fw, fh) < 16:
        raise ValueError("Jüri için yüz sınırları geçersiz.")
    return x, y, fw, fh


def _context_crop(rgb, bounds: Mapping[str, Any]):
    height, width = rgb.shape[:2]
    x, y, fw, fh = _face_bounds(bounds)
    x0 = max(0, int(x - fw * 0.35))
    x1 = min(width, int(math.ceil(x + fw * 1.35)))
    y0 = max(0, int(y - fh * 0.45))
    y1 = min(height, int(math.ceil(y + fh * 1.35)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError("Jüri için yüz kırpımı çok küçük.")
    return rgb[y0:y1, x0:x1]


def _encode_jpeg(image) -> bytes:
    import cv2

    for quality in JPEG_QUALITIES:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(encoded) <= MAX_CROP_BYTES:
            return encoded.tobytes()
    raise ValueError("Jüri için yüz kırpımı çok büyük.")


def face_crop_jpeg(png_bytes: bytes, bounds: Mapping[str, Any]) -> bytes:
    """A context crop around the face (seams, hairline, neck), ≤512 px JPEG.

    Retake's transport PNG already carries the pre-blended composite in RGB, so
    the jury sees exactly the photo the user would get.
    """
    import cv2

    crop = _context_crop(_decode_rgb(png_bytes), bounds)
    scale = min(1.0, CROP_EDGE / float(max(crop.shape[:2])))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return _encode_jpeg(crop)


def closeup_boxes(bounds: Mapping[str, Any], landmarks: Any, shape: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    """Eye/brow and mouth boxes (x0, y0, x1, y1): from the target mesh when it
    is valid, otherwise from typical positions inside the face box."""
    height, width = shape
    x, y, fw, fh = _face_bounds(bounds)
    boxes: dict[str, tuple[float, float, float, float]] = {}
    points = None
    try:
        from .fine_features import FEATURES, valid_mesh  # type: ignore[import-not-found]
    except ImportError:
        try:
            from fine_features import FEATURES, valid_mesh  # type: ignore[no-redef]
        except ImportError:
            FEATURES, valid_mesh = None, None
    if valid_mesh is not None:
        points = valid_mesh(landmarks, shape)
    if points is not None:
        eyes = points[list(FEATURES["eye_r"] + FEATURES["eye_l"] + FEATURES["brow_r"] + FEATURES["brow_l"])]
        lips = points[list(FEATURES["lips"])]
        for name, group, pad in (("eyes", eyes, 0.18), ("mouth", lips, 0.35)):
            gx0, gy0 = float(group[:, 0].min()), float(group[:, 1].min())
            gx1, gy1 = float(group[:, 0].max()), float(group[:, 1].max())
            px, py = (gx1 - gx0) * pad, (gy1 - gy0) * pad + 0.04 * fh
            boxes[name] = (gx0 - px, gy0 - py, gx1 + px, gy1 + py)
    else:
        boxes["eyes"] = (x + 0.02 * fw, y + 0.12 * fh, x + 0.98 * fw, y + 0.55 * fh)
        boxes["mouth"] = (x + 0.18 * fw, y + 0.58 * fh, x + 0.82 * fw, y + 0.98 * fh)
    result = {}
    for name, (x0, y0, x1, y1) in boxes.items():
        box = (max(0, int(x0)), max(0, int(y0)), min(width, int(math.ceil(x1))), min(height, int(math.ceil(y1))))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            raise ValueError("Jüri için yakın plan bölgesi çok küçük.")
        result[name] = box
    return result


def _fit(image, width: int, height: int, zoom_limit: float):
    """Scale into width×height (aspect kept), letterboxed on neutral grey."""
    import cv2
    import numpy as np

    scale = min(width / image.shape[1], height / image.shape[0], zoom_limit)
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    panel = np.full((height, width, 3), SHEET_BACKGROUND, np.uint8)
    top, left = (height - size[1]) // 2, (width - size[0]) // 2
    panel[top:top + size[1], left:left + size[0]] = resized
    return panel


def review_sheet_jpeg(png_bytes: bytes, bounds: Mapping[str, Any], landmarks: Any = None) -> bytes:
    """One JPEG per finalist: the face with its surroundings on the left, the
    same edit's eyes/brows (top) and mouth (bottom) enlarged on the right."""
    import numpy as np

    rgb = _decode_rgb(png_bytes)
    context = _context_crop(rgb, bounds)
    face_width = max(1, min(SHEET_HEIGHT, round(context.shape[1] * SHEET_HEIGHT / context.shape[0])))
    face = _fit(context, face_width, SHEET_HEIGHT, zoom_limit=SHEET_HEIGHT / context.shape[0])
    panel_height = (SHEET_HEIGHT - SHEET_GAP) // 2
    boxes = closeup_boxes(bounds, landmarks, rgb.shape[:2])
    closeups = [
        _fit(rgb[y0:y1, x0:x1], SHEET_CLOSEUP_WIDTH, panel_height, MAX_CLOSEUP_ZOOM)
        for x0, y0, x1, y1 in (boxes["eyes"], boxes["mouth"])
    ]
    sheet = np.full((SHEET_HEIGHT, face_width + SHEET_GAP + SHEET_CLOSEUP_WIDTH, 3), SHEET_BACKGROUND, np.uint8)
    sheet[:, :face_width] = face
    right = face_width + SHEET_GAP
    sheet[:panel_height, right:] = closeups[0]
    sheet[SHEET_HEIGHT - panel_height:, right:] = closeups[1]
    return _encode_jpeg(sheet)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):  # never forward face crops elsewhere
        return None


def _default_opener():
    return urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=ssl.create_default_context()))


def _normalize_verdict(payload: Any, count: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise JudgeError("Jüri yanıtı geçersiz.")
    decision = payload.get("decision")
    winner = payload.get("winner")
    votes = payload.get("votes")
    agreement = payload.get("agreement")
    if decision not in {"winner", "abstain"}:
        raise JudgeError("Jüri kararı geçersiz.")
    if winner is not None and (type(winner) is not int or not 0 <= winner < count):
        raise JudgeError("Jüri adayı geçersiz.")
    if type(votes) is not int or votes < 0 or votes > 32:
        raise JudgeError("Jüri oy sayısı geçersiz.")
    if isinstance(agreement, bool) or not isinstance(agreement, (int, float)) or not math.isfinite(agreement) or not 0 <= agreement <= 1:
        raise JudgeError("Jüri uzlaşı oranı geçersiz.")
    models = payload.get("models")
    models = [str(item)[:120] for item in models if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9._/-]{3,120}", item)] if isinstance(models, list) else []
    result = {"decision": decision, "winner": winner, "votes": votes, "agreement": float(agreement), "models": models[:4]}
    reason = payload.get("reason")
    if reason in ("NOT_ENOUGH_VOTES", "SPLIT_VOTE", "NARROW_MARGIN", "CONSENSUS"):
        result["reason"] = reason
    return result


def request_judgement(config: JudgeConfig, crops: list[bytes], opener=None, timeout: float = TIMEOUT_SECONDS,
                      layout: str | None = None) -> dict[str, Any]:
    if not MIN_CANDIDATES <= len(crops) <= MAX_CANDIDATES:
        raise ValueError("Jüriye 2–4 aday gönderilebilir.")
    if any(not crop.startswith(b"\xff\xd8\xff") or len(crop) > MAX_CROP_BYTES for crop in crops):
        raise ValueError("Jüri kırpımları JPEG ve 600 KB altında olmalı.")
    if sum(map(len, crops)) > 2_000_000:
        raise JudgeError("Jüri toplam kırpım boyutu aşıldı.", "payload-too-large")
    if layout not in (None, LAYOUT_FACE, LAYOUT_SHEET):
        raise ValueError("Jüri düzeni geçersiz.")
    payload: dict[str, Any] = {
        "task": TASK,
        "candidates": ["data:image/jpeg;base64," + base64.b64encode(crop).decode("ascii") for crop in crops],
    }
    if layout == LAYOUT_SHEET:
        payload["layout"] = LAYOUT_SHEET
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(validate_judge_url(config.url), data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer " + config.token,
        "User-Agent": "Retake-Engine/nvidia-judge",
    })
    try:
        with (opener or _default_opener()).open(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        code = ""
        try:
            parsed = json.loads(error.read(4096) or b"{}")
            candidate = parsed.get("error") if isinstance(parsed, dict) else None
            code = candidate if isinstance(candidate, str) and re.fullmatch(r"PROJECT_AI_[A-Z0-9_]{1,60}", candidate) else ""
        except (ValueError, OSError):
            code = ""
        raise JudgeError(f"Hafize jürisi HTTP {error.code}" + (f" ({code})" if code else ""), f"http-{error.code}" if 400 <= error.code <= 599 else "http-unexpected", code) from None
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        reason = getattr(error, "reason", error)
        category = "tls" if isinstance(reason, ssl.SSLError) else "timeout" if isinstance(reason, TimeoutError) else "network"
        raise JudgeError("Hafize jürisine ulaşılamadı: " + category, category) from None
    if status != 200 or len(raw) > MAX_RESPONSE_BYTES:
        raise JudgeError(f"Hafize jürisi beklenmeyen yanıt verdi (HTTP {status}).")
    try:
        payload = json.loads(raw)
    except ValueError:
        raise JudgeError("Hafize jürisi JSON döndürmedi.") from None
    return _normalize_verdict(payload, len(crops))


def judged_winner(verdict: Mapping[str, Any], count: int) -> int | None:
    """Re-apply consensus locally; the server's word alone is not enough."""
    winner = verdict.get("winner")
    if (
        verdict.get("decision") != "winner" or type(winner) is not int or not 0 <= winner < count
        or not isinstance(verdict.get("votes"), int) or verdict["votes"] < MIN_VOTES
        or not isinstance(verdict.get("agreement"), (int, float)) or verdict["agreement"] < MIN_AGREEMENT - 1e-9
    ):
        return None
    return winner


def build_judge(env: Mapping[str, str] | None = None, token_path: Path | None = None) -> Callable[[list[bytes]], dict[str, Any]] | None:
    """Per-job factory: a token added while the engine runs is picked up next job."""
    try:
        config = judge_config(env, token_path)
    except ValueError as error:
        print(f"[NVIDIA jüri] Devre dışı: {error}", flush=True)
        return None
    if config is None:
        return None

    def judge(images: list[bytes], bounds: Mapping[str, Any], landmarks: Any = None, opener=None) -> dict[str, Any]:
        """Review sheets first; an older Hafize that refuses the layout gets
        the plain face crops (one retry, nothing else changes)."""
        sheets: list[bytes] | None
        try:
            sheets = [review_sheet_jpeg(image, bounds, landmarks) for image in images]
        except ValueError as error:
            print(f"[NVIDIA jüri] İnceleme paftası hazırlanamadı ({error}); yüz kırpımları gönderiliyor.", flush=True)
            sheets = None
        if sheets is not None and sum(map(len, sheets)) <= 2_000_000:
            try:
                return {**request_judgement(config, sheets, opener=opener, layout=LAYOUT_SHEET), "layout": LAYOUT_SHEET}
            except JudgeError as error:
                if not (error.code == "http-400" and error.upstream_code == "PROJECT_AI_INVALID_INPUT"):
                    raise
                print("[NVIDIA jüri] Hafize inceleme paftasını henüz tanımıyor; yüz kırpımları gönderiliyor.", flush=True)
        crops = [face_crop_jpeg(image, bounds) for image in images]
        return {**request_judgement(config, crops, opener=opener), "layout": LAYOUT_FACE}

    judge.accepts_images = True  # type: ignore[attr-defined]
    return judge


def judge_headers(outcome: str, finalists: int = 0, verdict: Mapping[str, Any] | None = None, error=None) -> dict[str, str]:
    if outcome not in OUTCOMES:
        outcome = "error"
    headers = {"X-Retake-NVIDIA-Judge": outcome, "X-Retake-NVIDIA-Judge-Finalists": str(finalists)}
    if outcome == "error":
        code = getattr(error, "code", "unavailable") if isinstance(error, JudgeError) else "unavailable"
        headers["X-Retake-NVIDIA-Judge-Error"] = code if re.fullmatch(r"http-[45][0-9]{2}|tls|timeout|network|payload-too-large|unavailable|http-unexpected", str(code)) else "unavailable"
    upstream = getattr(error, "upstream_code", "") if isinstance(error, JudgeError) else ""
    if isinstance(upstream, str) and re.fullmatch(r"PROJECT_AI_[A-Z0-9_]{1,60}", upstream):
        headers["X-Retake-NVIDIA-Judge-Reason"] = upstream
    if verdict:
        reason = verdict.get("reason")
        if reason in ("NOT_ENOUGH_VOTES", "SPLIT_VOTE", "NARROW_MARGIN", "CONSENSUS"):
            headers["X-Retake-NVIDIA-Judge-Reason"] = reason
        headers["X-Retake-NVIDIA-Judge-Votes"] = str(int(verdict.get("votes") or 0))
        headers["X-Retake-NVIDIA-Judge-Agreement"] = f"{float(verdict.get('agreement') or 0.0):.3f}"
        headers["X-Retake-NVIDIA-Judge-Models"] = ",".join(verdict.get("models") or [])
        if verdict.get("layout") in (LAYOUT_FACE, LAYOUT_SHEET):
            headers["X-Retake-NVIDIA-Judge-Layout"] = verdict["layout"]
    return headers
