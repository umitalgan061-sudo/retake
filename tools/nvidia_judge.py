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


def face_crop_jpeg(png_bytes: bytes, bounds: Mapping[str, Any]) -> bytes:
    """A context crop around the face (seams, hairline, neck), ≤512 px JPEG.

    Retake's transport PNG already carries the pre-blended composite in RGB, so
    the jury sees exactly the photo the user would get.
    """
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Jüri için aday görseli okunamadı.")
    rgb = image[:, :, :3]
    height, width = rgb.shape[:2]
    try:
        x, y = float(bounds["x"]), float(bounds["y"])
        fw, fh = float(bounds["width"]), float(bounds["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Jüri için yüz sınırları geçersiz.") from error
    if not all(math.isfinite(value) for value in (x, y, fw, fh)) or min(fw, fh) < 16:
        raise ValueError("Jüri için yüz sınırları geçersiz.")
    x0 = max(0, int(x - fw * 0.35))
    x1 = min(width, int(math.ceil(x + fw * 1.35)))
    y0 = max(0, int(y - fh * 0.45))
    y1 = min(height, int(math.ceil(y + fh * 1.35)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError("Jüri için yüz kırpımı çok küçük.")
    crop = rgb[y0:y1, x0:x1]
    scale = min(1.0, CROP_EDGE / float(max(crop.shape[:2])))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    for quality in JPEG_QUALITIES:
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(encoded) <= MAX_CROP_BYTES:
            return encoded.tobytes()
    raise ValueError("Jüri için yüz kırpımı çok büyük.")


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


def request_judgement(config: JudgeConfig, crops: list[bytes], opener=None, timeout: float = TIMEOUT_SECONDS) -> dict[str, Any]:
    if not MIN_CANDIDATES <= len(crops) <= MAX_CANDIDATES:
        raise ValueError("Jüriye 2–4 aday gönderilebilir.")
    if any(not crop.startswith(b"\xff\xd8\xff") or len(crop) > MAX_CROP_BYTES for crop in crops):
        raise ValueError("Jüri kırpımları JPEG ve 600 KB altında olmalı.")
    if sum(map(len, crops)) > 2_000_000:
        raise JudgeError("Jüri toplam kırpım boyutu aşıldı.", "payload-too-large")
    body = json.dumps({
        "task": TASK,
        "candidates": ["data:image/jpeg;base64," + base64.b64encode(crop).decode("ascii") for crop in crops],
    }).encode("utf-8")
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
    return lambda crops: request_judgement(config, crops)


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
    return headers
