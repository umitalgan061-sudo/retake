"""Optional NVIDIA visual jury for Retake Ultra finalists, through Hafize's backend.

The local engine still generates and verifies every face. This module only asks
free hosted NVIDIA vision models (via Hafize's project-AI gateway, which holds the
NVIDIA key) which of 2–4 *already verified* Ultra finalists looks most like an
untouched camera photo: seams, tone/lighting mismatch, waxy skin, warped eyes or
teeth. It never generates pixels and never overrides identity: every finalist is
inside the local ArcFace tie band and passed the regional detail gate.

Data rule: only downscaled face crops of finalists leave the computer, only when
the user opted in for that job, only to an allow-listed Hafize address — or,
when no Hafize token is set up but this computer has NVIDIA_API_KEY (process
environment or a git-ignored .env file of the Retake folder), directly to
NVIDIA's own API host. Any error, timeout or split vote keeps the local winner.
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

DEFAULT_URL = "https://hafize-ua.web.app/api/project-ai/retake/vision-judge"
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
# Hafize's Cloud Run request window is 60 s; allow for a cold start on top.
TIMEOUT_SECONDS = 75.0
MIN_VOTES = 2
MIN_AGREEMENT = 2 / 3
OUTCOMES = ("off", "not-consented", "single-finalist", "kept", "switched", "abstain", "error")


class JudgeError(RuntimeError):
    """Jury unavailable; the caller keeps its local decision."""


def _ssl_context() -> ssl.SSLContext:
    # Corporate HTTPS inspection: trust the Windows certificate store when the
    # truststore package is available (the realism installer adds it).
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        return ssl.create_default_context()


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


# --- Direct NVIDIA transport (no Hafize token, NVIDIA_API_KEY on this PC) ---

DIRECT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
ROOT = Path(__file__).resolve().parents[1]
# Git-ignored local secret files of the Retake folder, after the process
# environment. RETAKE_ENV_FILE may name one more file explicitly.
ENV_FILES = (".env", "functions/.env", ".dev.vars")
# Hosted vision models of the free catalogue, asked in parallel; each casts
# one vote (the local consensus rule still needs >= 2 votes and 2/3).
DIRECT_MODELS = (
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "mistralai/mistral-large-4-1t-a52b",
    "z-ai/glm-5.3-flash",
)
DIRECT_MAX_RESPONSE_BYTES = 512 * 1024
SHEET_PANEL_HEIGHT = 360
DIRECT_SYSTEM = (
    "You are a meticulous senior photo retoucher. You compare edited versions of one photo and judge only "
    "photographic realism. You never identify people and never judge attractiveness. "
    "Answer with one JSON object and nothing else."
)


@dataclass(frozen=True)
class DirectConfig:
    key: str
    source: str


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
        if not sep or name.strip() != "NVIDIA_API_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        if value:
            return value
    return None


def direct_config(env: Mapping[str, str] | None = None, env_files: list[Path] | None = None) -> DirectConfig | None:
    """NVIDIA_API_KEY from the environment or a .env file; the source names where, never the value.

    With an explicit `env` (tests) no file is read unless `env_files` is given.
    """
    if env is None:
        env = os.environ
        if env_files is None:
            env_files = [ROOT / name for name in ENV_FILES]
    if env.get("RETAKE_NVIDIA_JUDGE", "").strip().lower() in {"0", "off", "false", "no"}:
        return None
    key = env.get("NVIDIA_API_KEY", "").strip()
    if key:
        return DirectConfig(key, "ortam değişkeni")
    files = list(env_files or [])
    explicit = env.get("RETAKE_ENV_FILE", "").strip()
    if explicit:
        files.insert(0, Path(explicit).expanduser())
    for path in files:
        key = _key_from_file(path)
        if key:
            return DirectConfig(key, path.name if path.parent == ROOT else str(path))
    return None


def _direct_prompt(count: int, sheet: bool) -> str:
    unit = "panel" if sheet else "image"
    layout = (f"The picture is one row of {count} panels numbered in their top-left corner. " if sheet
              else f"These are {count} images. ")
    return (
        layout + f"Each {unit} is a differently edited version of the same photo crop in which the face was "
        "replaced. Decide which one looks most convincingly like an untouched camera photograph: check skin "
        "texture (pores vs. plastic/waxy), blend edges at forehead, hairline, cheeks and jaw, skin tone versus "
        "neck and ears, light and shadow direction, eyes (catchlights, symmetry), lips and teeth, blur or "
        "over-sharpening, warping. If none is clearly more realistic, answer 0. "
        f'Reply exactly as {{"best": <{unit} number 1-{count} or 0>, "reason": "<at most 12 words>"}}'
    )


def contact_sheet(crops: list[bytes]) -> bytes:
    """One JPEG with the crops side by side, numbered 1..N (for one-image models)."""
    import cv2
    import numpy as np

    panels = []
    for number, data in enumerate(crops, 1):
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Jüri kırpımı okunamadı.")
        scale = SHEET_PANEL_HEIGHT / image.shape[0]
        image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), SHEET_PANEL_HEIGHT),
                           interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (44, 40), (255, 255, 255), -1)
        cv2.putText(image, str(number), (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        panels += [image, np.full((SHEET_PANEL_HEIGHT, 8, 3), 255, np.uint8)]
    ok, encoded = cv2.imencode(".jpg", np.hstack(panels[:-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("Karşılaştırma görseli kodlanamadı.")
    return encoded.tobytes()


def _post_direct(payload: dict[str, Any], key: str, opener=None, timeout: float = TIMEOUT_SECONDS) -> dict[str, Any]:
    request = urllib.request.Request(DIRECT_URL, data=json.dumps(payload).encode("utf-8"), method="POST", headers={
        "Content-Type": "application/json", "Accept": "application/json",
        "Authorization": "Bearer " + key, "User-Agent": "Retake-Engine/nvidia-judge",
    })
    try:
        with (opener or _default_opener()).open(request, timeout=timeout) as response:
            raw = response.read(DIRECT_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise JudgeError(f"NVIDIA HTTP {error.code}") from None
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise JudgeError("NVIDIA'ya ulaşılamadı: " + type(error).__name__) from None
    if len(raw) > DIRECT_MAX_RESPONSE_BYTES:
        raise JudgeError("NVIDIA yanıtı çok büyük.")
    try:
        return json.loads(raw)
    except ValueError:
        raise JudgeError("NVIDIA JSON döndürmedi.") from None


def parse_best(text: Any, count: int) -> int | None:
    """0..count from {"best": n}; 0 means no clear winner. None if unreadable."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            data = json.loads(match.group(0))
        except ValueError:
            continue
        best = data.get("best") if isinstance(data, dict) else None
        if isinstance(best, str) and best.strip().isdigit():
            best = int(best.strip())
        if type(best) is int and 0 <= best <= count:
            return best
    return None


def _model_vote(model: str, crops: list[bytes], order: list[int], key: str, poster) -> int | None:
    """One model's vote as an index into `crops` (-1 = no clear winner)."""
    shown = [crops[index] for index in order]
    for sheet in (False, True):
        images = [contact_sheet(shown)] if sheet else shown
        content: list[dict[str, Any]] = [{"type": "text", "text": _direct_prompt(len(shown), sheet)}]
        content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}}
                    for data in images]
        payload = {"model": model, "messages": [{"role": "system", "content": DIRECT_SYSTEM},
                                                 {"role": "user", "content": content}],
                   "temperature": 0.1, "top_p": 0.7, "max_tokens": 1024, "stream": False}
        try:
            response = poster(payload, key)
        except JudgeError as error:
            # A model that takes one image per request gets a contact sheet.
            if not sheet and re.search(r"HTTP (400|413|422)\b", str(error)):
                continue
            return None
        try:
            text = response["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            return None
        best = parse_best(text, len(shown))
        if best is None:
            return None
        return -1 if best == 0 else order[best - 1]
    return None


def request_direct(config: DirectConfig, crops: list[bytes], poster=None,
                   models: tuple[str, ...] = DIRECT_MODELS) -> dict[str, Any]:
    """Same verdict shape as the Hafize jury: decision/winner/votes/agreement/models.

    Each model sees the finalists in a different order (rotation), which
    cancels the position bias these models have.
    """
    if not MIN_CANDIDATES <= len(crops) <= MAX_CANDIDATES:
        raise ValueError("Jüriye 2–4 aday gönderilebilir.")
    poster = poster or _post_direct
    count = len(crops)
    orders = [[(start + offset) % count for offset in range(count)] for start in range(len(models))]
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        votes = list(pool.map(lambda job: _model_vote(job[0], crops, job[1], config.key, poster),
                              zip(models, orders)))
    answered = [(model, vote) for model, vote in zip(models, votes) if vote is not None]
    if not answered:
        raise JudgeError("NVIDIA görsel modellerinden okunabilir oy gelmedi.")
    tally: dict[int, int] = {}
    for _, vote in answered:
        if vote >= 0:
            tally[vote] = tally.get(vote, 0) + 1
    winner, top = max(tally.items(), key=lambda item: (item[1], -item[0])) if tally else (None, 0)
    agreement = top / len(answered)
    decision = "winner" if winner is not None and agreement > 0.5 else "abstain"
    return {"decision": decision, "winner": winner if decision == "winner" else None,
            "votes": len(answered), "agreement": round(agreement, 3),
            "models": [model for model, _ in answered]}


def judge_transport(env: Mapping[str, str] | None = None, token_path: Path | None = None,
                    env_files: list[Path] | None = None) -> tuple[str, str]:
    """("hafize" | "direct" | "none" | "invalid", detail without secrets)."""
    try:
        if judge_config(env, token_path):
            return "hafize", "Hafize jetonu"
    except ValueError as error:
        return "invalid", str(error)
    direct = direct_config(env, env_files)
    if direct:
        return "direct", direct.source
    return "none", ""


def judge_status(env: Mapping[str, str] | None = None, token_path: Path | None = None,
                 env_files: list[Path] | None = None) -> str:
    transport = judge_transport(env, token_path, env_files)[0]
    return {"hafize": "configured", "direct": "configured", "invalid": "invalid"}.get(transport, "not-configured")


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
    return urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=_ssl_context()))


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
    return {"decision": decision, "winner": winner, "votes": votes, "agreement": float(agreement), "models": models[:4]}


def request_judgement(config: JudgeConfig, crops: list[bytes], opener=None, timeout: float = TIMEOUT_SECONDS) -> dict[str, Any]:
    if not MIN_CANDIDATES <= len(crops) <= MAX_CANDIDATES:
        raise ValueError("Jüriye 2–4 aday gönderilebilir.")
    if any(not crop.startswith(b"\xff\xd8\xff") or len(crop) > MAX_CROP_BYTES for crop in crops):
        raise ValueError("Jüri kırpımları JPEG ve 600 KB altında olmalı.")
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
        raise JudgeError(f"Hafize jürisi HTTP {error.code}" + (f" ({code})" if code else "")) from None
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise JudgeError("Hafize jürisine ulaşılamadı: " + type(error).__name__) from None
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


def build_judge(env: Mapping[str, str] | None = None, token_path: Path | None = None,
                env_files: list[Path] | None = None) -> Callable[[list[bytes]], dict[str, Any]] | None:
    """Per-job factory: a token or key added while the engine runs is picked up next job.

    The Hafize gateway (key stays in Hafize) is preferred; without a Hafize
    token, NVIDIA_API_KEY of this computer reaches NVIDIA's API directly.
    """
    try:
        config = judge_config(env, token_path)
    except ValueError as error:
        print(f"[NVIDIA jüri] Devre dışı: {error}", flush=True)
        return None
    if config is not None:
        return lambda crops: request_judgement(config, crops)
    direct = direct_config(env, env_files)
    if direct is None:
        return None
    return lambda crops: request_direct(direct, crops)


def judge_headers(outcome: str, finalists: int = 0, verdict: Mapping[str, Any] | None = None) -> dict[str, str]:
    if outcome not in OUTCOMES:
        outcome = "error"
    headers = {"X-Retake-NVIDIA-Judge": outcome, "X-Retake-NVIDIA-Judge-Finalists": str(finalists)}
    if verdict:
        headers["X-Retake-NVIDIA-Judge-Votes"] = str(int(verdict.get("votes") or 0))
        headers["X-Retake-NVIDIA-Judge-Agreement"] = f"{float(verdict.get('agreement') or 0.0):.3f}"
        headers["X-Retake-NVIDIA-Judge-Models"] = ",".join(verdict.get("models") or [])
    return headers


def _selftest() -> int:
    """Checks the configured jury with synthetic, non-personal crops."""
    import cv2
    import numpy as np

    transport, detail = judge_transport()
    if transport in ("none", "invalid"):
        print("NVIDIA jürisi yapılandırılmamış" + (f": {detail}" if detail else "")
              + ". NVIDIA_API_KEY (ortam değişkeni ya da Retake klasöründeki .env) veya Hafize jetonu gerekli.")
        return 2
    print(f"NVIDIA jürisi yolu: {'Hafize' if transport == 'hafize' else 'doğrudan NVIDIA'} ({detail}).")
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:384, 0:320].astype(np.float32)
    scene = np.stack([90 + xx * 0.1, 110 + yy * 0.05, 140 + xx * 0.05], 2)
    cv2.ellipse(scene, (160, 190), (90, 120), 0, 0, 360, (130, 160, 210), -1)
    scene = np.clip(scene + rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)
    pasted = scene.copy()
    cv2.ellipse(pasted, (160, 190), (70, 95), 0, 0, 360, (90, 170, 240), -1)
    crops = [cv2.imencode(".jpg", image)[1].tobytes() for image in (scene, pasted)]
    try:
        verdict = build_judge()(crops)
    except Exception as error:
        print(f"NVIDIA jüri testi başarısız: {error}")
        return 1
    print(f"NVIDIA jürisi çalışıyor: karar={verdict['decision']} kazanan={verdict['winner']} "
          f"oy={verdict['votes']} uzlaşı={verdict['agreement']} modeller={verdict['models']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
