"""Retake doctor: one shareable report of what limits photo quality on this PC.

Run through ``tools/retake_doctor.ps1`` on the engine computer. It checks the
worktree, the hidden engine and its startup task, ``/health``, the GPU and
free disk space, the Python packages of the engine environment, the NVIDIA
jury (local token presence and Hafize's public status, optionally a synthetic
probe), whether the published site matches this worktree, and the scrubbed
engine event log.

Privacy: the report never contains photos, file names, the pairing code, the
Hafize project token, keys, the Windows user or computer name or local paths.
Every line passes through :func:`redact` before it is printed. The optional
jury probe sends two generated colour cards, never a photo. Standard library
only, so it also runs where the engine environment is broken.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

try:
    from .retake_event_log import EVENT_LOG_NAME, read_events, summarize_events, _SCRUBBERS
except ImportError:  # run directly from tools/
    from retake_event_log import EVENT_LOG_NAME, read_events, summarize_events, _SCRUBBERS

DOCTOR_VERSION = 1
ENGINE_URL = "http://127.0.0.1:8765"
HAFIZE_URL = "https://hafize-free.onrender.com"
SITE_URL = "https://retake06-ua.web.app"
TOKEN_FORMAT = re.compile(r"^[A-Za-z0-9_-]{43,128}$")
REVISION_LINE = re.compile(r'^QUALITY_PIPELINE_REVISION = "([^"]+)"$', re.MULTILINE)
SHELL_CACHE_LINE = re.compile(r'SHELL_CACHE\s*=\s*["\']([^"\']{1,60})["\']')
FULL_BOOST = "1024x1024"
DISK_WARN_GIB = 15.0
DISK_FAIL_GIB = 5.0
OTHER_VRAM_WARN_MIB = 2048
EVENT_WINDOW_SECONDS = 7 * 24 * 3600
PACKAGES = (
    "onnxruntime-gpu", "onnxruntime", "onnxruntime-directml", "numpy", "opencv-python",
    "opencv-python-headless", "pillow", "mediapipe", "insightface", "torch",
)
STATUS_LABEL = {"ok": "TAMAM", "warn": "UYARI", "fail": "SORUN", "info": "BILGI"}
SEVERITY = {"fail": 0, "warn": 1, "info": 2, "ok": 3}

Opener = Callable[..., tuple[int, dict, bytes]]
Runner = Callable[[list[str], float], tuple[int, str]]


@dataclass
class Check:
    area: str
    status: str
    title: str
    detail: str = ""
    fix: str = ""


@dataclass
class Context:
    root: Path
    home: Path
    opener: Opener
    run: Runner
    windows: dict = field(default_factory=dict)
    offline: bool = False
    probe_jury: bool = False
    now: float = field(default_factory=time.time)
    executable: str = sys.executable
    env: dict = field(default_factory=lambda: dict(os.environ))
    disk_usage: Callable[[str], Any] = shutil.disk_usage
    package_version: Callable[[str], str | None] = None  # type: ignore[assignment]

    @property
    def retake_dir(self) -> Path:
        return self.home / ".retake"


# ----------------------------------------------------------------- helpers

def redact(text: str, env: dict | None = None) -> str:
    """Remove paths, addresses, opaque strings and the user/computer name."""
    if not text:
        return ""
    env = os.environ if env is None else env
    cleaned = str(text)
    for key in ("USERPROFILE", "HOME"):
        value = env.get(key)
        if value and len(value) > 3:
            cleaned = cleaned.replace(value, "~")
    for pattern, replacement in _SCRUBBERS:
        cleaned = pattern.sub(replacement, cleaned)
    for key, placeholder in (("USERNAME", "<kullanıcı>"), ("USER", "<kullanıcı>"), ("COMPUTERNAME", "<bilgisayar>")):
        value = env.get(key)
        if value and len(value) >= 3:
            cleaned = re.sub(r"(?<!\w)" + re.escape(value) + r"(?!\w)", placeholder, cleaned, flags=re.IGNORECASE)
    return cleaned


def default_opener(url: str, *, headers: dict | None = None, data: bytes | None = None,
                   timeout: float = 10.0, method: str | None = None) -> tuple[int, dict, bytes]:
    """HTTP without redirects; loopback ignores any system proxy."""
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    handlers: list[Any] = [_NoRedirect()]
    if url.startswith(ENGINE_URL):
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read(4 * 1024 * 1024)
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read(65536) if error.fp else b""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - urllib hook
        return None


def default_runner(args: list[str], timeout: float) -> tuple[int, str]:
    try:
        completed = subprocess.run(args, capture_output=True, timeout=timeout, text=True,
                                   encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return completed.returncode, (completed.stdout or "")


def default_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _pairing_headers(ctx: Context) -> dict:
    token_path = ctx.retake_dir / "pairing-token"
    try:
        token = token_path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return {}
    return {"X-Retake-Token": token} if token else {}


def _gib(value: float) -> str:
    return f"{value / 1024 ** 3:.1f} GB"


def expected_revision(root: Path) -> str | None:
    try:
        match = REVISION_LINE.search((root / "tools" / "facefusion_local_server.py").read_text(encoding="utf-8"))
    except OSError:
        return None
    return match.group(1) if match else None


def solid_png(rgb: tuple[int, int, int], size: int = 64) -> bytes:
    """A valid PNG colour card with a lighter centre square (stdlib only)."""
    r, g, b = rgb
    light = tuple(min(255, value + 90) for value in rgb)
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            inside = size // 4 <= x < 3 * size // 4 and size // 4 <= y < 3 * size // 4
            rows.extend(light if inside else (r, g, b))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b"")


# ------------------------------------------------------------------ checks

def check_repo(ctx: Context) -> list[Check]:
    checks: list[Check] = []
    git = ["git", "-C", str(ctx.root)]
    code, head = ctx.run([*git, "rev-parse", "--short", "HEAD"], 15)
    if code != 0:
        return [Check("Depo", "fail", "Retake klasörü bir git deposu değil veya git bulunamadı.",
                      fix="Tanı aracını Retake deposunun içindeki tools klasöründen çalıştırın; git kurulu olmalı.")]
    head = head.strip()
    _, branch = ctx.run([*git, "rev-parse", "--abbrev-ref", "HEAD"], 15)
    branch = branch.strip() or "?"
    _, porcelain = ctx.run([*git, "status", "--porcelain"], 30)
    lines = [line for line in porcelain.splitlines() if line.strip()]
    modified = sum(1 for line in lines if not line.startswith("??"))
    untracked = len(lines) - modified
    revision = expected_revision(ctx.root)
    relation = "uzak depo denetlenmedi (çevrimdışı)"
    behind = 0
    if not ctx.offline:
        fetch_code, _ = ctx.run([*git, "fetch", "--quiet", "origin", "main"], 60)
        code, counts = ctx.run([*git, "rev-list", "--left-right", "--count", "HEAD...origin/main"], 15)
        parts = counts.split()
        if fetch_code == 0 and code == 0 and len(parts) == 2 and all(part.isdigit() for part in parts):
            ahead, behind = int(parts[0]), int(parts[1])
            relation = "origin/main ile aynı" if not ahead and not behind else f"{behind} commit geride, {ahead} commit ileride"
        else:
            relation = "GitHub'a ulaşılamadı; uzak sürüm denetlenmedi"
    status = "ok"
    fix = ""
    if behind:
        status, fix = "warn", "git pull --ff-only origin main ; .\\tools\\update_retake_ultra.ps1 -SkipPull -SkipDeploy"
    if branch != "main":
        status = "warn"
        fix = fix or "git switch main ; git pull --ff-only origin main"
    checks.append(Check("Depo", status, f"{branch} @ {head}; {relation}.",
                        detail=f"Depodaki kalite sürümü: {revision or 'bulunamadı'}.", fix=fix))
    if modified:
        checks.append(Check("Depo", "warn", f"Depoda {modified} dosyada yerel değişiklik var.",
                            detail="Güncelleme (git pull --ff-only) bu yüzden durabilir.",
                            fix="git status ile bakın; gerekmiyorsa git stash ile kenara alın."))
    if untracked:
        checks.append(Check("Depo", "info", f"Depoda {untracked} izlenmeyen dosya/klasör var (adları rapora yazılmadı)."))
    return checks


def check_background(ctx: Context) -> list[Check]:
    checks: list[Check] = []
    config = _read_json(ctx.retake_dir / "background-config.json")
    task = ctx.windows.get("task") if isinstance(ctx.windows.get("task"), dict) else None
    if task is not None and not task.get("exists"):
        checks.append(Check("Gizli motor", "info", "Windows açılış görevi kurulu değil; motor elle başlatılıyor olmalı.",
                            fix="Kalıcı kurulum için: .\\tools\\install_retake_autostart.ps1"))
    elif task is not None:
        state = str(task.get("state") or "?")
        last = task.get("last_result")
        status = "ok" if state == "Running" else "warn"
        detail = f"Son çalışma sonucu: {last}." if last not in (None, "") else ""
        fix = "" if state == "Running" else ".\\tools\\install_retake_autostart.ps1 -Action Start"
        if task.get("working_dir_matches") is False:
            status = "fail"
            detail = (detail + " Görev başka bir Retake klasörünü çalıştırıyor; güncellemeler bu motora ulaşmaz.").strip()
            fix = "Bu klasörden yeniden kurun: .\\tools\\install_retake_autostart.ps1 -Action Remove ; .\\tools\\install_retake_autostart.ps1"
        checks.append(Check("Gizli motor", status, f"Açılış görevi durumu: {state}.", detail=detail, fix=fix))
    port = ctx.windows.get("port") if isinstance(ctx.windows.get("port"), dict) else None
    if port and port.get("listening") and port.get("owner_is_retake") is False:
        checks.append(Check("Gizli motor", "fail", "8765 bağlantı noktasını Retake dışında bir program kullanıyor.",
                            fix="O programı kapatın; Retake motoru bu bağlantı noktasını bekler."))
    if config is None:
        if task is None or task.get("exists"):
            checks.append(Check("Gizli motor", "info", "Gizli motor ayar dosyası yok (elle başlatılan motor kullanılıyor olabilir)."))
        return checks
    boost = str(config.get("pixel_boost") or FULL_BOOST)
    parts = [f"çözünürlük {boost}", f"GFPGAN {'açık' if config.get('enhancer') else 'kapalı'}",
             f"eşleştirme {'açık' if config.get('pairing') else 'kapalı'}"]
    realism_off = bool(config.get("disable_realism"))
    realism_dir = Path(str(config.get("realism_dir") or ""))
    realism_ready = bool(str(config.get("realism_dir") or "")) and (realism_dir / "python.txt").is_file()
    parts.append("gerçekçilik " + ("kapalı (ayar)" if realism_off else "hazır" if realism_ready else "kurulu değil"))
    editors = [name for name in str(config.get("editors") or "").replace(";", ",").split(",") if name.strip()]
    if editors:
        parts.append(f"{len(editors)} ek düzenleyici")
    status = "ok"
    fix = ""
    if boost != FULL_BOOST:
        status = "warn"
        fix = ".\\tools\\install_retake_autostart.ps1 -FaceSwapperPixelBoost 1024x1024 (önce -Action Remove)"
    if not (Path(str(config.get("facefusion_dir") or "")) / "facefusion.py").is_file():
        status, fix = "fail", "FaceFusion klasörü taşınmış görünüyor; açılış görevini yeniden kurun."
    if not Path(str(config.get("python") or "")).is_file():
        status, fix = "fail", "Conda 'facefusion' ortamı değişmiş; açılış görevini yeniden kurun."
    checks.append(Check("Gizli motor", status, "Motor ayarları: " + ", ".join(parts) + ".", fix=fix))
    state = _read_json(ctx.retake_dir / "background-status.json")
    if state:
        age = ctx.now - float(state.get("updated_at") or 0)
        name = str(state.get("state") or "?")
        if name == "failed":
            checks.append(Check("Gizli motor", "fail", f"Gizli motor başlayamadı ({redact(str(state.get('error_type') or 'bilinmeyen hata'), ctx.env)}).",
                                fix=".\\tools\\install_retake_autostart.ps1 -Action Status ; gerekirse görevi yeniden kurun."))
        elif name == "waiting_for_port":
            checks.append(Check("Gizli motor", "warn", "Gizli motor 8765 bağlantı noktasının boşalmasını bekliyor (elle açılmış bir motor penceresi olabilir).",
                                fix="Elle açılmış motor penceresini kapatın."))
        else:
            checks.append(Check("Gizli motor", "info", f"Başlatıcı durumu: {name} ({_age(age)} önce güncellendi)."))
    return checks


def _age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 120:
        return f"{seconds} sn"
    if seconds < 7200:
        return f"{seconds // 60} dk"
    if seconds < 172800:
        return f"{seconds // 3600} sa"
    return f"{seconds // 86400} gün"


def check_engine(ctx: Context) -> tuple[list[Check], dict | None]:
    headers = _pairing_headers(ctx)
    try:
        status, _, body = ctx.opener(ENGINE_URL + "/health", headers=headers, timeout=8)
    except Exception as error:
        return [Check("Motor", "fail", "Motor yanıt vermiyor (127.0.0.1:8765).",
                      detail=redact(type(error).__name__, ctx.env),
                      fix="Açılış görevi varsa: .\\tools\\install_retake_autostart.ps1 -Action Start ; yoksa .\\tools\\start_facefusion_retake.ps1")], None
    if status == 401:
        return [Check("Motor", "fail", "Motor eşleştirme kodu istiyor ve yerel kod uyuşmuyor.",
                      fix="Motoru yeniden başlatın; kod %USERPROFILE%\\.retake\\pairing-token dosyasından okunur.")], None
    health = _json(body)
    if status != 200 or not isinstance(health, dict):
        return [Check("Motor", "fail", f"Motor sağlık yanıtı okunamadı (HTTP {status}).")], None
    checks: list[Check] = []
    revision = str(health.get("quality_revision") or "?")
    expected = expected_revision(ctx.root)
    if expected and revision != expected:
        checks.append(Check("Motor", "fail", f"Çalışan motor eski kalite sürümünde ({revision}); depo {expected}.",
                            fix=".\\tools\\update_retake_ultra.ps1 -SkipPull -SkipDeploy"))
    else:
        mode = "gizli motor" if health.get("background_mode") else "elle açılmış pencere"
        checks.append(Check("Motor", "ok", f"Motor çalışıyor: kalite sürümü {revision}, {mode}"
                            f"{', şu an fotoğraf işliyor' if health.get('busy') else ', boşta'}."))
    if not health.get("ok"):
        checks.append(Check("Motor", "fail", "Motor FaceFusion kurulumunu bulamıyor.", detail=redact(str(health.get("error") or ""), ctx.env)))
    if not health.get("cuda"):
        checks.append(Check("Motor", "fail", "CUDA kullanılamıyor: motor GPU yerine işlemciyle (çok yavaş, düşük ayarla) çalışır.",
                            detail=redact(str(health.get("provider_error") or ""), ctx.env),
                            fix="NVIDIA sürücüsünü ve 'onnxruntime-gpu' paketini kontrol edin (Paketler bölümüne bakın)."))
    boost = str(health.get("pixel_boost") or "?")
    if boost != FULL_BOOST:
        checks.append(Check("Motor", "warn", f"Yüz çözünürlüğü {boost}: ince ayrıntılar (kirpik, gözenek) düşük kalır.",
                            fix=".\\tools\\install_retake_autostart.ps1 -Action Remove ; .\\tools\\install_retake_autostart.ps1 -FaceSwapperPixelBoost 1024x1024"))
    optional = []
    optional.append("gerçekçilik (difüzyon) " + ("açık" if health.get("realism") else "kapalı"))
    optional.append("kişisel model " + ("var" if health.get("personal_model") else "yok"))
    optional.append("GFPGAN " + ("açık" if health.get("face_enhancer_enabled") else "kapalı"))
    editors = health.get("nim_editors_configured") or []
    if editors:
        optional.append(f"ek düzenleyici: {len(editors)}")
    checks.append(Check("Motor", "info", "İsteğe bağlı aşamalar: " + ", ".join(optional) + "."))
    scorer = health.get("identity_scorer") if isinstance(health.get("identity_scorer"), dict) else {}
    if scorer:
        if not scorer.get("enabled"):
            checks.append(Check("Motor", "warn", "Kalıcı ArcFace ölçümü kapalı; her ölçüm modelleri yeniden yükler (yavaş).",
                                fix="RETAKE_PERSISTENT_ARCFACE ortam değişkenini kaldırın."))
        elif int(scorer.get("failures") or 0) > 0:
            checks.append(Check("Motor", "warn", f"Kalıcı ArcFace ölçümü {scorer.get('failures')} kez hata verdi; yedek tek seferlik ölçüm kullanıldı.",
                                detail=redact(str(scorer.get("last_error") or ""), ctx.env)[:200]))
    cache = health.get("pass_cache") if isinstance(health.get("pass_cache"), dict) else None
    if cache and isinstance(cache.get("last_job"), dict):
        last = cache["last_job"]
        checks.append(Check("Motor", "info", f"Son işte geçiş önbelleği: {last.get('hits', 0)} isabet, {last.get('misses', 0)} kaçırma."))
    judge = str(health.get("nvidia_judge") or "?")
    if judge != "configured":
        checks.append(Check("Motor", "info" if judge == "not-configured" else "warn",
                            "NVIDIA görsel jürisi motorda " + ("kurulu değil." if judge == "not-configured" else f"geçersiz ({judge})."),
                            fix=".\\tools\\setup_hafize_judge.ps1 (ayrıntı Jüri bölümünde)"))
    return checks, health


def check_calibration(ctx: Context, health: dict | None) -> list[Check]:
    if health is None:
        return []
    try:
        status, _, body = ctx.opener(ENGINE_URL + "/api/calibration", headers=_pairing_headers(ctx), timeout=8)
    except Exception:
        return []
    data = _json(body) if status == 200 else None
    if not isinstance(data, dict):
        return []
    if not data.get("enabled"):
        return [Check("Öğrenme", "info", "Ultra öğrenmesi kapalı (her iş varsayılan sırayla aranır).")]
    banks = data.get("banks") if isinstance(data.get("banks"), dict) else {}
    jobs = sum(int(bank.get("jobs") or 0) for bank in banks.values() if isinstance(bank, dict))
    tests = sum(int(bank.get("self_tests") or 0) for bank in banks.values() if isinstance(bank, dict))
    return [Check("Öğrenme", "info", f"Ultra öğrenmesi açık: {len(banks)} profil, {jobs} iş, {tests} öz-test ölçümü.")]


def check_gpu(ctx: Context, health: dict | None) -> list[Check]:
    code, out = ctx.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,temperature.gpu,utilization.gpu",
                         "--format=csv,noheader,nounits"], 20)
    if code != 0 or not out.strip():
        return [Check("GPU", "warn", "nvidia-smi çalışmadı; GPU durumu okunamadı.",
                      fix="NVIDIA sürücüsünün kurulu olduğundan emin olun.")]
    fields = [part.strip() for part in out.strip().splitlines()[0].split(",")]
    if len(fields) < 6:
        return [Check("GPU", "warn", "nvidia-smi çıktısı tanınmadı.")]
    name, total, used, driver, temperature, utilization = fields[:6]
    checks = [Check("GPU", "ok", f"{name}: {used}/{total} MB kullanımda, sürücü {driver}, {temperature} °C, yük %{utilization}.")]
    code, apps = ctx.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], 20)
    others: dict[str, int] = {}
    if code == 0:
        for line in apps.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            base = re.split(r"[\\/]", parts[1])[-1] or "?"
            if base.lower().startswith("python"):
                continue
            try:
                memory = int(float(parts[2]))
            except ValueError:
                continue
            others[base] = others.get(base, 0) + memory
    other_total = sum(others.values())
    busy = bool(health and health.get("busy"))
    try:
        free = int(float(total)) - int(float(used))
    except ValueError:
        free = None
    if other_total >= OTHER_VRAM_WARN_MIB:
        listed = ", ".join(f"{redact(name, ctx.env)} {memory} MB" for name, memory in sorted(others.items(), key=lambda item: -item[1])[:4])
        checks.append(Check("GPU", "warn", f"Başka uygulamalar {other_total} MB ekran belleği kullanıyor ({listed}).",
                            detail="1024x1024 yüz geçişinde bellek hatası ve düşük çözünürlüğe inme riski.",
                            fix="Fotoğraf işlerken GPU kullanan uygulamaları (ör. ArcGIS Pro, 3B görüntüleyiciler) kapatın."))
    elif free is not None and not busy and free < 6000:
        checks.append(Check("GPU", "warn", f"Motor boştayken yalnız {free} MB ekran belleği boş.",
                            fix="GPU kullanan diğer uygulamaları kapatın; gerekirse bilgisayarı yeniden başlatın."))
    return checks


def check_disk(ctx: Context) -> list[Check]:
    checks: list[Check] = []
    seen: set[str] = set()
    for label, folder in (("Retake klasörü", str(ctx.root)), ("geçici klasör", tempfile.gettempdir()), ("kullanıcı klasörü", str(ctx.home))):
        try:
            usage = ctx.disk_usage(folder)
        except OSError:
            continue
        drive = (os.path.splitdrive(folder)[0] or folder[:1]).upper()
        if drive in seen:
            continue
        seen.add(drive)
        free = float(usage.free)
        status = "fail" if free < DISK_FAIL_GIB * 1024 ** 3 else "warn" if free < DISK_WARN_GIB * 1024 ** 3 else "ok"
        checks.append(Check("Disk", status, f"{label} sürücüsünde {_gib(free)} boş.",
                            fix="" if status == "ok" else "Geçici dosyaları temizleyin (Disk Temizleme); Retake işleri birkaç GB geçici alan kullanır."))
    return checks


def check_packages(ctx: Context) -> list[Check]:
    version = ctx.package_version or default_package_version
    found = {name: version(name) for name in PACKAGES}
    config = _read_json(ctx.retake_dir / "background-config.json") or {}
    engine_python = str(config.get("python") or "")
    same = bool(engine_python) and Path(engine_python).parent == Path(ctx.executable).parent
    where = "motor ortamı" if same else "bu Python ortamı (motor ortamı olmayabilir)"
    # "v" prefix: a four-part version (4.12.0.88) must not look like an IP.
    listed = ", ".join(f"{name} v{value}" for name, value in found.items() if value)
    checks = [Check("Paketler", "info", f"{where}: Python {sys.version.split()[0]}; {listed or 'tanınan paket yok'}.")]
    if found.get("onnxruntime") and found.get("onnxruntime-gpu"):
        checks.append(Check("Paketler", "warn", "Hem 'onnxruntime' hem 'onnxruntime-gpu' kurulu: CUDA sağlayıcısı kaybolabilir.",
                            fix="conda run -n facefusion python -m pip uninstall -y onnxruntime  (onnxruntime-gpu kalır)"))
    if found.get("onnxruntime-directml") and found.get("onnxruntime-gpu"):
        checks.append(Check("Paketler", "warn", "'onnxruntime-directml' ve 'onnxruntime-gpu' birlikte kurulu; CUDA yerine DirectML seçilebilir."))
    if same and not found.get("onnxruntime-gpu") and not found.get("onnxruntime"):
        checks.append(Check("Paketler", "fail", "Motor ortamında onnxruntime yok; FaceFusion çalışamaz.",
                            fix=".\\tools\\install_facefusion_windows.ps1 çıktısını kontrol edin."))
    return checks


def _jury_probe(ctx: Context, token: str) -> Check:
    cards = [solid_png((70, 130, 180)), solid_png((255, 140, 0))]
    payload = json.dumps({"task": "face-realism-rank", "candidates": [
        "data:image/png;base64," + base64.b64encode(card).decode("ascii") for card in cards]}).encode("utf-8")
    started = time.monotonic()
    try:
        status, _, body = ctx.opener(HAFIZE_URL + "/api/project-ai/retake/vision-judge", data=payload, timeout=190,
                                     method="POST", headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    except Exception as error:
        return Check("Jüri", "warn", f"Jüri denemesi tamamlanamadı ({redact(type(error).__name__, ctx.env)}).")
    elapsed = time.monotonic() - started
    reply = _json(body)
    if status != 200 or not isinstance(reply, dict):
        code = reply.get("error") if isinstance(reply, dict) else None
        code = code if isinstance(code, str) and re.fullmatch(r"PROJECT_AI_[A-Z0-9_]{1,60}", code) else ""
        fix = ""
        if status in (401, 403):
            fix = "Jeton özeti Render'daki HAFIZE_PROJECT_AI_TOKEN_HASHES ile eşleşmiyor: .\\tools\\setup_hafize_judge.ps1 ile değeri yeniden kopyalayıp Render'a girin."
        return Check("Jüri", "fail", f"Jüri denemesi başarısız: HTTP {status} {code}".strip() + f", {elapsed:.0f} sn.", fix=fix)
    votes = int(reply.get("votes") or 0)
    models = [str(model).split("/")[-1] for model in reply.get("models") or []][:6]
    status_name = "ok" if votes > 0 else "warn"
    return Check("Jüri", status_name, f"Sentetik jüri denemesi: {votes} oy, {len(models)} model yanıt verdi, {elapsed:.0f} sn.",
                 detail=("Yanıtlayan: " + ", ".join(models)) if models else "Hiçbir model geçerli oy vermedi.",
                 fix="" if votes else "Birkaç dakika sonra tekrar deneyin; sürerse Hafize Render günlüklerine bakın.")


def check_jury(ctx: Context, health: dict | None) -> list[Check]:
    checks: list[Check] = []
    token_path = ctx.retake_dir / "hafize-project-token"
    token = ""
    try:
        token = token_path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        pass
    if not token:
        return [Check("Jüri", "info", "NVIDIA jürisi bu bilgisayarda kurulmamış (isteğe bağlı danışman).",
                      fix="Kurmak için: .\\tools\\setup_hafize_judge.ps1")]
    if not TOKEN_FORMAT.match(token):
        return [Check("Jüri", "fail", "Yerel jüri jetonu bozuk görünüyor (içeriği rapora yazılmadı).",
                      fix=".\\tools\\setup_hafize_judge.ps1 -Rotate")]
    checks.append(Check("Jüri", "ok", "Yerel jüri jetonu mevcut ve biçimi geçerli (içeriği rapora yazılmadı)."))
    if ctx.offline:
        return checks
    try:
        status, _, body = ctx.opener(HAFIZE_URL + "/api/health", timeout=90)
    except Exception as error:
        checks.append(Check("Jüri", "warn", f"Hafize'ye ulaşılamadı ({redact(type(error).__name__, ctx.env)}); Render uyanıyor olabilir.",
                            fix="Bir dakika sonra tekrar çalıştırın."))
        return checks
    data = _json(body) if status == 200 else None
    gateway = data.get("projectAiGateway") if isinstance(data, dict) else None
    judge = gateway.get("visionJudge") if isinstance(gateway, dict) else None
    if not isinstance(judge, dict):
        checks.append(Check("Jüri", "warn", f"Hafize sağlık yanıtında jüri bilgisi yok (HTTP {status})."))
        return checks
    models = [str(model).split("/")[-1] for model in judge.get("models") or []]
    ready = bool(gateway.get("enabled")) and "retake" in (judge.get("projects") or [])
    layouts = judge.get("layouts") or []
    detail = f"Panel: {', '.join(models) or '—'}; yedek: {len(judge.get('reserveModels') or [])}."
    if ready:
        checks.append(Check("Jüri", "ok", f"Hafize jürisi hazır: {len(models)} model.", detail=detail))
    else:
        checks.append(Check("Jüri", "fail", "Hafize'de Retake jürisi etkin değil.", detail=detail,
                            fix="Render > hafize-free > Environment: HAFIZE_PROJECT_AI_TOKEN_HASHES değerini girip yeniden dağıtın."))
    if "review-sheet" not in layouts:
        checks.append(Check("Jüri", "warn", "Hafize inceleme paftasını tanımıyor (eski sürüm); jüri yalnız yüz kırpımı görür.",
                            fix="Render'da hafize-free için 'Manual Deploy > Deploy latest commit'."))
    roster = judge.get("health") if isinstance(judge.get("health"), dict) else None
    if roster:
        benched = [str(model).split("/")[-1] for model in roster.get("benched") or []]
        rows = []
        for model, stats in (roster.get("models") or {}).items():
            if isinstance(stats, dict) and stats.get("calls"):
                rows.append(f"{str(model).split('/')[-1]} {stats.get('valid', 0)}/{stats.get('calls', 0)}")
        if rows:
            checks.append(Check("Jüri", "warn" if benched else "info",
                                "Son jüri çağrıları (geçerli/toplam): " + ", ".join(rows[:8]) + ".",
                                detail=("Geçici olarak yedeğe alınan: " + ", ".join(benched)) if benched else ""))
    if health is not None and health.get("nvidia_judge") != "configured":
        checks.append(Check("Jüri", "warn", "Jeton var ama çalışan motor jüriyi görmüyor.",
                            fix=".\\tools\\update_retake_ultra.ps1 -SkipPull -SkipDeploy (motoru yeniden başlatır)"))
    if ctx.probe_jury:
        checks.append(_jury_probe(ctx, token))
    return checks


def check_site(ctx: Context) -> list[Check]:
    if ctx.offline:
        return []
    try:
        local = SHELL_CACHE_LINE.search((ctx.root / "sw.js").read_text(encoding="utf-8"))
    except OSError:
        local = None
    try:
        status, _, body = ctx.opener(SITE_URL + "/sw.js", timeout=20, headers={"Cache-Control": "no-cache"})
    except Exception as error:
        return [Check("Site", "warn", f"Yayındaki siteye ulaşılamadı ({redact(type(error).__name__, ctx.env)}).")]
    remote = SHELL_CACHE_LINE.search(body.decode("utf-8", "replace")) if status == 200 else None
    if not remote or not local:
        return [Check("Site", "warn", f"Site sürümü okunamadı (HTTP {status}).")]
    if remote.group(1) == local.group(1):
        return [Check("Site", "ok", f"Yayındaki site bu depoyla aynı sürümde ({local.group(1)}).")]
    return [Check("Site", "warn", f"Yayındaki site {remote.group(1)}, bu depo {local.group(1)}.",
                  detail="Site ile motor farklı sürümdeyse yeni ayarlar görünmeyebilir.",
                  fix="Depo geride ise güncelleyin; site gerideyse yeni yayını bekleyin ve sayfayı yenileyin (Ctrl+F5).")]


def check_events(ctx: Context) -> list[Check]:
    path = ctx.retake_dir / EVENT_LOG_NAME
    events = read_events(path, limit=2000)
    if not events:
        return [Check("Olaylar", "info", "Motor olay kaydı henüz yok; motor bu sürümle yeniden başlayınca oluşur.")]
    summary = summarize_events(events, since=ctx.now - EVENT_WINDOW_SECONDS)
    counts = summary["counts"]
    checks: list[Check] = []
    advice = {
        "memory": ("GPU bellek hatası", "Fotoğraf işlerken GPU kullanan diğer uygulamaları kapatın; çözünürlük otomatik düşürülmüş olabilir."),
        "realism": ("gerçekçilik aşaması hatası", "Gerçekçilik isteğe bağlıdır; sürerse motoru yeniden başlatın."),
        "jury_skip": ("jüri atlanması", "Jüri bölümüne bakın; yerel seçim her zaman korunur."),
        "identity": ("ArcFace ölçüm hatası", "Sürerse motoru yeniden başlatın."),
        "error": ("hata", "Sürerse bu raporu paylaşın."),
    }
    for category, (label, fix) in advice.items():
        if counts.get(category):
            checks.append(Check("Olaylar", "warn", f"Son 7 günde {counts[category]} kez {label}.", fix=fix))
    if counts.get("skipped"):
        checks.append(Check("Olaylar", "info", f"Son 7 günde {counts['skipped']} isteğe bağlı adım atlandı (aday, ince ayar vb.)."))
    checks.append(Check("Olaylar", "info", f"Son 7 günde {counts.get('result', 0)} tamamlanan iş, {summary['starts']} motor başlangıcı."))
    for row in summary["recent_results"]:
        checks.append(Check("Olaylar", "info", f"{_stamp(row['t'])} {row['msg']}"))
    for row in summary["recent_problems"][-5:]:
        checks.append(Check("Olaylar", "info", f"{_stamp(row['t'])} [{row['cat']}] {row['msg']}"))
    return checks


def _stamp(value: float) -> str:
    return time.strftime("%d.%m %H:%M", time.localtime(float(value)))


# ------------------------------------------------------------------ report

def collect(ctx: Context) -> list[Check]:
    checks: list[Check] = []
    checks += check_repo(ctx)
    checks += check_background(ctx)
    engine_checks, health = check_engine(ctx)
    checks += engine_checks
    checks += check_calibration(ctx, health)
    checks += check_gpu(ctx, health)
    checks += check_disk(ctx)
    checks += check_packages(ctx)
    checks += check_jury(ctx, health)
    checks += check_site(ctx)
    checks += check_events(ctx)
    for check in checks:
        check.title = redact(check.title, ctx.env)
        check.detail = redact(check.detail, ctx.env)
        check.fix = redact(check.fix, ctx.env)
    return checks


def render(checks: list[Check], *, now: float | None = None) -> str:
    counts = {status: sum(1 for check in checks if check.status == status) for status in STATUS_LABEL}
    lines = [
        f"===== RETAKE TANI RAPORU (doctor {DOCTOR_VERSION}) =====",
        f"Tarih: {time.strftime('%d.%m.%Y %H:%M', time.localtime(now or time.time()))} | "
        f"{counts['ok']} tamam, {counts['warn']} uyarı, {counts['fail']} sorun, {counts['info']} bilgi",
        "Not: Fotoğraf, dosya adı, jeton, kullanıcı adı ve yerel yol içermez.",
    ]
    area = None
    for check in checks:
        if check.area != area:
            area = check.area
            lines.append("")
            lines.append(f"[{area}]")
        lines.append(f"  {STATUS_LABEL.get(check.status, '?'):5} {check.title}")
        if check.detail:
            lines.append(f"        {check.detail}")
    steps = [check for check in sorted(checks, key=lambda check: SEVERITY.get(check.status, 9))
             if check.fix and check.status in ("fail", "warn")]
    lines.append("")
    if steps:
        lines.append("Önerilen adımlar (önem sırasıyla):")
        seen: set[str] = set()
        number = 0
        for check in steps:
            if check.fix in seen:
                continue
            seen.add(check.fix)
            number += 1
            lines.append(f"  {number}. {check.fix}")
    else:
        lines.append("Önerilen adım yok: kurulum en iyi kalite için hazır görünüyor.")
    lines.append("===== RAPOR SONU =====")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retake kurulumunu denetler ve paylaşılabilir bir rapor üretir.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--windows-facts", type=Path, help="PowerShell sarmalayıcısının topladığı Windows bilgileri (JSON).")
    parser.add_argument("--offline", action="store_true", help="GitHub, Hafize ve siteye bağlanma.")
    parser.add_argument("--probe-jury", action="store_true", help="Hafize jürisine iki sentetik renk kartı gönder (fotoğraf değil).")
    parser.add_argument("--json", action="store_true", help="Denetimleri JSON olarak yaz.")
    parser.add_argument("--output", type=Path, help="Raporu bu dosyaya UTF-8 olarak da yaz.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    windows: dict = {}
    if args.windows_facts:
        windows = _read_json(args.windows_facts) or {}
    ctx = Context(root=args.root.resolve(), home=args.home, opener=default_opener, run=default_runner,
                  windows=windows, offline=args.offline, probe_jury=args.probe_jury)
    checks = collect(ctx)
    text = json.dumps([asdict(check) for check in checks], ensure_ascii=False, indent=1) if args.json else render(checks, now=ctx.now)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.write(text.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
