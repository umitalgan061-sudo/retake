"""Opt-in, one-shot local FLUX worker. No listener, network, or persistent GPU."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

MODEL = "retake-local/flux2-klein-4b"
MAX_BODY = 32 * 1024 * 1024
MAX_RESPONSE = 32 * 1024 * 1024

def proof_path():
    return Path.home() / ".retake" / "flux2-managed.json"

def managed_config():
    try:
        cfg = json.loads(proof_path().read_text(encoding="utf-8-sig"))
        python = Path.home() / "retake-flux2-venv" / "Scripts" / "python.exe"
        model = Path.home() / "retake-models" / "FLUX.2-klein-4B"
        if (cfg.get("model") == MODEL and cfg.get("ready") is True
                and cfg.get("synthetic_edit_passed") is True
                and cfg.get("offload") == "sequential" and cfg.get("max_edge") == 512
                and python.is_file() and (model / "model_index.json").is_file()):
            return python, model
    except (OSError, ValueError, AttributeError):
        pass
    return None

def isolated_env():
    env = os.environ.copy()
    prefix = env.pop("CONDA_PREFIX", "")
    if prefix:
        root = os.path.normcase(os.path.abspath(prefix))
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
            if p and not (os.path.normcase(os.path.abspath(p)) == root
                          or os.path.normcase(os.path.abspath(p)).startswith(root + os.sep)))
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV"):
        env.pop(key, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    return env

def managed_request(port, path, payload=None, timeout=360):
    if port != 0:
        raise ValueError("Managed editor is not an HTTP endpoint")
    cfg = managed_config()
    if path == "/v1/health/ready" and payload is None:
        return cfg is not None
    if not cfg or path != "/v1/images/edits" or not isinstance(payload, dict) or payload.get("model") != MODEL:
        raise ValueError("Managed editor unavailable")
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_BODY:
        raise ValueError("Managed edit exceeds input limit")
    python, model = cfg
    runner = Path(__file__).with_name("local_flux2_editor.py")
    with tempfile.TemporaryDirectory(prefix="retake-flux2-job-") as folder:
        output = Path(folder) / "response.json"
        args = [str(python), str(runner), "--model-dir", str(model),
                "--offload", "sequential", "--max-edge", "512", "--one-shot-output", str(output)]
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        # subprocess.run waits for process exit (and kills on timeout), so
        # model allocations are gone before the next FaceFusion CUDA call.
        result = subprocess.run(args, input=data, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=min(360, timeout), check=False, env=isolated_env(), **options)
        if result.returncode or not output.is_file() or output.stat().st_size > MAX_RESPONSE:
            raise RuntimeError("Managed FLUX edit failed")
        return json.loads(output.read_text(encoding="utf-8"))
