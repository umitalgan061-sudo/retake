"""Persistent ArcFace scorer for one job (Retake v25).

Every identity measurement used to start a new Python process, import
FaceFusion and create the YOLO + ArcFace ONNX sessions (CUDA session set-up
alone takes seconds). A render measures 5-7 times (pool choice,
reinforcement, blends, final candidates, finishing steps), so with cached
FaceFusion passes most of a round was spent loading the same two models.

This keeps one worker alive with the models loaded and talks to it over a
line-based JSON protocol on stdin/stdout. It is purely an accelerator:
any start-up error, crash, timeout or malformed answer kills the worker and
the caller falls back to the one-shot process for that measurement. After
repeated failures the persistent path stays off until the engine restarts.
The worker is closed after every job and before the diffusion realism pass
so it never holds GPU memory that a heavier step needs.
"""
from __future__ import annotations

import json
import math
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

READY_PREFIX = "RETAKE_IDENTITY_READY="
REPORT_PREFIX = "RETAKE_IDENTITY_REPORT="
READY_TIMEOUT_SECONDS = 240.0
CALL_TIMEOUT_SECONDS = 300.0
MAX_FAILURES = 3
STDERR_TAIL_LINES = 40


def _pump(stream, lines: "queue.Queue[str | None]") -> None:
    try:
        for line in stream:
            lines.put(line.rstrip("\r\n"))
    except (OSError, ValueError):
        pass
    finally:
        lines.put(None)  # end of stream: the worker exited


def _drain(stream, tail: deque) -> None:
    try:
        for line in stream:
            tail.append(line.rstrip("\r\n"))
    except (OSError, ValueError):
        pass


class PersistentIdentityScorer:
    def __init__(
        self, facefusion_dir: Path, *, python: str | None = None, worker: Path | None = None,
        popen: Callable[..., Any] = subprocess.Popen, popen_options: dict[str, Any] | None = None,
        ready_timeout: float = READY_TIMEOUT_SECONDS, call_timeout: float = CALL_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.facefusion_dir = Path(facefusion_dir)
        self.python = python or sys.executable
        self.worker = Path(worker) if worker else Path(__file__).with_name("facefusion_identity_worker.py")
        self._popen = popen
        self._popen_options = dict(popen_options or {})
        self.ready_timeout = ready_timeout
        self.call_timeout = call_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._process: Any = None
        self._lines: "queue.Queue[str | None] | None" = None
        self._stderr: deque = deque(maxlen=STDERR_TAIL_LINES)
        self._next_id = 0
        self.failures = 0
        self.disabled = False
        self.stats = {"calls": 0, "starts": 0, "failures": 0, "closes": 0, "seconds": 0.0}
        self.last_error = ""

    # -- process management -------------------------------------------------
    def _alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _kill(self) -> None:
        process, self._process, self._lines = self._process, None, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except Exception:
            pass

    def _read_until(self, prefix: str, deadline: float, request_id: int | None = None) -> dict[str, Any]:
        assert self._lines is not None
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError("ArcFace sunucusu zamanında yanıt vermedi.")
            try:
                line = self._lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if not self._alive():
                    raise RuntimeError("ArcFace sunucusu kapandı.")
                continue
            if line is None:
                raise RuntimeError("ArcFace sunucusu kapandı.")
            if not line.startswith(prefix):
                continue  # FaceFusion's own log lines
            try:
                payload = json.loads(line[len(prefix):])
            except json.JSONDecodeError as error:
                raise RuntimeError(f"ArcFace sunucusu bozuk yanıt verdi: {error}") from None
            if not isinstance(payload, dict):
                raise RuntimeError("ArcFace sunucusu bozuk yanıt verdi.")
            if request_id is not None and payload.get("id") != request_id:
                continue  # an answer to an older, abandoned request
            return payload

    def _start(self) -> None:
        if not self.worker.is_file():
            raise FileNotFoundError(f"ArcFace çalışanı bulunamadı: {self.worker}")
        self._stderr = deque(maxlen=STDERR_TAIL_LINES)
        command = [self.python, str(self.worker), "--facefusion-dir", str(self.facefusion_dir), "--serve"]
        process = self._popen(
            command, cwd=str(self.facefusion_dir), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
            **self._popen_options
        )
        lines: "queue.Queue[str | None]" = queue.Queue()
        threading.Thread(target=_pump, args=(process.stdout, lines), daemon=True).start()
        threading.Thread(target=_drain, args=(process.stderr, self._stderr), daemon=True).start()
        self._process, self._lines = process, lines
        ready = self._read_until(READY_PREFIX, self._clock() + self.ready_timeout)
        if ready.get("ok") is not True:
            raise RuntimeError(str(ready.get("error") or "ArcFace modelleri yüklenemedi."))
        self.stats["starts"] += 1

    # -- public API ---------------------------------------------------------
    def score(self, centroid: list[float], probe_paths: list[Path]) -> list[float | None] | None:
        """Similarities, or None when the caller should use the one-shot path."""
        if self.disabled or not centroid or not probe_paths:
            return None
        with self._lock:
            started = self._clock()
            try:
                if not self._alive():
                    self._kill()
                    self._start()
                self._next_id += 1
                request_id = self._next_id
                request = {"id": request_id, "centroid": [float(value) for value in centroid],
                           "probes": [str(path) for path in probe_paths]}
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
                report = self._read_until(REPORT_PREFIX, self._clock() + self.call_timeout, request_id)
                if report.get("ok") is not True:
                    raise RuntimeError(str(report.get("error") or "ArcFace ölçümü başarısız."))
                values = report.get("similarities")
                if not isinstance(values, list) or len(values) != len(probe_paths):
                    raise RuntimeError("ArcFace sunucusu eksik sonuç döndürdü.")
                self.failures = 0
                self.stats["calls"] += 1
                self.stats["seconds"] += self._clock() - started
                return [
                    float(value) if type(value) in (int, float) and math.isfinite(value) and -1 <= value <= 1 else None
                    for value in values
                ]
            except Exception as error:
                self.failures += 1
                self.stats["failures"] += 1
                tail = " | ".join(list(self._stderr)[-3:])
                self.last_error = f"{type(error).__name__}: {error}" + (f" ({tail})" if tail else "")
                self._kill()
                if self.failures >= MAX_FAILURES:
                    self.disabled = True
                print(
                    f"[ArcFace] Kalıcı ölçüm sunucusu kullanılamadı ({self.last_error}); tek seferlik ölçüme dönülüyor"
                    + ("; bu oturumda kapatıldı." if self.disabled else "."),
                    flush=True
                )
                return None

    def close(self) -> None:
        """Free the models (end of a job, or before a heavier GPU step)."""
        with self._lock:
            if self._process is None:
                return
            if self._alive():
                try:
                    self._process.stdin.write(json.dumps({"op": "quit"}) + "\n")
                    self._process.stdin.flush()
                    self._process.wait(timeout=5)
                except Exception:
                    pass
            self._kill()
            self.stats["closes"] += 1

    def describe(self) -> dict[str, Any]:
        calls = self.stats["calls"]
        return {
            "enabled": not self.disabled,
            "running": self._alive(),
            "calls": calls,
            "starts": self.stats["starts"],
            "failures": self.stats["failures"],
            "mean_call_seconds": round(self.stats["seconds"] / calls, 2) if calls else None,
            "last_error": self.last_error[:300] if self.last_error else "",
        }
