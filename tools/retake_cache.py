"""Caches and timing memory for the local Retake engine.

- IdentityReportCache: the ArcFace screen of the 25 references only depends on
  those references, so its report is kept on disk and reused while they stay
  the same (it took ~40 s per request before).
- PassCache: FaceFusion pass outputs for a given target crop, sources, model,
  weight and pixel boost. "Regenerate" variants that keep the swapper weight
  reuse both GPU passes and only redo blending and head fitting.
- StageTimings: an exponential moving average of how long each job stage takes
  on this computer, used to show the remaining time.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CACHE_DIR = Path.home() / ".retake"


def digest(*parts: bytes | str) -> str:
    """SHA-256 over length-prefixed parts (unambiguous for any part contents)."""
    hasher = hashlib.sha256()
    for part in parts:
        data = part.encode("utf-8") if isinstance(part, str) else bytes(part)
        hasher.update(len(data).to_bytes(8, "big"))
        hasher.update(data)
    return hasher.hexdigest()


def references_digest(reference_bytes: Iterable[bytes], verifier: str) -> str:
    return digest(verifier, *reference_bytes)


class IdentityReportCache:
    def __init__(self, directory: Path | None = None, max_entries: int = 16) -> None:
        self.directory = (directory or DEFAULT_CACHE_DIR / "identity").expanduser()
        self.max_entries = max_entries
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            report = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return report if isinstance(report, dict) and report.get("ok") else None

    def put(self, key: str, report: dict[str, Any]) -> None:
        # Hard identity conflicts are not cached: the user is expected to fix
        # the references, and a retry should check again.
        if not isinstance(report, dict) or not report.get("ok") or report.get("hard_conflict"):
            return
        with self._lock:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = self._path(key).with_suffix(".tmp")
                temporary.write_text(json.dumps(report), encoding="utf-8")
                temporary.replace(self._path(key))
                entries = sorted(self.directory.glob("*.json"), key=lambda path: path.stat().st_mtime)
                for stale in entries[:-self.max_entries]:
                    stale.unlink(missing_ok=True)
            except OSError:
                pass  # a cache that cannot be written is only slower


class PassCache:
    """Small in-memory LRU of FaceFusion pass outputs (PNG bytes)."""

    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(target: bytes, sources: Iterable[bytes], model: str, weight: float, pixel_boost: str) -> str:
        return digest(model, f"{weight:.3f}", pixel_boost, target, *sources)

    def get(self, key: str) -> tuple[bytes, str] | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: str, output: bytes, used_boost: str) -> None:
        with self._lock:
            self._entries[key] = (output, used_boost)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


class StageTimings:
    """Remembers how long each stage takes here, to estimate the remaining time."""

    def __init__(self, path: Path | None = None, smoothing: float = 0.35) -> None:
        self.path = (path or DEFAULT_CACHE_DIR / "timings.json").expanduser()
        self.smoothing = smoothing
        self._lock = threading.Lock()
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            self.seconds = {str(k): float(v) for k, v in loaded.items() if float(v) > 0}
        except (OSError, ValueError, TypeError, AttributeError):
            self.seconds = {}

    def record(self, stage: str, seconds: float) -> None:
        if not seconds > 0:
            return
        with self._lock:
            previous = self.seconds.get(stage)
            self.seconds[stage] = seconds if previous is None else (
                previous + (seconds - previous) * self.smoothing
            )
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.seconds), encoding="utf-8")
            except OSError:
                pass

    def remaining(self, stages: list[str], current: int, elapsed_in_current: float) -> float | None:
        """Seconds left from stage index `current` on; None until every stage has a timing."""
        if any(stage not in self.seconds for stage in stages[current:]):
            return None
        left = sum(self.seconds[stage] for stage in stages[current:])
        return max(0.0, left - max(0.0, elapsed_in_current))
