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
import math
import os
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


# One Ultra round makes ~6-9 FaceFusion passes (identity pools, reinforcement,
# appearance, texture) per swapper weight. The 24 variants use three weights,
# so 8 entries thrashed: every weight change re-ran every pass (~3 min/round).
# Keep a whole job's passes; the cache is cleared after each job.
PASS_CACHE_MAX_ENTRIES = 96
PASS_CACHE_MAX_BYTES = 1536 * 1024 * 1024


class PassCache:
    """In-memory LRU of FaceFusion pass outputs (PNG bytes), bounded by count
    and by total bytes; job-local (cleared after every job)."""

    def __init__(self, max_entries: int = PASS_CACHE_MAX_ENTRIES, max_bytes: int | None = None) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = PASS_CACHE_MAX_BYTES if max_bytes is None else max(1, int(max_bytes))
        self._entries: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._last_job: dict[str, int] | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> "PassCache":
        """RETAKE_PASS_CACHE_MB (64..8192) overrides the byte bound."""
        value = (environ if environ is not None else os.environ).get("RETAKE_PASS_CACHE_MB", "")
        try:
            megabytes = int(value)
        except ValueError:
            return cls()
        if not 64 <= megabytes <= 8192:
            return cls()
        return cls(max_bytes=megabytes * 1024 * 1024)

    def describe(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_entries": self.max_entries, "max_mb": self.max_bytes // (1024 * 1024),
                "entries": len(self._entries), "mb": round(self._bytes / (1024 * 1024)),
                "hits": self._hits, "misses": self._misses,
                "last_job": dict(self._last_job) if self._last_job else None,
            }

    @staticmethod
    def key(target: bytes, sources: Iterable[bytes], model: str, weight: float, pixel_boost: str) -> str:
        return digest(model, f"{weight:.3f}", pixel_boost, target, *sources)

    def get(self, key: str) -> tuple[bytes, str] | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
                self._hits += 1
            else:
                self._misses += 1
            return value

    def put(self, key: str, output: bytes, used_boost: str) -> None:
        with self._lock:
            if len(output) > self.max_bytes:
                return  # larger than the whole budget: never cache
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= len(previous[0])
            self._entries[key] = (output, used_boost)
            self._bytes += len(output)
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                _, (evicted, _) = self._entries.popitem(last=False)
                self._bytes -= len(evicted)

    def clear(self) -> int:
        """Release job-local image bytes; keep model weights and identity reports."""
        with self._lock:
            count = len(self._entries)
            if self._hits or self._misses:
                # Diagnostics for /health: did the last job reuse its passes?
                self._last_job = {"hits": self._hits, "misses": self._misses,
                                  "entries": count, "mb": round(self._bytes / (1024 * 1024))}
            self._entries.clear()
            self._bytes = 0
            self._hits = 0
            self._misses = 0
            return count

    def __len__(self) -> int:
        with self._lock:
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


class IdentityScoreCache:
    """Job-local, bounded reuse of successful measurements of identical bytes.

    A score is scoped to the exact reference centroid and image bytes.
    Never retain image bytes, embeddings, paths or failed measurements.
    The caller constructs one instance per Ultra job, including all phases.
    """
    def __init__(self, max_entries=1024):
        self.max_entries=max(1,min(4096,int(max_entries)))
        self._entries=OrderedDict()
        self.hits=0
        self.measured=0
        self.failed=0

    def score(self, centroid, paths, measure):
        values=[None]*len(paths)
        if not centroid:
            return values
        try:
            scope=json.dumps(centroid,separators=(",",":"),allow_nan=False)
        except (ValueError,TypeError):
            return values
        groups={}
        reused={}
        def unchanged(index,key):
            try:
                return digest(scope,Path(paths[index]).read_bytes())==key
            except OSError:
                return False
        for index,path in enumerate(paths):
            try:
                key=digest(scope,Path(path).read_bytes())
            except OSError:
                self.failed+=1
                continue
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits+=1
                values[index]=self._entries[key]
                reused[index]=key
            else:
                groups.setdefault(key,[]).append(index)
        def verify_reused():
            for index,key in reused.items():
                if not unchanged(index,key):
                    self.hits-=1
                    self.failed+=1
                    values[index]=None
            return values
        if not groups:
            return verify_reused()
        keys=list(groups)
        probes=[paths[groups[key][0]] for key in keys]
        self.measured+=len(probes)
        try:
            scores=measure(probes)
        except Exception:
            self.failed+=len(probes)
            return verify_reused()
        if not isinstance(scores,(tuple,list)) or len(scores)!=len(probes):
            self.failed+=len(probes)
            return verify_reused()
        for key,value in zip(keys,scores):
            indices=groups[key]
            if type(value) not in (float,int) or not math.isfinite(value) or not -1<=value<=1:
                self.failed+=len(indices)
                continue
            # The measurement must belong to the unchanged bytes we keyed.
            if not unchanged(indices[0],key):
                self.failed+=len(indices)
                continue
            self._entries[key]=float(value)
            for position,idx in enumerate(indices):
                if position and not unchanged(idx,key):
                    self.failed+=1
                    continue
                if position:
                    self.hits+=1
                values[idx]=float(value)
            while len(self._entries)>self.max_entries:
                self._entries.popitem(last=False)
        return verify_reused()

    def describe(self):
        return {"hits":self.hits,"measured":self.measured,"failed":self.failed,"entries":len(self._entries)}
