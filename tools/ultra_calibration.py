"""Learned Ultra: which of the 24 variants look most natural for *this* person.

A GPU round takes minutes, so the seven-minute Ultra window usually runs only
the first few of the 24 variants. Their fixed order decided, in practice,
which settings were ever tried. This module learns a better order from
measured outcomes, per reference bank (person), on this computer only:

* Gold signal — self-tests. When the target is one of the person's own
  references, the original face is ground truth; every round reports SSIM and
  ΔE against it (``X-Retake-Self-Fidelity``). Fidelity = SSIM − 0.02·ΔE.
* Proxy signal — ordinary jobs. Regional photographic detail retained
  (``X-Retake-Detail-Score``), minus a penalty for the 128 px reinforcement.
  Used only until enough self-tests exist.

Order = upper confidence bound (mean + exploration bonus), so untested
variants still get tried across jobs while proven ones run first. Nothing here
changes what counts as a valid result: identity, detail and every other gate
stay in charge of the final choice. Stored values are numbers only (no images,
no embeddings); deleting the file resets the learning.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .retake_cache import DEFAULT_CACHE_DIR, digest
except ImportError:  # pragma: no cover - script execution
    from retake_cache import DEFAULT_CACHE_DIR, digest

STORE_VERSION = 1
DEFAULT_PATH = DEFAULT_CACHE_DIR / "ultra-calibration.json"
# Fidelity in self-tests is SSIM − DELTA_E_WEIGHT·ΔE. With this weight a JPEG
# re-save of the original scores ≈0.94 and the 25 Nov artificial result ≈0.55.
DELTA_E_WEIGHT = 0.02
REINFORCED_PROXY_PENALTY = 0.05
MIN_FIDELITY_SAMPLES = 3
MIN_PROXY_SAMPLES = 6
EXPLORATION = {"fidelity": 0.06, "proxy": 0.04}
MAX_BANKS = 8
_SELF_FIDELITY = re.compile(r"^ssim:(-?\d+(?:\.\d+)?),deltaE:(\d+(?:\.\d+)?)$")


def bank_key(reference_bytes: Iterable[bytes], weights: Sequence[int] | None = None) -> str:
    """Stable identifier of the person's reference bank (never stored in reverse)."""
    return digest("retake-ultra-calibration", *(bytes(item) for item in reference_bytes),
                  ",".join(str(int(w)) for w in (weights or ())))[:32]


def schedule_signature(profile_version: str, variations: Sequence[int]) -> str:
    """Learning resets whenever the meaning of a variant number may change."""
    return digest(profile_version, ",".join(str(v) for v in variations))[:16]


def parse_self_fidelity(value: Any) -> tuple[float, float] | None:
    match = _SELF_FIDELITY.match(str(value or "").strip())
    if not match:
        return None
    ssim, delta_e = float(match.group(1)), float(match.group(2))
    if not (math.isfinite(ssim) and -1 <= ssim <= 1 and math.isfinite(delta_e) and 0 <= delta_e <= 200):
        return None
    return ssim, delta_e


def round_quality(headers: Mapping[str, str]) -> tuple[str, float] | None:
    """('fidelity'|'proxy', value in 0..1) for one verified round, or None."""
    fidelity = parse_self_fidelity(headers.get("X-Retake-Self-Fidelity"))
    if fidelity is not None:
        ssim, delta_e = fidelity
        return "fidelity", max(0.0, min(1.0, ssim - DELTA_E_WEIGHT * delta_e))
    try:
        detail = float(headers.get("X-Retake-Detail-Score", ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(detail) or not 0 <= detail <= 1:
        return None
    if headers.get("X-Retake-Identity-Reinforced") == "on":
        detail -= REINFORCED_PROXY_PENALTY
    return "proxy", max(0.0, min(1.0, detail))


def _empty_stats() -> dict[str, float]:
    return {"n": 0, "sum": 0.0, "sq": 0.0}


def _valid_stats(value: Any) -> bool:
    return (
        isinstance(value, dict) and type(value.get("n")) is int and 0 <= value["n"] <= 1_000_000
        and all(isinstance(value.get(k), (int, float)) and math.isfinite(value[k]) for k in ("sum", "sq"))
    )


class CalibrationStore:
    """Thread-safe JSON store; a corrupt or foreign file is treated as empty."""

    def __init__(self, path: Path | None = None, signature: str = "", enabled: bool = True) -> None:
        self.path = (path or DEFAULT_PATH).expanduser()
        self.signature = signature
        self.enabled = enabled
        self._lock = threading.Lock()

    # ---- persistence -------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": STORE_VERSION, "signature": self.signature, "banks": {}}
        if (
            not isinstance(data, dict) or data.get("version") != STORE_VERSION
            or data.get("signature") != self.signature or not isinstance(data.get("banks"), dict)
        ):
            return {"version": STORE_VERSION, "signature": self.signature, "banks": {}}
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        banks = data["banks"]
        if len(banks) > MAX_BANKS:  # keep the most recently used people
            for key in sorted(banks, key=lambda k: banks[k].get("updated", 0))[: len(banks) - MAX_BANKS]:
                del banks[key]
        handle, temporary = tempfile.mkstemp(prefix=".ultra-calibration-", dir=str(self.path.parent))
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(data, stream, separators=(",", ":"))
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    # ---- learning ----------------------------------------------------
    def record(self, bank: str, observations: Sequence[tuple[int, Mapping[str, str]]], self_test: bool = False) -> int:
        """Add verified rounds of one job; returns how many were usable."""
        if not self.enabled or not bank:
            return 0
        # One original grid observation per variant per job. Repeated
        # finalists are correlated trials, not independent evidence.
        rows = []
        seen = set()
        for variation, headers in observations:
            if type(variation) is not int or not 0 <= variation < 24 or variation in seen:
                continue
            if headers.get("X-Retake-Detail-Gate") not in (None, "passed"):
                continue
            quality = round_quality(headers)
            if quality is None:
                continue
            if "X-Retake-Similarity" in headers:
                try:
                    identity = float(headers["X-Retake-Similarity"])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(identity) or not -1 <= identity <= 1:
                    continue
            seen.add(variation)
            rows.append((variation,quality))
        if not rows:
            return 0
        with self._lock:
            data = self._load()
            entry = data["banks"].setdefault(bank, {"variants": {}, "jobs": 0, "self_tests": 0})
            if not isinstance(entry.get("variants"), dict):
                entry["variants"] = {}
            for variation, (kind, value) in rows:
                slot = entry["variants"].setdefault(str(variation), {})
                stats = slot.get(kind)
                if not _valid_stats(stats):
                    stats = _empty_stats()
                stats["n"] += 1
                stats["sum"] += value
                stats["sq"] += value * value
                slot[kind] = stats
            entry["jobs"] = int(entry.get("jobs", 0)) + 1
            entry["self_tests"] = int(entry.get("self_tests", 0)) + (1 if self_test else 0)
            entry["updated"] = time.time()
            self._save(data)
        return len(rows)

    def _bank(self, bank: str) -> dict[str, Any]:
        with self._lock:
            entry = self._load()["banks"].get(bank)
        return entry if isinstance(entry, dict) and isinstance(entry.get("variants"), dict) else {"variants": {}}

    def schedule(self, bank: str, variations: Sequence[int]) -> tuple[list[int], str, int]:
        """(order, mode, samples). mode: learned-fidelity | learned-proxy | default."""
        order = list(variations)
        if not self.enabled or not bank:
            return order, "default", 0
        variants = self._bank(bank)["variants"]
        for kind, minimum in (("fidelity", MIN_FIDELITY_SAMPLES), ("proxy", MIN_PROXY_SAMPLES)):
            stats = {
                v: variants.get(str(v), {}).get(kind) for v in order
            }
            stats = {v: s for v, s in stats.items() if _valid_stats(s) and s["n"] > 0}
            total = sum(s["n"] for s in stats.values())
            if total < minimum:
                continue
            prior = sum(s["sum"] for s in stats.values()) / total
            bonus = EXPLORATION[kind]
            log_total = math.log(total + 1)

            def upper(v: int) -> float:
                s = stats.get(v)
                n = s["n"] if s else 0
                mean = s["sum"] / n if n else prior
                return mean + bonus * math.sqrt(log_total / (n + 1))

            ranked = sorted(order, key=lambda v: (-upper(v), order.index(v)))
            return ranked, "learned-" + kind, total
        return order, "default", 0

    def summary(self, bank: str | None = None) -> dict[str, Any]:
        """Numbers only, for the engine's /api/calibration endpoint."""
        with self._lock:
            data = self._load()
        banks = data["banks"] if bank is None else {bank: data["banks"].get(bank, {})}
        out: dict[str, Any] = {"enabled": self.enabled, "signature": self.signature, "banks": {}}
        for key, entry in banks.items():
            if not isinstance(entry, dict):
                continue
            rows = {}
            for variation, slot in (entry.get("variants") or {}).items():
                if not isinstance(slot, dict):
                    continue
                row = {}
                for kind in ("fidelity", "proxy"):
                    stats = slot.get(kind)
                    if _valid_stats(stats) and stats["n"] > 0:
                        row[kind] = {"n": stats["n"], "mean": round(stats["sum"] / stats["n"], 4)}
                if row:
                    rows[variation] = row
            out["banks"][key] = {
                "jobs": int(entry.get("jobs", 0) or 0), "self_tests": int(entry.get("self_tests", 0) or 0),
                "variants": rows,
            }
        return out

    def reset(self) -> None:
        with self._lock:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
