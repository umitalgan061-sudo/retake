"""Privacy-scrubbed operational event log for the hidden Retake engine.

The background engine (Task Scheduler, ``pythonw``) has no console, and its
raw output can contain the pairing code, temporary photo paths and image
diagnostics, so it used to be discarded entirely. That also discarded the
lines that explain a weaker photo: a GPU memory error that lowered the pixel
boost, a failing realism worker, a skipped NVIDIA jury or a finishing step
that could not run.

This module keeps only those operational lines, after removing anything that
could identify a person or a secret:

* lines mentioning the pairing code, tokens, keys or authorization are
  dropped completely, never stored in a reworded form;
* file paths, URLs (except the host), e-mail addresses, data URIs, long
  opaque strings (tokens, digests, base64) and non-loopback IP addresses are
  replaced with placeholders;
* informational chatter and HTTP access lines are not kept at all.

Each kept line is one JSON object per line (``{"t", "cat", "msg"}``) in
``~/.retake/engine-events.jsonl``; the file rotates at a small size so the
log can never grow without bound. Images, embeddings and filenames are never
written. Standard library only.
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Iterable

EVENT_LOG_NAME = "engine-events.jsonl"
EVENT_LOG_MAX_BYTES = 256 * 1024
MAX_MESSAGE_CHARS = 240
# Identical lines inside this window are written once (tracebacks in loops).
DUPLICATE_WINDOW_SECONDS = 2.0
# Hard ceiling against a runaway loop filling the disk.
MAX_EVENTS_PER_MINUTE = 120

CATEGORIES = (
    "start", "stop", "result", "jury", "memory", "realism", "jury_skip",
    "identity", "fallback", "detail", "skipped", "error",
)
# Categories that explain a weaker or missing result.
PROBLEM_CATEGORIES = ("memory", "realism", "jury_skip", "identity", "fallback", "detail", "skipped", "error")

_DROP = re.compile(
    r"eşleştirme|eslestirme|kodu bilen|pairing|token|jeton|bearer|authorization|"
    r"x-retake-token|api[_ -]?key|nvapi-|password|parola|şifre|secret|cookie",
    re.IGNORECASE,
)
_SCRUBBERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=_-]+"), "<veri>"),
    (re.compile(r"https?://([A-Za-z0-9.-]+)(?::\d+)?[^\s\"'<>)]*"), r"<adres:\1>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<e-posta>"),
    (re.compile(r"\\\\[^\s\"'<>|]+"), "<yol>"),
    (re.compile(r"\b[A-Za-z]:[\\/][^\s\"'<>|]*"), "<yol>"),
    (re.compile(r"(?<![\w.<:])/(?:[^\s\"'<>/]+/)+[^\s\"'<>]*"), "<yol>"),
    (re.compile(r"(?<![\w<])~[\\/][^\s\"'<>|]*"), "<yol>"),
    (re.compile(r"[A-Za-z0-9+/_=-]{32,}"), lambda match: "<gizli>" if _opaque(match.group(0)) else match.group(0)),
    (re.compile(r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
)

_ACCESS = re.compile(r"^\[Retake FaceFusion\]")


def _opaque(text: str) -> bool:
    """Tokens, keys, digests, base64 - but not a readable lowercase model id.

    "nemotron-3-nano-omni-30b-a3b-reasoning" is words joined by hyphens;
    secrets.token_urlsafe tokens mix cases, digests are one unbroken run.
    """
    words = re.split(r"[-_.]", text)
    readable = text == text.lower() and len(words) >= 3 and all(0 < len(word) <= 14 for word in words)
    return not readable
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("start", re.compile(r"^Retake kişisel yüz profili servisi|^Fotoğraf gerçekçiliği AÇIK", re.IGNORECASE)),
    ("memory", re.compile(
        r"bellek hatası|out of memory|outofmemory|cuda_error_out_of_memory|"
        r"cudaerrormemoryallocation|failed to allocate|bad_alloc",
        re.IGNORECASE,
    )),
    ("result", re.compile(r"^\[Ultra\] Sonuç:")),
    ("jury_skip", re.compile(r"^\[NVIDIA jüri\].*(atlandı|devre dışı|hazırlanamadı|tanımıyor)", re.IGNORECASE)),
    ("jury", re.compile(r"^\[NVIDIA jüri\]")),
    ("realism", re.compile(
        r"^\[Gerçekçilik\].*(başarısız|çalıştırılamadı|zaman aşımı|timeout)|"
        r"gerçekçiliği kullanılamıyor|finalist gerçekçilik atlandı",
        re.IGNORECASE,
    )),
    ("identity", re.compile(
        r"\[(?:Retake )?ArcFace\].*(ölçülemedi|başarısız|tutarsız|kapatılamadı|error|hata)",
        re.IGNORECASE,
    )),
    # The default identity model could not run and the fallback model was used.
    ("fallback", re.compile(r"kullanılamadı \(.*\); \S+ ile devam ediliyor", re.IGNORECASE)),
    # The delivered photo (or every candidate) failed the regional detail check.
    ("detail", re.compile(r"bölgesel ayrıntı kontrolünü geç", re.IGNORECASE)),
    ("skipped", re.compile(
        r"atlandı|skipped|kullanılamadı|uygulanamadı|oluşturulamadı|doğrulanamadı|geçersiz",
        re.IGNORECASE,
    )),
    ("error", re.compile(
        r"traceback \(most recent call last\)|^[A-Za-z_][\w.]*(?:Error|Exception)\b|"
        r"başarısız|\bhata\b|\bfailed\b|\berror\b|tamamlanamadı",
        re.IGNORECASE,
    )),
)


def scrub(text: str) -> str | None:
    """Return a shareable form of ``text``, or ``None`` when it must not be kept."""
    if not isinstance(text, str):
        return None
    if _DROP.search(text):
        return None
    cleaned = text
    for pattern, replacement in _SCRUBBERS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > MAX_MESSAGE_CHARS:
        cleaned = cleaned[: MAX_MESSAGE_CHARS - 1].rstrip() + "…"
    return cleaned or None


def classify(line: str) -> str | None:
    """Operational category of a raw engine line, or ``None`` to discard it."""
    if not line or not line.strip():
        return None
    if _ACCESS.match(line):
        return None
    # Indented traceback context ("  File ...", source lines) carries paths
    # and code, not the cause; the exception line itself is kept.
    if line[:1].isspace() and not re.search(r"bellek hatası|out of memory", line, re.IGNORECASE):
        return None
    for category, pattern in _RULES:
        if pattern.search(line):
            return category
    return None


class EventLog:
    """Append-only, rotating JSON-lines event file. Thread-safe; never raises."""

    def __init__(self, path: Path | str, *, max_bytes: int = EVENT_LOG_MAX_BYTES,
                 clock=time.time) -> None:
        self.path = Path(path)
        self.max_bytes = max(4096, int(max_bytes))
        self.clock = clock
        self._lock = threading.Lock()
        self._recent: dict[tuple[str, str], float] = {}
        self._minute = 0
        self._minute_count = 0

    def record(self, category: str, message: str, **numbers: float | int | str | None) -> bool:
        """Store one event; ``message`` is scrubbed again. Returns whether written."""
        if category not in CATEGORIES:
            return False
        cleaned = scrub(message) if message else ""
        if cleaned is None:
            return False
        now = float(self.clock())
        with self._lock:
            key = (category, cleaned)
            last = self._recent.get(key)
            if last is not None and now - last < DUPLICATE_WINDOW_SECONDS:
                return False
            minute = int(now // 60)
            if minute != self._minute:
                self._minute, self._minute_count = minute, 0
            if self._minute_count >= MAX_EVENTS_PER_MINUTE:
                return False
            self._minute_count += 1
            self._recent[key] = now
            if len(self._recent) > 256:
                self._recent = {k: v for k, v in self._recent.items() if now - v < DUPLICATE_WINDOW_SECONDS}
            row: dict[str, object] = {"t": round(now, 1), "cat": category, "msg": cleaned}
            for name, value in numbers.items():
                if re.fullmatch(r"[a-z_]{1,24}", name) and isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[name] = value
                elif re.fullmatch(r"[a-z_]{1,24}", name) and isinstance(value, str):
                    safe = scrub(value)
                    if safe is not None:
                        row[name] = safe[:40]
            data = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    size = self.path.stat().st_size
                except FileNotFoundError:
                    size = 0
                if size and size + len(data) > self.max_bytes:
                    os.replace(self.path, self.path.with_name(self.path.name + ".1"))
                with open(self.path, "ab") as handle:
                    handle.write(data)
                return True
            except OSError:
                return False

    def record_line(self, line: str) -> bool:
        category = classify(line)
        if category is None:
            return False
        return self.record(category, line)


class EventLogSink(io.TextIOBase):
    """Text stream for ``sys.stdout``/``sys.stderr`` of the hidden engine.

    Complete lines are classified; only operational lines reach the event
    log, everything else is discarded exactly like the former ``devnull``
    sink. ``fileno()`` points at ``os.devnull`` for libraries that need a
    real descriptor.
    """

    def __init__(self, log: EventLog) -> None:
        super().__init__()
        self.log = log
        self._buffer = ""
        self._lock = threading.Lock()
        self._null = open(os.devnull, "w", encoding="utf-8")

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return "utf-8"

    @property
    def errors(self) -> str:  # type: ignore[override]
        return "replace"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._null.fileno()

    def write(self, text: str) -> int:  # type: ignore[override]
        if not isinstance(text, str):
            text = str(text)
        try:
            with self._lock:
                self._buffer += text
                if len(self._buffer) > 65536:
                    # A huge line without newline (binary dump): keep the tail.
                    self._buffer = self._buffer[-4096:]
                *lines, self._buffer = self._buffer.split("\n")
            for line in lines:
                self.log.record_line(line.rstrip("\r"))
        except Exception:
            pass
        return len(text)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        try:
            with self._lock:
                rest, self._buffer = self._buffer, ""
            if rest.strip():
                self.log.record_line(rest)
            self._null.close()
        except Exception:
            pass
        super().close()


def read_events(path: Path | str, limit: int = 400) -> list[dict]:
    """Newest-last events from the log and its rotated predecessor."""
    base = Path(path)
    rows: list[dict] = []
    for candidate in (base.with_name(base.name + ".1"), base):
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("cat") in CATEGORIES and isinstance(row.get("t"), (int, float)):
                message = scrub(str(row.get("msg", "")))
                if message is None:
                    continue
                row["msg"] = message
                rows.append(row)
    rows.sort(key=lambda row: row["t"])
    return rows[-max(1, int(limit)):]


def summarize_events(events: Iterable[dict], *, since: float | None = None) -> dict:
    """Counts per category and the latest messages (already scrubbed)."""
    counts = {category: 0 for category in CATEGORIES}
    last: dict[str, dict] = {}
    problems: list[dict] = []
    results: list[dict] = []
    starts = 0
    for row in events:
        if since is not None and row.get("t", 0) < since:
            continue
        category = row.get("cat")
        if category not in counts:
            continue
        counts[category] += 1
        last[category] = row
        if category in PROBLEM_CATEGORIES:
            problems.append(row)
        elif category == "result":
            results.append(row)
        elif category == "start":
            starts += 1
    return {
        "counts": counts,
        "problems": sum(counts[category] for category in PROBLEM_CATEGORIES),
        "last": last,
        "recent_problems": problems[-6:],
        "recent_results": results[-3:],
        "starts": starts,
    }
