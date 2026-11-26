import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.retake_event_log import (
    EVENT_LOG_NAME, EventLog, EventLogSink, classify, read_events, scrub, summarize_events,
)
from tools import retake_background


class ScrubTest(unittest.TestCase):
    def test_secret_lines_are_dropped_not_reworded(self):
        for line in (
            "    EŞLEŞTİRME KODU: q8Zr2vX0b9yTq8Zr2vX0b9yTq8Zr2vX0b9yT",
            "Telefon/PWA eşleştirmesi AÇIK. şu kodu gir:",
            "Kodu bilen herkes bu motoru kullanabilir; paylaşma.",
            "Authorization: Bearer abc",
            "NVIDIA_API_KEY=nvapi-123",
            "proje jetonu geçersiz",
        ):
            self.assertIsNone(scrub(line), line)

    def test_paths_urls_and_opaque_strings_are_replaced(self):
        line = (r"[Gerçekçilik] Başarısız (kod 1): C:\Users\umit\AppData\Local\Temp\retake-x\target.png "
                "/tmp/retake/a b.png https://hafize-free.onrender.com/api/x?y=1 "
                "me@example.com 10.2.3.4 127.0.0.1 data:image/png;base64,iVBORw0KGgo= "
                + "f" * 64)
        cleaned = scrub(line)
        self.assertNotIn("umit", cleaned)
        self.assertNotIn("target.png", cleaned)
        self.assertNotIn("/tmp/retake", cleaned)
        self.assertNotIn("/api/x", cleaned)
        self.assertNotIn("example.com", cleaned)
        self.assertNotIn("10.2.3.4", cleaned)
        self.assertNotIn("iVBOR", cleaned)
        self.assertNotIn("f" * 40, cleaned)
        self.assertIn("<adres:hafize-free.onrender.com>", cleaned)
        self.assertIn("127.0.0.1", cleaned)
        self.assertIn("<yol>", cleaned)

    def test_ratios_model_names_and_numbers_survive(self):
        cleaned = scrub("[NVIDIA jüri] 3 finalist; karar=B, oy=6, uzlaşı=0.83; sonuç=agreed (3/4).")
        self.assertIn("uzlaşı=0.83", cleaned)
        self.assertIn("(3/4)", cleaned)
        self.assertEqual(scrub("hyperswap_1a_256: 1024x1024 bellek hatası; 768x768 ayarına düşülüyor."),
                         "hyperswap_1a_256: 1024x1024 bellek hatası; 768x768 ayarına düşülüyor.")

    def test_long_lines_are_capped(self):
        self.assertLessEqual(len(scrub("[Ultra] Aday atlandı: " + "kelime " * 200)), 240)


class ClassifyTest(unittest.TestCase):
    def test_operational_categories(self):
        cases = {
            "hyperswap_1a_256: 1024x1024 bellek hatası; 768x768 ayarına düşülüyor.": "memory",
            "[Gerçekçilik] Başarısız (kod 3): x": "realism",
            "Fotoğraf gerçekçiliği kullanılamıyor: ortam veya modeller eksik.": "realism",
            "[Ultra] Finalist gerçekçilik atlandı: RuntimeError": "realism",
            "[NVIDIA jüri] Atlandı, yerel seçim korundu: HTTP 502": "jury_skip",
            "[NVIDIA jüri] 3 finalist; karar=B, oy=6, uzlaşı=0.83; sonuç=agreed.": "jury",
            "[ArcFace] Benzerlik ölçülemedi: timeout": "identity",
            "[Retake] İnce bölge iyileştirmesi atlandı: ValueError": "skipped",
            "[Ultra] Sonuç: 19 doğrulanmış tur, kazanan 7. tur; benzerlik=0.712; süre 1510 sn.": "result",
            "RuntimeError: hyperswap işlemi başarısız (1024x1024).": "error",
            "Retake kişisel yüz profili servisi: http://127.0.0.1:8765 (profile=x)": "start",
            "hyperswap_1b_256 kullanılamadı (RuntimeError: x); alphaface_256 ile devam ediliyor.": "fallback",
            "Uyarı: son fotoğraf bölgesel ayrıntı kontrolünü geçemedi (eyes).": "detail",
            "Bölgesel ayrıntı kontrolünü geçen aday yok (3 aday).": "detail",
            "Gerçeklik ölçümü atlandı: ValueError": "skipped",
            "NVIDIA görsel jürisi kapalı: NVIDIA_API_KEY (ortam ya da Retake klasöründeki .env) yok.": None,
        }
        for line, expected in cases.items():
            self.assertEqual(classify(line), expected, line)

    def test_chatter_access_logs_and_traceback_context_are_discarded(self):
        for line in (
            "Kimlik profili geçişi tamamlandı (1024x1024).",
            '[Retake FaceFusion] "GET /health HTTP/1.1" 200 -',
            '  File "C:\\AI\\retake\\tools\\x.py", line 3, in f',
            "",
            "   ",
        ):
            self.assertIsNone(classify(line), line)


class EventLogTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.path = self.folder / EVENT_LOG_NAME
        self.now = [1000.0]
        self.log = EventLog(self.path, clock=lambda: self.now[0])

    def test_sink_keeps_only_operational_scrubbed_lines(self):
        sink = EventLogSink(self.log)
        print("Kimlik profili geçişi tamamlandı (1024x1024).", file=sink)
        print("    EŞLEŞTİRME KODU: q8Zr2vX0b9yTq8Zr2vX0b9yTq8Zr2vX0b9yT", file=sink)
        sink.write(r"[Gerçekçilik] Başarısız (kod 1): C:\Users\umit\x.png")
        sink.write("\nhyperswap_1a_256: 1024x1024 bellek hatası; 768x768 ayarına düşülüyor.\n")
        sink.close()
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn("umit", raw)
        self.assertNotIn("q8Zr2", raw)
        self.assertNotIn("Kimlik profili", raw)
        events = read_events(self.path)
        self.assertEqual([row["cat"] for row in events], ["realism", "memory"])

    def test_sink_is_a_usable_text_stream(self):
        sink = EventLogSink(self.log)
        self.assertTrue(sink.writable())
        self.assertFalse(sink.isatty())
        self.assertIsInstance(sink.fileno(), int)
        self.assertEqual(sink.encoding, "utf-8")
        self.assertEqual(sink.write("abc"), 3)
        sink.flush()
        sink.close()

    def test_duplicates_are_collapsed_and_rate_limited(self):
        for _ in range(5):
            self.log.record_line("[Ultra] Aday atlandı: ValueError")
        self.assertEqual(len(read_events(self.path)), 1)
        self.now[0] += 3
        self.assertTrue(self.log.record_line("[Ultra] Aday atlandı: ValueError"))
        for index in range(300):
            self.log.record("skipped", f"[Ultra] Aday atlandı: tur {index}")
        self.assertLessEqual(len(read_events(self.path, limit=1000)), 121)

    def test_rotation_bounds_disk_usage(self):
        log = EventLog(self.path, max_bytes=4096, clock=lambda: self.now[0])
        for index in range(400):
            self.now[0] += 61  # new minute each time: no rate limit
            log.record("skipped", f"[Ultra] Aday atlandı: tur {index} " + "x" * 40)
        rotated = self.path.with_name(self.path.name + ".1")
        self.assertTrue(rotated.is_file())
        self.assertLessEqual(self.path.stat().st_size, 4096)
        self.assertLessEqual(rotated.stat().st_size, 4096)
        events = read_events(self.path, limit=1000)
        self.assertTrue(events)
        self.assertEqual(events[-1]["msg"].split(" ")[4], "399")

    def test_unknown_category_and_unsafe_extras_are_refused(self):
        self.assertFalse(self.log.record("photo", "x"))
        self.log.record("start", "Gizli motor başlatıldı", revision="r8", boost="1024x1024",
                        **{"BadKey": 1, "code": True})
        row = read_events(self.path)[0]
        self.assertEqual(row["revision"], "r8")
        self.assertNotIn("BadKey", row)
        self.assertNotIn("code", row)

    def test_read_events_rescrubs_and_ignores_garbage(self):
        self.path.write_text(
            "not json\n"
            + json.dumps({"t": 1, "cat": "error", "msg": r"oops C:\Users\umit\a.png"}) + "\n"
            + json.dumps({"t": 2, "cat": "error", "msg": "jeton sızdı"}) + "\n"
            + json.dumps({"t": 3, "cat": "photo", "msg": "x"}) + "\n",
            encoding="utf-8",
        )
        events = read_events(self.path)
        self.assertEqual(len(events), 1)
        self.assertNotIn("umit", events[0]["msg"])

    def test_concurrent_writers_do_not_corrupt_lines(self):
        def worker(tag):
            for index in range(40):
                self.log.record("skipped", f"[Ultra] Aday atlandı: {tag}-{index}")
        threads = [threading.Thread(target=worker, args=(tag,)) for tag in "abc"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            json.loads(line)

    def test_summary_counts_and_recent_items(self):
        for line in (
            "[Ultra] Sonuç: 19 doğrulanmış tur, kazanan 7. tur; benzerlik=0.712; süre 1510 sn.",
            "hyperswap_1a_256: 1024x1024 bellek hatası; 768x768 ayarına düşülüyor.",
            "[NVIDIA jüri] Atlandı, yerel seçim korundu: HTTP 502",
        ):
            self.now[0] += 5
            self.log.record_line(line)
        summary = summarize_events(read_events(self.path))
        self.assertEqual(summary["counts"]["memory"], 1)
        self.assertEqual(summary["counts"]["jury_skip"], 1)
        self.assertEqual(summary["problems"], 2)
        self.assertEqual(len(summary["recent_results"]), 1)
        self.assertEqual(summarize_events(read_events(self.path), since=self.now[0] + 1)["problems"], 0)


class BackgroundRunnerTest(unittest.TestCase):
    def test_event_log_can_be_disabled(self):
        folder = Path(tempfile.mkdtemp())
        with patch.dict("os.environ", {"RETAKE_EVENT_LOG": "off"}):
            self.assertIsNone(retake_background.open_event_log(folder / "background-config.json"))
        with patch.dict("os.environ", {"RETAKE_EVENT_LOG": ""}):
            log = retake_background.open_event_log(folder / "background-config.json")
        self.assertEqual(log.path, folder / EVENT_LOG_NAME)

    def test_runner_failure_records_only_exception_type(self):
        folder = Path(tempfile.mkdtemp())
        config = folder / "background-config.json"
        config.write_text(json.dumps({"facefusion_dir": str(folder / "missing-user-dir")}), encoding="utf-8")
        with patch.object(sys, "argv", ["retake_background.py", "--config", str(config)]), \
                patch.dict("os.environ", {"RETAKE_EVENT_LOG": ""}):
            self.assertEqual(retake_background.main(), 1)
        events = read_events(folder / EVENT_LOG_NAME)
        self.assertEqual(events[-1]["cat"], "error")
        self.assertIn("FileNotFoundError", events[-1]["msg"])
        self.assertNotIn("missing-user-dir", events[-1]["msg"])
        status = json.loads((folder / "background-status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "failed")

    def test_runner_persists_scrubbed_operational_lines_from_the_engine(self):
        folder = Path(tempfile.mkdtemp())
        facefusion = folder / "FaceFusion"
        facefusion.mkdir()
        (facefusion / "facefusion.py").write_text("", encoding="utf-8")
        config = folder / "background-config.json"
        config.write_text(json.dumps({
            "facefusion_dir": str(facefusion), "python": sys.executable,
            "activated_path": "", "realism_dir": str(folder / "none"),
        }), encoding="utf-8")
        seen = {}

        def fake_main():
            seen["stdout"] = sys.stdout
            print("Telefon/PWA eşleştirmesi AÇIK. şu kodu gir:")
            print("    EŞLEŞTİRME KODU: q8Zr2vX0b9yTq8Zr2vX0b9yTq8Zr2vX0b9yT")
            print("Kimlik profili geçişi tamamlandı (1024x1024).")
            print("hyperswap_1a_256: 1024x1024 bellek hatası; 768x768 ayarına düşülüyor.")
            print(r"[Gerçekçilik] Başarısız (kod 1): C:\Users\umit\Temp\a.png", file=sys.stderr)
            return 0

        engine = type(sys)("facefusion_local_server")
        engine.main = fake_main
        engine.QUALITY_PIPELINE_REVISION = "r9"
        original = sys.stdout
        with patch.object(sys, "argv", ["retake_background.py", "--config", str(config)]), \
                patch.dict("os.environ", {"RETAKE_EVENT_LOG": ""}), \
                patch.dict(sys.modules, {"facefusion_local_server": engine}), \
                patch.object(retake_background, "port_open", return_value=False), \
                patch.object(sys, "executable", sys.executable):
            self.assertEqual(retake_background.main(), 1)  # code 0 -> scheduler restarts
        self.assertIs(sys.stdout, original)
        self.assertIsInstance(seen["stdout"], EventLogSink)
        raw = (folder / EVENT_LOG_NAME).read_text(encoding="utf-8")
        self.assertNotIn("q8Zr2", raw)
        self.assertNotIn("umit", raw)
        events = read_events(folder / EVENT_LOG_NAME)
        self.assertEqual([row["cat"] for row in events], ["start", "memory", "realism", "stop"])
        self.assertEqual(events[0]["revision"], "r9")


if __name__ == "__main__":
    unittest.main()
