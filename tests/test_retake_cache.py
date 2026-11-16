import tempfile
import unittest
from pathlib import Path

from tools.retake_cache import IdentityReportCache, PassCache, StageTimings, digest, references_digest


class DigestTest(unittest.TestCase):
    def test_parts_are_unambiguous_and_order_matters(self):
        self.assertNotEqual(digest(b"ab", b"c"), digest(b"a", b"bc"))
        self.assertNotEqual(references_digest([b"1", b"2"], "m"), references_digest([b"2", b"1"], "m"))
        self.assertNotEqual(references_digest([b"1"], "m1"), references_digest([b"1"], "m2"))


class IdentityReportCacheTest(unittest.TestCase):
    def test_report_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            report = {"ok": True, "median_similarity": 0.77, "outlier_indices": [1]}
            IdentityReportCache(Path(directory)).put("k", report)
            self.assertEqual(IdentityReportCache(Path(directory)).get("k"), report)
            self.assertIsNone(IdentityReportCache(Path(directory)).get("other"))

    def test_failures_and_conflicts_are_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = IdentityReportCache(Path(directory))
            cache.put("bad", {"ok": False})
            cache.put("conflict", {"ok": True, "hard_conflict": True})
            self.assertIsNone(cache.get("bad"))
            self.assertIsNone(cache.get("conflict"))

    def test_only_recent_reports_are_kept_and_corrupt_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = IdentityReportCache(Path(directory), max_entries=2)
            for key in ("a", "b", "c"):
                cache.put(key, {"ok": True, "key": key})
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)
            (Path(directory) / "broken.json").write_text("{not json")
            self.assertIsNone(cache.get("broken"))


class PassCacheTest(unittest.TestCase):
    def test_lru_keeps_recent_outputs(self):
        cache = PassCache(max_entries=2)
        cache.put("a", b"A", "1024x1024")
        cache.put("b", b"B", "768x768")
        self.assertEqual(cache.get("a"), (b"A", "1024x1024"))
        cache.put("c", b"C", "1024x1024")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(len(cache), 2)

    def test_key_depends_on_every_input(self):
        base = PassCache.key(b"t", [b"s"], "alphaface_256", 1.0, "1024x1024")
        self.assertEqual(base, PassCache.key(b"t", [b"s"], "alphaface_256", 1.0, "1024x1024"))
        for changed in (
            PassCache.key(b"T", [b"s"], "alphaface_256", 1.0, "1024x1024"),
            PassCache.key(b"t", [b"S"], "alphaface_256", 1.0, "1024x1024"),
            PassCache.key(b"t", [b"s"], "uniface_256", 1.0, "1024x1024"),
            PassCache.key(b"t", [b"s"], "alphaface_256", 0.94, "1024x1024"),
            PassCache.key(b"t", [b"s"], "alphaface_256", 1.0, "768x768"),
        ):
            self.assertNotEqual(base, changed)


class StageTimingsTest(unittest.TestCase):
    def test_estimate_needs_every_stage_and_smooths_new_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.json"
            timings = StageTimings(path)
            timings.record("a", 40)
            self.assertIsNone(timings.remaining(["a", "b"], 0, 0))
            timings.record("b", 100)
            self.assertEqual(timings.remaining(["a", "b"], 0, 10), 130)
            self.assertEqual(timings.remaining(["a", "b"], 1, 200), 0)
            timings.record("b", 200)
            self.assertAlmostEqual(StageTimings(path).seconds["b"], 135)
            timings.record("c", 0)
            self.assertNotIn("c", timings.seconds)


if __name__ == "__main__":
    unittest.main()
