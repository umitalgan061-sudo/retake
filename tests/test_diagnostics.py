"""v26: where the time went, and finest features in Ultra ties."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tools import facefusion_local_server as srv
from tools.retake_cache import PassCache
from tools.ultra_mode import (
    ULTRA_FINE_GAIN, fine_preference, fine_quality_from_headers, format_ultra_timing, prefer_ultra_candidate
)


def candidate(score, fine=None, sufficient=False, detail=None):
    headers = {"X-Retake-Similarity": str(score), "X-Retake-Detail-Gate": "passed"}
    if fine is not None:
        headers["X-Retake-Fine-Quality"] = str(fine)
    if detail is not None:
        headers["X-Retake-Detail-Score"] = str(detail)
    if sufficient:
        headers["X-Retake-Identity-Photo-Level"] = "0.70"
    return headers


class TimingFormatTest(unittest.TestCase):
    def test_timing_header_lists_counts_and_mean_seconds(self):
        value = format_ultra_timing(
            {"search": [70.2, 74.0, 72.0], "refine": [30.0, 32.4], "realism": [], "extra": [65.0]},
            judge_seconds=40.4, hold_seconds=0.0, total_seconds=2210.6
        )
        self.assertEqual(value, "search=3x72;refine=2x31;extra=1x65;judge=40;hold=0;total=2211")
        self.assertEqual(format_ultra_timing({}, 0, 0, 0), "search=0x0;judge=0;hold=0;total=0")
        self.assertIn("search=1x5", format_ultra_timing({"search": [5.0, float("nan"), -1.0]}, 0, float("inf"), 5))


class FineTieTest(unittest.TestCase):
    def test_parse_is_strict(self):
        self.assertEqual(fine_quality_from_headers({"X-Retake-Fine-Quality": "0.9123"}), 0.9123)
        for bad in ("", "nan", "1.5", "-0.1", "x"):
            self.assertIsNone(fine_quality_from_headers({"X-Retake-Fine-Quality": bad}))
        self.assertIsNone(fine_preference(candidate(.8, .90), candidate(.8)))
        self.assertIsNone(fine_preference(candidate(.8, .90), candidate(.8, .90 - ULTRA_FINE_GAIN / 2)))

    def test_more_natural_features_win_a_sufficient_tie(self):
        natural = candidate(.88, fine=.93, sufficient=True, detail=.80)
        waxy = candidate(.90, fine=.78, sufficient=True, detail=.82)
        with patch("tools.ultra_mode.identity_sufficiency", return_value=.80):
            self.assertTrue(prefer_ultra_candidate(natural, waxy))
            self.assertFalse(prefer_ultra_candidate(waxy, natural))

    def test_identity_still_comes_first_below_sufficiency(self):
        stronger = candidate(.84, fine=.70)
        natural = candidate(.80, fine=.95)
        self.assertFalse(prefer_ultra_candidate(natural, stronger), "no beauty for identity")
        tie = candidate(.8395, fine=.95)
        self.assertTrue(prefer_ultra_candidate(tie, stronger), "inside the tie band the eyes decide")
        self.assertFalse(prefer_ultra_candidate(candidate(.8399, fine=.70), candidate(.8398, fine=.95)))


class EngineStatsTest(unittest.TestCase):
    def tearDown(self):
        srv.IDENTITY_SCORER = None

    def test_stats_are_this_jobs_difference(self):
        cache = PassCache()
        cache.put("a", b"x", "1024x1024")
        cache.get("a")
        start = cache.describe()
        cache.get("a")
        cache.get("a")
        cache.get("missing")

        class Scorer:
            stats = {"calls": 10, "seconds": 5.0, "starts": 1, "failures": 0}

        scorer = Scorer()
        srv.IDENTITY_SCORER = scorer
        scorer_start = dict(scorer.stats)
        scorer.stats = {"calls": 14, "seconds": 6.0, "starts": 2, "failures": 1}
        self.assertEqual(
            srv.engine_stats(cache, start, scorer_start),
            "cache=2/1;arcface=4x0.25;arcface_starts=1;arcface_failures=1"
        )
        srv.IDENTITY_SCORER = None
        self.assertEqual(srv.engine_stats(None, None, None), "arcface=one-shot")

    def test_ultra_search_reports_timing_and_stats(self):
        now = [0.0]

        def fake_swap(*args):
            now[0] += 10.0
            return b"x", {"X-Retake-Similarity": "0.84", "X-Retake-Detail-Gate": "passed"}

        parsed = (b"t", [b"r"] * 25, ".png", [".png"] * 25, b"a", ".png",
                  {"variation": 0, "referenceWeights": [1] * 25, "faceBounds": {"x": 0, "y": 0, "width": 100, "height": 100}})
        with patch.object(srv, "process_swap", fake_swap):
            _, headers = srv.run_ultra_search(Path("."), "1024x1024", False, parsed, lambda *_a, **_k: None,
                                              clock=lambda: now[0], sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
                                              min_duration_seconds=0)
        self.assertTrue(headers["X-Retake-Ultra-Timing"].startswith("search=24x10;"))
        self.assertIn("total=240", headers["X-Retake-Ultra-Timing"])
        self.assertEqual(headers["X-Retake-Engine-Stats"], "arcface=one-shot")


if __name__ == "__main__":
    unittest.main()
