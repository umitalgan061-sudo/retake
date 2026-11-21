"""Ultra-mode scheduling and best-result integrity tests.

No GPU needed: evaluate the real HTTP orchestration with mocked FaceFusion.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tools import facefusion_local_server as srv
from tools.ultra_mode import (
    ULTRA_VARIATIONS, prefer_ultra_candidate, score_from_headers,
    should_continue_ultra, with_variation
)


def fake_request():
    return (
        b"original photo", [b"reference"] * 25, ".png", [".png"] * 25,
        b"original appearance", ".png", {
            "variation": 3, "referenceWeights": [1] * 25,
            "faceBounds": {"x": 10, "y": 15, "width": 200, "height": 200}
        }
    )


def score(score_value, realism=False, texture="average"):
    return {
        "X-Retake-Similarity": str(score_value),
        "X-Retake-Realism": "on" if realism else "off",
        "X-Retake-Texture-Source": texture,
        "X-Retake-Compositing": "preblended",
    }


class UltraSelectionTest(unittest.TestCase):
    def test_first_pass_is_always_variation_zero_on_untouched_original(self):
        parsed = fake_request()
        variant = with_variation(parsed, 0)
        self.assertEqual(variant[0], b"original photo")
        self.assertEqual(variant[1], parsed[1])
        self.assertEqual(variant[6]["variation"], 0)
        self.assertEqual(parsed[6]["variation"], 3)

    def test_invalid_cosine_never_displaces_verified_result(self):
        best = score("0.802")
        for invalid in ["nan", "inf", "1.2", "", "-1.5"]:
            self.assertIsNone(score_from_headers(score(invalid)))
            self.assertFalse(prefer_ultra_candidate(score(invalid), best))

    def test_favors_significant_identity_gain_and_natural_tiebreaks(self):
        best = score("0.800")
        self.assertTrue(prefer_ultra_candidate(best, None))
        self.assertTrue(prefer_ultra_candidate(score("0.811"), best))
        self.assertFalse(prefer_ultra_candidate(score("0.803"), best))
        self.assertTrue(prefer_ultra_candidate(score("0.799", realism=True), best))
        self.assertFalse(prefer_ultra_candidate(score("0.790", realism=True), best))
        self.assertFalse(prefer_ultra_candidate(score("0.800"), score("0.801", realism=True)))

    def test_four_rounds_pick_best_verified_face_not_necessarily_last(self):
        visits = []
        results = [score(0.68), score(0.72), score(0.81, realism=True), score(0.76)]

        def fake_swap(*args):
            parsed, progress = args[3], args[4]
            variant = parsed[6]["variation"]
            visits.append((parsed[0], variant))
            for i in range(len(srv.JOB_STAGES)):
                progress(i)
            return f"photo-{variant}".encode(), results[variant]

        positions = []
        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda step, cached=False: positions.append(step),
            )
        self.assertEqual(photo, b"photo-2")
        self.assertEqual(headers["X-Retake-Ultra"], "on")
        self.assertEqual(headers["X-Retake-Ultra-Rounds"], "4")
        self.assertEqual(headers["X-Retake-Ultra-Winner"], "3")
        self.assertEqual(positions, list(range(len(srv.JOB_STAGES) * 4)))
        self.assertEqual(visits, [(b"original photo", index) for index in ULTRA_VARIATIONS])

    def test_budget_stops_extra_rounds_without_sleep_or_fake_wait(self):
        now = [0.0]

        def clock():
            return now[0]

        def fake_swap(*args):
            now[0] += 125.0
            return b"best", score(0.83)

        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None, clock=clock
            )
        self.assertEqual(photo, b"best")
        self.assertEqual(headers["X-Retake-Ultra-Rounds"], "2")
        self.assertEqual(headers["X-Retake-Ultra-Elapsed"], "250.0")
        self.assertFalse(should_continue_ultra(2, 250))

    def test_failed_later_round_preserves_best_verified_result(self):
        attempts = []

        def fake_swap(*args):
            variation = args[3][6]["variation"]
            attempts.append(variation)
            if variation == 0:
                return b"good", score(0.85)
            raise srv.NoOpSwapError("bu varyasyonda yüz değişmedi")

        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None
            )
        self.assertEqual(photo, b"good")
        self.assertEqual(attempts, [0, 1, 2, 3])
        self.assertEqual(headers["X-Retake-Ultra-Winner"], "1")

    def test_every_failed_round_preserves_original_error_kind(self):
        with patch.object(srv, "process_swap", side_effect=srv.NoOpSwapError("no face")):
            with self.assertRaises(srv.NoOpSwapError):
                srv.run_ultra_search(
                    Path("."), "1024x1024", False, fake_request(),
                    lambda *_args, **_kwargs: None
                )


if __name__ == "__main__":
    unittest.main()
