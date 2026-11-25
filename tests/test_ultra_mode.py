"""Ultra-mode scheduling and best-result integrity tests.

No GPU needed: evaluate the real HTTP orchestration with mocked FaceFusion.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tools import facefusion_local_server as srv
from tools.ultra_mode import (
    ULTRA_VARIATIONS, prefer_ultra_candidate, rim_from_headers, score_from_headers,
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
        self.assertTrue(prefer_ultra_candidate(score("0.803"), best), "measurable identity improvement must not be discarded")
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
            return f"photo-{variant}".encode(), results[variant % len(results)]

        positions = []
        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda step, cached=False: positions.append(step),
                min_duration_seconds=0
            )
        self.assertEqual(photo, b"photo-2")
        self.assertEqual(headers["X-Retake-Ultra"], "on")
        self.assertEqual(headers["X-Retake-Ultra-Rounds"], "24")
        self.assertEqual(headers["X-Retake-Ultra-Winner"], "3")
        self.assertEqual(positions, list(range(len(srv.JOB_STAGES) * len(ULTRA_VARIATIONS))))
        self.assertEqual(visits, [(b"original photo", index) for index in ULTRA_VARIATIONS])


    def test_near_ties_cannot_accumulate_identity_loss_across_rounds(self):
        def fake_swap(*args):
            index = args[3][6]["variation"]
            return f"photo-{index}".encode(), {
                **score(.900 - index * .0015),
                "X-Retake-Rim-Fraction": str(max(0.0, .9 - index * .04))
            }
        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_a, **_k: None, min_duration_seconds=0
            )
        self.assertGreaterEqual(score_from_headers(headers), .898)
        self.assertEqual(float(headers["X-Retake-Ultra-Peak-Identity"]), .9)
        self.assertEqual(headers["X-Retake-Ultra-Attempted"], "24")

    def test_invalid_scores_are_attempted_but_not_verified(self):
        def fake_swap(*args):
            index = args[3][6]["variation"]
            return b"verified" if index == 1 else b"invalid", score(.84 if index == 1 else "nan")
        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_a, **_k: None, min_duration_seconds=0
            )
        self.assertEqual(photo, b"verified")
        self.assertEqual(headers["X-Retake-Ultra-Rounds"], "1")
        self.assertEqual(headers["X-Retake-Ultra-Attempted"], "24")

    def test_diffusion_is_run_only_for_best_identity_finalist(self):
        calls = []
        finalist_calls = []
        realism_config = {"python": Path("gpu"), "models": Path("weights")}

        def fake_swap(*args):
            variation = args[3][6]["variation"]
            active = bool(args[5] and args[5].get("realism"))
            calls.append((variation, active))
            if active:
                return b"realism", score(0.85, realism=True)
            return f"unfiltered-{variation}".encode(), score(0.84 if variation == 3 else 0.74)

        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None,
                caches={"realism": realism_config},
                on_finalist=lambda: finalist_calls.append("gpu"),
                min_duration_seconds=0
            )
        self.assertEqual(calls[:24], [(variation, False) for variation in ULTRA_VARIATIONS])
        self.assertEqual(calls[24:], [(3, True)])
        self.assertEqual(photo, b"realism")
        self.assertEqual(finalist_calls, ["gpu"])
        self.assertEqual(headers["X-Retake-Ultra-Realism-Policy"], "finalist-only")

    def test_bad_finalist_diffusion_cannot_overwrite_sharp_identity_result(self):
        def fake_swap(*args):
            if args[5] and args[5].get("realism"):
                return b"fake-face", score(0.65, realism=True)
            return b"strong-identity", score(0.84)
        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None,
                caches={"realism": {"python": Path("gpu")}},
                min_duration_seconds=0
            )
        self.assertEqual(photo, b"strong-identity")
        self.assertEqual(headers["X-Retake-Ultra-Rounds"], "24")

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
                lambda *_args, **_kwargs: None, clock=clock,
                min_duration_seconds=0
            )
        self.assertEqual(photo, b"best")
        self.assertEqual(headers["X-Retake-Ultra-Rounds"], "4")
        self.assertEqual(headers["X-Retake-Ultra-Elapsed"], "500.0")
        self.assertFalse(should_continue_ultra(4, 500))

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
                lambda *_args, **_kwargs: None,
                min_duration_seconds=0
            )
        self.assertEqual(photo, b"good")
        self.assertEqual(attempts, list(ULTRA_VARIATIONS))
        self.assertEqual(headers["X-Retake-Ultra-Winner"], "1")

    def test_seven_minimum_is_real_elapsed_time_and_does_not_reprocess_pixels(self):
        now = [0.0]
        holds = []
        seen = []

        def clock():
            return now[0]

        def advance(seconds):
            now[0] += seconds

        def fake_swap(*args):
            seen.append(args[3][6]["variation"])
            # The first GPU round costs 360s, second costs 80s.
            now[0] += 360 if len(seen) == 1 else 80
            return b"validated", score(0.86)

        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None, clock=clock,
                sleeper=advance, on_hold=holds.append,
                min_duration_seconds=420
            )
        self.assertEqual(photo, b"validated")
        self.assertEqual(seen, [0, 1])
        self.assertEqual(headers["X-Retake-Ultra-Elapsed"], "440.0")
        self.assertEqual(holds[-1], 0.0)

    def test_finishing_all_variants_early_uses_visible_minimum_hold(self):
        now = [0.0]
        holds = []
        attempts = []

        def fake_swap(*args):
            now[0] += 2
            attempts.append(args[3][6]["variation"])
            return b"good", score(0.9)

        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None,
                clock=lambda: now[0],
                sleeper=lambda delay: now.__setitem__(0, now[0] + delay),
                on_hold=holds.append,
                min_duration_seconds=420
            )
        self.assertEqual(len(attempts), 24)
        self.assertEqual(photo, b"good")
        self.assertEqual(headers["X-Retake-Ultra-Elapsed"], "420.0")
        self.assertGreater(len(holds), 1)
        self.assertGreater(holds[0], 0)
        self.assertEqual(holds[-1], 0.0)

    def test_detail_score_decides_ties_without_overriding_identity(self):
        a = {**score(.830), "X-Retake-Detail-Score": ".44"}
        b = {**score(.829), "X-Retake-Detail-Score": ".88"}
        c = {**score(.821), "X-Retake-Detail-Score": ".95"}
        self.assertTrue(prefer_ultra_candidate(b, a))
        self.assertFalse(prefer_ultra_candidate(c, a))

    def test_rejects_unverifiable_or_invalid_residual_halo_scores(self):
        for invalid in ("", "nan", "inf", "-0.1", "1.1", "bogus"):
            self.assertIsNone(rim_from_headers({"X-Retake-Rim-Fraction": invalid}))
        self.assertIsNone(rim_from_headers({}))
        self.assertAlmostEqual(rim_from_headers({"X-Retake-Rim-Fraction": ".19"}), .19)

    def test_visible_black_outline_breaks_only_identity_near_ties(self):
        dark = {**score(.838), "X-Retake-Rim-Fraction": ".19", "X-Retake-Detail-Score": ".85"}
        clean = {**score(.837), "X-Retake-Rim-Fraction": ".04", "X-Retake-Detail-Score": ".70"}
        worse = {**score(.821), "X-Retake-Rim-Fraction": ".00"}
        self.assertTrue(prefer_ultra_candidate(clean, dark), "visible seam improvement wins a narrow ArcFace tie")
        self.assertFalse(prefer_ultra_candidate(dark, clean), "same identity tie must reject darker outline")
        self.assertFalse(prefer_ultra_candidate(worse, dark), "a cleaner face must not replace a different identity")
        self.assertTrue(prefer_ultra_candidate(dark, worse), "meaningful identity gain stays first priority")

    def test_missing_halo_score_never_counts_as_perfect(self):
        dark = {**score(.83), "X-Retake-Rim-Fraction": ".19"}
        missing = score(.83)
        invalid = {**score(.83), "X-Retake-Rim-Fraction": "nan"}
        self.assertFalse(prefer_ultra_candidate(missing, dark))
        self.assertFalse(prefer_ultra_candidate(invalid, dark))
        self.assertFalse(prefer_ultra_candidate(dark, missing))

    def test_small_halo_difference_defers_to_existing_detail_tie_break(self):
        old = {**score(.83), "X-Retake-Rim-Fraction": ".10", "X-Retake-Detail-Score": ".40"}
        new = {**score(.83), "X-Retake-Rim-Fraction": ".09", "X-Retake-Detail-Score": ".88"}
        self.assertTrue(prefer_ultra_candidate(new, old))

    def test_every_failed_round_preserves_original_error_kind(self):
        with patch.object(srv, "process_swap", side_effect=srv.NoOpSwapError("no face")):
            with self.assertRaises(srv.NoOpSwapError):
                srv.run_ultra_search(
                    Path("."), "1024x1024", False, fake_request(),
                    lambda *_args, **_kwargs: None
                )


if __name__ == "__main__":
    unittest.main()


class FluxOutcomeTest(unittest.TestCase):
    def test_rejected_finalist_keeps_original_and_reports_real_outcome(self):
        def fake_swap(*args):
            if args[5] and args[5].get("nim_editors"):
                return b"rejected", {**score(.61), "X-Retake-FLUX-Status":"accepted"}
            return b"verified", score(.87)
        with patch.object(srv, "process_swap", fake_swap):
            image, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_a, **_k: None, min_duration_seconds=0,
                caches={"nim_editors":{"retake-local/flux2-klein-4b":0}}
            )
        self.assertEqual(image,b"verified")
        self.assertEqual(headers["X-Retake-FLUX-Status"],"finalist-not-selected")
