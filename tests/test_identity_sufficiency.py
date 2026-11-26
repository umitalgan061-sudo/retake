"""Identity sufficiency: above the person's own photo level, realism decides.

Synthetic headers and embeddings only; no GPU, no personal photos.
"""
from __future__ import annotations

import math
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import facefusion_local_server as srv
from tools.identity_consistency import identity_centroid, photo_level_similarities, photo_level_summary, similarity_to_centroid
from tools.ultra_mode import (
    identity_acceptable, identity_sufficiency, is_identity_sufficient, needs_extra_finalist, prefer_ultra_candidate
)

LEVEL = "0.8600"  # typical real photo of this person vs the centroid


def head(score, reinforced="off", detail=None, rim=None, level=LEVEL, before=None):
    headers = {
        "X-Retake-Similarity": f"{score:.4f}",
        "X-Retake-Identity-Reinforced": reinforced,
        "X-Retake-Detail-Gate": "passed",
        "X-Retake-Identity-Median-Similarity": "0.7700",
    }
    if level is not None:
        headers["X-Retake-Identity-Photo-Level"] = level
    if detail is not None:
        headers["X-Retake-Detail-Score"] = f"{detail:.4f}"
    if rim is not None:
        headers["X-Retake-Rim-Fraction"] = f"{rim:.5f}"
    if before is not None:
        headers["X-Retake-Similarity-Before"] = f"{before:.4f}"
    return headers


class PhotoLevelTest(unittest.TestCase):
    def test_leave_one_out_is_below_in_sample_similarity_and_skips_outliers(self):
        rng = random.Random(7)
        person = [rng.gauss(0, 1) for _ in range(64)]
        photos = [[p + rng.gauss(0, 0.9) for p in person] for _ in range(25)]
        stranger = [rng.gauss(0, 1) for _ in range(64)]
        embeddings = photos[:24] + [stranger]
        loo = photo_level_similarities(embeddings, exclude=[24])
        self.assertEqual(len(loo), 24)
        centroid = identity_centroid(embeddings, [24])
        in_sample = sorted(similarity_to_centroid(vector, centroid) for vector in photos[:24])
        # A photo never votes for itself: the honest level is lower.
        self.assertLess(sorted(loo)[12], in_sample[12])
        summary = photo_level_summary(loo)
        self.assertLessEqual(summary["photo_level_low"], summary["photo_level_similarity"])
        self.assertTrue(all(-1 <= value <= 1 and math.isfinite(value) for value in loo))
        self.assertEqual(photo_level_similarities(embeddings[:2]), [])
        self.assertEqual(photo_level_summary([0.8, 0.9]), {})


class PreferenceTest(unittest.TestCase):
    def test_sufficiency_threshold_and_validation(self):
        self.assertAlmostEqual(identity_sufficiency(head(.9)), 0.84)
        self.assertIsNone(identity_sufficiency(head(.9, level=None)))
        self.assertIsNone(identity_sufficiency(head(.9, level="1.5")))
        self.assertIsNone(identity_sufficiency(head(.9, level="nan")))
        self.assertTrue(is_identity_sufficient(head(.85)))
        self.assertFalse(is_identity_sufficient(head(.83)))

    def test_sharp_256px_render_beats_higher_cosine_128px_reinforcement(self):
        sharp = head(.86, reinforced="off", detail=.92)
        gamed = head(.91, reinforced="on", detail=.80)
        self.assertTrue(prefer_ultra_candidate(sharp, gamed))
        self.assertFalse(prefer_ultra_candidate(gamed, sharp))

    def test_among_sufficient_renders_detail_then_rim_decide_before_identity(self):
        self.assertTrue(prefer_ultra_candidate(head(.85, detail=.95), head(.90, detail=.85)))
        self.assertTrue(prefer_ultra_candidate(head(.85, rim=.01), head(.90, rim=.08)))
        # Equal realism evidence: identity breaks the tie.
        self.assertTrue(prefer_ultra_candidate(head(.90, detail=.90), head(.86, detail=.90)))

    def test_below_the_photo_level_identity_still_comes_first(self):
        self.assertTrue(prefer_ultra_candidate(head(.80, reinforced="on", detail=.7), head(.75, detail=.95)))
        self.assertFalse(prefer_ultra_candidate(head(.75, detail=.95), head(.80, reinforced="on", detail=.7)))
        # Reaching the level always beats staying below it.
        self.assertTrue(prefer_ultra_candidate(head(.85, reinforced="on"), head(.80)))
        # Without a measured level the old identity-first policy is unchanged.
        self.assertTrue(prefer_ultra_candidate(head(.91, reinforced="on", level=None), head(.86, level=None)))

    def test_no_extra_finalist_search_once_sufficient(self):
        self.assertFalse(needs_extra_finalist(head(.86, before=.85)))
        self.assertTrue(identity_acceptable(head(.85), peak_identity=.93))
        self.assertFalse(identity_acceptable(head(.80), peak_identity=.93))


class UltraLoopTest(unittest.TestCase):
    def run_search(self, rounds, judge=None):
        def fake_swap(*args):
            variation = args[3][6]["variation"]
            if variation not in rounds:
                raise RuntimeError("no more rounds")
            headers = dict(rounds[variation])
            return f"photo-{variation}".encode(), headers

        request = (b"t", [b"r"] * 25, ".png", [".png"] * 25, b"a", ".png",
                   {"variation": 0, "referenceWeights": [1] * 25, "nvidiaJudge": judge is not None,
                    "faceBounds": {"x": 10, "y": 10, "width": 100, "height": 100}})
        with patch.object(srv, "process_swap", fake_swap):
            return srv.run_ultra_search(Path("."), "1024x1024", False, request, lambda *_a, **_k: None,
                                        {"nvidia_judge": judge}, min_duration_seconds=0)

    def test_photo_level_result_is_not_dragged_back_to_a_reinforced_peak(self):
        photo, headers = self.run_search({
            0: head(.91, reinforced="on", detail=.78),
            1: head(.86, reinforced="off", detail=.93),
            2: head(.93, reinforced="on", detail=.75),
            3: head(.70, reinforced="off", detail=.99),
        })
        self.assertEqual(photo, b"photo-1")
        self.assertEqual(headers["X-Retake-Identity-Sufficient"], "yes")
        self.assertEqual(headers["X-Retake-Identity-Reinforced"], "off")

    def test_without_photo_level_the_strongest_identity_still_wins(self):
        photo, headers = self.run_search({
            0: head(.91, reinforced="on", detail=.78, level=None),
            1: head(.86, reinforced="off", detail=.93, level=None),
        })
        self.assertEqual(photo, b"photo-0")
        self.assertEqual(headers["X-Retake-Identity-Sufficient"], "unknown")

    def test_insufficient_best_reports_no(self):
        photo, headers = self.run_search({0: head(.70), 1: head(.72)})
        self.assertEqual(photo, b"photo-1")
        self.assertEqual(headers["X-Retake-Identity-Sufficient"], "no")


class InsideRenderTest(unittest.TestCase):
    def test_reinforcement_is_skipped_once_the_256px_pass_reaches_the_photo_level(self):
        from tools.ultra_quality import reinforcement_wanted
        from tools.ultra_mode import sufficiency_from_level
        threshold = sufficiency_from_level(0.86)
        self.assertAlmostEqual(threshold, 0.84)
        self.assertEqual(reinforcement_wanted(0.85, 0.91, threshold), (False, "identity-sufficient"))
        self.assertEqual(reinforcement_wanted(0.78, 0.83, threshold), (True, "gain"))
        self.assertEqual(reinforcement_wanted(0.78, 0.785, threshold), (False, "no-identity-gain"))
        self.assertEqual(reinforcement_wanted(None, 0.9, threshold), (False, "unscored"))
        # Without a measured level the old ArcFace-gain rule applies.
        self.assertEqual(reinforcement_wanted(0.85, 0.91, None), (True, "gain"))
        for bad in (None, True, "x", float("nan"), 0.2, 1.2):
            self.assertIsNone(sufficiency_from_level(bad))

    def test_sufficient_candidate_with_more_real_detail_wins_inside_a_render(self):
        from tools.ultra_quality import select_sufficient_realism
        identity = [0.90, 0.86, 0.80, None]
        detail = [0.78, 0.93, 0.99, 0.99]
        self.assertEqual(select_sufficient_realism(identity, detail, 0, 0.84), 1)
        # Not enough detail gain: keep the identity winner.
        self.assertEqual(select_sufficient_realism(identity, [0.90, 0.91, 0.99, 0.99], 0, 0.84), 0)
        # Below the level, or unmeasured, nothing changes.
        self.assertEqual(select_sufficient_realism([0.80, 0.79], [0.7, 0.99], 0, 0.84), 0)
        self.assertEqual(select_sufficient_realism(identity, detail, 0, None), 0)
        self.assertEqual(select_sufficient_realism(identity, [None, 0.93, 0.99, 0.99], 0, 0.84), 0)
        self.assertEqual(select_sufficient_realism(identity, detail[:2], 0, 0.84), 0)


if __name__ == "__main__":
    unittest.main()
