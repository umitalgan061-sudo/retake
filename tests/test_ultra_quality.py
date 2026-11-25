"""Unit tests of photorealistic local facial-detail retention scoring."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from tools.ultra_quality import REGIONS, choose_detail_safe_tie, regional_photo_detail

BOUNDS = {"x": 54, "y": 54, "width": 210, "height": 245}


class UltraPhotoDetailTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        image = np.full((360, 330, 3), (116, 144, 168), np.uint8)
        grain = rng.normal(0, 10, image.shape)
        self.original = np.clip(image.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    def test_same_photo_retains_detail_in_all_four_regions(self):
        score, ratios = regional_photo_detail(self.original, self.original.copy(), BOUNDS)
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(set(ratios), set(REGIONS))

    def test_blurred_eye_nose_mouth_and_beard_get_lower_score(self):
        blur = cv2.GaussianBlur(self.original, (0, 0), 5)
        first, _ = regional_photo_detail(self.original, self.original, BOUNDS)
        blurred, _ = regional_photo_detail(self.original, blur, BOUNDS)
        self.assertLess(blurred, first - 0.15)
        self.assertLess(blurred, 0.7)

    def test_over_sharpening_is_not_rewarded(self):
        blurry = cv2.GaussianBlur(self.original, (0, 0), 1.3)
        sharp = cv2.addWeighted(self.original, 2.9, blurry, -1.9, 0)
        score, _ = regional_photo_detail(self.original, sharp, BOUNDS)
        self.assertLess(score, 1.0)

    def test_no_claim_without_sufficient_face_details(self):
        score, ratios = regional_photo_detail(
            np.zeros((20, 20, 3), np.uint8), np.zeros((20, 20, 3), np.uint8),
            {"x": 1, "y": 1, "width": 4, "height": 4}
        )
        self.assertIsNone(score)
        self.assertEqual(ratios, {})

    def test_invalid_target_image_is_not_scored(self):
        score, _ = regional_photo_detail(
            self.original, self.original[:80], BOUNDS
        )
        self.assertIsNone(score)

    def test_detail_breaks_only_an_identity_near_tie(self):
        self.assertEqual(
            choose_detail_safe_tie([.8400, .8385], [.80, .88], 0), 1
        )
        self.assertEqual(
            choose_detail_safe_tie([.8400, .8300], [.80, .98], 0), 0
        )
        self.assertEqual(
            choose_detail_safe_tie([.8400, .8390], [.80, .81], 0), 0
        )


    def test_invalid_or_missing_identity_scores_do_not_change_winner(self):
        for invalid in (None, float("nan"), float("inf"), True, "0.84"):
            with self.subTest(invalid=invalid):
                self.assertEqual(
                    choose_detail_safe_tie([.84, invalid], [.50, .99], 0), 0
                )
                self.assertEqual(
                    choose_detail_safe_tie([invalid, .84], [.50, .99], 0), 0
                )

    def test_invalid_detail_and_out_of_range_scores_are_ignored(self):
        for invalid in (None, float("nan"), float("inf"), -1, 1.1, "1.0"):
            with self.subTest(invalid=invalid):
                self.assertEqual(
                    choose_detail_safe_tie([.84, .839], [.5, invalid], 0), 0
                )
                self.assertEqual(
                    choose_detail_safe_tie([.84, .839], [invalid, .9], 0), 0
                )

    def test_close_candidate_must_also_be_near_best_measured_identity(self):
        # A legacy winner can be behind an unselected candidate by .005
        # because the initial identity gate requires a meaningful .006 gain.
        # The extra quality tie-break must not trade away another .003.
        self.assertEqual(
            choose_detail_safe_tie([.835, .840, .833], [.5, .6, .99], 0), 0
        )

    def test_missing_or_invalid_parameters_leave_selection_unchanged(self):
        self.assertEqual(choose_detail_safe_tie([.84], [.8], 5), 5)
        self.assertEqual(choose_detail_safe_tie([.84, .839], [.8], 0), 0)
        self.assertEqual(
            choose_detail_safe_tie([.84, .839], [.8, .99], 0, identity_tolerance=float("nan")), 0
        )
        self.assertEqual(
            choose_detail_safe_tie([.84, .839], [.8, .99], 0, minimum_detail_gain=-.1), 0
        )

    def test_equal_detail_does_not_overrule_original_identity_choice(self):
        self.assertEqual(
            choose_detail_safe_tie([.840, .839], [.81, .825], 0), 0
        )


if __name__ == "__main__":
    unittest.main()



class DetailGateTest(unittest.TestCase):
    def test_sharp_regions_required_even_with_high_average(self):
        from tools.ultra_quality import detail_gate
        self.assertEqual(detail_gate(.95, {"eyes":.59,"mouth":1.0}), (False,"eyes-blur"))
        self.assertEqual(detail_gate(.95, {"eyes":1.0,"mouth":.64}), (False,"mouth-blur"))
        self.assertEqual(detail_gate(.95, {"eyes":2.8}), (False,"eyes-oversharpen"))
        self.assertTrue(detail_gate(.9, {"eyes":.9,"mouth":.8,"lower_face":2.1})[0])
        self.assertFalse(detail_gate(None, {})[0])

    def test_recovery_preserves_alpha_and_outside_pixels(self):
        import tempfile
        from pathlib import Path
        from tools.ultra_quality import balance_with_original, check_detail_gate
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rng=np.random.default_rng(42)
            original=np.clip(128+rng.normal(0,20,(360,330,3)),0,255).astype(np.uint8)
            softened=cv2.GaussianBlur(original,(0,0),4)
            alpha=np.zeros(original.shape[:2],np.uint8)
            alpha[54:299,54:264]=255
            candidate=np.dstack((softened,alpha))
            cv2.imwrite(str(root/"original.png"),original)
            cv2.imwrite(str(root/"candidate.png"),candidate)
            self.assertFalse(check_detail_gate(root/"original.png",root/"candidate.png",BOUNDS)[0])
            balance_with_original(root/"original.png",root/"candidate.png",root/"balanced.png",.60)
            balanced=cv2.imread(str(root/"balanced.png"),cv2.IMREAD_UNCHANGED)
            self.assertTrue(np.array_equal(balanced[:,:,3],alpha))
            self.assertTrue(np.array_equal(balanced[alpha==0],candidate[alpha==0]))
            self.assertGreater(np.std(balanced[alpha>0,:3]),np.std(candidate[alpha>0,:3]))

class RegionalCandidateSelectionTest(unittest.TestCase):
    def test_never_drifts_away_from_peak(self):
        from tools.ultra_quality import select_verified_regional_candidate
        self.assertEqual(select_verified_regional_candidate([.795,.801,.793],[.9,.8,.99],["head-fit","identity-only","identity-beard"],0),1)

    def test_verified_beard_can_win_a_tight_tie(self):
        from tools.ultra_quality import select_verified_regional_candidate
        self.assertEqual(select_verified_regional_candidate([.800,.798],[.85,.84],["head-fit","identity-beard"],0),1)

    def test_missing_or_degraded_regional_detail_cannot_win(self):
        from tools.ultra_quality import select_verified_regional_candidate
        for detail in (None,.50,float("nan")):
            self.assertEqual(select_verified_regional_candidate([.800,.798],[.85,detail],["identity-only","identity-beard"],0),0)

class NativeRegionalEditTest(unittest.TestCase):
    def test_flat_texture_can_still_have_measurable_regional_change(self):
        import tempfile
        from pathlib import Path
        from tools.ultra_quality import regional_edit_statistics
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = np.full((100,100,3),100,np.uint8)
            edited = original.copy()
            edited[55:90,20:80,0] = 130
            cv2.imwrite(str(root/"before.png"),original)
            cv2.imwrite(str(root/"after.png"),edited)
            values = regional_edit_statistics(root/"before.png",root/"after.png",
                {"x":10,"y":10,"width":80,"height":80})
            self.assertEqual(values["eyes"],0)
            self.assertGreater(values["mouth"],0)
            self.assertGreater(values["chin"],0)
            self.assertLessEqual(values["lower_face"],30)
    def test_invalid_bounds_or_input_do_not_create_a_fake_measurement(self):
        import tempfile
        from pathlib import Path
        from tools.ultra_quality import regional_edit_statistics
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root/"a.png"),np.zeros((50,50,3),np.uint8))
            self.assertEqual(regional_edit_statistics(root/"a.png",root/"a.png",
                {"x":0,"y":0,"width":float("nan"),"height":40}),{})
