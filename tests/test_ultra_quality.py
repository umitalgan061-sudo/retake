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


if __name__ == "__main__":
    unittest.main()

