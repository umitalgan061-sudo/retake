"""Synthetic regressions for the optional upper-face transition candidate."""
import unittest
import cv2
import numpy as np

from tools.retake_rim_cleanup import upper_seam_candidate, upper_seam_quality_guard


class UpperSeamTest(unittest.TestCase):
    def test_only_narrow_upper_edited_region_changes_and_alpha_is_preserved(self):
        original = np.full((240, 240, 3), (105, 145, 190), dtype=np.uint8)
        edited = np.concatenate((original.copy(), np.full((240, 240, 1), 255, np.uint8)), axis=2)
        mask = np.zeros((240, 240), np.uint8)
        cv2.ellipse(mask, (120, 120), (60, 75), 0, 0, 360, 1, -1)
        edited[mask > 0, :3] = (130, 165, 210)
        bounds = {"x": 60, "y": 45, "width": 120, "height": 150}
        candidate, report = upper_seam_candidate(original, edited, bounds)
        self.assertEqual(report["stage"], "candidate")
        self.assertGreater(report["pixels"], 0)
        np.testing.assert_array_equal(candidate[:, :, 3], edited[:, :, 3])
        np.testing.assert_array_equal(candidate[135:], edited[135:])
        np.testing.assert_array_equal(candidate[120, 120], edited[120, 120])
        np.testing.assert_array_equal(candidate[mask == 0], edited[mask == 0])

    def test_noop_and_invalid_bounds_do_not_modify(self):
        original = np.full((100, 100, 3), 125, dtype=np.uint8)
        for bounds in ({}, {"x": 20, "y": 20, "width": 60, "height": 60}):
            candidate, report = upper_seam_candidate(original, original.copy(), bounds)
            self.assertNotEqual(report["stage"], "candidate")
            np.testing.assert_array_equal(candidate, original)

    def test_guard_needs_border_gain_and_preserves_other_metrics(self):
        before = {"score": 81, "checks": {"border": 60, "detail": 90, "tone": 85}}
        self.assertTrue(upper_seam_quality_guard(before, {"score": 82, "checks": {"border": 64, "detail": 89, "tone": 85}}))
        for after in (
            {"score": 82, "checks": {"border": 61, "detail": 90, "tone": 85}},
            {"score": 78, "checks": {"border": 70, "detail": 90, "tone": 85}},
            {"score": 83, "checks": {"border": 70, "detail": 80, "tone": 85}},
            {"score": 83, "checks": {"border": 70, "detail": 90}},
            {"score": float("nan"), "checks": {"border": 70, "detail": 90, "tone": 85}},
        ):
            self.assertFalse(upper_seam_quality_guard(before, after))


if __name__ == "__main__":
    unittest.main()
