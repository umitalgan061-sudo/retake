"""Regression tests for dark halo removal around a FaceFusion swap."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.retake_rim_cleanup import repair_dark_rim, repair_dark_rim_file


class FaceRimCleanupTest(unittest.TestCase):
    bounds = {"x": 50.0, "y": 48.0, "width": 170.0, "height": 212.0}

    def make_source(self):
        return np.full((310, 270, 3), 140, np.uint8)

    def make_ink_outline(self):
        source = self.make_source()
        result = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
        cv2.ellipse(result, (135, 150), (80, 102), 0, 0, 360, (181, 181, 181, 255), -1)
        cv2.ellipse(result, (135, 150), (80, 102), 0, 0, 360, (10, 10, 10, 255), 1)
        return source, result

    def test_removes_1px_black_ring_without_changing_eye_or_alpha(self):
        source, result = self.make_ink_outline()
        repaired, stats = repair_dark_rim(source, result, self.bounds)
        self.assertGreater(stats["dark_line_pixels"], 100)
        self.assertGreater(stats["corrected_pixels"], 100)
        self.assertLess(np.count_nonzero(repaired[:, :, 0] < 20), 10)
        self.assertTrue(np.array_equal(repaired[:, :, 3], result[:, :, 3]))
        self.assertTrue(np.array_equal(repaired[150, 135], result[150, 135]))
        self.assertTrue(np.array_equal(repaired[0:20], result[0:20]))
        self.assertTrue(np.array_equal(repaired[-20:], result[-20:]))

    def test_continuous_three_pixel_ink_outline_is_actually_removed(self):
        source = self.make_source()
        edited = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
        cv2.ellipse(edited, (135, 150), (80, 102), 0, 0, 360, (181, 181, 181, 255), -1)
        # Median blur alone cannot fix a 3-px dark ring, as observed
        # on the real Retake output around forehead and lower jaw.
        cv2.ellipse(edited, (135, 150), (80, 102), 0, 0, 360, (5, 5, 5, 255), 3)
        outside = edited[0:25].copy()
        original_alpha = edited[:, :, 3].copy()
        cleaned, stats = repair_dark_rim(source, edited, self.bounds)
        self.assertGreater(stats["dark_line_pixels"], 200)
        self.assertGreater(stats["dark_line_reduction"], .50)
        self.assertLess(stats["dark_line_remaining"], stats["dark_line_pixels"] * .5)
        self.assertGreater(stats["corrected_pixels"], 200)
        # No dark ink should be left along the reconstructed face ellipse.
        self.assertLess(np.count_nonzero(cleaned[:, :, 0] < 20), 80)
        self.assertTrue(np.array_equal(cleaned[:, :, 3], original_alpha))
        self.assertTrue(np.array_equal(cleaned[0:25], outside))
        self.assertTrue(np.array_equal(cleaned[150, 135], edited[150, 135]))

    def test_dark_beard_inside_face_is_not_oversmoothed(self):
        source, edited = self.make_ink_outline()
        cv2.rectangle(edited, (108, 199), (165, 226), (28, 31, 35, 255), -1)
        beard = edited[201:225, 110:164].copy()
        cleaned, _ = repair_dark_rim(source, edited, self.bounds)
        self.assertTrue(np.array_equal(cleaned[201:225, 110:164], beard),
                        "real inner beard detail must not be inpainted")
        self.assertTrue(np.array_equal(cleaned[:, :, 3], edited[:, :, 3]))

    def test_short_one_pixel_jaw_ink_trace_is_detected(self):
        # A partial 110-degree outline occupies under 3% of the OLD wide
        # detection band: v21 silently skipped the visibly dark line.
        source = self.make_source()
        edited = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
        cv2.ellipse(edited, (135, 150), (80, 102), 0, 0, 360, (181, 181, 181, 255), -1)
        cv2.ellipse(edited, (135, 150), (80, 102), 0, 25, 145, (5, 5, 5, 255), 1)
        original_alpha = edited[:, :, 3].copy()
        cleaned, report = repair_dark_rim(source, edited, self.bounds)
        self.assertGreater(report["dark_line_pixels"], 35)
        self.assertGreater(report["dark_line_fraction"], 0.015)
        self.assertEqual(report["stage"], "repaired")
        self.assertGreater(report["dark_line_reduction"], .25)
        self.assertTrue(np.array_equal(cleaned[:, :, 3], original_alpha))
        self.assertTrue(np.array_equal(cleaned[150, 135], edited[150, 135]))
        self.assertTrue(np.array_equal(cleaned[0:24], edited[0:24]))

    def test_naturally_dark_target_outline_is_not_removed(self):
        source = self.make_source()
        edited = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
        cv2.ellipse(source, (135, 150), (80, 102), 0, 0, 360, (5, 5, 5), 2)
        edited[:, :, :3] = source
        result, report = repair_dark_rim(source, edited, self.bounds)
        self.assertEqual(report["dark_line_pixels"], 0)
        self.assertEqual(report["corrected_pixels"], 0)
        self.assertTrue(np.array_equal(result, edited))

    def test_no_halo_means_byte_for_byte_unchanged(self):
        source = self.make_source()
        same = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
        cleaned, stats = repair_dark_rim(source, same, self.bounds)
        self.assertEqual(stats["corrected_pixels"], 0)
        self.assertTrue(np.array_equal(cleaned, same))

    def test_invalid_bounds_and_shape_do_not_modify_an_image(self):
        source, result = self.make_ink_outline()
        for invalid in ({"x": 10, "y": 10, "width": -1, "height": 60}, {}, {"x": 0, "y": 0, "width": 12, "height": 13}):
            cleaned, stats = repair_dark_rim(source, result, invalid)
            self.assertEqual(stats["corrected_pixels"], 0)
            self.assertTrue(np.array_equal(cleaned, result))

    def test_file_candidate_can_be_arcface_verified_before_replacement(self):
        source, result = self.make_ink_outline()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            original = base / "source.png"
            current = base / "current.png"
            candidate = base / "candidate.png"
            cv2.imwrite(str(original), source)
            cv2.imwrite(str(current), result)
            stats = repair_dark_rim_file(original, current, candidate, self.bounds)
            self.assertGreater(stats["corrected_pixels"], 100)
            self.assertTrue(candidate.is_file())
            stored = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
            self.assertEqual(stored.shape, result.shape)
            self.assertTrue(np.array_equal(stored[:, :, 3], result[:, :, 3]))


if __name__ == "__main__":
    unittest.main()
