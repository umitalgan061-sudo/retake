"""Read-only A/B audit tests; no private profile images are stored in Git."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.audit_photo_quality import audit_result


class AuditPhotoQualityTest(unittest.TestCase):
    BOUNDS = {"x": 50.0, "y": 48.0, "width": 170.0, "height": 212.0}

    def test_detects_contour_and_does_not_touch_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = np.full((310, 270, 3), 140, np.uint8)
            result = target.copy()
            cv2.ellipse(result, (135, 150), (80, 102), 0, 0, 360, (181, 181, 181), -1)
            cv2.ellipse(result, (135, 150), (80, 102), 0, 0, 360, (5, 5, 5), 2)
            source = root / "source.png"
            output = root / "output.png"
            self.assertTrue(cv2.imwrite(str(source), target))
            self.assertTrue(cv2.imwrite(str(output), result))
            original_bytes, final_bytes = source.read_bytes(), output.read_bytes()

            report = audit_result(source, output, self.BOUNDS)

            self.assertIsNotNone(report["residual_dark_seam_fraction"])
            self.assertGreater(report["residual_dark_seam_fraction"], .01)
            self.assertEqual(original_bytes, source.read_bytes())
            self.assertEqual(final_bytes, output.read_bytes())

    def test_missing_image_reports_unknown_instead_of_inventing_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = audit_result(root / "missing.png", root / "missing-output.png", self.BOUNDS)
            self.assertIsNone(report["residual_dark_seam_fraction"])
            self.assertIsNone(report["face_detail_score"])


if __name__ == "__main__":
    unittest.main()
