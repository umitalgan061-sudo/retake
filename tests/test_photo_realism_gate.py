import unittest

try:
    import cv2
    import numpy as np
    from tools.photo_realism_gate import (
        photograph_texture_metrics, photographic_render_acceptable
    )
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV/NumPy bulunamadi")
class PhotographicRealismGateTests(unittest.TestCase):
    BOUNDS = {"x": 80.0, "y": 60.0, "width": 190.0, "height": 230.0}

    def base(self):
        rng = np.random.default_rng(31)
        image = np.full((400, 350, 3), 135, dtype=np.float32)
        image += rng.normal(0, 7, image.shape)
        return np.clip(np.rint(image), 0, 255).astype(np.uint8)

    def test_preserves_real_photo_texture(self):
        original = self.base()
        altered = np.clip(original.astype(np.int16) + 2, 0, 255).astype(np.uint8)
        self.assertTrue(photographic_render_acceptable(original, altered, self.BOUNDS)[0])
        self.assertGreater(photograph_texture_metrics(original, self.BOUNDS)["detail"], 2)

    def test_rejects_waxy_plastic_faces(self):
        original = self.base()
        smooth = cv2.GaussianBlur(original, (0, 0), 4.0)
        accepted, detail = photographic_render_acceptable(original, smooth, self.BOUNDS)
        self.assertFalse(accepted)
        self.assertIn("düzleştirilmiş", detail)

    def test_rejects_overprocessed_or_clipped_faces(self):
        original = self.base()
        clipped = original.copy()
        clipped[120:250, 100:250] = 255
        accepted, detail = photographic_render_acceptable(original, clipped, self.BOUNDS)
        self.assertFalse(accepted)
        self.assertIn("kırpılması", detail)

    def test_invalid_size_is_rejected_and_small_patch_is_unverifiable(self):
        image = self.base()
        self.assertFalse(photographic_render_acceptable(image, image[:100], self.BOUNDS)[0])
        small = {"x": 1, "y": 1, "width": 10, "height": 12}
        self.assertTrue(photographic_render_acceptable(image, image, small)[0])


if __name__ == "__main__":
    unittest.main()
