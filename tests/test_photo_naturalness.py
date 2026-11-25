import unittest

import cv2
import numpy as np

from tools import photo_naturalness as pn

H, W = 420, 360
CENTER, AXES = (180, 220), (95, 125)


def photo(seed=5):
    """A lit 'head' on a background with sensor grain and some skin texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    image = np.stack([60 + xx * 0.05, 80 + yy * 0.04, 95 + xx * 0.03], 2)
    head = (((xx - CENTER[0]) / AXES[0]) ** 2 + ((yy - CENTER[1]) / AXES[1]) ** 2) < 1.25
    light = 0.75 + 0.45 * (xx / W)                        # lit from the right
    skin = np.array([120, 150, 205], np.float32) * light[:, :, None]
    image[head] = skin[head]
    texture = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.5) * 6
    image += texture[:, :, None] * head[:, :, None]
    image += rng.normal(0, 2.5, image.shape)              # sensor grain everywhere
    return np.clip(image, 0, 255).astype(np.uint8)


def face_mask():
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, CENTER, AXES, 0, 0, 360, 1, -1)
    return mask


def edit(original, change):
    """Apply `change` (float image -> float image) inside the face, feathered."""
    soft = cv2.GaussianBlur(face_mask().astype(np.float32), (0, 0), 6)[:, :, None]
    changed = change(original.astype(np.float32))
    return np.clip(original * (1 - soft) + changed * soft, 0, 255).astype(np.uint8)


def other_texture(image, seed=9):
    """A 'new face': same light and grain, different skin texture."""
    rng = np.random.default_rng(seed)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.5) * 6
    return image + texture[:, :, None] * 0.8 + 3


class NaturalnessTest(unittest.TestCase):
    def score(self, result, original):
        return pn.naturalness(result, original, face_width=190.0, mask=face_mask())

    def test_the_original_photo_scores_full_marks(self):
        original = photo()
        report = pn.naturalness(original, original)
        self.assertEqual(report["score"], 100.0)

    def test_a_consistent_edit_scores_high(self):
        original = photo()
        good = edit(original, other_texture)
        report = self.score(good, original)
        self.assertGreater(report["score"], 80)

    def test_each_defect_is_caught_by_its_own_check(self):
        original = photo()
        good = self.score(edit(original, other_texture), original)
        mask = face_mask()[:, :, None].astype(np.float32)
        pasted = np.clip(original * (1 - mask) + np.clip(original * 1.15, 0, 255) * mask, 0, 255).astype(np.uint8)
        yellow = edit(original, lambda image: image + np.float32([-12, 2, 10]))
        waxy = edit(original, lambda image: cv2.GaussianBlur(image, (0, 0), 2.5))
        flipped = edit(original, lambda image: image * (1.3 - 0.6 * np.linspace(0, 1, W, dtype=np.float32))[None, :, None])
        cases = {"border": pasted, "tone": yellow, "detail": waxy, "light": flipped}
        for check, image in cases.items():
            with self.subTest(defect=check):
                report = self.score(image, original)
                self.assertLess(report["score"], good["score"] - 10)
                self.assertLess(report["checks"][check], good["checks"][check] - 20)
        smooth = self.score(edit(original, lambda image: cv2.bilateralFilter(
            image.astype(np.uint8), 9, 40, 9).astype(np.float32)), original)
        self.assertLess(smooth["checks"]["grain"], good["checks"]["grain"] - 20)

    def test_a_resaved_photo_is_limited_to_the_face_region(self):
        original = photo()
        good = edit(original, other_texture)
        resaved = cv2.imdecode(cv2.imencode(".jpg", good, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
        box = (CENTER[0] - AXES[0], CENTER[1] - AXES[1], 2 * AXES[0], 2 * AXES[1])
        report = pn.naturalness(resaved, original, region=box)
        self.assertIsNotNone(report["score"])
        self.assertGreater(report["score"], 60)

    def test_weakest_names_the_failing_check(self):
        self.assertEqual(pn.weakest({"checks": {"border": 90, "tone": 40}}), "ten uyumu")
        self.assertIsNone(pn.weakest({"checks": {"border": 90, "tone": 85}}))
        self.assertIsNone(pn.weakest({"checks": {}}))


if __name__ == "__main__":
    unittest.main()
