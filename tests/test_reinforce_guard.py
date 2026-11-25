"""The 128 px identity reinforcement must not blur large faces (synthetic images only)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.ultra_quality import check_reinforcement_detail, reinforcement_detail_decision


def textured(size=(1000, 800), seed=3):
    rng = np.random.default_rng(seed)
    base = rng.integers(70, 190, size=(*size, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (0, 0), 0.8)


def render_at(image, bounds, model_px):
    """Simulate a swapper that draws the face crop at `model_px` and upscales it."""
    x, y, w, h = (int(bounds[key]) for key in ("x", "y", "width", "height"))
    out = image.copy()
    face = out[y:y + h, x:x + w]
    small = cv2.resize(face, (model_px, max(1, round(model_px * h / w))), interpolation=cv2.INTER_AREA)
    out[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


class ReinforceGuardTest(unittest.TestCase):
    def check(self, single, reinforced, bounds):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "single.png", Path(folder) / "reinforced.png"
            cv2.imwrite(str(a), single)
            cv2.imwrite(str(b), reinforced)
            return check_reinforcement_detail(a, b, bounds)

    def test_large_face_rejects_128px_render_that_blurs_eyes_and_mouth(self):
        bounds = {"x": 170, "y": 200, "width": 460, "height": 600}
        single = render_at(textured(), bounds, 256)
        accepted, reason, ratios = self.check(single, render_at(single, bounds, 128), bounds)
        self.assertFalse(accepted)
        self.assertIn(reason, {"eyes-blur", "mouth-blur"})
        self.assertLess(ratios["eyes"], 0.85)

    def test_small_face_keeps_reinforcement_because_128px_loses_nothing(self):
        bounds = {"x": 300, "y": 300, "width": 110, "height": 140}
        single = textured()
        accepted, reason, _ratios = self.check(single, render_at(single, bounds, 128), bounds)
        self.assertTrue(accepted, reason)

    def test_identical_detail_passes_and_bad_inputs_fail_closed(self):
        bounds = {"x": 170, "y": 200, "width": 460, "height": 600}
        single = textured()
        self.assertTrue(self.check(single, single.copy(), bounds)[0])
        self.assertEqual(self.check(single, single[:500], bounds)[:2], (False, "unreadable"))
        self.assertEqual(reinforcement_detail_decision({}), (True, "unmeasured"))
        self.assertEqual(reinforcement_detail_decision({"eyes": float("nan")}), (False, "eyes-blur"))
        self.assertEqual(reinforcement_detail_decision({"eyes": 0.9, "mouth": 0.6, "nose": 0.1}), (False, "mouth-blur"))


if __name__ == "__main__":
    unittest.main()
