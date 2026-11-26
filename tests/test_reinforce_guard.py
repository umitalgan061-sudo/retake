"""The 128 px identity reinforcement must not blur large faces (synthetic images only)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.ultra_quality import (
    check_reinforcement_detail, fuse_reinforcement_detail, reinforcement_detail_decision, reinforcement_split_sigma
)


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


class DetailFusionTest(unittest.TestCase):
    def test_fusion_keeps_the_reinforced_structure_and_the_single_pass_detail(self):
        bounds = {"x": 170, "y": 200, "width": 460, "height": 600}
        base = textured()
        single = render_at(base, bounds, 256)
        reinforced = render_at(base, bounds, 128)
        x, y, w, h = 170, 200, 460, 600
        # The reinforcement also moves identity: a broad (low-frequency) change.
        shift = np.zeros(base.shape, np.float32)
        cv2.ellipse(shift, (x + w // 2, y + h // 2), (w // 3, h // 3), 0, 0, 360, (12, 6, -8), -1)
        shift = cv2.GaussianBlur(shift, (0, 0), 25)
        reinforced = np.clip(reinforced.astype(np.float32) + shift, 0, 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            a, b, c = (Path(folder) / name for name in ("single.png", "reinforced.png", "fused.png"))
            cv2.imwrite(str(a), single)
            cv2.imwrite(str(b), reinforced)
            sigma = fuse_reinforcement_detail(a, b, c, bounds)
            fused = cv2.imread(str(c))
            self.assertFalse(check_reinforcement_detail(a, b, bounds)[0])
            self.assertTrue(check_reinforcement_detail(a, c, bounds)[0])
        self.assertAlmostEqual(sigma, reinforcement_split_sigma(460))
        broad = lambda image: cv2.GaussianBlur(image.astype(np.float32), (0, 0), 12)
        face = (slice(y + 100, y + h - 100), slice(x + 100, x + w - 100))
        self.assertLess(np.abs(broad(fused) - broad(reinforced))[face].mean(), 1.0)
        self.assertGreater(np.abs(broad(single) - broad(reinforced))[face].mean(), 3.0)
        # Pixels neither swapper touched are unchanged.
        self.assertTrue(np.array_equal(fused[:150, :150], single[:150, :150]))

    def test_split_sigma_follows_the_face_size_within_bounds(self):
        self.assertEqual(reinforcement_split_sigma(128), 1.0)
        self.assertAlmostEqual(reinforcement_split_sigma(460), 0.5 * 460 / 128)
        self.assertEqual(reinforcement_split_sigma(5000), 4.0)
        self.assertEqual(reinforcement_split_sigma(float("nan")), 1.0)


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


class SelfFidelityTest(unittest.TestCase):
    """Ground-truth self-test metric on synthetic images."""

    def measure(self, original, edited, bounds):
        from tools.ultra_quality import self_fidelity
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a.png", Path(folder) / "b.png"
            cv2.imwrite(str(a), original)
            cv2.imwrite(str(b), edited)
            return self_fidelity(a, b, bounds)

    def test_identical_blurred_and_recoloured_faces_rank_as_expected(self):
        bounds = {"x": 170, "y": 200, "width": 460, "height": 600}
        original = textured()
        self.assertEqual(self.measure(original, original.copy(), bounds), {"ssim": 1.0, "delta_e": 0.0})
        blurred = self.measure(original, render_at(original, bounds, 128), bounds)
        self.assertLess(blurred["ssim"], 0.9)
        tinted = original.copy()
        tinted[:, :, 2] = np.clip(tinted[:, :, 2].astype(int) + 40, 0, 255).astype(np.uint8)
        self.assertGreater(self.measure(original, tinted, bounds)["delta_e"], 5)

    def test_bad_inputs_return_nothing(self):
        original = textured()
        self.assertEqual(self.measure(original, original[:400], {"x": 1, "y": 1, "width": 100, "height": 100}), {})
        self.assertEqual(self.measure(original, original, {"x": 1, "y": 1, "width": float("nan"), "height": 100}), {})
        self.assertEqual(self.measure(original, original, {"x": 1}), {})
