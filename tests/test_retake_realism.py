import unittest

import cv2
import numpy as np

from tools import retake_realism as rr

BOUNDS = {"x": 100.0, "y": 120.0, "width": 160.0, "height": 200.0}


def textured(seed, shape=(480, 400, 3)):
    rng = np.random.default_rng(seed)
    return np.clip(cv2.GaussianBlur(rng.uniform(0, 255, shape).astype(np.float32), (0, 0), 2.0), 0, 255).astype(np.uint8)


class FakeRenderer:
    """Stands in for the diffusion model: 'renders' a fixed photo."""

    def __init__(self, photo):
        self.photo = photo
        self.calls = []

    def render(self, crop, mask, strength, seed, identity=None, identity_scale=None):
        self.calls.append((crop.shape, float(mask.max()), strength, identity_scale))
        return cv2.resize(self.photo, (crop.shape[1], crop.shape[0]))


class RealismGeometryTest(unittest.TestCase):
    def test_crop_is_square_around_the_face_and_survives_the_image_edge(self):
        crop = rr.face_crop((480, 400), BOUNDS)
        self.assertEqual(crop.side, int(np.ceil(200 * rr.CROP_SCALE)))
        self.assertAlmostEqual(crop.x + crop.side / 2, 180, delta=1)
        image = textured(1)
        edge = rr.face_crop((480, 400), {"x": -20.0, "y": 300.0, "width": 160.0, "height": 200.0})
        patch = rr.read_crop(image, edge)
        self.assertEqual(patch.shape[:2], (edge.side, edge.side))
        written = rr.write_crop(image, edge, np.zeros_like(patch), np.ones(patch.shape[:2], np.float32))
        self.assertEqual(written.shape, image.shape)
        self.assertEqual(int(written[479, 0].max()), 0)

    def test_render_size_follows_the_crop_within_sd15_range(self):
        self.assertEqual(rr.render_size(200), 512)
        self.assertEqual(rr.render_size(640), 640)
        self.assertEqual(rr.render_size(2000), 768)
        self.assertEqual(rr.render_size(700) % 64, 0)

    def test_only_what_the_swap_changed_is_re_rendered(self):
        target = textured(2)
        swapped = target.copy()
        cv2.ellipse(swapped, (180, 230), (70, 95), 0, 0, 360, (90, 120, 170), -1)
        hair = (slice(120, 150), slice(110, 250))
        swapped[hair] = target[hair]  # the swap's mask kept the fringe
        crop = rr.face_crop(target.shape[:2], BOUNDS)
        mask = rr.face_mask(target.shape[:2], rr.changed_mask(swapped, target), BOUNDS, crop)
        self.assertGreater(float(mask[230 - crop.y, 180 - crop.x]), 0.95)
        self.assertLess(float(mask[128 - crop.y, 180 - crop.x]), 0.2)
        self.assertEqual(float(mask[5, 5]), 0.0)

    def test_render_keeps_the_swap_light_and_its_own_detail(self):
        target = textured(3)
        swapped = target.copy()
        cv2.ellipse(swapped, (180, 230), (70, 95), 0, 0, 360, (100, 130, 190), -1)
        # The 'render': fine texture on a much darker, bluer base.
        photo = np.clip(textured(9, (300, 300, 3)).astype(np.int16) - (90, 60, 40), 0, 255).astype(np.uint8)
        renderer = FakeRenderer(photo)
        out = rr.refine_face(renderer, swapped, target, BOUNDS, 0.55, identity=np.ones(512), identity_scale=1.3)
        self.assertEqual(renderer.calls[0][2:], (0.55, 1.3))
        centre = (slice(215, 245), slice(165, 195))
        # Broad colour is the swap's (the render's dark blue base is gone)...
        self.assertLess(np.abs(out[centre].mean(axis=(0, 1)) - swapped[centre].mean(axis=(0, 1))).max(), 6.0)
        # ...but there is fine detail now where the swap was flat.
        self.assertGreater(float(out[centre][:, :, 1].std()), float(swapped[centre][:, :, 1].std()) + 3.0)
        # Outside the face nothing moved.
        self.assertTrue(np.array_equal(out[:60], swapped[:60]))

    def test_nothing_to_do_without_a_changed_face(self):
        target = textured(4)
        renderer = FakeRenderer(textured(5, (300, 300, 3)))
        out = rr.refine_face(renderer, target.copy(), target, BOUNDS, 0.55)
        self.assertTrue(np.array_equal(out, target))
        self.assertEqual(renderer.calls, [])

    def test_plans_trade_strength_for_identity_guidance(self):
        self.assertGreaterEqual(len(rr.PLANS), 1)
        for identity_scale, strength in rr.PLANS:
            self.assertTrue(0.3 <= strength <= 0.8)
            self.assertTrue(0.5 <= identity_scale <= 2.0)


if __name__ == "__main__":
    unittest.main()
