"""Under-chin / throat transfer checks on synthetic faces (no photographs)."""
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools.lower_face_transfer import choose_and_transfer_lower_face
from tools.throat_transfer import (
    LIPS, THROAT_JAW, band_fade, throat_enabled, throat_grid, transfer_throat
)

SIZE = 360
SKIN = (120, 150, 200)     # BGR, a light skin tone
BEARD = (40, 55, 75)       # BGR, dark brown facial hair
COLLAR = (170, 90, 40)     # BGR, a blue shirt
BACKGROUND = (110, 110, 110)


def mesh(width=1.0, shift=(0.0, 0.0), scale=1.0):
    """478 points; only the ones the transfers read are placed meaningfully."""
    cx, cy = 180 + shift[0], 130 + shift[1]
    points = np.zeros((478, 2), np.float32)
    rng = np.random.default_rng(4)
    points[:] = np.column_stack([cx + rng.uniform(-30, 30, 478) * scale, cy + rng.uniform(-40, 10, 478) * scale])
    points[33] = (cx - 35 * scale, cy - 40 * scale)
    points[263] = (cx + 35 * scale, cy - 40 * scale)
    count = len(THROAT_JAW)
    for i, index in enumerate(THROAT_JAW):
        theta = np.pi * i / (count - 1)
        points[index] = (cx - 62 * width * scale * np.cos(theta), cy + 10 * scale + 60 * scale * np.sin(theta))
    for i, index in enumerate(LIPS):
        theta = 2 * np.pi * i / len(LIPS)
        points[index] = (cx + 22 * scale * np.cos(theta), cy + 32 * scale + 6 * scale * np.sin(theta))
    points[61], points[291] = (cx - 22 * scale, cy + 32 * scale), (cx + 22 * scale, cy + 32 * scale)
    return points


def paint(points, throat_colour, collar_from=None, texture=0.0, seed=1, beard=False):
    image = np.full((SIZE, SIZE, 3), BACKGROUND, np.uint8)
    jaw = np.round(points[list(THROAT_JAW)]).astype(np.int32)
    chin = points[152]
    # Neck: a column under the jaw, the throat colour on it.
    neck = np.array([[chin[0] - 48, chin[1] - 20], [chin[0] + 48, chin[1] - 20],
                     [chin[0] + 44, SIZE - 1], [chin[0] - 44, SIZE - 1]], np.int32)
    cv2.fillPoly(image, [neck], throat_colour)
    face = np.concatenate([jaw, np.array([[jaw[-1][0], 40], [jaw[0][0], 40]], np.int32)])
    cv2.fillPoly(image, [face], SKIN)
    if beard:
        # A beard covers the chin and jaw below the mouth and runs on down the throat.
        mouth_y = int(points[61][1]) + 8
        lower = np.concatenate([jaw, np.array([[jaw[-1][0], mouth_y], [jaw[0][0], mouth_y]], np.int32)])
        cv2.fillPoly(image, [lower], throat_colour)
        lips = np.round(points[list(LIPS)]).astype(np.int32)
        cv2.fillPoly(image, [lips], (90, 100, 170))
    if collar_from is not None:
        image[int(collar_from):, :] = COLLAR
    if texture:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, texture, image.shape[:2])
        region = np.all(image == np.array(throat_colour, np.uint8), axis=2)
        image = image.astype(np.float32)
        image[region] += noise[region][:, None]
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


class ThroatTransferTest(unittest.TestCase):
    def setUp(self):
        self.points = mesh()
        self.chin_y = float(self.points[152][1])
        self.collar_y = self.chin_y + 52
        self.target = paint(self.points, SKIN, collar_from=self.collar_y, texture=3.0)
        self.reference_points = mesh(shift=(6, -4), scale=1.08)
        self.reference = paint(self.reference_points, BEARD, texture=14.0, seed=7, beard=True)

    def throat_box(self):
        y0, y1 = int(self.chin_y + 4), int(self.chin_y + 22)
        x0, x1 = int(self.points[152][0] - 20), int(self.points[152][0] + 20)
        return slice(y0, y1), slice(x0, x1)

    def test_grid_starts_inside_the_jaw_and_narrows_down_the_neck(self):
        grid = throat_grid(self.points)
        self.assertEqual(grid.shape, (4, len(THROAT_JAW), 2))
        self.assertLess(grid[0, 6, 1], self.chin_y)              # row 0 lifted into the face
        self.assertGreater(grid[-1, 6, 1], self.chin_y + 30)     # neck row well below the chin
        self.assertLess(np.ptp(grid[-1, :, 0]), np.ptp(grid[0, :, 0]))
        fade = band_fade(grid, (SIZE, SIZE), 124)
        self.assertGreater(float(fade[int(self.chin_y + 6), int(self.points[152][0])]), 0.8)
        self.assertEqual(float(fade[5, 5]), 0.0)

    def test_reference_beard_reaches_the_throat_and_clothing_stays(self):
        result, report = transfer_throat(self.target.copy(), self.target, (self.reference, self.reference_points),
                                         self.points, face_width=124, strength=1.0)
        rows, cols = self.throat_box()
        before = cv2.cvtColor(self.target, cv2.COLOR_BGR2LAB)[rows, cols, 0].astype(float).mean()
        after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)[rows, cols, 0].astype(float).mean()
        self.assertLess(after, before - 25, "the throat must take the reference's beard darkness")
        self.assertGreater(float(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)[rows, cols].std()), 4.0,
                           "stubble texture comes with it")
        collar = result[int(self.collar_y) + 2:, :]
        np.testing.assert_array_equal(collar, self.target[int(self.collar_y) + 2:, :])
        np.testing.assert_array_equal(result[:, :60], self.target[:, :60])
        self.assertGreater(report["pixels"], 200)

    def test_scene_exposure_is_matched_without_lifting_the_beard(self):
        # A camera exposure difference is multiplicative (here 1.25× brighter).
        brighter = np.clip(self.reference.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
        dim, _ = transfer_throat(self.target.copy(), self.target, (self.reference, self.reference_points),
                                 self.points, face_width=124, strength=1.0)
        bright, report = transfer_throat(self.target.copy(), self.target, (brighter, self.reference_points),
                                         self.points, face_width=124, strength=1.0)
        rows, cols = self.throat_box()
        self.assertLess(abs(float(bright[rows, cols].mean()) - float(dim[rows, cols].mean())), 5.0)
        self.assertTrue(all(value < 1.0 for value in report["gain"]))
        self.assertLess(float(bright[rows, cols].mean()), float(self.target[rows, cols].mean()) - 20)

    def test_hidden_or_mismatched_throats_are_refused(self):
        turtleneck = paint(self.points, COLLAR)
        with self.assertRaisesRegex(ValueError, "Fotoğrafta boğaz"):
            transfer_throat(turtleneck.copy(), turtleneck, (self.reference, self.reference_points), self.points, 124)
        covered = paint(self.reference_points, COLLAR)
        with self.assertRaisesRegex(ValueError, "Referansta boğaz"):
            transfer_throat(self.target.copy(), self.target, (covered, self.reference_points), self.points, 124)
        turned = mesh(width=0.6)
        with self.assertRaisesRegex(ValueError, "pozu farklı"):
            transfer_throat(self.target.copy(), self.target, (paint(turned, BEARD, beard=True), turned), self.points, 124)
        with self.assertRaises(ValueError):
            transfer_throat(self.target.copy(), self.target, (self.reference, self.reference_points), self.points, 124, strength=0)

    def test_switch_and_lower_face_integration(self):
        with patch.dict(os.environ, {"RETAKE_THROAT_TRANSFER": "off"}):
            self.assertFalse(throat_enabled())
        with patch.dict(os.environ, {"RETAKE_THROAT_TRANSFER": ""}):
            self.assertTrue(throat_enabled())
        swap = self.target.copy()
        calls = []

        def fake_throat(image, target, candidate, points, face_width, strength):
            calls.append(strength)
            marked = image.copy()
            marked[0, 0] = (1, 2, 3)
            return marked, {"pixels": 1}

        lower = np.zeros(swap.shape[:2], np.float32)
        with patch("tools.lower_face_transfer._prepare_candidate", return_value=(swap.astype(np.float32), np.ones_like(lower), np.ones_like(lower))), \
                patch("tools.lower_face_transfer.lower_face_mask", return_value=np.ones_like(lower)), \
                patch("tools.lower_face_transfer.candidate_fit", return_value={"score": 1.0, "structure": 1.0}), \
                patch("tools.lower_face_transfer.transfer_throat", side_effect=fake_throat):
            result, _, _ = choose_and_transfer_lower_face(swap, self.target, [(self.reference, self.reference_points)],
                                                          self.points, 124, strength=0.8)
            self.assertEqual(tuple(result[0, 0]), (1, 2, 3))
            self.assertEqual(calls, [0.8])
            with patch.dict(os.environ, {"RETAKE_THROAT_TRANSFER": "off"}):
                untouched, _, _ = choose_and_transfer_lower_face(swap, self.target, [(self.reference, self.reference_points)],
                                                                 self.points, 124, strength=0.8)
            self.assertNotEqual(tuple(untouched[0, 0]), (1, 2, 3))
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
