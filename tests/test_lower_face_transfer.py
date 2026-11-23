"""Reference beard/moustache/chin transfer checks using 478-point test faces."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools.lower_face_transfer import (
    choose_and_transfer_lower_face, lower_face_mask, MOUSTACHE
)
from tools.retake_appearance import LOWER_FACE, LIPS
from tools.retake_detail import FACE_OVAL, INNER_MOUTH, MESH_POINTS


SIZE = 240


def face_mesh():
    rng = np.random.default_rng(10)
    points = np.zeros((MESH_POINTS, 2), np.float32)
    angle = rng.uniform(0, 2 * np.pi, MESH_POINTS)
    radius = np.sqrt(rng.uniform(0.0, 0.7, MESH_POINTS))
    points[:, 0] = 120 + 70 * radius * np.cos(angle)
    points[:, 1] = 125 + 95 * radius * np.sin(angle)
    for i, index in enumerate(FACE_OVAL):
        theta = -np.pi / 2 + 2 * np.pi * i / len(FACE_OVAL)
        points[index] = (120 + 70 * np.cos(theta), 125 + 95 * np.sin(theta))
    for i, index in enumerate(LOWER_FACE):
        theta = np.pi + np.pi * i / (len(LOWER_FACE) - 1)
        points[index] = (120 + 65 * np.cos(theta), 130 - 75 * np.sin(theta))
    # The moustache area runs across the upper lip but does not reach nose.
    for i, index in enumerate(MOUSTACHE):
        theta = np.pi + np.pi * i / (len(MOUSTACHE) - 1)
        points[index] = (120 + 25 * np.cos(theta), 166 + 13 * np.sin(theta))
    for i, index in enumerate(LIPS):
        theta = 2 * np.pi * i / len(LIPS)
        points[index] = (120 + 21 * np.cos(theta), 174 + 6 * np.sin(theta))
    for i, index in enumerate(INNER_MOUTH):
        theta = 2 * np.pi * i / len(INNER_MOUTH)
        points[index] = (120 + 13 * np.cos(theta), 174 + 3 * np.sin(theta))
    points[33], points[263] = (90, 100), (150, 100)
    points[70], points[300] = (93, 83), (148, 83)
    return points


class LowerFaceTransferTest(unittest.TestCase):
    def setUp(self):
        self.mesh = face_mesh()
        rng = np.random.default_rng(9)
        self.target = np.clip(120 + rng.normal(0, 12, (SIZE, SIZE, 3)), 0, 255).astype(np.uint8)
        self.swap = np.clip(self.target.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        self.reference = np.clip(self.target.astype(np.int16) - 22, 0, 255).astype(np.uint8)

    def test_mask_is_empty_outside_face_and_on_lips(self):
        mask = lower_face_mask(self.mesh, self.mesh, (SIZE, SIZE), face_width=140)
        self.assertEqual(mask.shape, (SIZE, SIZE))
        self.assertEqual(float(mask[30, 30]), 0)
        self.assertGreater(float(mask.max()), 0.5)
        self.assertEqual(float(mask[174, 120]), 0, "mouth/teeth must never be pasted from other photo")
        self.assertEqual(float(mask[90, 120]), 0, "eyes and brows are protected")

    def test_lower_face_candidates_preserve_eyes_mouth_background(self):
        with patch("tools.lower_face_transfer.candidate_fit", return_value={"structure": 0.9, "light": 0.5, "score": 1.0}):
            result, index, fits = choose_and_transfer_lower_face(
                self.swap, self.target, [(self.reference, self.mesh)],
                self.mesh, face_width=140, strength=0.94
            )
        self.assertEqual(index, 0)
        self.assertEqual(len(fits), 1)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.array_equal(result[25:40], self.swap[25:40]))
        self.assertTrue(np.array_equal(result[174, 120], self.swap[174, 120]))
        self.assertGreater(int(np.count_nonzero(cv2.absdiff(result, self.swap))), 50)

    def test_invalid_mesh_or_strength_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "478"):
            lower_face_mask(self.mesh[:20], self.mesh, (SIZE, SIZE), 140)
        with self.assertRaisesRegex(ValueError, "gücü"):
            choose_and_transfer_lower_face(
                self.swap, self.target, [(self.reference, self.mesh)],
                self.mesh, face_width=140, strength=2
            )

    def test_unchanged_input_is_not_mutated(self):
        snapshot = self.swap.copy()
        with patch("tools.lower_face_transfer.candidate_fit", return_value={"structure": .9, "light": .6, "score": 1.0}):
            choose_and_transfer_lower_face(
                self.swap, self.target, [(self.reference, self.mesh)],
                self.mesh, face_width=140, strength=.6
            )
        self.assertTrue(np.array_equal(self.swap, snapshot))


if __name__ == "__main__":
    unittest.main()
