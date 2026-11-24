"""Regressions for the user's colored outline and odd-sized native crops."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.retake_output_transport import (
    binary_preblended_rgba, encode_preblended_result, prepare_inference_target
)


class OutputTransportTest(unittest.TestCase):
    def test_odd_target_is_not_overwritten_or_rescaled(self):
        rng = np.random.default_rng(42)
        for height, width in ((81, 101), (80, 101), (81, 100), (80, 100)):
            with self.subTest(height=height, width=width), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / "target.png"
                image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(target), image))
                original_bytes = target.read_bytes()
                model_input = prepare_inference_target(target, {"width": width, "height": height}, root)
                padded = cv2.imread(str(model_input))
                self.assertEqual(padded.shape[:2], (height + height % 2, width + width % 2))
                np.testing.assert_array_equal(padded[:height, :width], image)
                self.assertEqual(target.read_bytes(), original_bytes)
                self.assertEqual(model_input == target, not (height % 2 or width % 2))

    def test_refuses_metadata_size_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.png"
            cv2.imwrite(str(target), np.zeros((81, 101, 3), dtype=np.uint8))
            with self.assertRaises(ValueError):
                prepare_inference_target(target, {"width": 100, "height": 81}, root)

    def test_binary_transport_preserves_rgb_through_canvas_premultiplication(self):
        target = np.full((4, 8, 3), [129, 151, 182], dtype=np.uint8)
        composite = np.empty((4, 8, 4), dtype=np.uint8)
        composite[:, :, :3] = [117, 156, 190]
        composite[:, :, 3] = [0, 1, 2, 5, 16, 64, 128, 255]
        wire = binary_preblended_rgba(target, composite)
        self.assertEqual(set(np.unique(wire[:, :, 3])), {0, 255})
        # Simulate browser canvas's 8-bit premultiplied RGB round trip.
        alpha = wire[:, :, 3:4].astype(float) / 255.0
        premultiplied = np.rint(wire[:, :, :3] * alpha)
        decoded = np.rint(premultiplied / np.maximum(alpha, 1 / 255)).astype(np.uint8)
        merged = np.where(alpha > 0, decoded, target)
        expected = np.where(composite[:, :, 3:4] > 0, composite[:, :, :3], target)
        np.testing.assert_array_equal(merged, expected)

    def test_unchanged_pixels_and_unauthorized_rgb_are_not_transmitted(self):
        target = np.full((6, 7, 3), 100, dtype=np.uint8)
        composite = cv2.cvtColor(target, cv2.COLOR_BGR2BGRA)
        composite[0, 0] = [255, 255, 255, 0]
        composite[2, 3] = [102, 104, 106, 1]
        result = binary_preblended_rgba(target, composite)
        self.assertEqual(int(np.count_nonzero(result[:, :, 3])), 1)
        np.testing.assert_array_equal(result[0, 0, :3], target[0, 0])
        self.assertEqual(result[2, 3, 3], 255)
        np.testing.assert_array_equal(result[2, 3, :3], composite[2, 3, :3])

    def test_png_roundtrip_at_odd_size_keeps_lossless_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = np.full((81, 101, 3), [131, 146, 173], dtype=np.uint8)
            edited = cv2.cvtColor(target, cv2.COLOR_BGR2BGRA)
            edited[:, :, 3] = 0
            edited[20:50, 20:50, :3] = [132, 147, 174]
            edited[20:50, 20:50, 3] = 1
            cv2.imwrite(str(root / "target.png"), target)
            cv2.imwrite(str(root / "final.png"), edited)
            payload = encode_preblended_result(root / "target.png", root / "final.png")
            result = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(result, binary_preblended_rgba(target, edited))
            self.assertEqual(result.shape, (81, 101, 4))

    def test_refuses_shape_or_channel_mismatch(self):
        target = np.zeros((81, 101, 3), dtype=np.uint8)
        for bad in (target, np.zeros((82, 102, 4), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                binary_preblended_rgba(target, bad)


if __name__ == "__main__":
    unittest.main()
