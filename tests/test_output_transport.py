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


class SmallFaceInputTest(unittest.TestCase):
    def test_preparation_preserves_source_and_outside_region(self):
        import tempfile
        from pathlib import Path
        from tools.retake_output_transport import prepare_small_face_inference
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=np.full((120,140,3),130,np.uint8)
            for x in range(30,100):
                image[20:100,x]=int(130+12*np.sin(x*1.4))
            path=root/"input.png"
            cv2.imwrite(str(path),image)
            original=path.read_bytes()
            bounds=dict(x=30,y=20,width=70,height=80)
            output,status=prepare_small_face_inference(path,bounds,7,root)
            self.assertEqual(status,"native-contrast")
            self.assertEqual(path.read_bytes(),original)
            result=cv2.imread(str(output))
            self.assertEqual(result.shape,image.shape)
            mask=np.zeros(image.shape[:2],bool);mask[20:100,30:100]=True
            self.assertTrue(np.array_equal(result[~mask],image[~mask]))
            self.assertGreater(np.count_nonzero(result!=image),20)
            self.assertLessEqual(np.max(np.abs(result.astype(int)-image.astype(int))),10)
            self.assertEqual(prepare_small_face_inference(path,bounds,0,root),(path,"original"))

    def test_flat_and_invalid_small_faces_do_not_invent_detail(self):
        import tempfile
        from pathlib import Path
        from tools.retake_output_transport import prepare_small_face_inference
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/"flat.png"
            cv2.imwrite(str(path),np.full((100,100,3),120,np.uint8))
            self.assertEqual(prepare_small_face_inference(path,dict(x=10,y=10,width=70,height=70),7,root),(path,"insufficient-evidence"))
            self.assertEqual(prepare_small_face_inference(path,dict(x=-10,y=10,width=70,height=70),7,root),(path,"invalid-bounds"))


class VisibleIdentityProbeTest(unittest.TestCase):
    def test_probe_matches_transport_and_never_scores_hidden_rgb(self):
        from tools.retake_output_transport import prepare_visible_identity_probe
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            target=np.full((81,101,3),120,np.uint8)
            composite=np.full((81,101,4),255,np.uint8)
            composite[:,:,3]=0
            composite[20:60,30:70,:3]=[100,130,160]
            composite[20:60,30:70,3]=1
            a,b=root/"target.png",root/"candidate.png"
            cv2.imwrite(str(a),target);cv2.imwrite(str(b),composite)
            path=prepare_visible_identity_probe(a,b,root)
            wire=cv2.imdecode(np.frombuffer(encode_preblended_result(a,b),np.uint8),cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(cv2.imread(str(path)),wire[:,:,:3])
            self.assertEqual(prepare_visible_identity_probe(a,b,root),path)
            composite[0,0,:3]=0
            cv2.imwrite(str(b),composite)
            self.assertEqual(prepare_visible_identity_probe(a,b,root),path)
            composite[30,40,:3]=[101,130,160]
            cv2.imwrite(str(b),composite)
            new_path=prepare_visible_identity_probe(a,b,root)
            self.assertNotEqual(new_path,path)
            np.testing.assert_array_equal(cv2.imread(str(path))[30,40],[100,130,160])
            self.assertEqual(prepare_visible_identity_probe(a,a,root),a)

    def test_mismatched_alpha_probe_fails_closed(self):
        from tools.retake_output_transport import prepare_visible_identity_probe
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            a,b=root/"target.png",root/"candidate.png"
            cv2.imwrite(str(a),np.zeros((80,80,3),np.uint8))
            cv2.imwrite(str(b),np.zeros((81,81,4),np.uint8))
            with self.assertRaises(ValueError):
                prepare_visible_identity_probe(a,b,root)
