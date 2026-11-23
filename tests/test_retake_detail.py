import json
import unittest

import cv2
import numpy as np

from tools import facefusion_local_server as srv
from tools.retake_detail import (
    FACE_OVAL,
    INNER_MOUTH,
    LEFT_BROW,
    MESH_POINTS,
    RIGHT_BROW,
    choose_and_transplant,
    match_broad_light,
    mouth_is_open,
    reference_skin,
    transplant_reference_face,
    transplant_region,
    validate_mesh,
    warp_piecewise_affine,
)

SIZE = 240


def synthetic_mesh(center=(120.0, 125.0), radius=(70.0, 90.0), seed=3):
    """A face-like 478-point mesh: oval contour, brows, eyes, mouth, interior points."""
    rng = np.random.default_rng(seed)
    cx, cy = center
    rx, ry = radius
    points = np.zeros((MESH_POINTS, 2), np.float32)
    # Interior points on a jittered grid inside the oval.
    angle = rng.uniform(0, 2 * np.pi, MESH_POINTS)
    distance = np.sqrt(rng.uniform(0.0, 0.8, MESH_POINTS))
    points[:, 0] = cx + rx * distance * np.cos(angle)
    points[:, 1] = cy + ry * distance * np.sin(angle)
    for order, index in enumerate(FACE_OVAL):
        theta = -np.pi / 2 + 2 * np.pi * order / len(FACE_OVAL)
        points[index] = (cx + rx * np.cos(theta), cy + ry * np.sin(theta))
    for order, index in enumerate(LEFT_BROW):
        points[index] = (cx - 50 + order * 9, cy - 40)
    for order, index in enumerate(RIGHT_BROW):
        points[index] = (cx + 14 + order * 9, cy - 40)
    points[33] = (cx - 30, cy - 25)
    points[263] = (cx + 30, cy - 25)
    for order, index in enumerate(INNER_MOUTH):
        theta = 2 * np.pi * order / len(INNER_MOUTH)
        points[index] = (cx + 18 * np.cos(theta), cy + 45 + 6 * np.sin(theta))
    return points


def textured(seed, size=SIZE):
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.uniform(0, 255, (size, size, 3)).astype(np.float32), (0, 0), 2.0)
    return np.clip(noise, 0, 255).astype(np.uint8)


class MeshWarpTest(unittest.TestCase):
    def test_a_shifted_mesh_moves_the_image_with_it(self):
        target_points = synthetic_mesh()
        source_points = target_points + np.float32([6, -4])
        source = textured(1)
        warped, coverage = warp_piecewise_affine(source, source_points, target_points, (SIZE, SIZE))
        self.assertEqual(warped.shape, (SIZE, SIZE, 3))
        inner = cv2.erode((coverage > 0).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        self.assertGreater(inner.sum(), 10000)
        expected = np.roll(np.roll(source.astype(np.float32), 4, axis=0), -6, axis=1)
        self.assertLess(float(np.abs(warped - expected)[inner].mean()), 1.0)
        # Nothing outside the face mesh.
        self.assertEqual(float(coverage[2, 2]), 0.0)

    def test_mesh_validation_rejects_wrong_shapes_and_far_points(self):
        good = synthetic_mesh().tolist()
        self.assertEqual(validate_mesh(good, SIZE, SIZE, 0).shape, (MESH_POINTS, 2))
        for bad in (good[:-1], [[1, 2, 3]] * MESH_POINTS, [[float("nan"), 1]] * MESH_POINTS, "x", None):
            with self.subTest(bad=str(bad)[:20]), self.assertRaises(ValueError):
                validate_mesh(bad, SIZE, SIZE, 0)
        far = [list(point) for point in good]
        far[5] = [SIZE * 3, 10]
        with self.assertRaisesRegex(ValueError, "dışında"):
            validate_mesh(far, SIZE, SIZE, 10)


class TransplantTest(unittest.TestCase):
    def setUp(self):
        self.points = synthetic_mesh()
        self.target = textured(5)
        self.swapped = np.clip(self.target.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        self.reference = textured(9)

    def test_the_face_comes_from_the_real_photo(self):
        result = transplant_reference_face(
            self.swapped, self.target, self.reference, self.points, self.points, face_width=140
        )
        self.assertEqual(result.dtype, np.uint8)
        # The fine detail is the real photo's (its broad light follows the swap).
        cheek = (slice(110, 135), slice(70, 95))
        detail = lambda image: image[cheek].astype(np.float32) - cv2.GaussianBlur(image, (0, 0), 8)[cheek]
        self.assertGreater(np.corrcoef(detail(result).ravel(), detail(self.reference).ravel())[0, 1], 0.9)
        # Outside the oval and above the brows the swap is untouched.
        self.assertTrue(np.array_equal(result[:20], self.swapped[:20]))
        self.assertTrue(np.array_equal(result[:, :30], self.swapped[:, :30]))

    def test_open_mouth_and_what_the_swap_kept_stay_untouched(self):
        swapped = self.swapped.copy()
        # A hand over the cheek: the swap model's mask kept the photo there.
        swapped[100:150, 140:175] = self.target[100:150, 140:175]
        result = transplant_reference_face(
            swapped, self.target, self.reference, self.points, self.points, face_width=140
        )
        hand = (slice(110, 140), slice(150, 165))
        self.assertLess(np.abs(result[hand].astype(int) - self.target[hand].astype(int)).mean(), 1.0)
        mouth = (slice(168, 172), slice(115, 125))
        self.assertTrue(np.array_equal(result[mouth], swapped[mouth]))

    def test_closed_lips_come_from_the_photo_an_open_mouth_does_not(self):
        open_mouth = synthetic_mesh()
        self.assertTrue(mouth_is_open(open_mouth))
        closed = open_mouth.copy()
        closed[13] = closed[14] = (120.0, 170.0)
        self.assertFalse(mouth_is_open(closed))
        region = transplant_region(open_mouth, (SIZE, SIZE), 140, keep_mouth=False)
        self.assertGreater(float(region[170, 120]), 0.99)

    def test_region_fades_in_below_the_forehead(self):
        region = transplant_region(self.points, (SIZE, SIZE), face_width=140)
        self.assertEqual(float(region[45, 120]), 0.0)        # upper forehead
        self.assertGreater(float(region[125, 120]), 0.99)     # mid face
        self.assertEqual(float(region[170, 120]), 0.0)       # open mouth


class RealFaceRequestTest(unittest.TestCase):
    base = {
        "referenceWeights": [1] * srv.MAX_REFERENCES,
        "faceBounds": {"x": 10, "y": 10, "width": 200, "height": 220},
        "targetSize": {"width": 320, "height": 320},
    }

    def test_real_face_metadata_is_optional_and_validated(self):
        self.assertIsNone(srv._validate_metadata(self.base, srv.MAX_REFERENCES)["realFace"])
        mesh = synthetic_mesh().tolist()
        many = {"targetLandmarks": mesh, "references": [{"landmarks": mesh}] * 3}
        validated = srv._validate_metadata({**self.base, "realFace": many}, srv.MAX_REFERENCES)["realFace"]
        self.assertEqual(len(validated["targetLandmarks"]), MESH_POINTS)
        self.assertEqual(len(validated["references"]), 3)
        lower = srv._validate_metadata(
            {**self.base, "realFace": {**many, "mode": "lower"}}, srv.MAX_REFERENCES
        )["realFace"]
        self.assertEqual(lower["mode"], "lower")
        self.assertEqual(validated["mode"], "full")
        with self.assertRaisesRegex(ValueError, "aktarım modu"):
            srv._validate_metadata({**self.base, "realFace": {**many, "mode": "unknown"}}, srv.MAX_REFERENCES)
        # The single-candidate form is still understood.
        single = {"referenceLandmarks": mesh, "targetLandmarks": mesh}
        validated = srv._validate_metadata({**self.base, "realFace": single}, srv.MAX_REFERENCES)["realFace"]
        self.assertEqual(len(validated["references"]), 1)
        for bad in ([], {"targetLandmarks": mesh}, {"referenceLandmarks": mesh, "targetLandmarks": mesh[:10]},
                    {"referenceLandmarks": [[-50, 1]] * MESH_POINTS, "targetLandmarks": mesh},
                    {"targetLandmarks": mesh, "references": []},
                    {"targetLandmarks": mesh, "references": [{"landmarks": mesh}] * 4},
                    {"targetLandmarks": mesh, "references": [mesh]}):
            with self.subTest(bad=str(bad)[:40]), self.assertRaisesRegex(ValueError, "Gerçek fotoğraf"):
                srv._validate_metadata({**self.base, "realFace": bad}, srv.MAX_REFERENCES)

    def multipart(self, real_face=None, real_photos=()):
        boundary = "b0undary"
        parts = []

        def add(name, payload, filename=None):
            head = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
            if filename:
                head += f'; filename="{filename}"\r\nContent-Type: image/png'
            parts.append(head.encode() + b"\r\n\r\n" + payload + b"\r\n")

        png = cv2.imencode(".png", self.reference_image())[1].tobytes()
        add("target", png, "t.png")
        add("appearance_profile", png, "a.png")
        for index in range(srv.MAX_REFERENCES):
            add("reference", png, f"r{index}.png")
        for index, photo in enumerate(real_photos):
            add("real_reference", photo, f"real{index}.png")
        metadata = dict(self.base)
        if real_face is not None:
            metadata["realFace"] = real_face
        add("metadata", json.dumps(metadata).encode())
        return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"

    @staticmethod
    def reference_image():
        return textured(2, 64)

    def test_the_real_photos_and_their_meshes_travel_together(self):
        mesh = synthetic_mesh().tolist()
        real = {"targetLandmarks": mesh, "references": [{"landmarks": mesh}, {"landmarks": mesh}]}
        photos = [cv2.imencode(".png", textured(seed))[1].tobytes() for seed in (4, 6)]
        body, content_type = self.multipart(real, photos)
        self.assertEqual(srv.parse_multipart(content_type, body)[-1]["realReferenceBytes"], photos)
        body, content_type = self.multipart(None)
        self.assertIsNone(srv.parse_multipart(content_type, body)[-1]["realReferenceBytes"])
        body, content_type = self.multipart(real, photos[:1])
        with self.assertRaisesRegex(ValueError, "fotoğrafı eksik"):
            srv.parse_multipart(content_type, body)
        body, content_type = self.multipart(real, photos + photos)
        with self.assertRaisesRegex(ValueError, "En fazla"):
            srv.parse_multipart(content_type, body)

    def test_an_unreadable_real_photo_falls_back_to_the_ai_swap(self):
        mesh = synthetic_mesh()
        target = textured(5)
        swapped = target.copy()
        swapped[60:180, 60:180] = 200
        references = [{"landmarks": mesh}]
        result, used = srv.apply_real_face(
            swapped, target, {"images": [b"not an image"], "references": references, "targetLandmarks": mesh}, 140
        )
        self.assertFalse(used)
        self.assertIs(result, swapped)
        # The person's photo: same features as the face, other light.
        photo = cv2.imencode(".png", np.clip(target.astype(np.int16) - 25, 0, 255).astype(np.uint8))[1].tobytes()
        # A broken candidate is skipped; the readable one is used.
        result, used = srv.apply_real_face(
            swapped, target,
            {"images": [b"broken", photo], "references": references * 2, "targetLandmarks": mesh}, 140
        )
        self.assertTrue(used)
        self.assertFalse(np.array_equal(result, swapped))
        # A photo whose features do not line up with the face is refused.
        stranger = cv2.imencode(".png", textured(9))[1].tobytes()
        result, used = srv.apply_real_face(
            swapped, target, {"images": [stranger], "references": references, "targetLandmarks": mesh}, 140
        )
        self.assertFalse(used)


class CandidateChoiceTest(unittest.TestCase):
    def test_the_candidate_whose_features_line_up_wins(self):
        points = synthetic_mesh()
        target = textured(5)
        swapped = np.clip(target.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        aligned = target.copy()                                    # same features as the photo
        misaligned = np.roll(target, 9, axis=1)                    # features shifted sideways
        result, chosen, fits = choose_and_transplant(
            swapped, target, [(misaligned, points), (aligned, points)], points, 140
        )
        self.assertEqual(chosen, 1)
        self.assertGreater(fits[1]["structure"], fits[0]["structure"] + 0.3)
        # Its detail is the photo's; its broad light is the swap's.
        cheek = (slice(115, 130), slice(75, 90))
        detail = lambda image: image[cheek].astype(np.float32) - cv2.GaussianBlur(image, (0, 0), 8)[cheek]
        self.assertGreater(np.corrcoef(detail(result).ravel(), detail(aligned).ravel())[0, 1], 0.9)
        self.assertLess(abs(float(result[cheek].mean()) - float(swapped[cheek].mean())), 6.0)

    def test_light_and_colour_are_matched_but_detail_is_kept(self):
        points = synthetic_mesh()
        target = textured(5)
        swapped = np.clip(target.astype(np.float32) * (1.15, 1.0, 0.85), 0, 255).astype(np.uint8)  # warm scene
        cool_photo = np.clip(target.astype(np.float32) * (0.8, 0.85, 1.1) - 20, 0, 255).astype(np.uint8)
        warped, coverage = warp_piecewise_affine(cool_photo, points, points, (SIZE, SIZE))
        weight = transplant_region(points, (SIZE, SIZE), 140) * coverage
        matched = match_broad_light(warped, swapped, weight, 140)
        inside = weight > 0.99
        for channel in range(3):
            self.assertLess(abs(float(matched[:, :, channel][inside].mean()) - float(swapped[:, :, channel][inside].mean())), 3.0)
        self.assertGreater(np.corrcoef(matched[:, :, 1][inside], cool_photo[:, :, 1][inside])[0, 1], 0.95)

    def test_downscaled_photo_does_not_alias(self):
        points = synthetic_mesh()
        rng = np.random.default_rng(1)
        stubble = (rng.uniform(0, 1, (SIZE * 6, SIZE * 6)) > 0.5).astype(np.uint8) * 255   # pixel-fine detail
        big = np.dstack([stubble] * 3)
        warped, coverage = warp_piecewise_affine(big, points * 6, points, (SIZE, SIZE))
        inner = cv2.erode((coverage > 0).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        # Area-averaged it is a flat grey; sampled it would be salt and pepper.
        self.assertLess(float(warped[:, :, 0][inner].std()), 30.0)

    def test_forehead_comes_from_the_photo_only_where_it_is_skin(self):
        points = synthetic_mesh()
        reference = np.full((SIZE, SIZE, 3), (120, 150, 200), np.uint8)    # skin
        reference[38:60, 60:120] = (30, 30, 35)                               # a dark fringe, left forehead
        warped, coverage = warp_piecewise_affine(reference, points, points, (SIZE, SIZE))
        skin = reference_skin(warped, points, coverage, 140)
        self.assertLess(float(skin[50, 90]), 0.05)
        self.assertGreater(float(skin[50, 150]), 0.95)
        region = transplant_region(points, (SIZE, SIZE), 140, skin)
        self.assertLess(float(region[50, 90]), 0.05)
        self.assertGreater(float(region[52, 150]), 0.6)
        self.assertEqual(float(transplant_region(points, (SIZE, SIZE), 140)[52, 150]), 0.0)


if __name__ == "__main__":
    unittest.main()
