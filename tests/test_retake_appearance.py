import unittest

import cv2
import numpy as np

from tools import facefusion_local_server as srv
from tools import retake_appearance as ra

SIZE = 300


def face_mesh(cx=150.0, cy=150.0, width=140.0):
    """478 points: lower-face contour, lips and cheek points where they belong."""
    points = np.tile(np.float32([[cx, cy]]), (478, 1))
    half = width / 2.0
    # Lower face: from under the nose round the jaw and back.
    for order, index in enumerate(ra.LOWER_FACE):
        theta = np.pi * order / (len(ra.LOWER_FACE) - 1)
        points[index] = (cx - half * 0.85 * np.cos(theta), cy + 15 + 70 * np.sin(theta))
    points[ra.LOWER_FACE[0]] = (cx, cy + 12)
    for order, index in enumerate(ra.LIPS):
        theta = 2 * np.pi * order / len(ra.LIPS)
        points[index] = (cx + 22 * np.cos(theta), cy + 40 + 8 * np.sin(theta))
    points[50] = (cx - 40, cy - 5)
    points[280] = (cx + 40, cy - 5)
    return points


def skin_with_beard(beard_l, stubble=False, seed=3):
    """A skin-toned face whose lower face is darker by `beard_l` (Lab L)."""
    rng = np.random.default_rng(seed)
    lab = np.zeros((SIZE, SIZE, 3), np.float32)
    lab[:] = (170, 140, 145)
    points = face_mesh()
    lower, _ = ra.face_regions(points, (SIZE, SIZE), 140)
    hair = rng.uniform(0, 1, (SIZE, SIZE)) < (0.5 if stubble else 1.0)
    lab[lower & hair, 0] -= beard_l
    lab[:, :, 0] += rng.normal(0, 1.5, (SIZE, SIZE))
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR), points


class BeardToneTest(unittest.TestCase):
    def test_profile_offset_reads_the_aligned_profile_layout(self):
        profile = np.full((682, 512, 3), (150, 170, 200), np.uint8)
        lower, cheeks = ra.profile_regions(profile.shape[:2])
        self.assertTrue(lower.any() and cheeks.any())
        self.assertFalse((lower & cheeks).any())
        profile[lower] = (60, 70, 85)
        offset = ra.profile_beard_offset(profile)
        self.assertLess(offset[0], -60)

    def test_a_light_swap_beard_is_darkened_towards_the_person(self):
        swap, points = skin_with_beard(20, stubble=True)
        alpha = np.full((SIZE, SIZE), 255, np.uint8)
        lab = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lower, cheeks = ra.face_regions(points, (SIZE, SIZE), 140)
        before = lab(swap)[lower].mean(0) - lab(swap)[cheeks].mean(0)
        out = ra.match_beard(swap, alpha, points, np.float32([-70, 0, 0]), 140)
        after = lab(out)[lower].mean(0) - lab(out)[cheeks].mean(0)
        self.assertLess(after[0], before[0] - 15)
        # Hair-like pixels take most of the change; bare skin barely moves.
        hairy = lower & (lab(swap)[:, :, 0] < lab(swap)[cheeks][:, 0].mean() - 10)
        bare = lower & ~hairy
        darkened = lambda mask: float((lab(swap)[mask][:, 0] - lab(out)[mask][:, 0]).mean())
        self.assertGreater(darkened(hairy), darkened(bare) * 1.5)
        # The upper face is untouched.
        self.assertTrue(np.array_equal(out[:100], swap[:100]))

    def test_never_lightens_and_stays_inside_the_matte(self):
        dark, points = skin_with_beard(80)
        alpha = np.full((SIZE, SIZE), 255, np.uint8)
        same = ra.match_beard(dark, alpha, points, np.float32([-40, 0, 0]), 140)
        lab = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lower, _ = ra.face_regions(points, (SIZE, SIZE), 140)
        self.assertGreaterEqual(float(lab(dark)[lower][:, 0].mean()), float(lab(same)[lower][:, 0].mean()) - 0.5)
        light, points = skin_with_beard(15)
        outside = ra.match_beard(light, np.zeros((SIZE, SIZE), np.uint8), points, np.float32([-70, 0, 0]), 140)
        self.assertTrue(np.array_equal(outside, light))

    def test_target_mesh_metadata_is_optional_and_validated(self):
        base = {
            "referenceWeights": [1] * srv.MAX_REFERENCES,
            "faceBounds": {"x": 10, "y": 10, "width": 200, "height": 220},
            "targetSize": {"width": 320, "height": 320},
        }
        self.assertIsNone(srv._validate_metadata(base, srv.MAX_REFERENCES)["targetLandmarks"])
        good = face_mesh().tolist()
        self.assertEqual(len(srv._validate_metadata({**base, "targetLandmarks": good}, srv.MAX_REFERENCES)["targetLandmarks"]), 478)
        for bad in ([], good[:10], [[5000, 5000]] * 478, "x"):
            with self.subTest(bad=str(bad)[:20]), self.assertRaisesRegex(ValueError, "Hedef yüz ağı"):
                srv._validate_metadata({**base, "targetLandmarks": bad}, srv.MAX_REFERENCES)

    def test_person_beard_needs_a_mesh_and_a_profile(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.png"
            image = np.full((682, 512, 3), (150, 170, 200), np.uint8)
            lower, _ = ra.profile_regions(image.shape[:2])
            image[lower] = (60, 70, 85)
            cv2.imwrite(str(profile), image)
            self.assertIsNone(srv.person_beard(profile, None))
            beard = srv.person_beard(profile, face_mesh().tolist())
            self.assertLess(beard["offset"][0], -60)
            self.assertIsNone(srv.person_beard(Path(directory) / "missing.png", face_mesh().tolist()))


if __name__ == "__main__":
    unittest.main()
