"""Mesh-guided finishing of eyes, brows, lips, nose tip and catchlights."""
from __future__ import annotations

import math
import unittest

try:
    import cv2
    import numpy as np
except ImportError:  # OpenCV/NumPy live in the FaceFusion environment.
    cv2 = None

if cv2 is not None:
    from tools.fine_features import (
        FEATURES, IRISES, accept_fine_refinement, describe_report, feature_mask, refine_fine_features
    )

BOUNDS = {"x": 90.0, "y": 100.0, "width": 220.0, "height": 260.0}


def ellipse(indices, centre, rx, ry, points):
    for k, index in enumerate(indices):
        theta = math.pi - 2 * math.pi * k / len(indices)
        points[index] = (centre[0] + rx * math.cos(theta), centre[1] + ry * math.sin(theta))


def synthetic_mesh(eye_ry=12.0):
    points = np.full((478, 2), (200.0, 220.0), np.float32)
    ellipse(FEATURES["eye_r"], (140, 170), 28, eye_ry, points)
    ellipse(FEATURES["eye_l"], (260, 170), 28, eye_ry, points)
    for eye, centre in (("eye_r", (140, 170)), ("eye_l", (260, 170))):
        iris, ring, _ = IRISES[eye]
        points[iris] = centre
        for k, index in enumerate(ring):
            angle = k * math.pi / 2
            points[index] = (centre[0] + 8 * math.cos(angle), centre[1] + 8 * math.sin(angle))
    for name, x0 in (("brow_r", 108), ("brow_l", 228)):
        indices = FEATURES[name]
        for k, index in enumerate(indices):
            row, column = divmod(k, 5)
            points[index] = (x0 + column * 16, 138 + row * 8)
    ellipse(FEATURES["lips"], (200, 290), 45, 16, points)
    ellipse(FEATURES["mouth_inner"], (200, 290), 38, 2, points)
    ellipse(FEATURES["nose_tip"], (200, 240), 15, 10, points)
    return points


def synthetic_photo(seed=3):
    rng = np.random.default_rng(seed)
    image = np.full((420, 400, 3), (150, 170, 205), np.float32)
    image += rng.normal(0, 4.0, image.shape)
    for centre in ((140, 170), (260, 170)):
        cv2.ellipse(image, centre, (27, 11), 0, 0, 360, (235, 238, 240), -1)
        cv2.circle(image, centre, 8, (40, 70, 110), -1)
        cv2.circle(image, centre, 3, (10, 10, 10), -1)
        for k in range(14):  # lashes
            x = centre[0] - 26 + 4 * k
            cv2.line(image, (x, centre[1] - 11), (x + 2, centre[1] - 19), (25, 25, 30), 1)
        cv2.circle(image, (centre[0] + 2, centre[1] - 3), 2, (252, 252, 252), -1)  # catchlight
    for x0 in (108, 228):  # brow hair
        for k in range(40):
            x = x0 + int(rng.integers(0, 64))
            y = 138 + int(rng.integers(0, 9))
            cv2.line(image, (x, y), (x + 3, y - 2), (40, 50, 60), 1)
    cv2.ellipse(image, (200, 290), (44, 15), 0, 0, 360, (95, 90, 175), -1)
    for k in range(18):  # lip creases
        x = 160 + 5 * k
        cv2.line(image, (x, 279), (x, 301), (80, 75, 150), 1)
    for k in range(10):  # nostril rim texture
        cv2.line(image, (188 + 3 * k, 236), (188 + 3 * k, 246), (110, 120, 160), 1)
    return np.clip(image, 0, 255).astype(np.uint8)


def blurred_in(image, points, names, sigma):
    out = image.copy()
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    for name in names:
        mask, _ = feature_mask(points, name, image.shape[:2])
        weight = np.clip(mask * 1.6, 0, 1)[:, :, None]
        out = (out * (1 - weight) + blurred * weight).astype(np.uint8)
    return out


def translate(image, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)


@unittest.skipIf(cv2 is None, "OpenCV/NumPy yüklü değil")
class FineFeatureTest(unittest.TestCase):
    def setUp(self):
        self.photo = synthetic_photo()
        self.mesh = synthetic_mesh()

    def test_soft_eyes_are_restored_where_they_moved(self):
        soft = blurred_in(self.photo, self.mesh, ("eye_r", "eye_l"), 2.2)
        result = translate(soft, 3, -2)  # a head fit moved the features
        refined, report = refine_fine_features(self.photo, result, self.mesh.tolist(), BOUNDS)
        self.assertEqual(report["status"], "applied")
        for eye in ("eye_r", "eye_l"):
            item = report["features"][eye]
            self.assertEqual(item["action"], "restore")
            self.assertLess(item["before"], 0.9)
            self.assertGreater(item["after"], item["before"] + 0.1)
            self.assertLess(abs(item["after"] - 1.0), abs(item["before"] - 1.0))
            self.assertAlmostEqual(item["shift"][0], 3, delta=0.7)
            self.assertAlmostEqual(item["shift"][1], -2, delta=0.7)
        self.assertEqual(report["features"]["lips"]["action"], "keep", "untouched features stay as they are")
        self.assertGreater(report["quality_after"], report["quality_before"])
        self.assertIn("eye_r:", describe_report(report))
        # Outside the eye masks the result is unchanged.
        np.testing.assert_array_equal(refined[380:, :], result[380:, :])

    def test_lost_catchlight_comes_back_from_the_photo(self):
        result = self.photo.copy()
        for centre in ((140, 170), (260, 170)):
            cv2.circle(result, (centre[0] + 2, centre[1] - 3), 3, (40, 70, 110), -1)
        refined, report = refine_fine_features(self.photo, result, self.mesh.tolist(), BOUNDS)
        self.assertEqual(report["catchlights"], {"eye_r": "restored", "eye_l": "restored"})
        self.assertEqual(report["catchlights_restored"], 2)
        self.assertGreater(int(refined[167, 142].mean()), 200)
        self.assertLess(int(result[167, 142].mean()), 120)

    def test_grey_eye_whites_are_pulled_towards_the_photo(self):
        result = self.photo.copy()
        for centre in ((140, 170), (260, 170)):
            grey = np.zeros(result.shape[:2], np.uint8)
            cv2.ellipse(grey, centre, (25, 9), 0, 0, 360, 1, -1)
            cv2.circle(grey, centre, 9, 0, -1)
            result[grey > 0] = (175, 185, 190)  # dull, slightly yellow-grey sclera
        refined, report = refine_fine_features(self.photo, result, self.mesh.tolist(), BOUNDS)
        self.assertEqual(report["sclera"], {"eye_r": "corrected", "eye_l": "corrected"})
        self.assertEqual(report["sclera_corrected"], 2)
        before = float(result[170, 125].mean())
        after = float(refined[170, 125].mean())
        target = float(self.photo[170, 125].mean())
        self.assertLess(abs(after - target), abs(before - target))
        # The iris itself is not recoloured.
        np.testing.assert_allclose(refined[170, 140].astype(float), result[170, 140].astype(float), atol=2)

    def test_present_catchlight_and_matching_detail_change_nothing(self):
        refined, report = refine_fine_features(self.photo, self.photo.copy(), self.mesh.tolist(), BOUNDS)
        self.assertEqual(report["status"], "unchanged")
        self.assertEqual(set(report["catchlights"].values()), {"present"})
        self.assertEqual(set(report["sclera"].values()), {"matched"})
        self.assertTrue(all(item["action"] == "keep" for item in report["features"].values()))
        np.testing.assert_array_equal(refined, self.photo)

    def test_ringing_lips_are_softened(self):
        result = self.photo.astype(np.float32)
        mask, _ = feature_mask(self.mesh, "lips", self.photo.shape[:2])
        sharp = result + 2.5 * (result - cv2.GaussianBlur(result, (0, 0), 1.0))
        result = np.clip(result + (sharp - result) * mask[:, :, None], 0, 255).astype(np.uint8)
        _, report = refine_fine_features(self.photo, result, self.mesh.tolist(), BOUNDS)
        lips = report["features"]["lips"]
        self.assertEqual(lips["action"], "soften")
        self.assertLess(lips["after"], lips["before"])

    def test_catchlight_never_lands_outside_a_dark_iris(self):
        result = self.photo.copy()
        for centre in ((140, 170), (260, 170)):
            # The iris is gone (e.g. a bright, washed-out render): no dot on white.
            cv2.circle(result, centre, 9, (155, 160, 165), -1)
        refined, report = refine_fine_features(self.photo, result, self.mesh.tolist(), BOUNDS)
        self.assertEqual(set(report["catchlights"].values()), {"misaligned"})

    def test_closed_eyes_keep_their_lids(self):
        mesh = synthetic_mesh(eye_ry=3.0)
        result = self.photo.copy()
        _, report = refine_fine_features(self.photo, result, mesh.tolist(), BOUNDS)
        self.assertEqual(report["catchlights"], {"eye_r": "closed", "eye_l": "closed"})

    def test_matte_hidden_pixels_stay_the_photo(self):
        soft = blurred_in(self.photo, self.mesh, ("eye_r", "eye_l"), 2.2)
        alpha = np.full(self.photo.shape[:2], 255, np.uint8)
        alpha[:, :200] = 0  # the right eye is outside the matte
        result = np.dstack([soft, alpha])
        refined, report = refine_fine_features(self.photo, result, self.mesh.tolist(), BOUNDS)
        self.assertEqual(refined.shape[2], 4)
        np.testing.assert_array_equal(refined[:, :, 3], alpha)
        np.testing.assert_array_equal(refined[:, :200, :3], soft[:, :200])
        self.assertEqual(report["features"]["eye_l"]["action"], "restore")
        self.assertNotIn("eye_r", report["features"])
        self.assertEqual(report["catchlights"]["eye_r"], "unmeasured")

    def test_missing_or_broken_mesh_is_a_no_op(self):
        for mesh in (None, [[0, 0]] * 10, [[float("nan"), 0]] * 478, [[5000, 5000]] * 478):
            refined, report = refine_fine_features(self.photo, self.photo, mesh, BOUNDS)
            self.assertIs(refined, self.photo)
            self.assertEqual(report["status"], "no-mesh")
        _, report = refine_fine_features(self.photo, self.photo[:100], self.mesh.tolist(), BOUNDS)
        self.assertEqual(report["status"], "unmeasured")

    def test_acceptance_needs_identity_detail_and_a_real_gain(self):
        report = {"status": "applied", "quality_before": 0.80, "quality_after": 0.86, "catchlights_restored": 0}
        self.assertEqual(accept_fine_refinement(report, 0.812, 0.814, True), (True, "applied"))
        self.assertEqual(accept_fine_refinement(report, 0.80, 0.814, True)[1], "identity-loss")
        self.assertEqual(accept_fine_refinement(report, None, 0.814, True)[1], "identity-unmeasured")
        self.assertEqual(accept_fine_refinement(report, 0.82, 0.814, False)[1], "detail-gate")
        flat = {**report, "quality_after": 0.805}
        self.assertEqual(accept_fine_refinement(flat, 0.82, 0.814, True)[1], "no-gain")
        self.assertTrue(accept_fine_refinement({**flat, "catchlights_restored": 1}, 0.82, 0.814, True)[0])
        self.assertTrue(accept_fine_refinement({**flat, "sclera_corrected": 2}, 0.82, 0.814, True)[0])
        self.assertEqual(accept_fine_refinement({"status": "no-mesh"}, 0.82, 0.814, True), (False, "no-mesh"))


if __name__ == "__main__":
    unittest.main()


class FineDeliveryGuardTest(unittest.TestCase):
    def test_independent_eye_regression_cannot_hide_in_average_gain(self):
        from tools.fine_features import fine_regions_preserved,accept_fine_refinement
        report={"status":"applied","quality_before":.60,"quality_after":.85,
                "catchlights_restored":1,
                "features":{"eye_r":{"before":.95,"after":.4},"eye_l":{"before":.4,"after":.95}}}
        self.assertFalse(fine_regions_preserved(report))
        self.assertEqual(accept_fine_refinement(report,.84,.84,True),(False,"region-regression"))
        report["features"]["eye_r"]["after"]=.92
        self.assertTrue(accept_fine_refinement(report,.84,.84,True)[0])

    def test_rejected_finish_reports_no_restored_lights_or_changed_ratios(self):
        from tools.fine_features import delivered_fine_report
        proposal={"features":{"eye_r":{"before":.5,"after":.9,"action":"restore"}},
                  "catchlights":{"eye_r":"restored","eye_l":"present"},
                  "sclera":{"eye_r":"corrected"},"catchlights_restored":1,"sclera_corrected":1}
        actual=delivered_fine_report(proposal,False)
        self.assertEqual(actual["features"]["eye_r"]["after"],.5)
        self.assertEqual(actual["catchlights_restored"],0)
        self.assertEqual(actual["catchlights"]["eye_r"],"not-applied")
        self.assertEqual(actual["sclera_corrected"],0)
        self.assertEqual(proposal["features"]["eye_r"]["after"],.9)
        self.assertIs(delivered_fine_report(proposal,True),proposal)
