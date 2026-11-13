import unittest

try:
    import cv2
    import numpy as np
    from tools import retake_compositing as rc
except ImportError:  # OpenCV/NumPy live in the FaceFusion environment.
    cv2 = None

HEIGHT, WIDTH = 512, 392
FACE = {"x": 96.0, "y": 116.0, "width": 200.0, "height": 270.0}


def synthetic_swap(tone=(0.82, 0.86, 0.95), offset=-6.0):
    """A grainy 'photo' of a skin-coloured head and a smoother, darker face
    pasted into it with a narrow feather, as a face swapper would."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    background = np.stack([90 + xx * 0.1, 110 + yy * 0.08, 120 + xx * 0.06], 2)
    head = (((xx - 196) / 115) ** 2 + ((yy - 250) / 155) ** 2) < 1
    target = background.copy()
    target[head] = np.array([120, 150, 205], np.float32) * (0.8 + 0.25 * xx[head] / WIDTH)[:, None]
    target[190:200, 125:270] *= 0.5
    target = np.clip(target + rng.normal(0, 3.0, target.shape), 0, 255).astype(np.uint8)

    raw = cv2.GaussianBlur(target, (0, 0), 1.5).astype(np.float32) * np.array(tone, np.float32) + offset
    region = ((((xx - 196) / 85) ** 2 + ((yy - 260) / 112) ** 2) < 1).astype(np.float32)
    opacity = cv2.GaussianBlur(region, (0, 0), 2.0)[:, :, None]
    swapped = np.clip(target * (1 - opacity) + raw * opacity, 0, 255).astype(np.uint8)
    return target, swapped


def seam_strength(image, target):
    """99.9th percentile gradient of the smoothed change below the eye line
    (plain skin only) — a visible border shows up as a ridge here."""
    change = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) - \
        cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    change = cv2.GaussianBlur(change, (0, 0), 3)[225:]
    return float(np.percentile(np.hypot(
        cv2.Sobel(change, cv2.CV_32F, 1, 0), cv2.Sobel(change, cv2.CV_32F, 0, 1)
    ), 99.9))


@unittest.skipIf(cv2 is None, "OpenCV/NumPy yüklü değil")
class SeamlessCompositeTest(unittest.TestCase):
    def test_tone_mismatch_no_longer_leaves_a_border(self):
        target, swapped = synthetic_swap()
        composite, alpha = rc.seamless_face_composite(target, swapped, FACE)
        self.assertLess(seam_strength(composite, target), seam_strength(swapped, target) * 0.5)
        # The face interior takes on the surrounding skin tone.
        centre = (slice(250, 280), slice(180, 210))
        before = np.abs(swapped[centre].astype(np.float32) - target[centre]).mean()
        after = np.abs(composite[centre].astype(np.float32) - target[centre]).mean()
        self.assertLess(after, before * 0.4)

    def test_pixels_outside_the_matte_are_byte_identical(self):
        target, swapped = synthetic_swap()
        composite, alpha = rc.seamless_face_composite(target, swapped, FACE)
        self.assertTrue((composite[alpha == 0] == target[alpha == 0]).all())
        self.assertEqual(int(alpha[0].max()), 0)
        self.assertEqual(int(alpha[:, 0].max()), 0)
        self.assertEqual(int(alpha[250, 196]), 255)
        self.assertLess(float((alpha > 0).mean()), 0.45)

    def test_features_inside_the_face_are_kept(self):
        target, swapped = synthetic_swap()
        # Give the swap a feature the target does not have.
        cv2.circle(swapped, (196, 300), 14, (40, 40, 60), -1)
        composite, _ = rc.seamless_face_composite(target, swapped, FACE)
        inside = composite[300, 196].astype(np.float32)
        beside = composite[300, 236].astype(np.float32)
        self.assertGreater(float(np.abs(inside - beside).max()), 60)

    def test_unchanged_swap_returns_the_target_with_an_empty_matte(self):
        target, _ = synthetic_swap()
        composite, alpha = rc.seamless_face_composite(target, target.copy(), FACE)
        self.assertTrue((composite == target).all())
        self.assertFalse(alpha.any())

    def test_soft_swap_is_brought_up_to_the_photo_sharpness(self):
        target, swapped = synthetic_swap(tone=(1.0, 1.0, 1.0), offset=-3.0)
        composite, alpha = rc.seamless_face_composite(target, swapped, FACE)
        core = (alpha == 255).astype(np.float32)
        core[:, :] = 0
        core[230:300, 150:240] = 1
        luma = lambda image: rc._luma(image.astype(np.float32))
        self.assertGreater(
            rc._detail_energy(luma(composite), core),
            rc._detail_energy(luma(swapped), core) * 1.3
        )


@unittest.skipIf(cv2 is None, "OpenCV/NumPy yüklü değil")
class HeadFitTest(unittest.TestCase):
    def head_extent(self, image, background_free):
        mask = np.abs(image.astype(np.float32) - background_free.astype(np.float32)).sum(2) > 60
        columns = np.where(mask[250])[0]
        rows = np.where(mask[:, 196])[0]
        return int(columns.max() - columns.min()), int(rows.max() - rows.min())

    def make(self):
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        background = np.stack([90 + xx * 0.1, 110 + yy * 0.08, 120 + xx * 0.06], 2).astype(np.uint8)
        image = background.copy()
        head = (((xx - 196) / 115) ** 2 + ((yy - 250) / 155) ** 2) < 1
        image[head] = (120, 150, 205)
        return image, background

    def test_head_gets_longer_and_narrower_without_touching_the_crop_border(self):
        image, background = self.make()
        alpha = np.zeros((HEIGHT, WIDTH), np.uint8)
        warped, out_alpha = rc.fit_head_geometry(image, alpha, FACE, 0.95, 1.07)
        width_before, height_before = self.head_extent(image, background)
        width_after, height_after = self.head_extent(warped, background)
        self.assertAlmostEqual(width_after / width_before, 0.95, delta=0.025)
        self.assertAlmostEqual(height_after / height_before, 1.07, delta=0.025)
        border = np.concatenate([out_alpha[0], out_alpha[-1], out_alpha[:, 0], out_alpha[:, -1]])
        self.assertEqual(int(border.max()), 0)
        self.assertTrue((warped[out_alpha == 0] == image[out_alpha == 0]).all())

    def test_head_can_also_get_shorter_and_wider(self):
        image, background = self.make()
        warped, _ = rc.fit_head_geometry(image, np.zeros((HEIGHT, WIDTH), np.uint8), FACE, 1.06, 0.94)
        width_before, height_before = self.head_extent(image, background)
        width_after, height_after = self.head_extent(warped, background)
        self.assertGreater(width_after, width_before)
        self.assertLess(height_after, height_before)

    def test_warp_never_folds_over(self):
        for scale_x, scale_y in ((rc.HEAD_SCALE_MIN, rc.HEAD_SCALE_MAX), (rc.HEAD_SCALE_MAX, rc.HEAD_SCALE_MIN)):
            map_x, map_y, _ = rc.head_fit_maps((HEIGHT, WIDTH), FACE, scale_x, scale_y)
            self.assertGreater(float(np.diff(map_x, axis=1).min()), 0)
            self.assertGreater(float(np.diff(map_y, axis=0).min()), 0)

    def test_identity_scale_is_a_no_op_and_out_of_range_is_rejected(self):
        image, _ = self.make()
        alpha = np.zeros((HEIGHT, WIDTH), np.uint8)
        same, same_alpha = rc.fit_head_geometry(image, alpha, FACE, 1.0, 1.0)
        self.assertIs(same, image)
        self.assertIs(same_alpha, alpha)
        with self.assertRaises(ValueError):
            rc.fit_head_geometry(image, alpha, FACE, 0.8, 1.0)

    def test_face_at_the_crop_edge_fades_the_warp_out(self):
        image, _ = self.make()
        edge_face = {"x": 0.0, "y": 116.0, "width": 200.0, "height": 270.0}
        _, _, displacement = rc.head_fit_maps((HEIGHT, WIDTH), edge_face, 0.95, 1.07)
        self.assertLess(float(displacement.max()), 0.01)


@unittest.skipIf(cv2 is None, "OpenCV/NumPy yüklü değil")
class OddTargetSizeTest(unittest.TestCase):
    """FaceFusion rounds outputs to even sizes; odd targets are padded, then cropped back."""

    def test_odd_target_is_padded_to_even_and_cropped_back_exactly(self):
        import tempfile
        from pathlib import Path
        from tools.facefusion_local_server import crop_to_size, pad_image_to_even

        rng = np.random.default_rng(3)
        original = rng.integers(0, 255, (1811, 1387, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.png"
            cv2.imwrite(str(path), original)
            size = pad_image_to_even(path)
            padded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        self.assertEqual(size, (1387, 1811))
        self.assertEqual(padded.shape[:2], (1812, 1388))
        self.assertTrue((padded[:1811, :1387] == original).all(), "no pixel of the photo is resampled")
        cropped = crop_to_size(padded, {"width": 1387, "height": 1811})
        self.assertTrue((cropped == original).all())

    def test_even_target_is_left_alone(self):
        import tempfile
        from pathlib import Path
        from tools.facefusion_local_server import crop_to_size, pad_image_to_even

        original = np.zeros((1024, 784, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.png"
            cv2.imwrite(str(path), original)
            before = path.read_bytes()
            self.assertEqual(pad_image_to_even(path), (784, 1024))
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(crop_to_size(original, {"width": 784, "height": 1024}).shape, original.shape)
        # Larger mismatches are not hidden by cropping.
        self.assertEqual(crop_to_size(original, {"width": 700, "height": 1024}).shape, original.shape)

    def test_blend_accepts_facefusion_even_outputs_for_an_odd_target(self):
        import tempfile
        from pathlib import Path
        from tools import facefusion_local_server as srv

        target, swapped = synthetic_swap()
        odd_target = target[:511, :391]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_path = root / "target.png"
            cv2.imwrite(str(target_path), odd_target)
            srv.pad_image_to_even(target_path)
            padded = cv2.imread(str(target_path))
            swap_padded = padded.copy()
            swap_padded[:511, :391] = swapped[:511, :391]
            for name in ("identity.png", "appearance.png"):
                cv2.imwrite(str(root / name), swap_padded)
            srv.blend_lower_face(
                target_path, root / "identity.png", root / "appearance.png", root / "out.png",
                FACE, 0.2, {"width": 391, "height": 511}
            )
            result = cv2.imread(str(root / "out.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(result.shape[:2], (511, 391))


if __name__ == "__main__":
    unittest.main()
