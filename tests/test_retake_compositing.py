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


def lit_face_pair():
    """A photo of a 3D-lit face (side light, nose highlight) and a flat-lit
    swap of it with a moustache only the swap has."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    cx, cy, rx, ry = 196, 250, 115, 155
    u, v = (xx - cx) / rx, (yy - cy) / ry
    inside = (u * u + v * v) < 1
    shade = np.clip(0.5 * u + 0.3 * v + np.sqrt(np.clip(1 - u * u - v * v, 0, 1)) * 0.81, 0.15, 1)
    nose = np.exp(-((xx - cx) / 9) ** 2 - ((yy - (cy - 10)) / 45) ** 2) * 0.35
    skin = np.array([120, 150, 205], np.float32)
    background = np.stack([90 + xx * 0.1, 110 + yy * 0.08, 120 + xx * 0.06], 2)
    photo = background.copy()
    photo[inside] = (skin * (0.55 + 0.55 * (shade + nose))[:, :, None])[inside]
    swap = background.copy()
    swap[inside] = skin * 0.9
    cv2.ellipse(swap, (cx, cy + 60), (40, 14), 0, 0, 360, (60, 60, 80), -1)
    rng = np.random.default_rng(1)
    photo = np.clip(photo + rng.normal(0, 2, photo.shape), 0, 255).astype(np.uint8)
    swap = np.clip(swap + rng.normal(0, 2, swap.shape), 0, 255).astype(np.uint8)
    moustache_free = inside & (np.hypot(xx - cx, (yy - cy - 60) * 2.8) > 70)
    bounds = {"x": cx - rx, "y": cy - ry, "width": 2 * rx, "height": 2 * ry}
    return photo, swap, bounds, moustache_free


@unittest.skipIf(cv2 is None, "OpenCV/NumPy yüklü değil")
class ShadingTransferTest(unittest.TestCase):
    def gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def shading_match(self, image, photo, region):
        a = cv2.GaussianBlur(self.gray(image), (0, 0), 3)[region]
        b = cv2.GaussianBlur(self.gray(photo), (0, 0), 3)[region]
        return float(np.corrcoef(a, b)[0, 1])

    def test_the_photo_3d_shading_is_carried_onto_a_flat_swap(self):
        photo, swap, bounds, region = lit_face_pair()
        flat, _ = rc.seamless_face_composite(photo, swap, bounds, shading_strength=0)
        lit, _ = rc.seamless_face_composite(photo, swap, bounds)
        self.assertLess(self.shading_match(flat, photo, region), 0.6)
        self.assertGreater(self.shading_match(lit, photo, region), 0.85)

    def test_features_only_the_new_face_has_are_kept(self):
        photo, swap, bounds, _ = lit_face_pair()
        lit, _ = rc.seamless_face_composite(photo, swap, bounds)
        moustache = (slice(305, 315), slice(176, 216))
        self.assertLess(abs(float(self.gray(lit)[moustache].mean()) - float(self.gray(swap)[moustache].mean())), 4)

    def test_luminance_change_is_bounded_and_overshoot_is_safe(self):
        photo, swap, bounds, region = lit_face_pair()
        weight = np.zeros(photo.shape[:2], np.float32)
        weight[region] = 1
        out = rc.transfer_shading(swap.astype(np.float32), photo.astype(np.float32), weight, 230.0)
        gain = self.gray(np.clip(out, 0, 255).astype(np.uint8))[region] / np.maximum(self.gray(swap)[region], 1)
        self.assertLess(float(gain.max()), np.exp(rc.SHADING_MAX_DELTA * rc.SHADING_STRENGTH) + 0.08)
        # Overshoot from the multi-band blend (values outside 0..255) must not produce NaNs.
        overshoot = swap.astype(np.float32) * 1.4 - 60
        safe = rc.transfer_shading(overshoot, photo.astype(np.float32), weight, 230.0)
        self.assertTrue(np.isfinite(safe).all())
        self.assertGreaterEqual(float(safe.min()), 0)
        same = rc.transfer_shading(swap.astype(np.float32), photo.astype(np.float32), weight, 230.0, strength=0)
        self.assertTrue(np.array_equal(same, swap.astype(np.float32)))

    def test_colour_blotches_take_the_photo_skin_tone_but_a_beard_keeps_its_colour(self):
        photo, swap, bounds, _ = lit_face_pair()
        swap = swap.copy()
        cv2.circle(swap, (196, 255), 22, (95, 110, 235), -1)    # red nose tip only the swap has
        lab = lambda image: cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
        out, _ = rc.seamless_face_composite(photo, swap, bounds)
        nose = (slice(248, 262), slice(189, 203))
        red_before = float(lab(swap)[nose][:, :, 1].mean() - lab(photo)[nose][:, :, 1].mean())
        red_after = float(lab(out)[nose][:, :, 1].mean() - lab(photo)[nose][:, :, 1].mean())
        self.assertGreater(red_before, 15)
        self.assertLess(red_after, red_before * 0.5)
        moustache = (slice(305, 315), slice(176, 216))
        self.assertLess(abs(float(lab(out)[moustache][:, :, 1].mean() - lab(swap)[moustache][:, :, 1].mean())), 6)

def jaw_controls(chin_drop=30.0):
    controls = []
    for i in range(21):
        angle = np.pi * i / 20
        start = [392 - np.cos(angle) * 220, 560 + np.sin(angle) * 300]
        along = 1 - abs(i - 10) / 10
        controls.append({"from": start, "to": [start[0], start[1] + chin_drop * along]})
    anchors = [[332, 420], [452, 420], [392, 500], [352, 600], [432, 600], [392, 410], [392, 590], [392, 610]]
    return controls, anchors


@unittest.skipIf(cv2 is None, "OpenCV/NumPy yüklü değil")
class JawlineWarpTest(unittest.TestCase):
    SHAPE = (1024, 784)

    def chin_y(self, image):
        column = image[:, 392].astype(int)
        return int(np.where(np.abs(column - np.array([60, 80, 100])).sum(1) > 30)[0].max())

    def face(self):
        image = np.full((*self.SHAPE, 3), (60, 80, 100), np.uint8)
        cv2.ellipse(image, (392, 560), (220, 300), 0, 0, 360, (140, 170, 210), -1)
        return image

    def test_chin_follows_the_controls_while_eyes_nose_and_mouth_stay(self):
        controls, anchors = jaw_controls(30)
        image = self.face()
        out, alpha = rc.warp_jawline(image, np.zeros(self.SHAPE, np.uint8), controls, anchors, 440.0)
        self.assertGreater(self.chin_y(out) - self.chin_y(image), 18)
        dx, dy = rc.jaw_displacement_field(self.SHAPE, controls, anchors, 440.0)
        for x, y in anchors:
            self.assertLess(float(np.hypot(dx[y, x], dy[y, x])), 0.05)
        self.assertGreater(int(alpha[862, 392]), 0, "moved pixels are covered by the matte")

    def test_warp_is_smooth_and_leaves_the_crop_border_alone(self):
        controls, anchors = jaw_controls(30)
        dx, dy = rc.jaw_displacement_field(self.SHAPE, controls, anchors, 440.0)
        det = (1 + np.diff(dx, axis=1)[:-1]) * (1 + np.diff(dy, axis=0)[:, :-1]) \
            - np.diff(dx, axis=0)[:, :-1] * np.diff(dy, axis=1)[:-1]
        self.assertGreater(float(det.min()), 0.3, "no folding or tearing")
        for edge in (dx[0], dx[-1], dx[:, 0], dx[:, -1], dy[0], dy[-1], dy[:, 0], dy[:, -1]):
            self.assertLess(float(np.abs(edge).max()), 1e-3)

    def test_shifts_are_capped_and_zero_scale_is_a_no_op(self):
        controls, anchors = jaw_controls(400)
        dx, dy = rc.jaw_displacement_field(self.SHAPE, controls, anchors, 440.0)
        self.assertLess(float(np.hypot(dx, dy).max()), 440 * rc.JAW_MAX_SHIFT_RATIO * 1.15)
        image = self.face()
        alpha = np.zeros(self.SHAPE, np.uint8)
        same, same_alpha = rc.warp_jawline(image, alpha, controls, anchors, 440.0, scale=0)
        self.assertIs(same, image)
        self.assertIs(same_alpha, alpha)


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
