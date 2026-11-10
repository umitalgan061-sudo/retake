import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.facefusion_local_server import (
    FACE_SWAP_VARIATIONS,
    FACE_MASK_REGIONS,
    FACE_MASK_TYPES,
    MAX_REFERENCES,
    _validate_metadata,
    face_mask_cli_arguments,
    validate_reference_count,
    FACE_ENHANCER_BLEND,
    FACE_ENHANCER_INTERNAL_BLEND,
    FACE_ENHANCER_MODEL,
    FACE_ENHANCER_WEIGHT,
    adjust_face_geometry,
    face_geometry_affine_matrix,
    run_facefusion,
)


class FaceFusionSemanticMaskArgumentsTest(unittest.TestCase):
    def test_exactly_25_reference_images_are_required(self):
        self.assertEqual(MAX_REFERENCES, 25)
        self.assertIsNone(validate_reference_count(25))
        with self.assertRaisesRegex(ValueError, "tam olarak 25"):
            validate_reference_count(24)

    def test_cli_uses_facefusion_392_mask_type_tokens(self):
        args = face_mask_cli_arguments()
        index = args.index("--face-mask-types")
        self.assertEqual(args[index + 1:index + 1 + len(FACE_MASK_TYPES)], ["box", "occlusion", "region"])
        self.assertNotIn("box,occlusion", args)

    def test_cli_enables_bisenet_face_parser_and_xseg_occlusion_model(self):
        args = face_mask_cli_arguments()
        self.assertIn("--face-parser-model", args)
        self.assertIn("bisenet_resnet_34", args)
        self.assertIn("--face-occluder-model", args)
        self.assertIn("xseg_1", args)

    def test_gfpgan_is_a_low_blend_optional_stage(self):
        self.assertEqual(FACE_ENHANCER_MODEL, "gfpgan_1.4")
        self.assertEqual(FACE_ENHANCER_BLEND, 25)
        self.assertEqual(FACE_ENHANCER_INTERNAL_BLEND, 100)
        self.assertEqual(FACE_ENHANCER_WEIGHT, 0.25)

    def test_parser_keeps_glasses_out_of_the_replacement_region(self):
        args = face_mask_cli_arguments()
        index = args.index("--face-mask-regions")
        regions = args[index + 1:index + 1 + len(FACE_MASK_REGIONS)]
        self.assertEqual(regions, list(FACE_MASK_REGIONS))
        self.assertNotIn("glasses", regions)
        self.assertIn("skin", regions)
        self.assertIn("mouth", regions)
        self.assertIn("upper-lip", regions)
        self.assertIn("lower-lip", regions)
        self.assertNotIn("hair", regions)

    def test_alternate_render_balances_are_bounded_and_distinct(self):
        self.assertEqual(len(FACE_SWAP_VARIATIONS), 4)
        self.assertEqual(FACE_SWAP_VARIATIONS[0], (1.0, 0.20, 0.98, 1.05))
        self.assertEqual(len(set(FACE_SWAP_VARIATIONS)), 4)
        for swapper_weight, detail_strength, width_scale, height_scale in FACE_SWAP_VARIATIONS:
            self.assertGreaterEqual(swapper_weight, 0.92)
            self.assertLessEqual(swapper_weight, 1.0)
            self.assertGreater(detail_strength, 0)
            self.assertLessEqual(detail_strength, 0.30)
            self.assertGreaterEqual(width_scale, 0.94)
            self.assertLessEqual(width_scale, 1.02)
            self.assertGreaterEqual(height_scale, 1.0)
            self.assertLessEqual(height_scale, 1.08)

    def test_face_geometry_scales_around_the_face_center(self):
        bounds = {"x": 100.0, "y": 200.0, "width": 200.0, "height": 300.0}
        matrix = face_geometry_affine_matrix(bounds, 0.98, 1.05)
        self.assertAlmostEqual(matrix[0][0], 0.98)
        self.assertAlmostEqual(matrix[0][2], 4.0)
        self.assertAlmostEqual(matrix[1][1], 1.05)
        self.assertAlmostEqual(matrix[1][2], -17.5)

    def test_face_geometry_moves_rgb_and_alpha_together_without_opaque_edges(self):
        cv2 = MagicMock()
        cv2.IMREAD_UNCHANGED = 1
        cv2.INTER_CUBIC = 2
        cv2.INTER_LINEAR = 3
        cv2.BORDER_REFLECT_101 = 4
        cv2.BORDER_CONSTANT = 5
        image = MagicMock()
        image.ndim = 3
        image.shape = (1024, 784, 4)
        image.__getitem__.side_effect = lambda key: "rgb" if isinstance(key[2], slice) else "alpha"
        cv2.imread.return_value = image
        cv2.warpAffine.side_effect = ("warped-rgb", "warped-alpha")
        cv2.split.return_value = ("blue", "green", "red")
        cv2.merge.side_effect = lambda channels: tuple(channels)
        numpy = SimpleNamespace(float32="float32", asarray=lambda values, dtype: values)

        with patch.dict("sys.modules", {"cv2": cv2, "numpy": numpy}):
            adjust_face_geometry(
                Path("/hybrid.png"),
                Path("/adjusted.png"),
                {"x": 100.0, "y": 200.0, "width": 200.0, "height": 300.0},
                0.98,
                1.05
            )

        calls = cv2.warpAffine.call_args_list
        self.assertEqual(calls[0].args[0], "rgb")
        self.assertEqual(calls[1].args[0], "alpha")
        for actual, expected in zip(calls[0].args[1], [[0.98, 0.0, 4.0], [0.0, 1.05, -17.5]]):
            for actual_value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(actual_value, expected_value)
        self.assertEqual(calls[0].kwargs["borderMode"], cv2.BORDER_REFLECT_101)
        self.assertEqual(calls[1].kwargs["borderMode"], cv2.BORDER_CONSTANT)
        self.assertEqual(calls[1].kwargs["borderValue"], 0)
        cv2.imwrite.assert_called_once_with(
            "/adjusted.png", ("blue", "green", "red", "warped-alpha")
        )

    def test_metadata_defaults_to_first_variant_and_rejects_invalid_variants(self):
        metadata = {
            "referenceWeights": [1] * MAX_REFERENCES,
            "faceBounds": {"x": 0, "y": 0, "width": 100, "height": 100}
        }
        self.assertEqual(_validate_metadata(metadata, MAX_REFERENCES)["variation"], 0)
        metadata["variation"] = len(FACE_SWAP_VARIATIONS)
        with self.assertRaisesRegex(ValueError, "Görünüm alternatifi"):
            _validate_metadata(metadata, MAX_REFERENCES)
        metadata["variation"] = True
        with self.assertRaisesRegex(ValueError, "Görünüm alternatifi"):
            _validate_metadata(metadata, MAX_REFERENCES)

    @patch("tools.facefusion_local_server.subprocess.run")
    def test_facefusion_command_uses_the_selected_swap_strength(self, subprocess_run):
        subprocess_run.return_value = SimpleNamespace(returncode=1, stderr="", stdout="")
        run_facefusion(
            Path("/facefusion"),
            Path("/target.png"),
            [Path("/reference.png")],
            Path("/output.png"),
            "alphaface_256",
            "1024x1024",
            0.94
        )
        command = subprocess_run.call_args.args[0]
        weight_index = command.index("--face-swapper-weight")
        self.assertEqual(command[weight_index + 1], "0.94")


if __name__ == "__main__":
    unittest.main()
