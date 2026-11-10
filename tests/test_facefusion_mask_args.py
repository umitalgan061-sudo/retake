import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
        self.assertEqual(FACE_SWAP_VARIATIONS[0], (1.0, 0.20))
        self.assertEqual(len(set(FACE_SWAP_VARIATIONS)), 4)
        for swapper_weight, detail_strength in FACE_SWAP_VARIATIONS:
            self.assertGreaterEqual(swapper_weight, 0.92)
            self.assertLessEqual(swapper_weight, 1.0)
            self.assertGreater(detail_strength, 0)
            self.assertLessEqual(detail_strength, 0.30)

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
