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
    HEAD_FIT_MAX,
    HEAD_FIT_MIN,
    MAX_TARGET_HEIGHT,
    MAX_TARGET_WIDTH,
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
        self.assertEqual(FACE_SWAP_VARIATIONS[0], (1.0, 0.20, 1.0))
        self.assertEqual(len(set(FACE_SWAP_VARIATIONS)), 4)
        strengths = set()
        for swapper_weight, detail_strength, head_fit_strength in FACE_SWAP_VARIATIONS:
            self.assertGreaterEqual(swapper_weight, 0.92)
            self.assertLessEqual(swapper_weight, 1.0)
            self.assertGreater(detail_strength, 0)
            self.assertLessEqual(detail_strength, 0.30)
            self.assertGreaterEqual(head_fit_strength, 0.0)
            self.assertLessEqual(head_fit_strength, 1.5)
            strengths.add(head_fit_strength)
        # One alternative must leave the photo's own head proportions alone.
        self.assertIn(0.0, strengths)

    def test_metadata_accepts_native_resolution_crops_and_bounded_head_fit(self):
        metadata = {
            "referenceWeights": [1] * MAX_REFERENCES,
            "faceBounds": {"x": 300, "y": 400, "width": 700, "height": 900},
            "targetSize": {"width": 1500, "height": 1960},
            "headFit": {"scaleX": 0.96, "scaleY": 1.05}
        }
        validated = _validate_metadata(metadata, MAX_REFERENCES)
        self.assertEqual(validated["targetSize"], {"width": 1500, "height": 1960})
        self.assertEqual(validated["headFit"], {"scaleX": 0.96, "scaleY": 1.05})

        defaults = _validate_metadata({
            "referenceWeights": [1] * MAX_REFERENCES,
            "faceBounds": {"x": 0, "y": 0, "width": 100, "height": 100}
        }, MAX_REFERENCES)
        self.assertEqual(defaults["targetSize"], {"width": 784, "height": 1024})
        self.assertEqual(defaults["headFit"], {"scaleX": 1.0, "scaleY": 1.0})

        for bad_size in ({"width": MAX_TARGET_WIDTH + 1, "height": 1024}, {"width": 784, "height": MAX_TARGET_HEIGHT + 1},
                         {"width": 784.5, "height": 1024}, {"width": True, "height": 1024}, "784x1024"):
            with self.subTest(size=bad_size), self.assertRaisesRegex(ValueError, "kırpım boyutu"):
                _validate_metadata({**metadata, "targetSize": bad_size}, MAX_REFERENCES)
        for bad_fit in ({"scaleX": HEAD_FIT_MIN - 0.01, "scaleY": 1.0}, {"scaleX": 1.0, "scaleY": HEAD_FIT_MAX + 0.01},
                        {"scaleX": float("nan"), "scaleY": 1.0}, {"scaleX": 1.0}, [1.0, 1.0]):
            with self.subTest(fit=bad_fit), self.assertRaisesRegex(ValueError, "Kafa oranı"):
                _validate_metadata({**metadata, "headFit": bad_fit}, MAX_REFERENCES)
        with self.assertRaisesRegex(ValueError, "kırpımının dışında"):
            _validate_metadata({**metadata, "targetSize": {"width": 784, "height": 1024}}, MAX_REFERENCES)

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
