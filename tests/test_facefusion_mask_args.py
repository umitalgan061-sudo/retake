import unittest

from tools.facefusion_local_server import (
    FACE_MASK_REGIONS,
    FACE_MASK_TYPES,
    face_mask_cli_arguments,
)


class FaceFusionSemanticMaskArgumentsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
