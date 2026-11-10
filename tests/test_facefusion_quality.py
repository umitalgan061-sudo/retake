import unittest

from tools.facefusion_quality import PIXEL_BOOST_FALLBACKS, is_near_noop


class FaceFusionQualityTests(unittest.TestCase):
    def test_1024_falls_back_to_768_then_512(self):
        self.assertEqual(PIXEL_BOOST_FALLBACKS["1024x1024"], ("768x768", "512x512"))

    def test_768_falls_back_to_512(self):
        self.assertEqual(PIXEL_BOOST_FALLBACKS["768x768"], ("512x512",))

    def test_512_has_no_further_fallback(self):
        self.assertEqual(PIXEL_BOOST_FALLBACKS["512x512"], ())

    def test_noop_guard_is_conservative(self):
        self.assertTrue(is_near_noop(0.0002, 0.1, 0.9))
        self.assertFalse(is_near_noop(0.02, 1.2, 8.0))
        self.assertFalse(is_near_noop(0.0002, 0.9, 2.5))


if __name__ == "__main__":
    unittest.main()
