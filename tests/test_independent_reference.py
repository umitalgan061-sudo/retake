import unittest
from tools.identity_selection import select_texture_reference

class IndependentReferenceTest(unittest.TestCase):
    def test_exact_target_cannot_be_direct_texture_reference(self):
        weights=[3]+[2]*24
        report={'reference_median_similarities':[.95]+[.8]*24,'outlier_indices':[]}
        self.assertEqual(select_texture_reference(weights,report),0)
        self.assertNotEqual(select_texture_reference(weights,report,excluded_index=0),0)
