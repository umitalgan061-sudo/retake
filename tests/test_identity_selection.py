import unittest

from tools.identity_selection import plan_identity_source_sets


class IdentitySourcePlanTests(unittest.TestCase):
    def report(self, n=25):
        return {
            "outlier_indices": [],
            "reference_median_similarities": [0.50 + i * 0.01 for i in range(n)]
        }

    def test_all_twenty_five_always_first_and_weighted(self):
        weights = [1, 2, 3, 1, 2] * 5
        pools = plan_identity_source_sets(weights, self.report())
        self.assertEqual(pools[0][0], "all-25")
        self.assertEqual(len(pools[0][1]), sum(weights))
        self.assertEqual(pools[0][1].count(2), 3)
        self.assertEqual(pools[1][0], "consensus")

    def test_consensus_is_high_agreement_and_excludes_outlier(self):
        report = self.report()
        report["outlier_indices"] = [24]
        pools = plan_identity_source_sets([1] * 25, report)
        self.assertEqual([name for name, _ in pools], ["all-25", "consensus"])
        self.assertEqual(len(pools[1][1]), 12)
        self.assertNotIn(24, pools[1][1])
        self.assertEqual(pools[1][1], tuple(range(12, 24)))

    def test_pose_matched_pool_needs_six_valid_descriptors(self):
        distances = [None] * 25
        for i in range(5):
            distances[i] = 0.04 + i * 0.01
        self.assertEqual(len(plan_identity_source_sets([1] * 25, self.report(), distances)), 2)
        for i in range(5, 9):
            distances[i] = 0.04 + i * 0.01
        pools = plan_identity_source_sets([1] * 25, self.report(), distances)
        self.assertEqual(pools[2][0], "pose-match")
        self.assertEqual(len(pools[2][1]), 8)
        self.assertEqual(pools[2][1], tuple(range(8)))

    def test_pose_pool_excludes_known_outlier(self):
        distances = [0.10 + i * 0.01 for i in range(25)]
        distances[0] = 0.001
        report = self.report()
        report["outlier_indices"] = [0]
        pools = plan_identity_source_sets([1] * 25, report, distances)
        self.assertEqual([name for name, _ in pools], ["all-25", "consensus", "pose-match"])
        self.assertNotIn(0, pools[-1][1])

    def test_old_report_uses_full_set_and_invalid_weights_fail(self):
        self.assertEqual(
            plan_identity_source_sets([1] * 25, {"centroid": [1.0]}),
            [("all-25", tuple(range(25)))]
        )
        with self.assertRaisesRegex(ValueError, "ağırlıkları"):
            plan_identity_source_sets([0] * 25, self.report())
        with self.assertRaisesRegex(ValueError, "ağırlıkları"):
            plan_identity_source_sets([True] * 25, self.report())


if __name__ == "__main__":
    unittest.main()
