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
        self.assertEqual([name for name, _ in pools], ["all-25", "trusted-24", "consensus"])
        self.assertEqual(len(pools[1][1]), 24)
        self.assertNotIn(24, pools[1][1])
        self.assertEqual(pools[1][1], tuple(range(24)))
        self.assertEqual(len(pools[2][1]), 12)
        self.assertNotIn(24, pools[2][1])
        self.assertEqual(pools[2][1], tuple(range(12, 24)))

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
        self.assertEqual([name for name, _ in pools], ["all-25", "trusted-24", "consensus", "pose-match"])
        self.assertNotIn(0, pools[-1][1])

    def test_trusted_full_keeps_quality_weights_without_a_flagged_photo(self):
        report = self.report()
        report["outlier_indices"] = [1, 4]
        weights = [3, 1, 2, 1, 3] * 5
        pools = plan_identity_source_sets(weights, report)
        trusted = dict(pools)["trusted-23"]
        self.assertEqual(trusted.count(0), 3)
        self.assertEqual(trusted.count(2), 2)
        self.assertNotIn(1, trusted)
        self.assertNotIn(4, trusted)

    def test_single_texture_is_a_real_photo_with_identity_and_sharpness(self):
        from tools.identity_selection import select_texture_reference
        report = self.report()
        weights = [1] * 25
        weights[7] = 3
        self.assertEqual(select_texture_reference(weights, report), 7)
        report["outlier_indices"] = [7]
        self.assertNotEqual(select_texture_reference(weights, report), 7)

    def test_single_texture_can_match_target_pose_instead_of_global_best(self):
        from tools.identity_selection import select_texture_reference
        report = self.report()
        weights = [1] * 25
        weights[23] = 3
        poses = [0.2 + i * 0.01 for i in range(25)]
        poses[:8] = [0.02 + i * 0.01 for i in range(8)]
        self.assertEqual(select_texture_reference(weights, report, poses), 7)
        self.assertIsNone(select_texture_reference(weights, {"outlier_indices": []}))

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


class SourcePoolExplorationTests(unittest.TestCase):
    def test_canonical_round_uses_best_measured_pool(self):
        from tools.identity_selection import choose_identity_pool
        self.assertEqual(choose_identity_pool([.78,.82,.81],0),1)

    def test_later_rounds_only_explore_verified_close_sources(self):
        from tools.identity_selection import choose_identity_pool
        scores=[.80,.81,.805,.62,None,float("nan")]
        choices={choose_identity_pool(scores,i) for i in (4,7,9,12,15,17,19,21,23)}
        self.assertEqual(choices,{0,2})

    def test_missing_alternatives_do_not_override_a_good_pool(self):
        from tools.identity_selection import choose_identity_pool
        self.assertEqual(choose_identity_pool([None,.81,.65],7),1)

    def test_texture_variants_do_not_reuse_excluded_reference(self):
        from tools.identity_selection import select_texture_reference
        weights=[1]*25
        report={"outlier_indices":[24],"reference_median_similarities":[.50+i*.01 for i in range(25)]}
        poses=[i*.01 for i in range(25)]
        choices={select_texture_reference(weights,report,poses,excluded_index=7,variation=i)
                 for i in (0,3,7,11,15,19,23)}
        self.assertEqual(choices,{4,5,6})
