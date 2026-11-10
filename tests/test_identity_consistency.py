import unittest

from tools.identity_consistency import analyze_identity_embeddings


class IdentityConsistencyTests(unittest.TestCase):
    def test_identical_references_are_coherent(self):
        embeddings = [[1.0, 0.0, 0.0] for _ in range(25)]
        report = analyze_identity_embeddings(embeddings)
        self.assertEqual(report["count"], 25)
        self.assertEqual(report["outlier_indices"], [])
        self.assertFalse(report["hard_conflict"])
        self.assertAlmostEqual(report["median_similarity"], 1.0)

    def test_single_clear_outlier_is_reported_not_silently_removed(self):
        embeddings = [[1.0, 0.0, 0.0] for _ in range(24)]
        embeddings.append([-1.0, 0.0, 0.0])
        names = [f"reference-{index + 1:02d}.jpg" for index in range(25)]
        report = analyze_identity_embeddings(embeddings, names)
        self.assertEqual(report["outlier_indices"], [24])
        self.assertEqual(report["outlier_files"], ["reference-25.jpg"])
        self.assertFalse(report["hard_conflict"])

    def test_incoherent_collection_is_flagged(self):
        embeddings = [
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]
        ] * 5
        self.assertTrue(analyze_identity_embeddings(embeddings)["hard_conflict"])

    def test_invalid_embeddings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "normu sıfır"):
            analyze_identity_embeddings([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "boyutları"):
            analyze_identity_embeddings([[1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
