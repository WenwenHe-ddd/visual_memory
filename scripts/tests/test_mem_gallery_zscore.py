import unittest
import numpy as np

from scripts.evaluation.mem_gallery_zscore import group_scores


class GalleryZScoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(12, 7))
        self.x /= np.linalg.norm(self.x, axis=1, keepdims=True)
        self.q = rng.normal(size=(3, 7))
        self.q /= np.linalg.norm(self.q, axis=1, keepdims=True)

    def test_full_pool_matches_direct_covariance_inverse(self):
        raw, weighted, detail = group_scores(self.x, self.q, [(list(range(12)), 0)], .03, 1.)
        centered = self.x - self.x.mean(axis=0)
        covariance = centered.T @ centered / 11
        white = self.x @ np.linalg.solve(covariance + .03*np.eye(7), .03*self.q[0])
        expected = raw + raw.std()*np.tanh((white-white.mean())/white.std())
        np.testing.assert_allclose(weighted, expected, atol=1e-13)
        np.testing.assert_allclose(detail[0]['white'], white, atol=1e-13)

    def test_master_queries_singleton_and_unassigned_text(self):
        groups = [([0, 1, 2, 3], 1), ([4], 2), ([5, 6], 2)]
        raw, weighted, _ = group_scores(self.x, self.q, groups, .03, 1.)
        np.testing.assert_allclose(raw[:4], self.x[:4] @ self.q[1])
        np.testing.assert_allclose(raw[4:7], self.x[4:7] @ self.q[2])
        np.testing.assert_array_equal(raw[7:], weighted[7:])
        np.testing.assert_allclose(raw[7:], self.x[7:] @ self.q[0])
        self.assertEqual(weighted[4], raw[4])
        self.assertNotEqual(weighted[5], weighted[6])

    def test_alpha_zero_preserves_original_scores_and_order(self):
        groups = [([0, 3, 7], 1), ([2, 8], 2)]
        raw, weighted, _ = group_scores(self.x, self.q, groups, .03, 0.)
        np.testing.assert_array_equal(raw, weighted)
        np.testing.assert_array_equal(np.argsort(-raw, kind='stable'), np.argsort(-weighted, kind='stable'))


if __name__ == '__main__':
    unittest.main()
