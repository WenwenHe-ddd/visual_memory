import unittest
import numpy as np
from scripts.memory.ridge_whitening import fit, score, standardize


class RidgeWhiteningTests(unittest.TestCase):
    def test_matches_full_covariance_inverse(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(5, 8)); x /= np.linalg.norm(x, axis=1, keepdims=True)
        q = rng.normal(size=8); q /= np.linalg.norm(q)
        v, values = fit(x)
        centered = x - x.mean(0)
        cov = centered.T @ centered / (len(x) - 1)
        ridge = .01
        expected = x @ np.linalg.solve(cov + ridge * np.eye(8), ridge * q)
        np.testing.assert_allclose(score(q, x, v, values, ridge), expected, atol=1e-13)

    def test_two_leaves_gap_and_z_invariance(self):
        x = np.array([[1., 0., 0.], [0., 1., 0.]])
        q = np.array([.8, .6, 0.])
        v, values = fit(x)
        for ridge in [.0001, .01, 1., 100.]:
            result = score(q, x, v, values, ridge)
            self.assertAlmostEqual(result[0] - result[1], .2 * ridge / (values[0] + ridge), places=12)
            np.testing.assert_array_equal(standardize(result)[0], [1., -1.])

    def test_legitimate_tie_and_singleton(self):
        x = np.array([[1., 0.], [0., 1.]])
        v, values = fit(x)
        result = score(np.array([1., 1.]) / np.sqrt(2), x, v, values, .01)
        np.testing.assert_array_equal(standardize(result)[0], [0., 0.])
        v, values = fit(x[:1])
        np.testing.assert_array_equal(standardize(score(np.array([.8, .6]), x[:1], v, values, .01))[0], [.8])

    def test_rank_deficiency_and_large_ridge(self):
        x = np.tile([1., 0., 0.], (4, 1)); q = np.array([.8, .6, 0.])
        v, values = fit(x)
        np.testing.assert_array_equal(values, [0., 0., 0.])
        np.testing.assert_allclose(score(q, x, v, values, .01), x @ q)
        np.testing.assert_array_equal(standardize(score(q, x, v, values, .01))[0], np.zeros(4))
        x = np.eye(3); v, values = fit(x)
        np.testing.assert_allclose(score(q, x, v, values, 1e12), x @ q, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
