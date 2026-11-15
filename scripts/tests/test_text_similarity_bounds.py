"""Bounds must contain unseen member scores using aggregate-only state."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import dataclasses
import unittest

import numpy as np

from scripts.graph.text_similarity_bounds import TextBoundIndex


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class TextSimilarityBoundTests(unittest.TestCase):
    def test_known_spherical_cap_tight_boundary_and_inside(self):
        angle = np.deg2rad(20)
        members = np.array([[np.cos(angle), np.sin(angle)],
                            [np.cos(angle), -np.sin(angle)]])
        index = TextBoundIndex.fit(members, [[0, 1]])
        query = [np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60))]
        exact = np.cos(np.deg2rad(40))
        bound = index.upper_bounds(query)[0]
        self.assertGreaterEqual(bound, exact)
        self.assertLess(bound - exact, 1e-10)
        self.assertEqual(index.upper_bounds([1, 0])[0], 1)
        opposite_bound = index.upper_bounds([-1, 0])[0]
        self.assertGreaterEqual(opposite_bound, -np.cos(angle))
        self.assertLess(opposite_bound, -.9)  # Negative upper bounds are valid.
        self.assertLessEqual(bound, index.radius_upper_bounds(query)[0])
        self.assertAlmostEqual(index.mean_scores(query)[0], np.mean(members @ unit(query)))

    def test_singletons_and_equal_score_ties(self):
        vectors = np.array([[1., 0.], [0., 1.], [1., 0.]])
        index = TextBoundIndex.fit(vectors, [[0], [1], [2]])
        query = unit([1., 1.])
        exact = vectors @ query
        upper = index.upper_bounds(query)
        self.assertTrue(np.all(upper >= exact))
        np.testing.assert_allclose(upper, exact, atol=1e-12, rtol=0)
        self.assertEqual(upper[0], upper[2])
        self.assertFalse(upper[2] < exact[0])  # A same-score unseen item cannot be skipped.
        np.testing.assert_allclose(index.mean_scores(query), exact, atol=1e-15)

    def test_antipodal_zero_and_near_zero_centers_fall_back(self):
        for vectors in (np.array([[1., 0.], [-1., 0.]]),
                        np.array([[1., 0.], [-1., 1e-15]])):
            index = TextBoundIndex.fit(vectors, [[0, 1]])
            np.testing.assert_array_equal(index.directions, np.zeros((1, 2)))
            for query in ([1., 0.], [0., 1.], [-1., 0.]):
                self.assertEqual(index.upper_bounds(query)[0], 1.)
                self.assertGreaterEqual(index.radius_upper_bounds(query)[0],
                                        np.max(unit(vectors) @ unit(query)))

    def test_random_queries_all_actual_member_scores_below_bounds(self):
        rng = np.random.default_rng(572)
        for dimension in (1, 2, 17, 384):
            with self.subTest(dimension=dimension):
                centers = rng.normal(size=(4, dimension))
                vectors = np.vstack([center + .35 * rng.normal(size=(7, dimension)) for center in centers])
                groups = [list(range(i * 7, (i + 1) * 7)) for i in range(4)]
                index = TextBoundIndex.fit(vectors, groups)
                normalized = unit(vectors)
                queries = np.vstack([rng.normal(size=(20, dimension)), normalized])
                for query in queries:
                    true_scores = np.clip(normalized @ unit(query), -1, 1)
                    maxima = np.array([max(true_scores[group]) for group in groups])
                    self.assertTrue(np.all(index.upper_bounds(query) >= maxima))
                    self.assertTrue(np.all(index.radius_upper_bounds(query) >= maxima))
                    np.testing.assert_allclose(index.mean_scores(query),
                                               [np.mean(true_scores[group]) for group in groups], atol=1e-14)

    def test_near_endpoint_angles_remain_conservative(self):
        for angle in (0., 1e-12, 1e-8, 1e-5, np.pi / 2 - 1e-12):
            vectors = np.array([[np.cos(angle), np.sin(angle)],
                                [np.cos(angle), -np.sin(angle)]])
            index = TextBoundIndex.fit(vectors, [[0, 1]])
            for query_angle in (0., angle, angle + 1e-12, np.pi / 2, np.pi - 1e-12, np.pi):
                query = [np.cos(query_angle), np.sin(query_angle)]
                self.assertGreaterEqual(index.upper_bounds(query)[0],
                                        min(1., float(np.max(unit(vectors) @ unit(query)))))

    def test_scale_invariance_without_overflow_or_underflow(self):
        vectors = np.array([[1., 2., -1.], [1., .2, .3], [-1., .5, .2]])
        groups = [[0, 1], [2]]
        base = TextBoundIndex.fit(vectors, groups)
        scaled = TextBoundIndex.fit(vectors * np.array([1e300, 1e-300, 7.])[:, None], groups)
        np.testing.assert_allclose(base.centers, scaled.centers, atol=1e-15)
        for query in (np.array([.3, .4, .8]), np.array([.3, .4, .8]) * 1e300):
            np.testing.assert_allclose(base.upper_bounds(query), scaled.upper_bounds(query), atol=1e-13)
            np.testing.assert_allclose(base.mean_scores(query), scaled.mean_scores(query), atol=1e-15)

    def test_aggregate_only_state_owns_arrays_and_roundtrips(self):
        vectors = np.array([[1., 0., 0.], [.8, .6, 0.], [0., 0., 1.]])
        index = TextBoundIndex.fit(vectors, [[0, 1], [2]])
        self.assertEqual({field.name for field in dataclasses.fields(index)},
                         {"centers", "directions", "min_cosine", "radii", "counts"})
        self.assertFalse(hasattr(index, "__dict__"))
        before = index.upper_bounds([1., 0., 1.])
        vectors[:] = 42
        np.testing.assert_array_equal(before, index.upper_bounds([1., 0., 1.]))
        state = index.export_state()
        restored = TextBoundIndex.from_state(state)
        np.testing.assert_array_equal(before, restored.upper_bounds([1., 0., 1.]))
        self.assertEqual(state["centers"].shape, (2, 3))
        state["centers"][:] = 0
        np.testing.assert_array_equal(before, index.upper_bounds([1., 0., 1.]))
        with self.assertRaises(ValueError):
            index.centers[0, 0] = 0

    def test_invalid_vectors_queries_and_partitions(self):
        for vectors in ([], [1., 2.], [[0., 0.]], [[np.nan, 1.]], [[np.inf, 1.]], [[1j, 1.]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                TextBoundIndex.fit(vectors, [[0]])
        vectors = np.eye(3)
        for groups in ([], [[]], [[0], [1]], [[0, 1], [1, 2]], [[0, 1, 3]],
                       [[0, 1, -1]], [[True, 1, 2]], [[0., 1, 2]], [0, 1, 2]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                TextBoundIndex.fit(vectors, groups)
        index = TextBoundIndex.fit(vectors, [[0, 1, 2]])
        for query in ([0., 0., 0.], [1., 0.], [[1., 0., 0.]], [np.nan, 0., 1.], [1j, 0, 1]):
            with self.subTest(query=query), self.assertRaises(ValueError):
                index.upper_bounds(query)

    def test_invalid_saved_aggregate_state_rejected(self):
        state = TextBoundIndex.fit(np.eye(3), [[0, 1], [2]]).export_state()
        for field, invalid in (
            ("schema_version", 99), ("normalization", "none"), ("dimension", 12),
            ("numerical_margin", .03), ("counts", [0, 1]),
            ("min_cosine", [1.1, 1]), ("radii", [-1, 0]),
            ("directions", np.ones((2, 3))), ("centers", np.ones((2, 3))),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                TextBoundIndex.from_state({**state, field: invalid})


if __name__ == "__main__":
    unittest.main()
