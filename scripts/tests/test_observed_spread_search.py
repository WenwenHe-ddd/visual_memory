import unittest

import numpy as np

from scripts.graph.observed_spread_search import ObservedSpreadSearch


class ObservedSpreadTests(unittest.TestCase):
    def test_counter_and_trace_repeatability(self):
        rng = np.random.default_rng(5)
        vectors = rng.normal(size=(30, 12))
        index = ObservedSpreadSearch.fit(vectors, [list(range(i, i + 3)) for i in range(0, 30, 3)])
        q = rng.normal(size=12)
        for variant in ("offset", "radius_ratio", "max_both"):
            traced = index.search(q, list(range(10)), 5, prediction=variant)
            plain = index.search(q, list(range(10)), 5, record_trace=False, prediction=variant)
            self.assertEqual(traced["selected_indices"], plain["selected_indices"])
            self.assertEqual(traced["scalar_coordinate_products"], (10 + traced["scored_leaves"]) * 12)
            self.assertFalse(traced["exact_guarantee"])
            self.assertEqual(plain["trace"], [])
            self.assertEqual(len(set(traced["scored_leaf_indices"])), traced["scored_leaves"])

    def test_cannot_stop_before_output_budget_and_ties_are_stable(self):
        vectors = np.repeat([[1., 0.]], 6, axis=0)
        index = ObservedSpreadSearch.fit(vectors, [[i] for i in range(6)])
        result = index.search(np.array([1., 0.]), list(reversed(range(6))), 5)
        self.assertEqual(result["selected_indices"], list(range(5)))
        self.assertEqual(result["scored_leaves"], 6)

    def test_prediction_can_miss_and_is_not_called_a_bound(self):
        # First cluster has no spread; a later cluster's average is low but one
        # member is an exact query match. This explicitly tests the limitation.
        vectors = np.array([[.8, .6], [1., 0.], [-1., 0.]])
        index = ObservedSpreadSearch.fit(vectors, [[0], [1, 2]])
        result = index.search(np.array([1., 0.]), [0, 1], 1)
        self.assertEqual(result["selected_indices"], [0])
        self.assertFalse(result["exact_guarantee"])
        self.assertTrue(any(t["action"] == "empirical_stop" for t in result["trace"]))

    def test_centroid_reuse_preserves_search_and_counts_the_saved_dot(self):
        rng = np.random.default_rng(11)
        index = ObservedSpreadSearch.fit(rng.normal(size=(40, 16)),
                                        [list(range(i, i + 4)) for i in range(0, 40, 4)])
        for _ in range(20):
            q = rng.normal(size=16)
            direct = index.search(q, list(range(10)), 5)
            reused = index.search(q, list(range(10)), 5, reuse_centroid=True)
            self.assertEqual(direct["selected_indices"], reused["selected_indices"])
            np.testing.assert_allclose(direct["selected_scores"], reused["selected_scores"], atol=1e-12)
            self.assertEqual(reused["exact_leaf_dot_count"],
                             reused["scored_leaves"] - reused["centroid_inferred_score_count"])
            self.assertLess(reused["scalar_coordinate_products"], direct["scalar_coordinate_products"])

    def test_centroid_reuse_resolves_ambiguous_ties_with_direct_scores(self):
        index = ObservedSpreadSearch.fit(np.repeat([[.8, .6]], 8, axis=0),
                                        [[0, 1], [2, 3], [4, 5], [6, 7]])
        result = index.search(np.array([.8, .6]), [3, 2, 1, 0], 5, reuse_centroid=True)
        self.assertEqual(result["selected_indices"], list(range(5)))
        self.assertGreater(result["exact_leaf_dot_count"], 4)

    def test_singleton_refinement_updates_spread_before_stopping(self):
        index = ObservedSpreadSearch.fit(np.repeat([[.8, .6]], 8, axis=0),
                                        [[i] for i in range(8)])
        result = index.search(np.array([.8, .6]), list(reversed(range(8))), 5,
                              reuse_centroid=True)
        self.assertEqual(result["selected_indices"], list(range(5)))
        self.assertEqual(result["visited_masters"], 8)

    def test_batch_and_scalar_refinement_use_same_reduction_order(self):
        index = ObservedSpreadSearch.fit(np.repeat([[.8, .6]], 12, axis=0),
                                        [list(range(i, i + 3)) for i in range(0, 12, 3)])
        result = index.search(np.array([.8, .6]), [3, 2, 1, 0], 10, reuse_centroid=True)
        self.assertEqual(result["selected_indices"], list(range(10)))
        initial = sum(t.get("direct_leaf_dots", 0) for t in result["trace"])
        refined = sum(len(t.get("tie_refinements", [])) for t in result["trace"])
        self.assertEqual(result["exact_leaf_dot_count"], initial + refined)


if __name__ == "__main__":
    unittest.main()
