"""Correctness and cost tests for text-only progressive exact search."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]

import numpy as np

from scripts.graph.progressive_text_bounds import ProgressiveTextBounds


def expected(index, query, limit):
    q = np.asarray(query, dtype=np.float64)
    q /= np.linalg.norm(q)
    scores = np.einsum("ij,j->i", index.vectors, q, optimize=False)
    ids = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:limit]
    return ids, scores


class ProgressiveTextBoundTests(unittest.TestCase):
    def test_random_exact_top_l_and_accounting(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            x = rng.normal(size=(29, 17))
            groups = [list(range(i, min(i+4, len(x)))) for i in range(0, len(x), 4)]
            index = ProgressiveTextBounds.fit(x, groups)
            query = rng.normal(size=x.shape[1])
            order = rng.permutation(len(groups)).tolist()
            for limit in (1, 3, 7, 29):
                with self.subTest(seed=seed, limit=limit):
                    result = index.search(query, order, limit)
                    ids, scores = expected(index, query, limit)
                    self.assertEqual(result["selected_indices"], ids)
                    np.testing.assert_array_equal(result["selected_scores"], scores[ids])
                    self.assertEqual(len(set(result["scored_indices"])), result["exact_score_count"])
                    self.assertEqual(result["pruned_leaf_count"]+result["exact_score_count"], len(x))
                    cost = result["cost"]
                    self.assertEqual(cost["query_projection_scalar_products"], result["projected_components"]*x.shape[1])
                    self.assertEqual(cost["exact_leaf_scalar_products"], result["exact_score_count"]*x.shape[1])
                    self.assertEqual(cost["leaf_partial_scalar_products"], sum(
                        row["active_before"] for row in result["trace"] if row["action"] == "project_component"))
                    self.assertEqual(cost["total_scalar_products"], sum(cost[k] for k in (
                        "mean_dot_scalar_products", "query_projection_scalar_products",
                        "leaf_partial_scalar_products", "exact_leaf_scalar_products")))

    def test_identical_leaves_and_stable_ties(self):
        for n, dimension in ((1, 8), (20, 8), (8, 20)):
            with self.subTest(n=n, dimension=dimension):
                vector = np.arange(1, dimension+1, dtype=float)
                index = ProgressiveTextBounds.fit(np.tile(vector, (n, 1)), [list(range(n))])
                for limit in (1, n):
                    result = index.search(vector, [0], limit)
                    self.assertEqual(result["selected_indices"], list(range(limit)))
                    self.assertEqual(result["projected_components"], 0)

    def test_duplicate_leaves_across_groups_preserve_original_index_ties(self):
        x = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        index = ProgressiveTextBounds.fit(x, [[2], [1, 3], [0]])
        result = index.search([1, 0, 0], [0, 1, 2], 1)
        self.assertEqual(result["selected_indices"], [0])

    def test_every_partial_bound_contains_actual_scores(self):
        for nearly_low_rank in (False, True):
            rng = np.random.default_rng(817)
            x = rng.normal(size=(31, 23))
            if nearly_low_rank:
                x = rng.normal(size=(31, 2)) @ rng.normal(size=(2, 23)) + 1e-13*x
            index = ProgressiveTextBounds.fit(x, [list(range(len(x)))])
            for trial in range(8):
                query = rng.normal(size=x.shape[1])
                query /= np.linalg.norm(query)
                actual = np.einsum("ij,j->i", index.vectors, query, optimize=False)
                partial = np.zeros(len(x))
                projected_sq = 0.
                mean_score = float(index.mean @ query)
                ids = np.arange(len(x))
                for k in range(index.rank+1):
                    with self.subTest(nearly_low_rank=nearly_low_rank, trial=trial, k=k):
                        if k:
                            projection = float(index.basis[k-1] @ query)
                            projected_sq += projection**2
                            partial += index.coordinates[:, k-1]*projection
                        lower, upper = index._bounds(ids, k, partial, mean_score, projected_sq)
                        self.assertTrue(np.all(actual >= lower))
                        self.assertTrue(np.all(actual <= upper))

    def test_easy_direction_prunes_leaves_without_full_scores(self):
        rng = np.random.default_rng(4)
        x = np.zeros((40, 64))
        x[:, 0] = np.linspace(-1, 1, len(x))
        x[:, 1] = np.sqrt(1-x[:, 0]**2)
        x[:, 2:] += rng.normal(scale=1e-5, size=(40, 62))
        groups = [[i] for i in range(len(x))]
        index = ProgressiveTextBounds.fit(x, groups)
        query = np.zeros(64)
        query[0] = 1
        result = index.search(query, list(reversed(range(len(x)))), 3)
        ids, _ = expected(index, query, 3)
        self.assertEqual(result["selected_indices"], ids)
        self.assertLess(result["exact_score_count"], len(x))
        self.assertLess(result["cost"]["total_scalar_products"], len(x)*64)

    def test_trace_disabled_preserves_scores_counts_and_decisions(self):
        rng = np.random.default_rng(96)
        x = rng.normal(size=(21, 37))
        index = ProgressiveTextBounds.fit(x, [[i] for i in range(21)])
        query = rng.normal(size=37)
        traced = index.search(query, list(range(21)), 5)
        silent = index.search(query, list(range(21)), 5, record_trace=False)
        self.assertEqual(silent["trace"], [])
        for key in traced:
            if key != "trace":
                self.assertEqual(traced[key], silent[key])

    def test_invalid_partitions_rejected(self):
        for bad in ([], [[0, 0], [1]], [[0]], [[0], [1], [2]], [[False], [1]]):
            with self.subTest(groups=bad), self.assertRaises(ValueError):
                ProgressiveTextBounds.fit([[1, 0], [0, 1]], bad)

    def test_invalid_query_limit_and_order_rejected(self):
        index = ProgressiveTextBounds.fit([[1, 0], [0, 1]], [[0], [1]])
        for query, order, limit in [
            ([0, 0], [0, 1], 1), ([1, 0, 0], [0, 1], 1),
            ([1, 0], [0, 0], 1), ([1, 0], [0, 1], 0),
            ([1, 0], [0, 1], 3), ([1, 0], [0, 1], True),
        ]:
            with self.subTest(query=query, order=order, limit=limit), self.assertRaises(ValueError):
                index.search(query, order, limit)


if __name__ == "__main__":
    unittest.main()
