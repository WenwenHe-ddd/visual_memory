"""Exact text-pivot tree behavior, pruning safety, ties, and honest dot costs."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

import numpy as np

from scripts.graph.hierarchical_text_bounds import HierarchicalTextBoundIndex


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x), axis=-1, keepdims=True)
    x = x / scale
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class HierarchicalTextBoundTests(unittest.TestCase):
    def assert_matches_full(self, index, query, order, limit, policy):
        scores = np.array([vector @ unit(query) for vector in index.vectors])
        expected = sorted(range(len(scores)), key=lambda j: (-scores[j], j))[:limit]
        result = index.search(query, order, limit, policy=policy)
        self.assertEqual(result["selected_indices"], expected)
        np.testing.assert_array_equal(result["selected_scores"], scores[expected])
        self.assertEqual(result["scored_leaves"], result["bound_dot_count"] + result["leaf_dot_count"])
        self.assertEqual(result["exact_dot_count"], result["scored_leaves"])
        self.assertEqual(result["scalar_coordinate_products"], result["scored_leaves"] * index.dimension)
        self.assertLessEqual(result["exact_dot_count"], index.n_members)
        self.assertEqual(len(result["scored_leaf_indices"]), len(set(result["scored_leaf_indices"])))
        self.assertEqual(result["additional_nonleaf_dot_count"], 0)
        for event in result["trace"]:
            if event["action"] == "bound":
                node = index.nodes[event["node_id"]]
                descendants = [leaf for leaf in node.members if leaf != node.pivot_index]
                self.assertGreaterEqual(event["upper_bound"], max(scores[descendants]))
            if event["action"] in {"prune", "stop_frontier"}:
                for leaf in event["unscored_members"]:
                    self.assertLess(scores[leaf], event["threshold"])
        quiet = index.search(query, order, limit, policy=policy, record_trace=False)
        self.assertEqual(quiet["trace"], [])
        self.assertEqual({k: v for k, v in result.items() if k != "trace"},
                         {k: v for k, v in quiet.items() if k != "trace"})
        return result

    def test_random_groups_and_limits_match_full_scan_both_policies(self):
        rng = np.random.default_rng(902)
        for dimension in (2, 19, 384):
            vectors = rng.normal(size=(23, dimension))
            groups = [list(range(0, 7)), list(range(7, 11)), [11], list(range(12, 23))]
            index = HierarchicalTextBoundIndex.fit(vectors, groups)
            for query in rng.normal(size=(5, dimension)):
                for limit in (1, 3, 10, 23):
                    for policy in ("clip", "best_bound"):
                        with self.subTest(dimension=dimension, limit=limit, policy=policy):
                            self.assert_matches_full(index, query, [3, 1, 0, 2], limit, policy)

    def test_identical_vectors_preserve_global_stable_ties(self):
        vectors = np.tile([.3, .4, .5], (12, 1))
        index = HierarchicalTextBoundIndex.fit(vectors, [[7, 1, 10], [11, 5], [0, 2, 3, 4, 6, 8, 9]])
        for policy in ("clip", "best_bound"):
            result = self.assert_matches_full(index, vectors[0], [1, 0, 2], 4, policy)
            self.assertEqual(result["selected_indices"], [0, 1, 2, 3])
            self.assertEqual(result["scored_leaves"], 12)

    def test_singletons_and_opposite_vectors(self):
        vectors = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.], [1., 0.]])
        for groups in ([[0], [1], [2], [3], [4]], [[0, 1, 2, 3, 4]]):
            index = HierarchicalTextBoundIndex.fit(vectors, groups)
            for policy in ("clip", "best_bound"):
                result = self.assert_matches_full(index, [1., 0.], list(reversed(range(len(groups)))), 2, policy)
                self.assertEqual(result["selected_indices"], [0, 4])
                if len(groups) == 5:
                    self.assertEqual(result["bound_dot_count"], 0)
                    self.assertEqual(result["leaf_dot_count"], 5)

    def test_pruning_reduces_total_coordinate_products_not_just_leaf_counter(self):
        vectors = np.vstack([np.array([[1., 0.]]),
                             np.tile([-1., 0.], (15, 1)),
                             np.tile([0., 1.], (15, 1))])
        index = HierarchicalTextBoundIndex.fit(vectors, [[0], list(range(1, 16)), list(range(16, 31))])
        for policy in ("clip", "best_bound"):
            result = self.assert_matches_full(index, [1., 0.], [0, 1, 2], 1, policy)
            self.assertEqual(result["selected_indices"], [0])
            self.assertEqual(result["scored_leaves"], 3)
            self.assertEqual(result["bound_dot_count"], 2)
            self.assertEqual(result["leaf_dot_count"], 1)
            self.assertEqual(result["scalar_coordinate_products"], 6)
            self.assertLess(result["scalar_coordinate_products"], vectors.size)

    def test_every_member_is_a_unique_pivot_and_children_exclude_it(self):
        rng = np.random.default_rng(124)
        index = HierarchicalTextBoundIndex.fit(rng.normal(size=(31, 7)), [list(range(15)), list(range(15, 31))])
        self.assertEqual(len(index.nodes), 31)
        self.assertEqual(sorted(node.pivot_index for node in index.nodes), list(range(31)))
        for node in index.nodes:
            self.assertIn(node.pivot_index, node.members)
            children = [index.nodes[child] for child in node.children]
            child_members = [member for child in children for member in child.members]
            self.assertEqual(sorted(child_members), [member for member in node.members if member != node.pivot_index])
            self.assertEqual(len(child_members), len(set(child_members)))

    def test_fit_is_deterministic_and_owns_text_vectors(self):
        raw = np.array([[1., 0.], [.8, .6], [0., 1.], [-1., 0.]])
        first = HierarchicalTextBoundIndex.fit(raw, [[0, 1, 2], [3]])
        second = HierarchicalTextBoundIndex.fit(raw.copy(), [[0, 1, 2], [3]])
        self.assertEqual(first.nodes, second.nodes)
        np.testing.assert_array_equal(first.vectors, second.vectors)
        raw[:] = 99
        np.testing.assert_array_equal(first.vectors, second.vectors)
        with self.assertRaises(ValueError):
            first.vectors[0, 0] = 0
        state = first.export_state()
        self.assertEqual(state["kind"], "text_member_pivot_band_forest")
        state["text_vectors"][:] = 0
        np.testing.assert_array_equal(first.vectors, second.vectors)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            HierarchicalTextBoundIndex.fit([[0., 0.]], [[0]])
        with self.assertRaises(ValueError):
            HierarchicalTextBoundIndex.fit(np.eye(3), [[0, 1], [1, 2]])
        index = HierarchicalTextBoundIndex.fit(np.eye(3), [[0, 1], [2]])
        for order, limit, policy, query in (
            ([0, 0], 1, "clip", [1, 0, 0]), ([0], 1, "clip", [1, 0, 0]),
            ([0, 1], 0, "clip", [1, 0, 0]), ([0, 1], 4, "clip", [1, 0, 0]),
            ([0, 1], True, "clip", [1, 0, 0]), ([0, 1], 1, "unknown", [1, 0, 0]),
            ([0, 1], 1, "clip", [0, 0, 0]), ([0, 1], 1, "clip", [1, np.nan, 0]),
        ):
            with self.subTest(order=order, limit=limit, policy=policy, query=query), self.assertRaises(ValueError):
                index.search(query, order, limit, policy=policy)


if __name__ == "__main__":
    unittest.main()
