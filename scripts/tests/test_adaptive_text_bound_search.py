"""Control-flow checks for bound-guided traversal without encoding any model."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

import numpy as np

from scripts.analysis.evaluate_text_similarity_bounds import adaptive_search


class AdaptiveTextBoundSearchTests(unittest.TestCase):
    def recording_callback(self, permitted):
        called = []

        def score_cluster(index):
            called.append(index)
            self.assertIn(index, permitted, "A skipped or stopped cluster was scored")
            return permitted[index]

        return called, score_cluster

    def test_low_next_bound_skips_but_later_high_bound_is_visited(self):
        called, callback = self.recording_callback({
            0: [(0, .7)],
            2: [(2, .9)],
        })
        result = adaptive_search([0, 1, 2], [.8, .2, .95], [[0], [1], [2]], callback, 1)
        self.assertEqual(called, [0, 2])
        self.assertEqual(result["selected_indices"], [2])
        self.assertEqual(result["skipped_indices"], [1])
        self.assertEqual(result["tail_indices"], [])
        self.assertEqual([step["action"] for step in result["trace"]], ["visit", "skip", "visit"])
        self.assertEqual(result["trace"][1]["remaining_max_before"], .95)

    def test_stops_only_when_all_remaining_bounds_are_below_cutoff(self):
        called, callback = self.recording_callback({0: [(0, .8), (1, .7)]})
        result = adaptive_search([0, 2, 1, 3], [.8, .6, .3, .4],
                                 [[0, 1], [2], [3], [4]], callback, 2)
        self.assertEqual(called, [0])
        self.assertEqual(result["selected_indices"], [0, 1])
        self.assertEqual(result["tail_indices"], [2, 1, 3])
        self.assertEqual(result["scored_leaf_count"], 2)
        self.assertEqual(result["stop_reason"], "all_remaining_bounds_below_current_cutoff")
        self.assertEqual([step["action"] for step in result["trace"]], ["visit", "stop"])
        self.assertLess(result["trace"][-1]["remaining_max_before"],
                        result["trace"][-1]["threshold_before"])

    def test_equal_bound_must_continue_and_ties_use_global_leaf_order(self):
        called, callback = self.recording_callback({
            0: [(0, .5)], 1: [(1, .5)], 2: [(2, .5)],
        })
        result = adaptive_search([1, 0, 2], [.5, .5, .5], [[0], [1], [2]], callback, 1)
        self.assertEqual(called, [1, 0, 2])
        self.assertEqual(result["selected_indices"], [0])
        self.assertEqual(result["selected_scores"], [.5])
        self.assertEqual(result["skipped_indices"], [])
        self.assertEqual(result["tail_indices"], [])
        self.assertEqual([step["action"] for step in result["trace"]], ["visit"] * 3)

    def test_low_bounds_cannot_stop_before_collecting_requested_leaf_count(self):
        called, callback = self.recording_callback({
            0: [(0, .8)], 1: [(1, -.4)], 2: [(2, -.6)],
        })
        result = adaptive_search([0, 1, 2], [.9, -.3, -.5], [[0], [1], [2]], callback, 3)
        self.assertEqual(called, [0, 1, 2])
        self.assertEqual(result["selected_indices"], [0, 1, 2])
        self.assertEqual(result["scored_leaf_count"], 3)
        self.assertTrue(all(step["threshold_before"] is None for step in result["trace"]))
        self.assertEqual(result["trace"][-1]["threshold_after"], -.6)

    def test_incomplete_wrong_order_duplicate_and_nonfinite_callback_results_rejected(self):
        cases = [
            ([[0, 1]], {0: [(0, .3)]}),
            ([[0, 1]], {0: [(1, .3), (0, .4)]}),
            ([[0, 1]], {0: [(0, .3), (0, .4)]}),
            ([[0]], {0: [(0, np.nan)]}),
            ([[0]], {0: [(0, np.inf)]}),
            # Duplicate membership across clusters is rejected when the callback
            # tries to score the same leaf twice, rather than silently overwriting.
            ([[0], [0]], {0: [(0, .3)], 1: [(0, .4)]}),
        ]
        for members, payloads in cases:
            with self.subTest(members=members, payloads=payloads):
                _, callback = self.recording_callback(payloads)
                with self.assertRaises(ValueError):
                    adaptive_search(list(range(len(members))), [1.] * len(members),
                                    members, callback, 1)


if __name__ == "__main__":
    unittest.main()
