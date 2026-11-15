"""Behavioral checks for caption-driven partitions and compressed visual updates.

The visual vectors below are retained by the tests as a truth oracle only.  The
memory under test must retain just two visual coordinates per leaf.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy
import json
import unittest

import numpy as np

from scripts.graph.caption_ward_visual import (
    CaptionWardVisualMemory,
    choose_caption_partition,
    update_visual_partition,
)


BASIS = np.eye(4, dtype=np.float64)[:, :2]


def member_sets(masters):
    return {frozenset(master["member_ids"]) for master in masters.values()}


def master_for(masters, members):
    expected = set(members)
    return next(master for master in masters.values()
                if set(master["member_ids"]) == expected)


def make_master(master_id, members, visual_sum):
    return {
        "id": master_id,
        "member_ids": list(members),
        "n": len(members),
        "visual_sum": np.asarray(visual_sum, dtype=np.float64),
        "source_overlaps": {},
    }


class CaptionPartitionTests(unittest.TestCase):
    def test_startup_and_identical_caption_degeneracy(self):
        # Startup singletons avoid claiming an evidence-based two-item cut.
        for count in (1, 2):
            with self.subTest(count=count):
                result = choose_caption_partition(np.tile([1., 0., 0.], (count, 1)))
                self.assertEqual(result["selected_k"], count)
                self.assertEqual(sorted(map(len, result["partition"])), [1] * count)
                self.assertEqual(result["candidate_scores"], [])
        result = choose_caption_partition(np.tile([1., 0., 0.], (4, 1)))
        self.assertEqual(result["partition"], [[0, 1, 2, 3]])
        self.assertEqual(result["selected_k"], 1)

    def test_duplicate_groups_and_all_exact_cut_sizes(self):
        result = choose_caption_partition(np.array([
            [1., 0.], [1., 0.], [0., 1.], [0., 1.],
        ]))
        self.assertEqual({frozenset(group) for group in result["partition"]},
                         {frozenset([0, 1]), frozenset([2, 3])})
        self.assertEqual(result["selected_k"], 2)
        self.assertEqual([row["k"] for row in result["candidate_scores"]], [2, 3])
        self.assertAlmostEqual(result["candidate_scores"][0]["score"], 1.)
        self.assertAlmostEqual(result["candidate_scores"][1]["score"], .5)
        # Zero-height merges must still yield exactly K groups, rather than
        # dropping candidate K values because a height-based cut collapses ties.
        self.assertTrue(all(np.isfinite(row["score"])
                            for row in result["candidate_scores"]))

    def test_positive_rescaling_of_each_caption_does_not_change_partition(self):
        vectors = np.array([
            [1., .03, 0.], [1., -.03, 0.], [0., .03, 1.],
            [0., -.03, 1.], [-1., .02, 0.], [-1., -.02, 0.],
        ])
        base = choose_caption_partition(vectors)
        scaled = choose_caption_partition(vectors * np.array([2., 30., .1, 4., .5, 7.])[:, None])
        self.assertEqual(base["partition"], scaled["partition"])
        np.testing.assert_allclose(
            [row["score"] for row in base["candidate_scores"]],
            [row["score"] for row in scaled["candidate_scores"]], atol=1e-13,
        )

    def test_equal_silhouettes_prefer_fewer_clusters(self):
        # Four orthogonal unit vectors have identical pairwise distances:
        # every non-singleton silhouette and the singleton convention are zero.
        result = choose_caption_partition(np.eye(4))
        self.assertEqual(result["selected_k"], 2)
        np.testing.assert_allclose(
            [row["score"] for row in result["candidate_scores"]], 0., atol=1e-13,
        )


class VisualAggregateTests(unittest.TestCase):
    def assert_projection_invariant(self, masters, coords):
        for master in masters.values():
            expected = np.sum([coords[member] for member in master["member_ids"]], axis=0)
            np.testing.assert_allclose(BASIS.T @ master["visual_sum"], expected, atol=1e-13)
            self.assertEqual(master["n"], len(master["member_ids"]))

    def test_append_and_unchanged_master_preserve_known_sums_and_identity(self):
        unchanged = make_master("M1", ["stay"], [0., 0., 1., 0.])
        old = {
            "M1": unchanged,
            "M2": make_master("M2", ["a", "b"], [.6, .6, .8, .8]),
        }
        coords = {"stay": [0., 0.], "a": [.6, 0.], "b": [0., .6]}
        before = copy.deepcopy(old)
        masters, _, _ = update_visual_partition(
            old, coords, [["stay"], ["a", "b", "c"]],
            {"c": np.array([0., 0., 1., 0.])}, BASIS, 3,
        )
        stable = master_for(masters, ["stay"])
        self.assertEqual(stable["id"], "M1")
        np.testing.assert_array_equal(stable["visual_sum"], unchanged["visual_sum"])
        np.testing.assert_allclose(
            master_for(masters, ["a", "b", "c"])["visual_sum"], [.6, .6, 1.8, .8], atol=1e-13,
        )
        for key in before:
            np.testing.assert_array_equal(old[key]["visual_sum"], before[key]["visual_sum"])
            self.assertEqual(old[key]["member_ids"], before[key]["member_ids"])
        coords["c"] = [0., 0.]
        self.assert_projection_invariant(masters, coords)

    def test_partial_migration_has_only_unobserved_direction_error_and_conserves_sum(self):
        # True vectors: a=(.6,0,.8,0), b=(0,.6,0,.8), c=(0,0,1,0).
        # Splitting a/b distributes their unavailable residual equally.
        old = {"M1": make_master("M1", ["a", "b"], [.6, .6, .8, .8])}
        coords = {"a": [.6, 0.], "b": [0., .6]}
        masters, _, next_index = update_visual_partition(
            old, coords, [["a", "c"], ["b"]],
            {"c": np.array([0., 0., 1., 0.])}, BASIS, 2,
        )
        ac = master_for(masters, ["a", "c"])["visual_sum"]
        b = master_for(masters, ["b"])["visual_sum"]
        np.testing.assert_allclose(ac, [.6, 0., 1.4, .4], atol=1e-13)
        np.testing.assert_allclose(b, [0., .6, .4, .4], atol=1e-13)
        np.testing.assert_allclose(ac - [.6, 0., 1.8, 0.], [0., 0., -.4, .4], atol=1e-13)
        self.assertGreater(np.linalg.norm(ac - [.6, 0., 1.8, 0.]), .5)
        np.testing.assert_allclose(ac + b, [.6, .6, 1.8, .8], atol=1e-13)
        coords["c"] = [0., 0.]
        self.assert_projection_invariant(masters, coords)

        # Recombining all portions recovers the true global sum despite the
        # deliberately nonzero individual-cluster approximation error.
        merged, _, _ = update_visual_partition(
            masters, coords, [["a", "b", "c"]], {}, BASIS, next_index,
        )
        np.testing.assert_allclose(
            master_for(merged, ["a", "b", "c"])["visual_sum"], [.6, .6, 1.8, .8], atol=1e-13,
        )

    def test_simultaneous_cross_cluster_migrations_are_order_independent(self):
        old = {
            "M1": make_master("M1", ["a", "b"], [.6, .6, .8, .8]),
            "M2": make_master("M2", ["c", "d"], [0., 0., 1., 1.]),
        }
        coords = {"a": [.6, 0.], "b": [0., .6], "c": [0., 0.], "d": [0., 0.]}
        left, _, _ = update_visual_partition(
            old, coords, [["a", "c"], ["b", "d"]], {}, BASIS, 3,
        )
        right, _, _ = update_visual_partition(
            old, coords, [["b", "d"], ["a", "c"]], {}, BASIS, 3,
        )
        for members, expected in [(["a", "c"], [.6, 0., .9, .9]),
                                  (["b", "d"], [0., .6, .9, .9])]:
            np.testing.assert_allclose(master_for(left, members)["visual_sum"], expected, atol=1e-13)
            np.testing.assert_allclose(
                master_for(left, members)["visual_sum"],
                master_for(right, members)["visual_sum"], atol=1e-13,
            )
        np.testing.assert_allclose(
            sum(master["visual_sum"] for master in left.values()), [.6, .6, 1.8, 1.8], atol=1e-13,
        )
        self.assert_projection_invariant(left, coords)

    def test_unequal_split_allocates_hidden_residual_by_member_count(self):
        old = {"M1": make_master("M1", ["a", "b", "c"], [.6, .6, 1.8, .8])}
        coords = {"a": [.6, 0.], "b": [0., .6], "c": [0., 0.]}
        masters, _, _ = update_visual_partition(
            old, coords, [["a", "b"], ["c"]], {}, BASIS, 2,
        )
        np.testing.assert_allclose(
            master_for(masters, ["a", "b"])["visual_sum"], [.6, .6, 1.2, 8 / 15], atol=1e-13,
        )
        np.testing.assert_allclose(
            master_for(masters, ["c"])["visual_sum"], [0., 0., .6, 4 / 15], atol=1e-13,
        )
        np.testing.assert_allclose(
            sum(master["visual_sum"] for master in masters.values()), [.6, .6, 1.8, .8], atol=1e-13,
        )
        self.assert_projection_invariant(masters, coords)

    def test_same_members_reordered_preserve_master_ids_and_exact_arrays(self):
        old = {
            "M1": make_master("M1", ["a", "b"], [.6, .6, .8, .8]),
            "M2": make_master("M2", ["c"], [0., 0., 1., 0.]),
        }
        coords = {"a": [.6, 0.], "b": [0., .6], "c": [0., 0.]}
        updated, audit, next_index = update_visual_partition(
            old, coords, [["c"], ["b", "a"]], {}, BASIS, 3,
        )
        self.assertEqual(next_index, 3)
        self.assertEqual(set(audit["unchanged_master_ids"]), {"M1", "M2"})
        self.assertEqual(audit["changed_master_ids"], [])
        for master_id, previous in old.items():
            current = master_for(updated, previous["member_ids"])
            self.assertEqual(current["id"], master_id)
            np.testing.assert_array_equal(current["visual_sum"], previous["visual_sum"])


class CaptionWardMemoryTests(unittest.TestCase):
    def test_visual_inputs_never_control_caption_partition(self):
        text_vectors = [[1., 0.], [0., 1.], [1., .02], [.02, 1.], [1., -.02]]
        first = CaptionWardVisualMemory(BASIS)
        second = CaptionWardVisualMemory(BASIS)
        expected_sum_first = np.zeros(4)
        expected_sum_second = np.zeros(4)
        for index, text_vector in enumerate(text_vectors):
            image_id = f"image_{index}"
            v1 = np.eye(4)[index % 4]
            v2 = np.array([index + 1., 4. - index, -2., 3.])
            first.add(image_id, f"full caption {index}", text_vector, v1)
            second.add(image_id, f"full caption {index}", text_vector, v2)
            self.assertEqual(member_sets(first.masters), member_sets(second.masters))
            expected_sum_first += v1
            expected_sum_second += v2 / np.linalg.norm(v2)
            np.testing.assert_allclose(
                sum(master["visual_sum"] for master in first.masters.values()),
                expected_sum_first, atol=1e-12,
            )
            np.testing.assert_allclose(
                sum(master["visual_sum"] for master in second.masters.values()),
                expected_sum_second, atol=1e-12,
            )
            for memory in (first, second):
                for master in memory.masters.values():
                    coords = np.sum([memory.image_nodes[member]["visual_coords"]
                                     for member in master["member_ids"]], axis=0)
                    np.testing.assert_allclose(BASIS.T @ master["visual_sum"], coords, atol=1e-12)

    def test_input_aliases_cannot_change_stored_content_or_visual_statistics(self):
        memory = CaptionWardVisualMemory(BASIS)
        visual = np.array([.6, 0., .8, 0.])
        text = np.array([1., 0.])
        dialogue = [{"role": "user", "content": "complete original dialogue"}]
        memory.add("a", "full original caption", text, visual, dialogue, "2026-10-20")
        visual[:] = 100.
        text[:] = -100.
        dialogue[0]["content"] = "later mutation"
        np.testing.assert_allclose(master_for(memory.masters, ["a"])["visual_sum"],
                                   [.6, 0., .8, 0.], atol=1e-13)
        self.assertEqual(memory.image_nodes["a"]["dialogue"][0]["content"],
                         "complete original dialogue")
        self.assertEqual(memory.image_nodes["a"]["caption"], "full original caption")
        np.testing.assert_allclose(memory.image_nodes["a"]["visual_coords"], [.6, 0.], atol=1e-13)

    def test_snapshot_exports_only_two_visual_coordinates_per_leaf(self):
        memory = CaptionWardVisualMemory(BASIS)
        for index, visual in enumerate([[.6, 0., .8, 0.], [0., .6, 0., .8], [0., 0., 1., 0.]]):
            memory.add(f"image_{index}", f"caption {index}", [1., 0.], visual,
                       [{"role": "user", "content": f"dialogue {index}"}], "2026-10-20")
        snapshot = json.loads(json.dumps(memory.snapshot(), allow_nan=False))
        arrays = memory.export_visual_arrays()
        self.assertEqual(len(snapshot["masters"]), 1)
        expected_array_keys = {"shared_basis"}
        for master in snapshot["masters"]:
            self.assertNotIn("visual_sum", master)
            self.assertNotIn("visual_mean", master)
            expected_array_keys.update([master["visual_sum_ref"], master["visual_mean_ref"]])
            np.testing.assert_allclose(arrays[master["visual_mean_ref"]],
                                       arrays[master["visual_sum_ref"]] / master["n"])
        self.assertEqual(set(arrays), expected_array_keys)
        self.assertEqual(snapshot["leaf_visual_dim"], 2)
        for node in snapshot["image_nodes"]:
            self.assertEqual(len(node["visual_coords"]), 2)
            self.assertEqual({key for key in node if "visual" in key}, {"visual_coords"})
            self.assertFalse({"image_embedding", "image_path", "raw_image", "visual_vector"} & set(node))
        # Exported files are detached; an external consumer cannot silently
        # change the live basis or any master by mutating an exported array.
        for array in arrays.values():
            array[:] = 99.
        np.testing.assert_array_equal(memory.basis, BASIS)
        np.testing.assert_allclose(
            next(iter(memory.masters.values()))["visual_sum"], [.6, .6, 1.8, .8], atol=1e-13,
        )

    def test_saved_prefix_is_unchanged_after_future_caption_reorganization(self):
        memory = CaptionWardVisualMemory(BASIS)
        for index, visual in enumerate(np.eye(4)[:3]):
            memory.add(str(index), f"caption {index}", [1., 0.], visual)
        prefix = memory.snapshot()
        prefix_json = json.dumps(prefix, sort_keys=True, allow_nan=False)
        prefix_arrays = memory.export_visual_arrays()
        original_arrays = {key: value.copy() for key, value in prefix_arrays.items()}
        self.assertEqual(len(prefix["masters"]), 1)

        memory.add("future", "distinct future caption", [0., 1.], [0., 0., 0., 1.])
        self.assertEqual(len(memory.masters), 2)
        self.assertEqual(json.dumps(prefix, sort_keys=True, allow_nan=False), prefix_json)
        self.assertEqual({node["id"] for node in prefix["image_nodes"]}, {"0", "1", "2"})
        for key, value in original_arrays.items():
            np.testing.assert_array_equal(prefix_arrays[key], value)


if __name__ == "__main__":
    unittest.main()
