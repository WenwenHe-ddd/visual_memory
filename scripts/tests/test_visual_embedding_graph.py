"""Behavioral checks for streaming visual clustering and aggregate distances."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy
import json
import math
import unittest

from scripts.graph.visual_embedding_graph import (
    APPROXIMATE_DISTANCE,
    EXACT_DISTANCE,
    METHOD,
    VisualEmbeddingGraph,
)


def unit(vector):
    length = math.sqrt(sum(part * part for part in vector))
    return [part / length for part in vector]


def polar(degrees):
    angle = math.radians(degrees)
    return [math.cos(angle), math.sin(angle)]


def insert(graph, node_id, vector):
    return graph.insert(
        node_id, vector,
        dialogue=[{"role": "user", "content": f"complete dialogue {node_id}"}],
        caption=f"complete caption {node_id}",
        timestamp="2026-10-19T08:00:00+08:00",
        image_path=f"/images/{node_id}.png",
    )


class VisualEmbeddingGraphTests(unittest.TestCase):
    def test_first_master_zero_threshold_and_zero_distance(self):
        graph = VisualEmbeddingGraph()
        result = insert(graph, "first", [3, 4])
        self.assertTrue(result["new_master"])
        self.assertEqual(result["master_id"], "M1")
        self.assertEqual(result["candidates"], [])
        self.assertEqual(graph.masters[0]["visual_mean"], [0.6, 0.8])
        self.assertEqual(graph.masters[0]["join_threshold"], 0)
        self.assertEqual(graph.edges["first"]["distance"], 0)
        self.assertEqual(graph.edges["first"]["distance_semantics"], EXACT_DISTANCE)
        graph.validate()

    def test_zero_similarity_fails_strict_bootstrap_threshold(self):
        graph = VisualEmbeddingGraph()
        insert(graph, "x", [1, 0])
        result = insert(graph, "y", [0, 1])
        self.assertTrue(result["new_master"])
        self.assertEqual(result["master_id"], "M2")
        self.assertEqual(result["threshold_after"], 0)
        self.assertFalse(result["candidates"][0]["eligible"])

    def test_new_master_keeps_negative_nearest_master_similarity(self):
        graph = VisualEmbeddingGraph()
        insert(graph, "x", [1, 0])
        result = insert(graph, "opposite", [-1, 0])
        self.assertTrue(result["new_master"])
        self.assertEqual(result["threshold_after"], -1)
        # A singleton threshold is a bootstrap rule, not its mean similarity.
        self.assertEqual(graph.edges["opposite"]["distance"], 0)
        joined = insert(graph, "y", [0, 1])
        self.assertFalse(joined["new_master"])
        self.assertEqual(joined["master_id"], "M2")
        self.assertEqual(joined["threshold_before"], -1)
        self.assertAlmostEqual(joined["threshold_after"], math.sqrt(0.5))
        graph.validate()

    def test_choose_highest_similarity_among_eligible_not_largest_margin(self):
        graph = VisualEmbeddingGraph()
        insert(graph, "zero", polar(0))
        insert(graph, "sixty", polar(60))
        separated = insert(graph, "hundred", polar(100))
        self.assertTrue(separated["new_master"])
        # The first center is at 30 degrees, not at its first member (0).
        self.assertAlmostEqual(separated["threshold_after"], math.cos(math.radians(70)))
        result = insert(graph, "forty", polar(40))
        first, second = result["candidates"]
        self.assertTrue(first["eligible"] and second["eligible"])
        self.assertGreater(first["similarity"], second["similarity"])
        self.assertLess(first["similarity"] - first["threshold"],
                        second["similarity"] - second["threshold"])
        self.assertEqual(result["master_id"], "M1")

    def test_ties_use_creation_order_and_threshold_filter_precedes_ranking(self):
        graph = VisualEmbeddingGraph()
        insert(graph, "x", [1, 0])
        insert(graph, "y", [0, 1])
        result = insert(graph, "diagonal", [1, 1])
        self.assertEqual(result["candidates"][0]["similarity"],
                         result["candidates"][1]["similarity"])
        self.assertEqual(result["master_id"], "M1")
        # M1's center has moved toward the diagonal and its threshold tightened.
        result = insert(graph, "thirty", polar(30))
        self.assertEqual(result["master_id"], "M1")
        result = insert(graph, "fifty", polar(50))
        self.assertGreater(result["candidates"][0]["similarity"],
                           result["candidates"][1]["similarity"])
        self.assertFalse(result["candidates"][0]["eligible"])
        self.assertTrue(result["candidates"][1]["eligible"])
        self.assertEqual(result["master_id"], "M2")

    def test_identical_image_at_threshold_one_is_rejected_strictly(self):
        graph = VisualEmbeddingGraph()
        insert(graph, "a", [1, 0])
        insert(graph, "b", [1, 0])
        self.assertEqual(graph.masters[0]["join_threshold"], 1)
        result = insert(graph, "c", [1, 0])
        self.assertTrue(result["new_master"])
        self.assertEqual(result["threshold_after"], 1)

    def test_aggregate_mean_is_exact_but_individual_old_distances_are_approximate(self):
        graph = VisualEmbeddingGraph()
        vectors = [[1, 0, 0], [0.6, 0.8, 0], [1, 0.4, 0.2], [1, 0.5, 0.2]]
        retained_in_test_only = []
        saw_individual_error = False
        for index, vector in enumerate(vectors):
            previous_edges = copy.deepcopy(graph.edges)
            result = insert(graph, str(index), vector)
            self.assertEqual(result["master_id"], "M1")
            retained_in_test_only.append(unit(vector))
            mean = [sum(parts) / len(retained_in_test_only)
                    for parts in zip(*retained_in_test_only)]
            center = unit(mean)
            true_distances = [1 - sum(a * b for a, b in zip(member, center))
                              for member in retained_in_test_only]
            stored_distances = [edge["distance"] for edge in graph.edges.values()]
            for actual, expected in zip(graph.masters[0]["visual_mean"], mean):
                self.assertAlmostEqual(actual, expected, places=14)
            self.assertAlmostEqual(sum(stored_distances) / len(stored_distances),
                                   sum(true_distances) / len(true_distances), places=14)
            self.assertAlmostEqual(stored_distances[-1], true_distances[-1], places=14)
            for image_id, previous in previous_edges.items():
                self.assertAlmostEqual(graph.edges[image_id]["distance"] - previous["distance"],
                                       result["old_edges_distance_delta"], places=14)
                self.assertEqual(graph.edges[image_id]["distance_semantics"], APPROXIMATE_DISTANCE)
            if len(retained_in_test_only) > 1:
                self.assertAlmostEqual(graph.masters[0]["join_threshold"],
                                       1 - sum(true_distances) / len(true_distances), places=14)
            saw_individual_error |= any(abs(a - b) > 1e-5
                                        for a, b in zip(stored_distances, true_distances))
            graph.validate()
        self.assertTrue(saw_individual_error,
                        "a shared correction must not be mislabeled as exact per-image distance")

    def test_snapshot_resume_preserves_decisions_order_and_complete_content(self):
        original = VisualEmbeddingGraph()
        for node_id, degrees in [("zero", 0), ("sixty", 60), ("hundred", 100)]:
            insert(original, node_id, polar(degrees))
        snapshot = json.loads(json.dumps(original.to_dict(), allow_nan=False))
        restored = VisualEmbeddingGraph.from_dict(snapshot)
        self.assertEqual(restored.insertion_order, ["zero", "sixty", "hundred"])
        for node_id, degrees in [("forty", 40), ("negative", 200), ("near", 205)]:
            self.assertEqual(insert(original, node_id, polar(degrees)),
                             insert(restored, node_id, polar(degrees)))
        self.assertEqual(restored.to_dict(), original.to_dict())
        self.assertEqual(restored.image_nodes["zero"]["caption"], "complete caption zero")
        self.assertEqual(restored.image_nodes["zero"]["dialogue"],
                         [{"role": "user", "content": "complete dialogue zero"}])
        self.assertEqual(restored.image_nodes["zero"]["timestamp"], "2026-10-19T08:00:00+08:00")

    def test_no_individual_vectors_in_snapshot_or_decision_and_no_aliasing(self):
        graph = VisualEmbeddingGraph()
        vector = [0.6, 0.8]
        dialogue = [{"role": "user", "content": "keep complete"}]
        decision = graph.insert("a", vector, dialogue=dialogue, caption="full", timestamp=None)
        vector[0] = 99
        dialogue[0]["content"] = "changed"
        insert(graph, "b", [1, 1])
        snapshot = graph.to_dict()
        self.assertEqual(snapshot["method"], METHOD)
        self.assertEqual(set(snapshot["masters"][0]),
                         {"id", "visual_mean", "member_count", "join_threshold"})
        for node in snapshot["image_nodes"]:
            self.assertEqual(set(node), {"id", "image_id", "image_path", "dialogue", "caption", "timestamp"})
        for edge in snapshot["edges"]:
            self.assertEqual(set(edge), {"master_id", "image_id", "distance", "distance_semantics"})
        self.assertNotIn("embedding", json.dumps(decision))
        self.assertEqual(graph.image_nodes["a"]["dialogue"][0]["content"], "keep complete")
        snapshot["masters"][0]["visual_mean"][0] = 99
        self.assertNotEqual(snapshot["masters"][0]["visual_mean"], graph.masters[0]["visual_mean"])
        graph.validate()

    def test_invalid_inputs_do_not_mutate_graph(self):
        graph = VisualEmbeddingGraph()
        insert(graph, "a", [1, 0])
        before = graph.to_dict()
        for vector in ([], [0, 0], [1, 2, 3], [math.nan, 0], [math.inf, 1], [True, 0], "12"):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                insert(graph, "bad", vector)
            self.assertEqual(graph.to_dict(), before)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            insert(graph, "a", [1, 0])
        with self.assertRaisesRegex(ValueError, "JSON"):
            graph.insert("bad", [1, 0], dialogue={"value": math.nan}, caption="full", timestamp=None)
        self.assertEqual(graph.to_dict(), before)

    def test_invalid_saved_payloads_are_rejected_instead_of_silently_dropped(self):
        graph = VisualEmbeddingGraph()
        insert(graph, "a", [1, 0])
        insert(graph, "b", [0.8, 0.6])
        base = graph.to_dict()
        mutations = [
            lambda s: s["image_nodes"][0].update(embedding=[1, 0]),
            lambda s: s["masters"][0].update(shared_caption="unexpected"),
            lambda s: s["masters"][0].update(member_count=99),
            lambda s: s["masters"][0].update(join_threshold=0.2),
            lambda s: s["masters"][0].update(visual_mean=[0, 0]),
            lambda s: s["edges"][0].update(distance=0.9),
            lambda s: s["edges"][0].update(distance=float("nan")),
            lambda s: s["edges"][0].update(master_id="missing"),
            lambda s: s["image_nodes"].append(copy.deepcopy(s["image_nodes"][0])),
            lambda s: s["edges"].reverse(),
            lambda s: s.update(schema_version=True),
        ]
        for mutate in mutations:
            snapshot = copy.deepcopy(base)
            mutate(snapshot)
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                VisualEmbeddingGraph.from_dict(snapshot)

    def test_empty_graph_round_trip_and_extreme_finite_vectors(self):
        empty = VisualEmbeddingGraph()
        self.assertEqual(VisualEmbeddingGraph.from_dict(empty.to_dict()).to_dict(), empty.to_dict())
        for vector in ([1e308, 1e308], [1e-300, 1e-300]):
            graph = VisualEmbeddingGraph()
            insert(graph, "a", vector)
            self.assertAlmostEqual(graph.masters[0]["visual_mean"][0], math.sqrt(0.5))
            graph.validate()


if __name__ == "__main__":
    unittest.main()
