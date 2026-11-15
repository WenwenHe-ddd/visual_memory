"""Independent checks of the one-arrival Brand prototype.

No CLIP model is created here. Synthetic input bytes exercise the feeder; actual
Brand checks inspect output metadata and an existing post-hoc cosine CSV only.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import weakref

import numpy as np

import run_no_original as pilot
from no_replay_sketch import NoReplaySketch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def audit(run_directory: Path | None = None):
    checks = []
    details = {}

    def check(name, passed, **information):
        checks.append({"name": name, "passed": bool(passed), **information})

    class SyntheticEncoder:
        embedding_dim = 32
        metadata = {"kind": "synthetic_test_only"}

        def __init__(self):
            self.paths = []  # Observer state, never part of clustering memory.
            self.tokens = []
            self.vector_refs = []

        def encode(self, path):
            path = Path(path)
            assert path.is_file()
            token = int(path.read_text())
            self.paths.append(path)
            self.tokens.append(token)
            vector = np.random.default_rng(98700 + token).normal(size=self.embedding_dim)
            vector /= np.linalg.norm(vector)
            self.vector_refs.append(weakref.ref(vector))
            return vector

    def inspect_core(graph):
        allowed = {"policy", "sketches", "nearest", "seen", "log", "last_edges", "method"}
        assert set(vars(graph)) == allowed
        assert list(inspect.signature(graph.add).parameters) == ["image_id", "z"]
        for sketch in graph.sketches:
            arrays = {key: value for key, value in vars(sketch).items() if isinstance(value, np.ndarray)}
            assert set(arrays) == {"mean", "basis", "coordinates", "singular_values"}
            assert sketch.basis.shape == (sketch.dimension, sketch.rank)
            assert sketch.coordinates.shape == (sketch.n, sketch.rank)
            assert sketch.rank <= 2
            assert not any(callable(value) or isinstance(value, Path) for value in vars(sketch).values())
        assert not any(callable(value) or isinstance(value, Path) for value in vars(graph).values())
        # Each child's only visual numbers are the at-most-two coordinates.
        # Neighbor identities and connectivity are graph topology, not a saved
        # visual distance, norm, hash projection, or per-member error estimate.
        assert all(type(neighbor) is int and neighbor in graph.seen and neighbor != member
                   for member, neighbor in graph.nearest.items())
        assert all(set(edge) == {"a", "b", "kept"} for edge in graph.last_edges)
        assert all(set(edge) == {"a", "b", "kept"}
                   for event in graph.log for edge in event["edges"])

    def inspect_serialized(directory, graph, nodes, label):
        output = json.loads((directory / "graph.json").read_text())
        check(label + "_payload_preserved", output["image_nodes"] == nodes)
        text = json.dumps(output)
        check(label + "_no_image_path_key", '"image_path"' not in text)
        check(label + "_no_image_bytes", not any(directory.glob("*.png")) and not any(directory.glob("*.jpg")))
        nearest = output["nearest_neighbors"]
        check(label + "_nearest_state_contains_ids_only", all(
            type(neighbor) is int and str(neighbor) in output["stream_indices"] and str(neighbor) != str(member)
            for member, neighbor in nearest.items()))
        check(label + "_selected_edges_contain_no_visual_scalars", all(
            set(edge) == {"a", "b", "kept"} for edge in output["selected_connections"]))
        state_path = directory / output["vector_storage"]["path"]
        check(label + "_npz_digest", hashlib.sha256(state_path.read_bytes()).hexdigest() == output["vector_storage"]["sha256"])
        with np.load(state_path, allow_pickle=False) as arrays:
            expected = {master["id"] + "_" + key for master in output["masters"]
                        for key in ("mean", "basis", "coordinates", "singular_values")}
            check(label + "_only_documented_visual_arrays", set(arrays.files) == expected)
            ids = {node["id"] for node in nodes}
            check(label + "_edge_id_coverage", len(output["edges"]) == len(ids)
                  and {edge["image_id"] for edge in output["edges"]} == ids)
            resolved = []
            for edge in output["edges"]:
                assert set(edge) == {"image_id", "master_id", "residual_coordinates_ref"}
                ref = edge["residual_coordinates_ref"]
                assert set(ref) == {"path", "key", "row"}
                assert ref["path"] == "visual_state.npz"
                row = arrays[ref["key"]][ref["row"]]
                assert row.ndim == 1 and row.size <= 2
                resolved.append((ref["key"], ref["row"]))
            check(label + "_coordinate_indices_unique", len(resolved) == len(set(resolved)))
            check(label + "_children_have_only_at_most_two_visual_coordinates", True)
            for master in output["masters"]:
                mid = master["id"]
                assert master["visual_mean_ref"]["path"] == "visual_state.npz"
                assert master["residual_basis_ref"]["path"] == "visual_state.npz"
                assert np.array_equal(arrays[master["visual_mean_ref"]["key"]], arrays[mid + "_mean"])
                assert arrays[mid + "_basis"].shape == (output["embedding_dim"], master["rank"])
                assert arrays[mid + "_coordinates"].shape == (master["member_count"], master["rank"])
                for edge in output["edges"]:
                    if edge["master_id"] == mid:
                        row_index = edge["residual_coordinates_ref"]["row"]
                        stream_index = master["member_indices"][row_index]
                        assert output["stream_indices"][str(stream_index)] == edge["image_id"]
                summary = master["sketch_summary"]
                if not summary["mean_is_exact"]:
                    assert "estimated" in summary["mean_semantics"]
            check(label + "_all_npz_references_resolve", True)
        if graph is not None:
            check(label + "_serialized_mean_flags_match_live", all(
                master["sketch_summary"]["mean_is_exact"] == sketch.mean_is_exact
                for master, sketch in zip(output["masters"], graph.sketches)))
        return output

    with tempfile.TemporaryDirectory(prefix="audit_no_original_") as temporary:
        workspace = Path(temporary)
        entries = []
        for index in range(48):
            path = workspace / f"fixture_{index}.txt"
            path.write_text(str(index))
            entries.append({"id": f"opaque_{index}", "image_id": f"opaque_{index}",
                            "image_path": str(path), "dialogue": {"user": str(index), "assistant": "reply"},
                            "caption": "完整 caption " + str(index), "timestamp": "2026-10-19"})
        encoder = SyntheticEncoder()
        feeder = pilot.ArrivalInput(encoder, entries)
        graph = pilot.NoOriginalGraph(cut=False)
        running_sum = np.zeros(encoder.embedding_dim)
        nodes = []
        max_error = 0.0
        split_count = 0
        approximate_encounters = 0
        for index in range(len(entries)):
            node, vector = feeder.next()
            assert not any(path.exists() for path in encoder.paths)
            assert "image_path" not in node
            running_sum += vector
            nodes.append(node)
            event = graph.add(index, vector)
            inspect_core(graph)
            assert not any(np.shares_memory(vector, value) for sketch in graph.sketches
                           for value in sketch.array_state().values())
            del vector
            gc.collect()
            assert all(ref() is None for ref in encoder.vector_refs)
            estimate = sum((sketch.n * sketch.mean for sketch in graph.sketches), start=np.zeros(encoder.embedding_dim))
            max_error = max(max_error, float(np.max(np.abs(estimate - running_sum))))
            split_count += event["splits"]
            approximate_encounters += event["approximate_mean_master_count"]
        check("core_has_no_encoder_path_callback_or_vector_cache", True)
        check("core_has_no_persistent_per_child_distance_or_extra_visual_summary", True)
        check("all_temporary_inputs_gone_before_each_add", True)
        check("incoming_embedding_objects_released_after_add", True)
        check("one_encoder_call_per_input", encoder.tokens == list(range(len(entries))))
        check("feeder_consumed_sources_not_retained", not feeder.pending and len(feeder.consumed_ids) == len(entries))
        check("synthetic_global_weighted_mean_invariant", max_error < 1e-11, max_abs_error=max_error)
        details["synthetic_stream"] = {"n": len(entries), "dimension": encoder.embedding_dim,
                                       "splits": split_count, "approximate_mean_encounters": approximate_encounters}
        try:
            feeder.next()
        except StopIteration:
            check("exhausted_feeder_cannot_replay", True)
        else:
            check("exhausted_feeder_cannot_replay", False)
        duplicate_encoder = SyntheticEncoder()
        duplicate = pilot.ArrivalInput(duplicate_encoder, [entries[0], entries[0]])
        duplicate.next()
        try:
            duplicate.next()
        except ValueError:
            check("duplicate_feeder_id_rejected_before_second_encode", len(duplicate_encoder.tokens) == 1)
        else:
            check("duplicate_feeder_id_rejected_before_second_encode", False)
        before = len(graph.log)
        try:
            graph.add(0, np.ones(encoder.embedding_dim) / np.sqrt(encoder.embedding_dim))
        except ValueError:
            check("duplicate_graph_id_rejected_without_mutation", len(graph.log) == before)
        else:
            check("duplicate_graph_id_rejected_without_mutation", False)
        directory = workspace / "synthetic_output"
        graph.save(directory, nodes)
        inspect_serialized(directory, graph, nodes, "synthetic")
        try:
            graph.save(workspace / "invalid_payload", [entries[0]] + nodes[1:])
        except ValueError:
            check("save_rejects_image_path_payload", True)
        else:
            check("save_rejects_image_path_payload", False)

        # Force a lossy split independent of the graph policy, then check flags
        # survive serialization. Synthetic vectors are test oracle data only.
        sketch = NoReplaySketch(max_rank=2)
        for index in range(10):
            v = np.random.default_rng(6500 + index).normal(size=32)
            sketch.append(index, v / np.linalg.norm(v))
        parent_mean = sketch.mean.copy()
        children = sketch.split([list(range(5)), list(range(5, 10))])
        check("lossy_split_means_marked_approximate", all(not child.mean_is_exact for child in children))
        check("lossy_split_preserves_weighted_parent_mean", np.allclose(
            sum(child.n * child.mean for child in children) / 10, parent_mean, atol=1e-12))
        graph.sketches = children
        graph.seen = list(range(10))
        graph.nearest = {}
        graph.last_edges = []
        graph.log = []
        directory = workspace / "lossy_split_output"
        graph.save(directory, nodes[:10])
        inspect_serialized(directory, graph, nodes[:10], "lossy_split")

    if run_directory is not None:
        run_directory = Path(run_directory)
        summary = json.loads((run_directory / "run_summary.json").read_text())
        output = json.loads((run_directory / "graph.json").read_text())
        source = json.loads((ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json").read_text())
        original = {node["id"]: {key: value for key, value in node.items() if key != "image_path"}
                    for node in source["image_nodes"]}
        nodes = [original[node["id"]] for node in output["image_nodes"]]
        output = inspect_serialized(run_directory, None, nodes, "brand")
        calls = json.loads((run_directory / "encoder_calls.json").read_text())
        check("brand_recorded_code_hashes_match_current_files", all(
            hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest
            for filename, digest in summary["code_sha256"].items()))
        check("brand_source_graph_hash_matches", hashlib.sha256(
            (ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json").read_bytes()
        ).hexdigest() == summary["source_graph_sha256"])
        check("brand_one_arrival_call_per_image", len(calls) == len(nodes)
              and len({call["image_id"] for call in calls}) == len(nodes)
              and all(call["purpose"] == "arrival_only" for call in calls))
        check("brand_reports_zero_reencoding", summary["runtime"]["old_image_reencoding_calls"] == 0)
        check("brand_reports_zero_persistent_nearest_distance_bytes", summary["storage"]["nearest_distance_scalar_bytes"] == 0)
        check("brand_global_weighted_mean_invariant", summary["audit"]["max_global_embedding_sum_abs_error"] < 1e-10)
        check("brand_no_claim_of_measured_local_mean_error", summary["audit"]["error_bounds_are_conservative_not_observed_error"])
        logs = [json.loads(line) for line in (run_directory / "admission_log.jsonl").read_text().splitlines()]
        check("brand_prefix_coverage", [row["prefix_count"] for row in logs] == list(range(1, len(nodes) + 1)))
        check("brand_each_prefix_uses_only_arrived_ids", all(
            sorted(i for group in row["memberships"] for i in group) == list(range(row["prefix_count"])) for row in logs))
        check("brand_logs_do_not_persist_per_edge_visual_scalars", all(
            set(edge) == {"a", "b", "kept"} for row in logs for edge in row["edges"]))
        # Existing scalar benchmark matrix is read only after the run finished.
        # No original image bytes or old CLIP embedding is accessed.
        cosine = {}
        with (ROOT / "reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv").open() as handle:
            for row in csv.DictReader(handle):
                if row["task"] == "Brand_Memory_Test":
                    cosine[frozenset((row["earlier_id"], row["later_id"]))] = float(row["cosine"])
        indices = output["stream_indices"]
        true_nn_hits = 0
        nearest_count = len(output["nearest_neighbors"])
        for index, neighbor in output["nearest_neighbors"].items():
            image_id = indices[str(index)]
            neighbor_id = indices[str(neighbor)]
            actual = 1 - cosine[frozenset((image_id, neighbor_id))]
            minimum = min(1 - cosine[frozenset((image_id, other["id"]))]
                          for other in nodes if other["id"] != image_id)
            true_nn_hits += abs(actual - minimum) < 1e-7
        details["brand_posthoc_neighbor_identity_audit"] = {
            "nearest_edges": nearest_count, "true_nearest_neighbor_hits": true_nn_hits,
            "true_nearest_neighbor_fraction": true_nn_hits / nearest_count if nearest_count else None,
            "stored_edge_distance_error": "not applicable: no per-edge distance is persisted",
            "matrix_loaded_by_admission": False,
            "matrix_source": "previously measured pairwise_cosines.csv; posthoc only",
        }

    result = {"all_passed": all(row["passed"] for row in checks), "checks": checks, "details": details,
              "scope": "Independent source and object-state inspection plus synthetic runtime and serialized artifact checks.",
              "limits": ["Not a complete operating-system syscall or information-flow proof.",
                         "The external benchmark source images remain unchanged; the simulated memory has no old-image path or image bytes.",
                         "The test feeder knows future input paths externally; the clustering object only receives the current ID/vector.",
                         "Children retain at most two visual coordinates; graph neighbor IDs are topology, and master means/bases/statistics remain shared visual state.",
                         "No old Brand image is reencoded by this audit. Local exact centroid errors cannot be measured without the forbidden originals/vectors.",
                         "A singleton master is necessarily its arriving vector; low-rank master state is not claimed to erase all recoverable image information."]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "independent_audit.json")
    args = parser.parse_args()
    result = audit(args.run)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_passed": result["all_passed"], "check_count": len(result["checks"]),
                      "details": result["details"], "output": str(args.output)}, indent=2))
    raise SystemExit(0 if result["all_passed"] else 1)
