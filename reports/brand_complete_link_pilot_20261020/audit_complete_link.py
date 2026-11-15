"""Independent synthetic and saved-artifact audit; no image/encoder access.

Full synthetic vectors are test oracles only. Real-image checks read serialized
memory, admission logs and provenance; they never open a benchmark image.
"""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import weakref

import numpy as np

from complete_link import (
    CompleteLinkGraph, HERE, ROOT, NoReplaySketch, max_diameter,
    partition, represented_distances, write_json,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(groups):
    return sorted(tuple(sorted(group)) for group in groups)


def explicit_distances(sketches, ids):
    """Test-only synthetic oracle; production does not materialize these rows."""
    rows = {member: sketch.mean + sketch.basis @ sketch.coordinates[row]
            for sketch in sketches for row, member in enumerate(sketch.member_ids)}
    vectors = np.stack([rows[member] for member in ids])
    denominator = np.linalg.norm(vectors, axis=1)
    denominator = denominator[:, None] * denominator[None, :]
    cosine = np.divide(vectors @ vectors.T, denominator,
                       out=np.zeros_like(denominator), where=denominator > 0)
    cosine[np.abs(1 - cosine) <= np.finfo(float).eps * vectors.shape[1]] = 1
    result = 1 - np.clip(cosine, -1, 1)
    np.fill_diagonal(result, 0)
    return result


def run_audit():
    checks = []
    details = {"scope": "No encoder instantiated; no benchmark image opened."}

    def check(name, passed, **information):
        checks.append({"name": name, "passed": bool(passed), **information})

    def inspect_live(graph):
        assert set(vars(graph)) == {
            "policy", "sketches", "nearest", "seen", "log", "last_edges", "method"}
        assert graph.nearest == {} and graph.last_edges == []
        assert list(inspect.signature(graph.add).parameters) == ["image_id", "z"]
        flattened = [i for sketch in graph.sketches for i in sketch.member_ids]
        assert len(flattened) == len(set(flattened)) == len(graph.seen)
        assert set(flattened) == set(graph.seen)
        for sketch in graph.sketches:
            arrays = {k: v for k, v in vars(sketch).items() if isinstance(v, np.ndarray)}
            assert set(arrays) == {"mean", "basis", "coordinates", "singular_values"}
            assert sketch.rank <= 2
            assert sketch.mean.shape == (sketch.dimension,)
            assert sketch.basis.shape == (sketch.dimension, sketch.rank)
            assert sketch.coordinates.shape == (sketch.n, sketch.rank)
            assert sketch.singular_values.shape == (sketch.rank,)
            assert all(np.isfinite(v).all() for v in arrays.values())
            np.testing.assert_allclose(
                sketch.basis.T @ sketch.basis, np.eye(sketch.rank), atol=1e-11)
            np.testing.assert_allclose(
                sketch.coordinates.sum(axis=0), 0, atol=1e-10)
            np.testing.assert_allclose(
                sketch.coordinates.T @ sketch.coordinates,
                np.diag(sketch.singular_values ** 2), atol=1e-10)
        assert all(event["edges"] == [] for event in graph.log)

    def inspect_saved(directory, expected_nodes=None, name=None):
        name = name or str(directory.relative_to(HERE))
        graph = json.loads((directory / "graph.json").read_text())
        if expected_nodes is not None:
            check(name + ":payload", graph["image_nodes"] == expected_nodes)
        check(name + ":original_path_absent", '"image_path"' not in json.dumps(graph))
        check(name + ":no_raster_files", not any(
            path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
            for path in directory.rglob("*") if path.is_file()))
        check(name + ":no_persistent_pair_or_tree_state",
              graph["nearest_neighbors"] == {} and graph["selected_connections"] == [])
        state = directory / graph["vector_storage"]["path"]
        check(name + ":vector_hash", digest(state) == graph["vector_storage"]["sha256"])
        with np.load(state, allow_pickle=False) as arrays:
            expected = {master["id"] + "_" + field for master in graph["masters"]
                        for field in ("mean", "basis", "coordinates", "singular_values")}
            check(name + ":only_documented_visual_arrays", set(arrays.files) == expected)
            stream_ids = graph["stream_indices"]
            expected_ids = [node["id"] for node in graph["image_nodes"]]
            observed_ids = [edge["image_id"] for edge in graph["edges"]]
            check(name + ":coverage",
                  len(observed_ids) == len(set(observed_ids)) == len(expected_ids)
                  and set(observed_ids) == set(expected_ids))
            coordinate_locations = set()
            loaded = []
            for master in graph["masters"]:
                mid, count, rank = master["id"], master["member_count"], master["rank"]
                assert 0 <= rank <= 2
                assert arrays[mid + "_mean"].shape == (graph["embedding_dim"],)
                assert arrays[mid + "_basis"].shape == (graph["embedding_dim"], rank)
                assert arrays[mid + "_coordinates"].shape == (count, rank)
                assert arrays[mid + "_singular_values"].shape == (rank,)
                assert master["visual_mean_ref"] == {
                    "path": "visual_state.npz", "key": mid + "_mean"}
                assert master["residual_basis_ref"] == {
                    "path": "visual_state.npz", "key": mid + "_basis"}
                s = NoReplaySketch()
                s.member_ids = list(master["member_indices"])
                for field in ("mean", "basis", "coordinates", "singular_values"):
                    setattr(s, field, arrays[mid + "_" + field].copy())
                loaded.append(s)
                np.testing.assert_allclose(s.coordinates.sum(axis=0), 0, atol=1e-10)
                np.testing.assert_allclose(s.basis.T @ s.basis, np.eye(rank), atol=1e-11)
                for edge in graph["edges"]:
                    if edge["master_id"] != mid:
                        continue
                    assert set(edge) == {"image_id", "master_id", "residual_coordinates_ref"}
                    ref = edge["residual_coordinates_ref"]
                    assert set(ref) == {"path", "key", "row"}
                    assert ref["path"] == "visual_state.npz"
                    assert ref["key"] == mid + "_coordinates"
                    assert stream_ids[str(s.member_ids[ref["row"]])] == edge["image_id"]
                    coordinate_locations.add((ref["key"], ref["row"]))
            check(name + ":shape_and_reference_integrity", True)
            check(name + ":only_two_coordinates_per_child",
                  len(coordinate_locations) == len(expected_ids))
            array_bytes = sum(arrays[key].nbytes for key in arrays.files)
        log = [json.loads(line) for line in (directory / "admission_log.jsonl").read_text().splitlines()]
        check(name + ":one_event_per_image",
              [event["prefix_count"] for event in log] == list(range(1, len(expected_ids) + 1)))
        for event in log:
            count = event["prefix_count"]
            assert sorted(i for group in event["memberships"] for i in group) == list(range(count))
            assert len(event["memberships"]) == event["cluster_count"]
            assert event["edges"] == []
            assert not any(key in event for key in ("tree", "pair_distances", "merge_heights"))
            if "applied_cut" in event:
                assert event["maximum_cluster_diameter_at_decision"] <= event["applied_cut"] + 1e-12
        check(name + ":prefix_only_membership_and_decision_diameter", True)
        final_ids = list(range(len(expected_ids)))
        final_groups = [s.member_ids for s in loaded]
        assert canonical(final_groups) == canonical(log[-1]["memberships"])
        saved_distances = represented_distances(loaded, final_ids)
        observed_final_diameter = max_diameter(saved_distances, final_groups, final_ids)
        check(name + ":saved_post_compression_diameter",
              np.isclose(observed_final_diameter,
                         log[-1]["maximum_cluster_diameter_after_compression"], atol=1e-12))
        summary_path = directory / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            check(name + ":storage_accounting",
                  summary["storage"]["sketch_array_bytes"] == array_bytes
                  and summary["storage"]["npz_file_bytes"] == state.stat().st_size
                  and summary["storage"]["stored_pair_distance_bytes"] == 0)
            check(name + ":source_graph_hash",
                  summary["source_graph_sha256"] == digest(
                      ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json"))
            check(name + ":code_hashes",
                  all(digest(ROOT / path) == expected
                      for path, expected in summary["code_sha256"].items()))
            check(name + ":global_first_moment_logged",
                  summary["audit"]["max_global_embedding_sum_abs_error"] < 1e-10
                  and all(event["audit_global_embedding_sum_max_abs_error"] < 1e-10 for event in log))
            check(name + ":no_reencoding_logged",
                  summary["runtime"]["encoder_calls"] == len(expected_ids)
                  and summary["runtime"]["old_image_reencoding_calls"] == 0)
        return graph, log

    # Independent synthetic distance and coordinate-frame invariance oracles.
    rng = np.random.default_rng(602020)
    truth = rng.normal(size=(40, 32))
    truth /= np.linalg.norm(truth, axis=1, keepdims=True)
    sketches = []
    for members in (list(range(8)), list(range(8, 17)), list(range(17, 18))):
        sketch = NoReplaySketch()
        for member in members:
            sketch.append(member, truth[member])
        sketches.append(sketch)
    ids = list(reversed(range(18)))
    factored = represented_distances(sketches, ids)
    explicit = explicit_distances(sketches, ids)
    check("factored_cosine_matches_synthetic_explicit_oracle",
          np.allclose(factored, explicit, atol=1e-12),
          maximum_abs_error=float(np.max(np.abs(factored - explicit))))
    rotated = [s.clone() for s in sketches]
    for s in rotated:
        if s.rank:
            rotation, _ = np.linalg.qr(rng.normal(size=(s.rank, s.rank)))
            s.basis = s.basis @ rotation
            s.coordinates = s.coordinates @ rotation
    check("independent_local_basis_rotations_preserve_distances",
          np.allclose(factored, represented_distances(rotated, ids), atol=1e-12))
    check("distance_matrix_symmetric_finite_bounded_zero_diagonal",
          np.isfinite(factored).all() and (factored >= 0).all()
          and (factored <= 2).all() and np.array_equal(factored, factored.T)
          and np.array_equal(np.diag(factored), np.zeros(len(ids))))
    zero = NoReplaySketch()
    zero.append(100, np.zeros(32))
    unit = NoReplaySketch()
    unit.append(101, truth[0])
    check("zero_reconstruction_has_explicit_finite_convention",
          np.array_equal(represented_distances([zero, unit], [100, 101]),
                         np.array([[0., 1.], [1., 0.]])))

    # Frozen cold-start convention and chaining example; no user-selected cutoff.
    chain = np.array([[0., .1, .35], [.1, 0., .1], [.35, .1, 0.]])
    groups, info = partition(chain, [0, 1, 2])
    check("chain_does_not_connect_all_three",
          sorted(map(len, groups)) == [1, 2] and np.isclose(info["cut"], .225))
    check("singleton_start", partition(np.zeros((1, 1)), [7])[0] == [[7]])
    check("two_nonidentical_deferred",
          canonical(partition(np.array([[0., .2], [.2, 0.]]), [0, 1])[0]) == [(0,), (1,)])
    check("two_duplicates_merge",
          partition(np.zeros((2, 2)), [0, 1])[0] == [[0, 1]])
    tied = np.full((5, 5), .25)
    np.fill_diagonal(tied, 0)
    groups, info = partition(tied, list(range(5)))
    check("constant_positive_heights_explicit_keep_all",
          groups == [list(range(5))] and info["cut_reason"] == "constant_merge_heights_keep_all")
    groups, info = partition(np.zeros((5, 5)), list(range(5)))
    check("all_duplicates_merge", groups == [list(range(5))])
    groups, info = partition(factored, ids)
    check("complete_link_precompression_diameter_bound",
          max_diameter(factored, groups, ids) <= info["applied_cut"] + 1e-12)

    # Complete streaming path with vectors owned only by the synthetic test.
    graph = CompleteLinkGraph()
    check("empty_graph_has_no_visual_memory", graph.sketches == [] and graph.seen == [])
    largest_moment_error = 0.
    largest_distance_error = 0.
    all_released = True
    for index in range(len(truth)):
        current = truth[index].copy()
        reference = weakref.ref(current)
        graph.add(index, current)
        inspect_live(graph)
        assert not any(np.shares_memory(current, array) for s in graph.sketches
                       for array in s.array_state().values())
        del current
        gc.collect()
        all_released &= reference() is None
        estimated = sum((s.n * s.mean for s in graph.sketches), start=np.zeros(32))
        largest_moment_error = max(largest_moment_error, float(
            np.max(np.abs(estimated - truth[:index + 1].sum(axis=0)))))
        distance = represented_distances(graph.sketches, graph.seen)
        largest_distance_error = max(largest_distance_error, float(np.max(np.abs(
            distance - explicit_distances(graph.sketches, graph.seen)))))
    check("live_storage_shape_topology_and_coordinate_invariants", True)
    check("current_vector_objects_released", all_released)
    check("dynamic_global_first_moment_preserved", largest_moment_error < 1e-10,
          maximum_abs_error=largest_moment_error)
    check("all_prefix_factored_geometry_matches_oracle", largest_distance_error < 1e-12,
          maximum_abs_error=largest_distance_error)
    prefix = CompleteLinkGraph()
    for i, z in enumerate(truth[:12]):
        prefix.add(i, z)
    fields = ("memberships", "cut", "cut_reason", "approximate_mean_master_count",
              "maximum_cluster_diameter_at_decision", "maximum_cluster_diameter_after_compression")
    check("synthetic_physically_truncated_prefix_matches",
          all(all(a.get(field) == b.get(field) for field in fields)
              for a, b in zip(prefix.log, graph.log)))
    before_events = len(graph.log)
    try:
        graph.add(0, truth[0])
    except ValueError:
        rejected_duplicate = len(graph.log) == before_events
    else:
        rejected_duplicate = False
    check("duplicate_arrival_rejected_without_mutation", rejected_duplicate)
    for label, invalid in (("nonunit", np.ones(32)), ("nonfinite", np.full(32, np.nan))):
        try:
            graph.add(40, invalid)
        except ValueError:
            passed = len(graph.log) == before_events
        else:
            passed = False
        check(label + "_arrival_rejected_without_mutation", passed)
    parent = sketches[0]
    children = parent.split([parent.member_ids[::2], parent.member_ids[1::2]])
    check("coordinate_only_split_preserves_represented_geometry",
          np.allclose(represented_distances([parent], parent.member_ids),
                      represented_distances(children, parent.member_ids), atol=1e-12))
    check("lossy_split_centers_honestly_marked_approximate",
          not parent.representation_is_exact and all(not s.mean_is_exact for s in children))
    check("split_preserves_weighted_center",
          np.allclose(sum(s.n * s.mean for s in children), parent.n * parent.mean, atol=1e-12))
    with tempfile.TemporaryDirectory(prefix="complete_link_audit_") as temporary:
        directory = Path(temporary)
        nodes = [{"id": f"synthetic_{i}", "dialogue": {"user": str(i)},
                  "caption": "完整 caption", "timestamp": "synthetic"}
                 for i in range(len(truth))]
        graph.save(directory, nodes)
        inspect_saved(directory, nodes, "synthetic_serialization")
    details["synthetic"] = {
        "images": len(truth), "dimension": 32,
        "splits": sum(e["splits"] for e in graph.log),
        "approximate_mean_encounters": sum(e["approximate_mean_master_count"] for e in graph.log),
        "maximum_global_moment_error": largest_moment_error,
    }

    # Actual Brand artifacts: metadata and scalar logs only; never image input.
    source = json.loads((ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json").read_text())
    source_nodes = {node["id"]: {k: v for k, v in node.items() if k != "image_path"}
                    for node in source["image_nodes"]}
    inspected = []
    for directory in sorted((HERE / "runs").glob("*/complete_link")):
        if not (directory / "run_summary.json").exists():
            continue
        graph_json = json.loads((directory / "graph.json").read_text())
        expected = [source_nodes[node["id"]] for node in graph_json["image_nodes"]]
        inspect_saved(directory, expected)
        calls = json.loads((directory.parent / "encoder_calls.json").read_text())
        image_ids = [node["id"] for node in graph_json["image_nodes"]]
        check(str(directory.parent.name) + ":arrival_log_one_encode_per_image",
              len(calls) == len(image_ids) == len(set(image_ids))
              and [call["image_id"] for call in calls] == image_ids
              and all(call["purpose"] == "arrival_only"
                      and call["temporary_input_deleted_before_admission"] for call in calls))
        inspected.append(str(directory.relative_to(HERE)))
    prefix_directory = HERE / "validation/prefix_12/complete_link"
    original_directory = HERE / "runs/original/complete_link"
    if (prefix_directory / "run_summary.json").exists() and (original_directory / "run_summary.json").exists():
        expected = [{k: v for k, v in node.items() if k != "image_path"}
                    for node in source["image_nodes"][:12]]
        _, prefix_log = inspect_saved(prefix_directory, expected)
        original_log = [json.loads(line) for line in
                        (original_directory / "admission_log.jsonl").read_text().splitlines()]
        check("brand_physically_truncated_prefix_matches",
              len(prefix_log) == 12 and all(
                  all(a.get(field) == b.get(field) for field in fields)
                  for a, b in zip(prefix_log, original_log)))
        inspected.append(str(prefix_directory.relative_to(HERE)))
    details["actual_artifact_directories"] = inspected
    result = {
        "schema_version": 1, "passed": all(row["passed"] for row in checks),
        "check_count": len(checks), "failed_count": sum(not row["passed"] for row in checks),
        "checks": checks, "details": details,
        "limitations": [
            "Actual temporary-file deletion and call counts are verified from saved logs; synthetic checks independently verify core vector release.",
            "Actual local centroid errors and original-image distance errors are not measured; no image is opened and no model is instantiated.",
            "Complete-link diameter checks apply to pre-update compressed geometry; post-update drift is audited separately.",
            "Tests establish these invariants and do not establish clustering quality or large-scale efficiency.",
        ],
    }
    write_json(HERE / "independent_audit.json", result)
    print(json.dumps({key: result[key] for key in ("passed", "check_count", "failed_count")}))
    if not result["passed"]:
        print(json.dumps([row for row in checks if not row["passed"]], ensure_ascii=False))
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run_audit()
