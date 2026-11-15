"""Independent synthetic review; never reads Brand embeddings or changes policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from run_streaming_pilot import PrefixEncoder, StreamingGraph


HERE = Path(__file__).resolve().parent


def run_stream(vectors, policy="local_scale_nn"):
    graph = StreamingGraph(policy)
    calls = []
    maximum_mean_error = 0.0
    maximum_distance_formula_error = 0.0
    maximum_energy_error = 0.0
    for i, z in enumerate(vectors):
        visible_count = i + 1

        def reencode(j):
            assert 0 <= j < visible_count, (j, visible_count)
            calls.append((j, visible_count))
            return vectors[j].copy()

        approximate_distances = graph.distances(z)
        for sketch in graph.sketches:
            temporary_reconstruction = sketch.reconstruct()
            norms = np.linalg.norm(temporary_reconstruction, axis=1) * np.linalg.norm(z)
            dots = temporary_reconstruction @ z
            expected = 1 - np.clip(
                np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0), -1, 1
            )
            for j, distance in zip(sketch.member_ids, expected):
                maximum_distance_formula_error = max(
                    maximum_distance_formula_error,
                    abs(approximate_distances[j]["distance"] - float(distance)),
                )
            del temporary_reconstruction
        graph.add(i, z, reencode)
        if policy == "verified_local_scale_nn":
            for j, edge in graph.nearest.items():
                other = edge["neighbor"]
                actual_distance = 1 - float(
                    vectors[j] @ vectors[other]
                    / (np.linalg.norm(vectors[j]) * np.linalg.norm(vectors[other]))
                )
                assert edge["exact_up_to_roundoff"]
                assert abs(edge["distance"] - actual_distance) < 1e-12
        flat = [j for sketch in graph.sketches for j in sketch.member_ids]
        assert sorted(flat) == list(range(visible_count))
        assert len(flat) == len(set(flat))
        for sketch in graph.sketches:
            actual = vectors[sketch.member_ids]
            mean_error = float(np.max(np.abs(sketch.mean - actual.mean(axis=0))))
            energy_error = abs(
                float(np.sum((actual - sketch.reconstruct()) ** 2))
                - sketch.cumulative_discarded_scatter_energy
            )
            maximum_mean_error = max(maximum_mean_error, mean_error)
            maximum_energy_error = max(maximum_energy_error, energy_error)
            # Only permitted arrays occur in the live cluster state.
            assert {
                name for name, value in vars(sketch).items() if isinstance(value, np.ndarray)
            } == {"mean", "basis", "singular_values", "coordinates"}
            assert sketch.basis.shape == (vectors.shape[1], sketch.rank)
            assert sketch.coordinates.shape == (sketch.n, sketch.rank)
    assert maximum_mean_error < 1e-12
    assert maximum_distance_formula_error < 1e-11
    assert maximum_energy_error < 1e-10
    return graph, {
        "reencode_calls": len(calls),
        "future_calls": sum(j >= visible for j, visible in calls),
        "max_mean_error": maximum_mean_error,
        "max_cosine_formula_error": maximum_distance_formula_error,
        "max_discarded_energy_error": maximum_energy_error,
        "split_masters": sum(event["split_masters"] for event in graph.log),
        "approximate_distance_queries": sum(event["query_distances_approximate"] for event in graph.log),
        "master_count": len(graph.sketches),
    }


def canonical_prefix(log):
    return [
        {key: row[key] for key in ("memberships", "cut_log_ratio", "edges")}
        for row in log
    ]


def main():
    scenarios = []
    for policy in ("local_scale_nn", "uncut_nn", "verified_local_scale_nn"):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            vectors = rng.normal(size=(30, 12))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            graph, audit = run_stream(vectors, policy)
            prefix, _ = run_stream(vectors[:12], policy)
            assert canonical_prefix(graph.log[:12]) == canonical_prefix(prefix.log)
            scenarios.append({"policy": policy, "seed": seed, **audit})

    # Roundtrip the exported external arrays and image-to-master row mapping.
    # This verifies persistence, not resume (the isolated driver has no loader).
    nodes = [
        {"id": f"synthetic:{i}", "full_caption": f"caption {i}", "dialogue": [str(i)],
         "timestamp": f"time {i}", "image_path": f"synthetic-image-{i}"}
        for i in range(len(vectors))
    ]
    with tempfile.TemporaryDirectory(prefix="brand_stream_review_") as temp:
        saved = graph.save(temp, nodes)
        loaded_graph = json.loads((Path(temp) / "graph.json").read_text())
        assert loaded_graph["image_nodes"] == nodes
        assert saved["image_nodes"] == nodes
        with np.load(Path(temp) / "visual_state.npz") as state:
            for edge in loaded_graph["edges"]:
                index = int(edge["image_id"].split(":")[1])
                master = edge["master_id"]
                ref = edge["residual_coordinates_ref"]
                reconstructed = state[f"{master}_mean"] + state[f"{master}_basis"] @ state[ref["key"]][ref["row"]]
                owner = next(s for s in graph.sketches if index in s.member_ids)
                np.testing.assert_allclose(reconstructed, owner.reconstruct([index])[0], atol=1e-14)
        assert loaded_graph["vector_storage"]["sha256"] == hashlib.sha256(
            (Path(temp) / "visual_state.npz").read_bytes()
        ).hexdigest()

    # Exact duplicates are especially sensitive to a bare ``distance == 0``.
    duplicate_failures = []
    for seed in range(30):
        vector = np.random.default_rng(seed).normal(size=512)
        vector /= np.linalg.norm(vector)
        graph, audit = run_stream(np.vstack((vector, vector)))
        if len(graph.sketches) != 1:
            duplicate_failures.append({
                "seed": seed,
                "unit_norm_squared": float(vector @ vector),
                "distance": graph.nearest[0]["distance"],
                "master_count": len(graph.sketches),
            })

    # Existing HOME relationship: R2 and R4 should become the close pair after
    # a third observation; two nonidentical observations remain provisional.
    gram = np.array([[1, .7623, .9330], [.7623, 1, .7789], [.9330, .7789, 1]])
    home = np.linalg.cholesky(gram)
    home_graph, home_audit = run_stream(home)
    home_groups = sorted(sorted(s.member_ids) for s in home_graph.sketches)
    assert home_groups == [[0, 2], [1]]
    assert home_graph.log[1]["cluster_count"] == 2

    # Guard proves unseen images cannot be encoded even if paths are known.
    class DummyEncoder:
        def __init__(self):
            self.calls = 0

        def encode(self, path):
            self.calls += 1
            return np.array([1., 0.])

    encoder = DummyEncoder()
    guard = PrefixEncoder(encoder, [{"image_path": "a"}, {"image_path": "b"}])
    guard.visible_count = 1
    guard.encode(0, "arrival")
    for future in (1, 2, -1):
        try:
            guard.encode(future, "invalid")
        except ValueError:
            pass
        else:
            raise AssertionError("future-image access accepted")
    assert encoder.calls == 1

    report = {
        "driver_sha256": hashlib.sha256((HERE / "run_streaming_pilot.py").read_bytes()).hexdigest(),
        "status": "passed" if not duplicate_failures else "duplicate_boundary_bug_found",
        "synthetic_stream_count": len(scenarios),
        "scenarios": scenarios,
        "prefix_invariance": "all tested prefixes identical with or without later data",
        "historical_full_embedding_cache": "none in reviewed state and source; test owns synthetic input arrays only",
        "future_encode_guard": "passed",
        "verified_policy_all_stored_edges_match_true_cosines": "passed",
        "persisted_arrays_rows_payload_and_sha_roundtrip": "passed; not a resume test",
        "home_three_image_groups": home_groups,
        "home_audit": home_audit,
        "identical_512d_vector_two_arrival_tests": 30,
        "identical_vector_failures": duplicate_failures,
        "known_design_limits": [
            "Nearest edges can remain stale insertion-time estimates after lossy basis changes.",
            "Otsu always splits a nonconstant edge distribution; it does not test whether two classes exist.",
            "self.log retains full prefix memberships and edges, so diagnostic RAM is quadratic in stream length.",
            "Numerical array storage accounting excludes diagnostics, IDs, scalar metadata and temporary SVD workspace.",
        ],
    }
    (HERE / "independent_streaming_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "scenarios"}, indent=2))


if __name__ == "__main__":
    main()
