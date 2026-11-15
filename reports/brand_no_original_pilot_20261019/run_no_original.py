"""Brand streaming pilot with one-time image input and no old-image capability.

The clustering object receives only a current normalized vector and opaque ID.
It has no image paths, encoder, callbacks or original-vector cache. Splits operate
on stored residual coordinates and explicitly propagate approximate mean status.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent / "brand_dynamic_pilot_20261019"
sys.path.insert(0, str(PREVIOUS))
from run_streaming_pilot import ImageEncoder, StreamingGraph, write_json
sys.path.insert(0, str(HERE))
from no_replay_sketch import NoReplaySketch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NoOriginalGraph(StreamingGraph):
    """Sparse candidate topology; no per-image visual scalars beyond coordinates.

    Only neighbor IDs persist. Edge distances are recomputed from compressed
    endpoints for each arrival and discarded. This does not recover old neighbor
    candidates omitted earlier, so the topology is not guaranteed exact NN.
    """
    def __init__(self, cut=True):
        super().__init__("local_scale_nn" if cut else "uncut_nn")
        self.method = "no_original_local_scale_nn" if cut else "no_original_uncut_nn"

    def distances(self, z):
        output = {}
        for s in self.sketches:
            dot = float(z @ s.mean) + s.coordinates @ (s.basis.T @ z)
            norm2 = float(s.mean @ s.mean) + 2 * s.coordinates @ (s.basis.T @ s.mean)
            norm2 += np.sum(s.coordinates**2, axis=1)
            norms = np.sqrt(np.maximum(norm2, 0))
            cos = np.divide(dot, norms * np.linalg.norm(z), out=np.zeros_like(dot), where=norms > 0)
            cos[np.abs(1 - cos) <= np.finfo(float).eps * s.dimension] = 1.0
            exact = s.reconstruction_frobenius_error_upper_bound <= np.finfo(float).eps * max(s.dimension, s.n)
            for i, d in zip(s.member_ids, 1 - np.clip(cos, -1, 1)):
                output[i] = {"distance": float(d), "exact_up_to_roundoff": bool(exact)}
        return output

    def add(self, image_id, z):
        if image_id in self.seen:
            raise ValueError("each image may arrive only once")
        z = np.asarray(z, dtype=float)
        if z.ndim != 1 or not np.isfinite(z).all() or not np.isclose(z @ z, 1):
            raise ValueError("expected normalized embedding")
        start = time.perf_counter()
        distances = self.distances(z)
        prefix = len(self.seen) + 1
        locations = {i: (s, row) for s in self.sketches for row, i in enumerate(s.member_ids)}
        temporary_neighbors = {}
        refreshed = {}
        for i, j in self.nearest.items():
            key = tuple(sorted((i, j)))
            if key not in refreshed:
                a, ai = locations[i]
                b, bi = locations[j]
                # Two transient reconstructed vectors; neither is a cached
                # original image embedding, and neither survives this method.
                left = a.mean + a.basis @ a.coordinates[ai]
                right = b.mean + b.basis @ b.coordinates[bi]
                denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                cosine = float(left @ right) / denominator if denominator else 0.0
                if abs(1 - cosine) <= np.finfo(float).eps * z.size:
                    cosine = 1.0
                refreshed[key] = {"distance": float(1 - np.clip(cosine, -1, 1)),
                                  "exact_up_to_roundoff": a.representation_is_exact and b.representation_is_exact}
                del left, right
            temporary_neighbors[i] = {"neighbor": j, **refreshed[key], "measured_at_prefix": prefix}
        if distances:
            nearest = min(distances, key=lambda i: (distances[i]["distance"], i))
            temporary_neighbors[image_id] = {"neighbor": nearest, **distances[nearest], "measured_at_prefix": prefix}
            for old_id, row in distances.items():
                if old_id not in temporary_neighbors or row["distance"] < temporary_neighbors[old_id]["distance"]:
                    temporary_neighbors[old_id] = {"neighbor": image_id, **row, "measured_at_prefix": prefix}
        self.seen.append(image_id)
        # The existing partition routine consumes only transient edge scalars.
        partition_view = StreamingGraph(self.policy)
        partition_view.seen = self.seen
        partition_view.nearest = temporary_neighbors
        groups, edges, cut = partition_view._partition()
        self.nearest = {i: row["neighbor"] for i, row in temporary_neighbors.items()}
        destination = {i: g for g, members in enumerate(groups) for i in members}
        pieces = defaultdict(list)
        splits = 0
        for sketch in self.sketches:
            parts = defaultdict(list)
            for i in sketch.member_ids:
                parts[destination[i]].append(i)
            if len(parts) == 1:
                pieces[next(iter(parts))].append(sketch)
            else:
                splits += 1
                keys = sorted(parts)
                children = sketch.split([parts[k] for k in keys])
                for k, child in zip(keys, children):
                    pieces[k].append(child)
        singleton = NoReplaySketch(max_rank=2)
        singleton.append(image_id, z)
        pieces[destination[image_id]].append(singleton)
        updated = []
        merges = 0
        for group in range(len(groups)):
            combined = pieces[group][0]
            for piece in pieces[group][1:]:
                combined = combined.merged(piece)
                merges += 1
            updated.append(combined)
        self.sketches = updated
        # Persist topology only, including in diagnostic logs; historical edge
        # weights would otherwise be extra visual numbers for each image.
        self.last_edges = [{key: e[key] for key in ("a", "b", "kept")} for e in edges]
        elapsed = time.perf_counter() - start
        event = {"prefix_count": prefix, "arriving_id": image_id, "cluster_count": len(groups),
                 "memberships": groups, "cut_log_ratio": cut,
                 "cut_ratio": float(np.exp(cut)) if cut is not None else None,
                 "splits": splits, "merges": merges, "admission_seconds": elapsed,
                 "query_distances_exact": sum(row["exact_up_to_roundoff"] for row in distances.values()),
                 "query_distances_approximate": sum(not row["exact_up_to_roundoff"] for row in distances.values()),
                 "approximate_mean_master_count": sum(not s.mean_is_exact for s in updated),
                 "sketch_array_bytes": sum(s.summary()["array_bytes"] for s in updated),
                 "edges": self.last_edges}
        self.log.append(event)
        return event

    def save(self, directory, nodes):
        if any("image_path" in node for node in nodes):
            raise ValueError("memory payload must not retain an original image path")
        result = super().save(directory, nodes)
        result["method"] = self.method
        result["original_image_retention"] = "none; admission inputs are discarded immediately after their one encoder call"
        result["master_mean_semantics"] = "Full-dimensional running estimate; may be approximate after a lossy split. See per-master mean_is_exact and error bounds."
        result["reencoding_supported"] = False
        result["strict_two_coordinates_per_image"] = True
        result["distance_semantics"] = "No edge distance is retained. Candidate-edge distances are recomputed transiently from current master means, bases and member coordinates. The candidate topology need not contain true nearest neighbors."
        write_json(Path(directory) / "graph.json", result)
        return result


class ArrivalInput:
    """External test feeder; consumes each item once and exposes no replay API.

    Original benchmark files stay outside the memory. One temporary incoming
    file simulates an upload; it is unlinked immediately after the encoder call,
    before clustering begins. No consumed source path is retained by this feeder.
    """
    def __init__(self, encoder, entries):
        self.encoder = encoder
        self.pending = [dict(entry) for entry in entries]
        self.calls = []
        self.consumed_ids = set()

    def next(self):
        if not self.pending:
            raise StopIteration
        entry = self.pending.pop(0)
        image_id = entry["id"]
        if image_id in self.consumed_ids:
            raise ValueError("image already consumed")
        source = Path(entry.pop("image_path"))
        index = len(self.calls)
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="brand_arrival_") as temporary:
            incoming = Path(temporary) / ("incoming" + source.suffix)
            shutil.copyfile(source, incoming)
            content_sha = sha(incoming)
            t = time.perf_counter()
            vector = np.asarray(self.encoder.encode(incoming), dtype=float)
            encoder_seconds = time.perf_counter() - t
            incoming.unlink()
            assert not incoming.exists()
        self.consumed_ids.add(image_id)
        self.calls.append({"index": index, "image_id": image_id, "purpose": "arrival_only",
                           "input_sha256": content_sha, "encoder_seconds": encoder_seconds,
                           "input_seconds": time.perf_counter() - start,
                           "temporary_input_deleted_before_admission": True})
        return entry, vector


def run(encoder, entries, output, cut=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    feeder = ArrivalInput(encoder, entries)
    graph = NoOriginalGraph(cut=cut)
    nodes = []
    # Audit-only single aggregate vector; never a per-image vector cache.
    true_global_sum = np.zeros(encoder.embedding_dim)
    global_error = 0.0
    start = time.perf_counter()
    for index in range(len(entries)):
        node, vector = feeder.next()
        nodes.append(node)
        true_global_sum += vector
        graph.add(index, vector)
        del vector
        estimated_sum = sum((s.n * s.mean for s in graph.sketches), start=np.zeros(encoder.embedding_dim))
        local_error = float(np.max(np.abs(estimated_sum - true_global_sum)))
        global_error = max(global_error, local_error)
        graph.log[-1]["audit_global_embedding_sum_max_abs_error"] = local_error
    seconds = time.perf_counter() - start
    assert not feeder.pending
    assert len(feeder.calls) == len(nodes) == len(feeder.consumed_ids)
    assert global_error < 1e-10
    serialized = graph.save(output, nodes)
    # Exact metadata equality except deliberate removal of original image path.
    assert nodes == [{k: v for k, v in row.items() if k != "image_path"} for row in entries]
    assert all("image_path" not in n for n in serialized["image_nodes"])
    summaries = [s.summary() for s in graph.sketches]
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "method": graph.method,
               "image_count": len(nodes), "master_count": len(graph.sketches),
               "runtime": {"stream_seconds_excluding_model_load_and_posthoc_evaluation": seconds,
                           "encoder_seconds": sum(c["encoder_seconds"] for c in feeder.calls),
                           "admission_seconds": sum(e["admission_seconds"] for e in graph.log),
                           "encoder_calls": len(feeder.calls), "old_image_reencoding_calls": 0},
               "storage": {"sketch_array_bytes": sum(s["array_bytes"] for s in summaries),
                           "nearest_distance_scalar_bytes": 0,
                           "full_per_image_float64_comparator_bytes": len(nodes) * encoder.embedding_dim * 8,
                           "full_per_image_float32_comparator_bytes": len(nodes) * encoder.embedding_dim * 4,
                           "npz_file_bytes": (output / "visual_state.npz").stat().st_size,
                           "note": "Array bytes exclude JSON IDs, bounded-error scalar metadata, diagnostic logs, transient SVD workspace and model weights."},
               "audit": {"all_temporary_inputs_deleted_before_admission": all(c["temporary_input_deleted_before_admission"] for c in feeder.calls),
                         "payload_preserved_except_removed_image_path": True,
                         "retained_image_path_fields": 0,
                         "retained_per_image_visual_scalars_besides_coordinates": 0,
                         "max_global_embedding_sum_abs_error": global_error,
                         "exact_master_mean_count": sum(s.mean_is_exact for s in graph.sketches),
                         "approximate_master_mean_count": sum(not s.mean_is_exact for s in graph.sketches),
                         "actual_local_master_mean_errors": "not measured on Brand: no old-image encoding or original-vector retention",
                         "error_bounds_are_conservative_not_observed_error": True,
                         "query_distances_exact": sum(e["query_distances_exact"] for e in graph.log),
                         "query_distances_approximate": sum(e["query_distances_approximate"] for e in graph.log)},
               "masters": summaries, "encoder": encoder.metadata,
               "source_graph_sha256": sha(ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json"),
               "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                   HERE / "run_no_original.py", HERE / "no_replay_sketch.py",
                   PREVIOUS / "run_streaming_pilot.py", PREVIOUS / "streaming_sketch.py")},
               "limits": ["A lossy split produces approximate full-dimensional centers, never relabeled exact without proof.",
                          "Only candidate neighbor IDs persist; recomputing their distances cannot recover previously omitted old candidates.",
                          "Two coordinates per image do not imply two numbers total: master bases and means also require storage.",
                          "Benchmark originals are external test fixtures, unchanged; they are not retained by the memory graph."]}
    write_json(output / "run_summary.json", summary)
    write_json(output / "encoder_calls.json", feeder.calls)
    return graph, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "original_order")
    parser.add_argument("--order", choices=["original", "reverse", "shuffle_20261019", "shuffle_20261020"], default="original")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--uncut", action="store_true")
    args = parser.parse_args()
    entries = json.loads((ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json").read_text())["image_nodes"]
    if args.order == "reverse":
        entries.reverse()
    elif args.order.startswith("shuffle_"):
        order = np.random.default_rng(int(args.order.split("_")[-1])).permutation(len(entries))
        entries = [entries[int(i)] for i in order]
    if args.limit is not None:
        if not 1 <= args.limit <= len(entries):
            raise ValueError("invalid prefix length")
        entries = entries[:args.limit]
    start = time.perf_counter()
    encoder = ImageEncoder(str(ROOT / "models/clip-vit-base-patch32"),
                           revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", device="cpu", local_files_only=True)
    load_seconds = time.perf_counter() - start
    graph, summary = run(encoder, entries, args.output, cut=not args.uncut)
    summary["runtime"]["model_load_seconds"] = load_seconds
    summary["input_order"] = args.order
    write_json(args.output / "run_summary.json", summary)
    print(json.dumps({"output": str(args.output), "masters": len(graph.sketches), "runtime": summary["runtime"],
                      "storage": summary["storage"], "audit": summary["audit"]}, indent=2))


if __name__ == "__main__":
    main()
