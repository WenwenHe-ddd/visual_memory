"""Isolated Brand pilot: sequential CLIP, local rank-2 sketches, sparse 1-NN graph.

Admission never imports the evaluator, brand labels, captions or a pairwise matrix.
The policy only receives opaque sequential IDs, the arriving vector and a guarded
callback for re-encoding already seen images when a lossy master splits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from graph.visual_image_encoder import ImageEncoder
from streaming_sketch import ClusterSketch


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def otsu_cut(values):
    """Exact 1-D two-class variance split, without a histogram/bin parameter.

    Cut only between distinct values. Ties choose the earliest split. Constant
    data has no identifiable weak class, so retain all finite edges.
    """
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) < 2 or x[0] == x[-1]:
        return None
    sums = np.cumsum(x)
    candidates = np.flatnonzero(x[:-1] < x[1:])
    left = candidates + 1
    right = len(x) - left
    delta = sums[candidates] / left - (sums[-1] - sums[candidates]) / right
    scores = left * right / len(x)**2 * delta**2
    p = int(candidates[int(np.argmax(scores))])
    return float((x[p] + x[p + 1]) / 2)


class StreamingGraph:
    def __init__(self, policy="local_scale_nn"):
        if policy not in {"local_scale_nn", "verified_local_scale_nn", "uncut_nn"}:
            raise ValueError(policy)
        self.policy = policy
        self.sketches = []
        self.nearest = {}  # one neighbor and scalar edge distance per image
        self.seen = []
        self.log = []
        self.last_edges = []

    def assignments(self):
        return {i: f"M{min(s.member_ids) + 1:03d}" for s in self.sketches for i in s.member_ids}

    def distances(self, z):
        """Query compressed members without materializing historical n x d vectors."""
        output = {}
        for s in self.sketches:
            dot = float(z @ s.mean) + s.coordinates @ (s.basis.T @ z)
            norm2 = float(s.mean @ s.mean) + 2 * s.coordinates @ (s.basis.T @ s.mean)
            norm2 += np.sum(s.coordinates**2, axis=1)
            norms = np.sqrt(np.maximum(norm2, 0))
            # A zero reconstruction contains no direction; cosine=0 is explicit.
            cos = np.divide(dot, norms * np.linalg.norm(z), out=np.zeros_like(dot), where=norms > 0)
            cos[np.abs(1 - cos) <= np.finfo(float).eps * s.dimension] = 1.0
            loss = s.cumulative_discarded_scatter_energy
            numerical = np.finfo(float).eps**2 * max(1, s.dimension * s.n)
            exact = loss <= numerical
            for i, d in zip(s.member_ids, 1 - np.clip(cos, -1, 1)):
                output[i] = {"distance": float(d), "exact_up_to_roundoff": bool(exact)}
        return output

    def _partition(self):
        edges = {}
        for i, row in self.nearest.items():
            pair = tuple(sorted((i, row["neighbor"])))
            if pair not in edges or row["distance"] < edges[pair]["distance"]:
                edges[pair] = {"a": pair[0], "b": pair[1], **{k: v for k, v in row.items() if k != "neighbor"}}
        finite = []
        for row in edges.values():
            d = row["distance"]
            scale = min(self.nearest[row["a"]]["distance"], self.nearest[row["b"]]["distance"])
            weight = 0.0 if d == 0 else (float(np.log(d / scale)) if scale > 0 else None)
            row["log_local_ratio"] = weight
            if weight is not None:
                finite.append(weight)
        cut = otsu_cut(finite) if self.policy != "uncut_nn" else None
        parent = {i: i for i in self.seen}

        def root(i):
            while i != parent[i]:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for row in edges.values():
            if self.policy == "uncut_nn":
                keep = True
            elif len(edges) < 2:
                # One nonzero edge cannot establish an empirical edge distribution.
                keep = row["distance"] == 0
            else:
                w = row["log_local_ratio"]
                keep = w is not None and (cut is None or w <= cut)
            row["kept"] = keep
            if keep:
                a, b = root(row["a"]), root(row["b"])
                parent[max(a, b)] = min(a, b)
        groups = defaultdict(list)
        for i in self.seen:
            groups[root(i)].append(i)
        return sorted(groups.values(), key=min), list(edges.values()), cut

    def add(self, image_id, z, reencode, verify_encode=None):
        if image_id in self.seen:
            raise ValueError("duplicate arrival")
        z = np.asarray(z, dtype=float)
        if z.ndim != 1 or not np.isfinite(z).all() or not np.isclose(z @ z, 1):
            raise ValueError("expected normalized image embedding")
        before = self.assignments()
        started = time.perf_counter()
        distances = self.distances(z)
        query_exact = sum(r["exact_up_to_roundoff"] for r in distances.values())
        query_approximate = len(distances) - query_exact
        prefix = len(self.seen) + 1
        verify_calls = 0
        verify_seconds = 0.0
        if self.policy == "verified_local_scale_nn" and distances:
            checked = {}
            best = float("inf")
            for i in sorted(distances, key=lambda i: (distances[i]["distance"], i)):
                row = distances[i]
                improves_old = i not in self.nearest or row["distance"] < self.nearest[i]["distance"]
                if not row["exact_up_to_roundoff"] and not (improves_old or row["distance"] < best):
                    continue
                if not row["exact_up_to_roundoff"]:
                    start_check = time.perf_counter()
                    old = np.asarray((verify_encode or reencode)(i), dtype=float)
                    verify_seconds += time.perf_counter() - start_check
                    verify_calls += 1
                    cosine = float((z @ old) / (np.linalg.norm(z) * np.linalg.norm(old)))
                    if abs(1 - cosine) <= np.finfo(float).eps * z.size:
                        cosine = 1.0
                    row = {"distance": float(1 - np.clip(cosine, -1, 1)), "exact_up_to_roundoff": True}
                    del old
                checked[i] = row
                best = min(best, row["distance"])
            # Every retained edge has an exact distance; approximate candidate
            # screening can still omit a true neighbor. This is not certified NN.
            distances = checked
        if distances:
            nearest = min(distances, key=lambda i: (distances[i]["distance"], i))
            self.nearest[image_id] = {"neighbor": nearest, **distances[nearest], "measured_at_prefix": prefix}
            for old_id, row in distances.items():
                if old_id not in self.nearest or row["distance"] < self.nearest[old_id]["distance"]:
                    self.nearest[old_id] = {"neighbor": image_id, **row, "measured_at_prefix": prefix}
        self.seen.append(image_id)
        groups, edges, cut = self._partition()
        destination = {i: g for g, group in enumerate(groups) for i in group}
        pieces = defaultdict(list)
        rebuild_calls = 0
        exact_reconstructions = 0
        split_masters = 0
        reencode_seconds = 0.0
        for s in self.sketches:
            parts = defaultdict(list)
            for i in s.member_ids:
                parts[destination[i]].append(i)
            if len(parts) == 1:
                pieces[next(iter(parts))].append(s)
                continue
            split_masters += 1
            numerical = np.finfo(float).eps**2 * max(1, s.dimension * s.n)
            lossless = s.cumulative_discarded_scatter_energy <= numerical
            for g, members in parts.items():
                child = ClusterSketch(max_rank=2)
                for i in members:
                    if lossless:
                        vector = s.reconstruct([i])[0]
                        exact_reconstructions += 1
                    else:
                        t = time.perf_counter()
                        vector = reencode(i)
                        reencode_seconds += time.perf_counter() - t
                        rebuild_calls += 1
                    child.append(i, vector)
                    del vector
                pieces[g].append(child)
        singleton = ClusterSketch(max_rank=2)
        singleton.append(image_id, z)
        pieces[destination[image_id]].append(singleton)
        updated = []
        merges = 0
        for g in range(len(groups)):
            combined = pieces[g][0]
            for piece in pieces[g][1:]:
                combined = combined.merged(piece)
                merges += 1
            updated.append(combined)
        self.sketches = updated
        self.last_edges = edges
        elapsed = time.perf_counter() - started
        assignments = self.assignments()
        event = {
            "prefix_count": prefix, "arriving_id": image_id,
            "cluster_count": len(updated), "memberships": groups,
            "cut_log_ratio": cut,
            "cut_ratio": float(np.exp(cut)) if cut is not None else None,
            "old_members_changing_master_id": sum(assignments[i] != m for i, m in before.items()),
            "split_masters": split_masters, "merges": merges,
            "split_reencode_calls": rebuild_calls,
            "lossless_split_reconstructions": exact_reconstructions,
            "split_reencode_seconds": reencode_seconds,
            "edge_verification_calls": verify_calls,
            "edge_verification_seconds": verify_seconds,
            "admission_seconds_including_reencode": elapsed,
            "admission_seconds_excluding_reencode": max(0, elapsed - reencode_seconds - verify_seconds),
            "query_distances_exact": query_exact,
            "query_distances_approximate": query_approximate,
            "sketch_array_bytes": sum(s.summary()["array_bytes"] for s in updated),
            "nearest_edge_count": len(self.nearest),
            "nearest_distance_scalar_bytes": len(self.nearest) * 8,
            "discarded_scatter_energy": sum(s.cumulative_discarded_scatter_energy for s in updated),
            "edges": edges,
        }
        self.log.append(event)
        return event

    def save(self, directory, nodes):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {}
        masters = []
        image_edges = []
        assignments = self.assignments()
        for s in self.sketches:
            mid = assignments[s.member_ids[0]]
            for name, value in s.array_state().items():
                arrays[f"{mid}_{name}"] = value
            masters.append({"id": mid, "member_count": s.n, "rank": s.rank,
                            "visual_mean_ref": {"path": "visual_state.npz", "key": f"{mid}_mean"},
                            "residual_basis_ref": {"path": "visual_state.npz", "key": f"{mid}_basis"},
                            "sketch_summary": s.summary(), "member_indices": s.member_ids})
            for row, i in enumerate(s.member_ids):
                image_edges.append({"image_id": nodes[i]["id"], "master_id": mid,
                                    "residual_coordinates_ref": {"path": "visual_state.npz", "key": f"{mid}_coordinates", "row": row}})
        np.savez(directory / "visual_state.npz", **arrays)
        graph = {"schema_version": 1, "method": self.policy, "embedding_dim": self.sketches[0].dimension,
                 "masters": masters, "image_nodes": [nodes[i] for i in self.seen], "edges": image_edges,
                 "stream_indices": {str(i): nodes[i]["id"] for i in self.seen},
                 "nearest_neighbors": self.nearest, "selected_connections": self.last_edges,
                 "vector_storage": {"path": "visual_state.npz", "sha256": sha(directory / "visual_state.npz")},
                 "distance_semantics": "New-to-old cosine uses the old local rank-2 reconstruction. Stored nearest edges retain insertion-time estimates; no claim of exact nearest neighbors after truncation."}
        write_json(directory / "graph.json", graph)
        with (directory / "admission_log.jsonl").open("w") as f:
            for event in self.log:
                f.write(json.dumps(event, allow_nan=False) + "\n")
        return graph


class PrefixEncoder:
    """No vector cache. Re-encoding unseen/future images is a hard error."""
    def __init__(self, encoder, nodes):
        self.encoder = encoder
        self.nodes = nodes
        self.visible_count = 0
        self.calls = []

    def encode(self, index, purpose):
        if not 0 <= index < self.visible_count:
            raise ValueError("attempt to encode a future image")
        start = time.perf_counter()
        result = np.asarray(self.encoder.encode(self.nodes[index]["image_path"]), dtype=float)
        self.calls.append({"image_index": index, "visible_count": self.visible_count,
                           "purpose": purpose, "seconds": time.perf_counter() - start})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["local_scale_nn", "verified_local_scale_nn", "uncut_nn"], default="verified_local_scale_nn")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json"
    nodes = json.loads(source.read_text())["image_nodes"]
    if args.limit is not None:
        if not 1 <= args.limit <= len(nodes):
            raise ValueError("limit outside the Brand stream")
        nodes = nodes[:args.limit]
    load_start = time.perf_counter()
    encoder = ImageEncoder(str(ROOT / "models/clip-vit-base-patch32"),
                           revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                           device="cpu", local_files_only=True)
    load_seconds = time.perf_counter() - load_start
    callback = PrefixEncoder(encoder, nodes)
    graph = StreamingGraph(args.policy)
    start = time.perf_counter()
    for index in range(len(nodes)):
        callback.visible_count = index + 1
        vector = callback.encode(index, "arrival")
        graph.add(index, vector, lambda i: callback.encode(i, "split_rebuild"),
                  lambda i: callback.encode(i, "edge_verification"))
        del vector
    run_seconds = time.perf_counter() - start
    output_graph = graph.save(args.output, nodes)
    calls_before_audit = list(callback.calls)
    # Post-admission audit reads one original vector at a time; it cannot change decisions.
    mean_error = 0.0
    squared_reconstruction_error = 0.0
    energy_error = 0.0
    summaries = []
    for s in graph.sketches:
        total = np.zeros(s.dimension)
        error = 0.0
        for index in s.member_ids:
            vector = callback.encode(index, "posthoc_geometry_audit")
            total += vector
            residual = vector - s.reconstruct([index])[0]
            error += float(residual @ residual)
            del vector, residual
        local_mean_error = float(np.max(np.abs(total / s.n - s.mean)))
        mean_error = max(mean_error, local_mean_error)
        squared_reconstruction_error += error
        energy_error = max(energy_error, abs(error - s.cumulative_discarded_scatter_energy))
        summaries.append({"members": s.member_ids, "n": s.n, "rank": s.rank,
                          "mean_max_abs_error": local_mean_error, "actual_reconstruction_squared_error": error,
                          **s.summary()})
    assert mean_error < 1e-12
    assert energy_error < 1e-10
    assert output_graph["image_nodes"] == nodes
    calls = Counter(c["purpose"] for c in calls_before_audit)
    max_n = len(nodes)
    total_bytes = sum(s.summary()["array_bytes"] for s in graph.sketches)
    result = {
        "policy": args.policy, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Brand only, actual prefix-only sequential encoding, no full image embedding cache",
        "image_count": max_n, "master_count": len(graph.sketches),
        "source_graph": str(source), "source_graph_sha256": sha(source),
        "code_sha256": {p.name: sha(p) for p in (Path(__file__), HERE / "streaming_sketch.py")},
        "image_sha256": {n["id"]: sha(n["image_path"]) for n in nodes},
        "encoder": encoder.metadata,
        "runtime": {"model_load_seconds": load_seconds, "stream_seconds_excluding_model_load_and_posthoc_audit": run_seconds,
                    "arrival_encode_seconds": sum(c["seconds"] for c in calls_before_audit if c["purpose"] == "arrival"),
                    "admission_seconds_excluding_reencode": sum(e["admission_seconds_excluding_reencode"] for e in graph.log),
                    "split_reencode_seconds": sum(e["split_reencode_seconds"] for e in graph.log),
                    "edge_verification_seconds": sum(e["edge_verification_seconds"] for e in graph.log),
                    "encode_calls": dict(calls)},
        "storage": {"sketch_array_bytes": total_bytes,
                    "nearest_distance_scalar_bytes": len(graph.nearest)*8,
                    "all_individual_embeddings_float64_bytes_comparator": max_n*encoder.embedding_dim*8,
                    "all_individual_embeddings_float32_bytes_comparator": max_n*encoder.embedding_dim*4,
                    "peak_prefix_sketch_array_bytes": max(e["sketch_array_bytes"] for e in graph.log),
                    "npz_file_bytes": (args.output / "visual_state.npz").stat().st_size,
                    "note": "Numeric array budget only; excludes IDs, JSON, encoder weights and temporary SVD workspace. All sketch arrays use float64; float32 baseline shown separately."},
        "audit": {"future_image_calls": sum(c["image_index"] >= c["visible_count"] for c in callback.calls),
                  "payload_preserved_exactly": output_graph["image_nodes"] == nodes,
                  "max_master_mean_abs_error": mean_error,
                  "reconstruction_rms_per_image": float(np.sqrt(squared_reconstruction_error / max_n)),
                  "max_discarded_energy_audit_error": energy_error,
                  "posthoc_audit_encode_calls": len(callback.calls)-len(calls_before_audit),
                  "query_distances_exact": sum(e["query_distances_exact"] for e in graph.log),
                  "query_distances_approximate": sum(e["query_distances_approximate"] for e in graph.log),
                  "old_members_rebuilt_by_encoding": sum(e["split_reencode_calls"] for e in graph.log)},
        "masters": summaries,
        "limits": ["Rank two is a user-selected compression rank, not a promise of lossless directions.",
                   "Nearest edges are insertion-time estimates; future observations do not recover discarded non-nearest edges.",
                   "Otsu learns a boundary and assumes useful strong/weak edge separation; it has no manually fixed cosine cutoff.",
                   "The 30-image pilot does not establish large-library scaling or generalization."]}
    write_json(args.output / "run_summary.json", result)
    write_json(args.output / "encoder_calls.json", callback.calls)
    print(json.dumps({"output": str(args.output), "masters": len(graph.sketches),
                      "runtime": result["runtime"], "storage": result["storage"], "audit": result["audit"]}, indent=2))


if __name__ == "__main__":
    main()
