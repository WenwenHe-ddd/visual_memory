"""Controlled Brand streams; each arrival is encoded once for both policies.

Each policy has independent persistent memory. The transient current embedding
is delivered to both and then discarded. Evaluation is imported only afterwards.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy

from complete_link import CompleteLinkGraph, HERE, ROOT, STRICT
from run_no_original import ArrivalInput, ImageEncoder, NoOriginalGraph, sha, write_json

PREVIOUS = HERE.parent / "brand_dynamic_pilot_20261019"


def run_stream(encoder, entries, output, include_baseline=True):
    feeder = ArrivalInput(encoder, entries)
    methods = {"complete_link": CompleteLinkGraph()}
    if include_baseline:
        methods["previous_nn"] = NoOriginalGraph()
    nodes = []
    total = np.zeros(encoder.embedding_dim)
    errors = {name: 0.0 for name in methods}
    start = time.perf_counter()
    for index in range(len(entries)):
        node, vector = feeder.next()
        nodes.append(node)
        total += vector
        for name, graph in methods.items():
            graph.add(index, vector)
            error = float(np.max(np.abs(sum(
                (s.n * s.mean for s in graph.sketches), start=np.zeros_like(total)) - total)))
            errors[name] = max(errors[name], error)
            graph.log[-1]["audit_global_embedding_sum_max_abs_error"] = error
        del vector
    seconds = time.perf_counter() - start
    assert not feeder.pending
    assert nodes == [{k: v for k, v in e.items() if k != "image_path"} for e in entries]
    assert max(errors.values()) < 1e-10
    encoder_seconds = sum(c["encoder_seconds"] for c in feeder.calls)
    for name, graph in methods.items():
        directory = output / name
        graph.save(directory, nodes)
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "method": graph.method,
            "image_count": len(nodes), "master_count": len(graph.sketches),
            "cluster_sizes": sorted([s.n for s in graph.sketches], reverse=True),
            "singleton_count": sum(s.n == 1 for s in graph.sketches),
            "runtime": {
                "shared_stream_wall_seconds": seconds,
                "encoder_seconds_shared_between_methods": encoder_seconds,
                "admission_seconds": sum(e["admission_seconds"] for e in graph.log),
                "diagnostic_seconds": sum(e.get("compressed_geometry_diagnostic_seconds", 0) for e in graph.log),
                "encoder_calls": len(feeder.calls), "old_image_reencoding_calls": 0,
                "note": "One encoder pass shared by the two independent methods; timing excludes model load and posthoc evaluation."
            },
            "audit": {
                "all_temporary_inputs_deleted_before_admission": all(c["temporary_input_deleted_before_admission"] for c in feeder.calls),
                "retained_image_path_fields": 0,
                "retained_per_image_visual_scalars_besides_coordinates": 0,
                "max_global_embedding_sum_abs_error": errors[name],
                "approximate_master_mean_count": sum(not s.mean_is_exact for s in graph.sketches),
                "prefixes_exceeding_cut_after_compression": sum(e.get("clusters_exceeding_decision_cut_after_compression", 0) > 0 for e in graph.log),
                "true_local_centroid_errors": "not measured; no original reencoding or vector cache",
            },
            "storage": {
                "sketch_array_bytes": sum(s.summary()["array_bytes"] for s in graph.sketches),
                "npz_file_bytes": (directory / "visual_state.npz").stat().st_size,
                "full_per_image_float64_comparator_bytes": len(nodes) * encoder.embedding_dim * 8,
                "full_per_image_float32_comparator_bytes": len(nodes) * encoder.embedding_dim * 4,
                "stored_pair_distance_bytes": 0,
                "note": "Master means and bases are included; JSON, logs, model weights and transient working memory are excluded.",
            },
            "encoder": encoder.metadata, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "source_graph_sha256": sha(ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json"),
            "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                HERE / "complete_link.py", HERE / "run_pilot.py", STRICT / "run_no_original.py",
                STRICT / "no_replay_sketch.py", PREVIOUS / "run_streaming_pilot.py",
                PREVIOUS / "streaming_sketch.py", ROOT / "scripts/graph/visual_image_encoder.py")},
        }
        write_json(directory / "run_summary.json", summary)
    write_json(output / "encoder_calls.json", feeder.calls)
    return methods


def main():
    entries = json.loads((ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json").read_text())["image_nodes"]
    encoder_start = time.perf_counter()
    encoder = ImageEncoder(str(ROOT / "models/clip-vit-base-patch32"),
                           revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                           device="cpu", local_files_only=True)
    load_seconds = time.perf_counter() - encoder_start
    orders = {"original": list(range(len(entries))), "reverse": list(reversed(range(len(entries))))}
    for seed in (20261019, 20261020):
        orders[f"shuffle_{seed}"] = np.random.default_rng(seed).permutation(len(entries)).tolist()
    for name, order in orders.items():
        methods = run_stream(encoder, [entries[i] for i in order], HERE / "runs" / name)
        print(json.dumps({"order": name, "sizes": {
            k: sorted([s.n for s in graph.sketches], reverse=True) for k, graph in methods.items()}}), flush=True)
    # Independent fresh stream sees a physically truncated input list.
    prefix = run_stream(encoder, entries[:12], HERE / "validation/prefix_12", include_baseline=False)["complete_link"]
    original = [json.loads(line) for line in (HERE / "runs/original/complete_link/admission_log.jsonl").read_text().splitlines()]
    for a, b in zip(prefix.log, original):
        for key in ("memberships", "cut", "cut_reason", "approximate_mean_master_count"):
            if isinstance(a[key], float):
                np.testing.assert_allclose(a[key], b[key], atol=1e-12)
            else:
                assert a[key] == b[key], (key, a[key], b[key])
    write_json(HERE / "validation/prefix_audit.json", {
        "physically_truncated_prefix_12_matches": True, "model_load_seconds": load_seconds,
        "fresh_streams": 5, "total_encoder_calls": 132,
        "within_stream_old_image_reencoding_calls": 0,
        "orders_predeclared": list(orders), "rank": 2,
        "no_parameter_sweep_or_evaluation_feedback": True,
    })


if __name__ == "__main__":
    main()
