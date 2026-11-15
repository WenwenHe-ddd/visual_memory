"""Real-encoder prefix replay and fixed order sensitivity probes, no vector cache."""
from pathlib import Path
import json
import sys
import time

import numpy as np

from run_streaming_pilot import HERE, ROOT, ImageEncoder, PrefixEncoder, StreamingGraph, write_json
sys.path.insert(0, str(HERE / "evaluation"))
from evaluate_pilot import evaluate, write_report


def run(encoder, nodes, output):
    callback = PrefixEncoder(encoder, nodes)
    graph = StreamingGraph("verified_local_scale_nn")
    start = time.perf_counter()
    for i in range(len(nodes)):
        callback.visible_count = i + 1
        z = callback.encode(i, "arrival")
        graph.add(i, z, lambda j: callback.encode(j, "split_rebuild"),
                  lambda j: callback.encode(j, "edge_verification"))
        del z
    elapsed = time.perf_counter() - start
    graph.save(output, nodes)
    write_json(output / "encoder_calls.json", callback.calls)
    assert all(c["image_index"] < c["visible_count"] for c in callback.calls)
    return graph, callback, elapsed


def main():
    source = ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json"
    nodes = json.loads(source.read_text())["image_nodes"]
    encoder = ImageEncoder(str(ROOT / "models/clip-vit-base-patch32"),
                           revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", local_files_only=True)
    base = HERE / "validation"
    base.mkdir(exist_ok=True)
    # Replay a physically truncated input. Prefix length is a test case, not a
    # clustering parameter; the primary reference ran the complete stream.
    replay, callback, seconds = run(encoder, nodes[:12], base / "prefix_12")
    primary = [json.loads(line) for line in (HERE / "verified_local_scale_nn/admission_log.jsonl").read_text().splitlines()]
    for a, b in zip(replay.log, primary):
        assert a["memberships"] == b["memberships"]
        assert a["cluster_count"] == b["cluster_count"]
        if a["cut_log_ratio"] is None:
            assert b["cut_log_ratio"] is None
        else:
            np.testing.assert_allclose(a["cut_log_ratio"], b["cut_log_ratio"], atol=1e-12)
        assert [(e["a"], e["b"], e["kept"]) for e in a["edges"]] == [(e["a"], e["b"], e["kept"]) for e in b["edges"]]
        np.testing.assert_allclose([e["distance"] for e in a["edges"]], [e["distance"] for e in b["edges"]], atol=1e-12)
    try:
        callback.encode(12, "must_fail_future_access")
    except ValueError:
        future_guard_works = True
    else:
        raise AssertionError("future access accepted")
    result = {"prefix_replay": {"length": 12, "all_memberships_and_edges_match_full_run_prefixes": True,
                                "future_access_rejected": future_guard_works},
              "order_choice": "Reverse plus two preselected seeded permutations, no model or clustering tuning across orders.",
              "orders": []}
    orders = {"reverse": list(reversed(range(len(nodes))))}
    for seed in (20261019, 20261020):
        orders[f"shuffle_{seed}"] = np.random.default_rng(seed).permutation(len(nodes)).tolist()
    for name, indices in orders.items():
        selected = [nodes[i] for i in indices]
        directory = base / name
        graph, calls, seconds = run(encoder, selected, directory)
        assignment = {selected[i]["id"]: mid for i, mid in graph.assignments().items()}
        # All assignments are already final before evaluation data is opened.
        report = evaluate(assignment, name=f"候选核验动态聚类 — {name}")
        write_report(report, directory)
        assert not report["errors"]
        result["orders"].append({"name": name, "input_order_ids": [n["id"] for n in selected],
                                 "masters": len(graph.sketches), "silhouette": report["geometry"]["cosine_silhouette"],
                                 "close_pairs": report["visual_sanity"]["close_pairs_kept"],
                                 "different_pairs": report["visual_sanity"]["different_pairs_separated"],
                                 "stream_seconds": seconds, "encoder_calls": len(calls.calls),
                                 "future_image_calls": sum(c["image_index"] >= c["visible_count"] for c in calls.calls),
                                 "array_bytes": sum(s.summary()["array_bytes"] for s in graph.sketches)})
        write_json(base / "stream_order_audit.json", result)
        print(json.dumps(result["orders"][-1]), flush=True)
    write_json(base / "stream_order_audit.json", result)


if __name__ == "__main__":
    main()
