"""Fresh-stream prefix/order probes; every stream encodes each arrival once."""
import json
import sys
import numpy as np

from run_no_original import HERE, ROOT, PREVIOUS, ImageEncoder, run, write_json
sys.path.insert(0, str(PREVIOUS / "evaluation"))
from evaluate_pilot import evaluate


def main():
    entries = json.loads((ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json").read_text())["image_nodes"]
    encoder = ImageEncoder(str(ROOT / "models/clip-vit-base-patch32"),
                           revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", local_files_only=True)
    validation = HERE / "validation"
    validation.mkdir(exist_ok=True)
    prefix, _ = run(encoder, entries[:12], validation / "prefix_12")
    original_log = [json.loads(line) for line in (HERE / "original_order/admission_log.jsonl").read_text().splitlines()]
    for a, b in zip(prefix.log, original_log):
        assert a["memberships"] == b["memberships"]
        assert a["approximate_mean_master_count"] == b["approximate_mean_master_count"]
        assert [(e["a"], e["b"], e["kept"]) for e in a["edges"]] == [(e["a"], e["b"], e["kept"]) for e in b["edges"]]
        if a["cut_log_ratio"] is None:
            assert b["cut_log_ratio"] is None
        else:
            np.testing.assert_allclose(a["cut_log_ratio"], b["cut_log_ratio"], atol=1e-12)
    result = {"prefix_12_matches_full_stream": True, "orders": []}
    orders = {"reverse": list(reversed(range(len(entries))))}
    for seed in (20261019, 20261020):
        orders[f"shuffle_{seed}"] = np.random.default_rng(seed).permutation(len(entries)).tolist()
    original = json.loads((HERE / "original_order/graph.json").read_text())
    original_assignment = {e["image_id"]: e["master_id"] for e in original["edges"]}
    for name, indices in orders.items():
        selected = [entries[i] for i in indices]
        output = validation / name
        graph, summary = run(encoder, selected, output)
        assignments = {selected[i]["id"]: m for i, m in graph.assignments().items()}
        report = evaluate(assignments, name=f"无原图回放 — {name}")
        assert not report["errors"]
        write_json(output / "evaluation.json", report)
        ids = list(assignments)
        disagreements = sum((assignments[a] == assignments[b]) != (original_assignment[a] == original_assignment[b])
                            for i, a in enumerate(ids) for b in ids[i+1:])
        row = {"name": name, "masters": summary["master_count"],
               "silhouette": report["geometry"]["cosine_silhouette"],
               "close_pairs": report["visual_sanity"]["close_pairs_kept"],
               "different_pairs": report["visual_sanity"]["different_pairs_separated"],
               "pairwise_assignment_disagreements_with_original": disagreements,
               "runtime": summary["runtime"], "audit": summary["audit"]}
        result["orders"].append(row)
        write_json(validation / "order_audit.json", result)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
