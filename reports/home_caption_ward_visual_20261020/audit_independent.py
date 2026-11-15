#!/usr/bin/env python3
"""Independent artifact audit; no Home image encoding and no core imports."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path("/storage/hewenwen/VisualMemory")
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage

checks = []
maxima = {}


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, condition):
    checks.append({"check": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)


def close(name, actual, expected, tolerance=1e-11):
    delta = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0.))
    family = name.split("/")[0]
    maxima[family] = max(maxima.get(family, 0.), delta)
    check(name, delta <= tolerance)


def canonical(groups):
    return {frozenset(group) for group in groups}


def normalise_like_cache(vector):
    vector = np.asarray(vector, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    vector /= np.max(np.abs(vector))
    vector /= np.linalg.norm(vector)
    return vector


def independent_silhouette(distances, labels):
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    scores = np.zeros(len(labels))
    for group in groups:
        if len(group) == 1:
            continue
        own = distances[np.ix_(group, group)].sum(axis=1) / (len(group) - 1)
        others = [distances[np.ix_(group, other)].mean(axis=1)
                  for other in groups if other[0] != group[0]]
        nearest = np.min(others, axis=0)
        denominator = np.maximum(own, nearest)
        scores[group] = np.divide(nearest - own, denominator,
                                  out=np.zeros_like(own), where=denominator != 0)
    return float(scores.mean())


def main():
    started = time.perf_counter()
    manifest = load(OUT / "input_manifest.json")
    rows, calibration = manifest["home"], manifest["calibration"]
    protocol = load(OUT / "protocol.json")
    results = load(OUT / "results.json")
    state = load(OUT / "memory/graph.json")
    audits = [json.loads(line) for line in (OUT / "insertion_audit.jsonl").read_text().splitlines()]
    ids = [r["image_id"] for r in rows]
    index = {image_id: i for i, image_id in enumerate(ids)}
    check("home_unique_ids", len(ids) == len(index) == 89)
    check("exact_89_prefixes", [a["n_images"] for a in audits] == list(range(1, 90)))
    check("incoming_order", [a["image_id"] for a in audits] == ids)
    historical_unavailable = []
    migrated_code = []
    for path, expected in protocol["source_sha256"].items():
        source = Path(path)
        if not source.exists() and ("graph_memory_threshold_caption_full_v5" in path
                or "query_image_caption/top5/dialogue_full_caption" in path):
            historical_unavailable.append({"path": path, "original_sha256": expected})
        elif source.suffix == ".py" and source.exists() and digest(source) != expected:
            migrated_code.append({"path": path, "original_sha256": expected,
                                  "current_sha256": digest(source)})
        else:
            check(f"source_sha/{path}", digest(path) == expected)
    frozen_root = ROOT / "data/independent_visual_inputs"
    frozen_protocol = load(frozen_root / "protocol.json")
    for name, expected in frozen_protocol["export_hashes"].items():
        check(f"frozen_input_sha/{name}", digest(frozen_root / name) == expected)

    natural = [json.loads(line) for line in (ROOT / "results/natural_observation/memories.jsonl").read_text().splitlines()]
    external_hashes = {image for record in natural
                       if record["task"] != protocol["domain"] for image in record["images"]}
    home_hashes = {r["image_sha256"] for r in rows}
    check("home_unique_hashes", len(home_hashes) == 89)
    check("calibration_exact_unique_external_set",
          {r["image_sha256"] for r in calibration} == external_hashes and len(calibration) == 344)
    check("calibration_no_home_overlap", not home_hashes.intersection(external_hashes))
    for row in [*rows, *calibration]:
        check(f"image_sha/{row['image_sha256']}", digest(row["source_path"]) == row["image_sha256"])
    raw_dialog = load(ROOT / "data/MemEye/data/dialog/Home_Renovation_Interior_Design.json")
    chronology = []
    for session in raw_dialog["multi_session_dialogues"]:
        for turn in session["dialogues"]:
            for image_number, image_path in enumerate(turn.get("input_image", []), 1):
                chronology.append((f"{protocol['domain']}/{turn['round']}#image{image_number}",
                                   session["date"], Path(image_path).name))
    check("original_nonopen_dialog_order",
          chronology == [(r["image_id"], r["timestamp"], r["filename"]) for r in rows])
    check("chronological_dates", [r["timestamp"] for r in rows] == sorted(r["timestamp"] for r in rows))
    source_root = ROOT / "data/independent_visual_inputs"
    source_captions = {record["image_id"]: record["caption"] for record in
                       [json.loads(line) for line in (source_root / "arrivals.jsonl").read_text().splitlines()]}
    reader_records = {record["id"]: record for record in
                      [json.loads(line) for line in (source_root / "retrieval_memories.jsonl").read_text().splitlines()]}
    for row in rows:
        dialogue, caption = reader_records[row["memory_id"]]["reader_text"].split("\nMemory description: ", 1)
        check(f"frozen_caption/{row['image_id']}",
              row["caption"] == caption == source_captions[row["image_id"]])
        check(f"frozen_dialogue/{row['image_id']}", row["dialogue"] == dialogue)

    basis = np.load(OUT / "memory/basis.npy", allow_pickle=False)
    sums = np.load(OUT / "memory/master_sums.npy", allow_pickle=False)
    check("basis_shape", basis.shape == (512, 2))
    close("basis_orthonormal", basis.T @ basis, np.eye(2))
    nodes = state["image_nodes"]
    check("leaf_order", [n["id"] for n in nodes] == ids)
    expected_leaf_keys = {"id", "caption", "dialogue", "timestamp", "visual_coords", "master_id"}
    for node, row in zip(nodes, rows):
        check(f"strict_leaf_schema/{node['id']}", set(node) == expected_leaf_keys)
        check(f"two_visual_coords/{node['id']}",
              len(node["visual_coords"]) == 2 and np.isfinite(node["visual_coords"]).all())
        check(f"leaf_content/{node['id']}",
              all(node[k] == row[k] for k in ("caption", "dialogue", "timestamp")))
    check("only_expected_memory_files",
          {p.name for p in (OUT / "memory").iterdir()} == {"graph.json", "master_sums.npy", "basis.npy"})
    check("no_persistent_text_embedding_cache",
          state["text_embedding_cache"]["persisted"] is False)
    check("sums_shape_and_dtype", sums.shape == (len(state["masters"]), 512) and sums.dtype == np.float64)
    check("basis_dtype", basis.dtype == np.float64)
    z = np.array([n["visual_coords"] for n in nodes])
    check("final_partition_exact_cover",
          sorted(m for c in state["masters"] for m in c["member_ids"]) == sorted(ids))
    for row_num, master in enumerate(state["masters"]):
        check(f"sum_reference/{master['id']}", master["visual_sum_ref"] ==
              {"path": "master_sums.npy", "row": row_num})
        check(f"mean_reference/{master['id']}", master["visual_mean_ref"] ==
              {"path": "master_sums.npy", "row": row_num, "scale": 1 / master["n"]})
        check(f"master_count/{master['id']}", master["n"] == len(master["member_ids"]))
        members = [index[m] for m in master["member_ids"]]
        close(f"projection/{master['id']}", basis.T @ sums[row_num], z[members].sum(axis=0))

    gram = np.eye(89)
    seen = set()
    gram_path = ROOT / "reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv"
    for pair in csv.DictReader(gram_path.open()):
        if pair["earlier_id"] in index and pair["later_id"] in index:
            i, j = index[pair["earlier_id"]], index[pair["later_id"]]
            key = tuple(sorted((i, j)))
            check(f"gram_unique/{i}/{j}", key not in seen and i != j)
            seen.add(key)
            gram[i, j] = gram[j, i] = float(pair["cosine"])
    check("gram_complete", len(seen) == 3916)
    check("gram_positive_semidefinite", np.linalg.eigvalsh(gram).min() >= -1e-10)
    close("global_total_norm", np.sum(sums.sum(axis=0) ** 2), gram.sum(), 1e-8)
    previous_encoder = load(ROOT / "results/visual_embedding_graph/encoder_metadata.json")
    check("historical_gram_encoder_metadata_identical", protocol["image_encoder"] == previous_encoder)

    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    text_model = SentenceTransformer(protocol["text_encoder"]["model_path"],
                                     device="cpu", local_files_only=True)
    embeddings = []
    for row in rows:
        token_count = len(text_model.tokenizer(row["caption"], truncation=False)["input_ids"])
        check(f"caption_token_count/{row['image_id']}",
              token_count == audits[len(embeddings)]["caption_token_count"] and token_count <= 512)
        embeddings.append(normalise_like_cache(text_model.encode(
            row["caption"], normalize_embeddings=True, show_progress_bar=False)))
    embeddings = np.array(embeddings)
    old = {}
    coefficients = {}
    old_owners = {}
    independent_final_metrics = []
    for n, audit in enumerate(audits, 1):
        prefix = f"prefix_{n}"
        clusters = {c["master_id"]: c for c in audit["clusters"]}
        owners = {member: master_id for master_id, c in clusters.items() for member in c["member_ids"]}
        check(f"prefix_exact_cover/{n}", sorted(owners) == sorted(ids[:n]) and
              sum(c["n"] for c in clusters.values()) == n)
        check(f"prefix_candidate_k/{n}", [c["k"] for c in audit["candidate_scores"]] ==
              ([] if n <= 2 else list(range(2, n))))
        if n <= 2:
            check(f"startup_rule/{n}", audit["selected_k"] == n)
        else:
            x = embeddings[:n].copy()
            x /= np.max(np.abs(x), axis=1, keepdims=True)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            tree = linkage(x, method="ward", metric="euclidean")
            cuts = cut_tree(tree, n_clusters=list(range(2, n)))
            distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
            scores = [independent_silhouette(distances, cuts[:, k - 2]) for k in range(2, n)]
            close(f"silhouette_scores/{n}", scores,
                  [r["score"] for r in audit["candidate_scores"]], 2e-12)
            # Match the documented numerical tie rule while visiting K in the
            # descending Ward merge order used for choosing the best cut.
            best_score = -np.inf
            best_k = n
            for k in range(n - 1, 1, -1):
                score = scores[k - 2]
                if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and k < best_k):
                    best_score, best_k = score, k
            labels = cuts[:, best_k - 2]
            groups = [[ids[i] for i in np.flatnonzero(labels == label)] for label in np.unique(labels)]
            check(f"independent_selected_k/{n}", best_k == audit["selected_k"])
            check(f"independent_selected_membership/{n}",
                  canonical(groups) == canonical(c["member_ids"] for c in clusters.values()))
        unchanged = {old_id for old_id, c in old.items()
                     if any(set(c["member_ids"]) == set(new["member_ids"]) for new in clusters.values())}
        check(f"unchanged_ids/{n}", unchanged == set(audit["unchanged_master_ids"]))
        check(f"retired_ids/{n}", set(old) - set(clusters) == set(audit["retired_master_ids"]))
        for old_id in unchanged:
            check(f"unchanged_members_preserve_id/{n}/{old_id}",
                  old_id in clusters and set(old[old_id]["member_ids"]) == set(clusters[old_id]["member_ids"]))
        relation_changes = sum((old_owners[left] == old_owners[right]) != (owners[left] == owners[right])
                               for i, left in enumerate(ids[:n-1]) for right in ids[i+1:n-1])
        check(f"relation_changes/{n}", relation_changes == audit["old_pair_relation_changes"])
        new_coefficients = {}
        overlap_expected = []
        split_sources = []
        for old_id, c in old.items():
            destinations = sum(bool(set(c["member_ids"]) & set(new["member_ids"])) for new in clusters.values())
            if destinations > 1:
                split_sources.append(old_id)
        check(f"split_sources/{n}", set(split_sources) == set(audit["split_master_ids"]))
        for new_id, c in clusters.items():
            new_coefficients[new_id] = np.zeros(n)
            approx = False
            for old_id, old_c in old.items():
                overlap = set(c["member_ids"]) & set(old_c["member_ids"])
                if overlap:
                    fraction = len(overlap) / old_c["n"]
                    new_coefficients[new_id][:n-1] += coefficients[old_id] * fraction
                    overlap_expected.append((old_id, new_id, len(overlap), old_c["n"], fraction))
                    approx |= old_c["approximate"] or len(overlap) != old_c["n"]
            if ids[n-1] in c["member_ids"]:
                new_coefficients[new_id][-1] += 1
            check(f"approximation_lineage/{n}/{new_id}", c["approximate"] == approx)
        check(f"overlap_accounting/{n}", sorted(overlap_expected) == sorted(
            (r["old_master_id"], r["new_master_id"], r["count"], r["old_n"], r["fraction"])
            for r in audit["source_overlaps"]))
        close(f"coefficient_conservation/{n}", sum(new_coefficients.values()), np.ones(n))
        h = gram[:n, :n] - z[:n] @ z[:n].T
        check(f"residual_gram_psd/{n}", np.linalg.eigvalsh(h).min() >= -1e-10)
        oracle_by_id = {r["master_id"]: r for r in audit["visual_center_error"]["clusters"]}
        metrics = []
        for master_id, c in clusters.items():
            a = new_coefficients[master_id] / c["n"]
            b = np.array([float(image_id in c["member_ids"]) / c["n"] for image_id in ids[:n]])
            delta = a - b
            squared_error = float(delta @ h @ delta)
            true_squared_norm = float(b @ gram[:n, :n] @ b)
            projected = b @ z[:n]
            approx_squared_norm = float(projected @ projected + a @ h @ a)
            inner = float(projected @ projected + a @ h @ b)
            cosine = float(np.clip(inner / np.sqrt(true_squared_norm * approx_squared_norm), -1, 1))
            metric = {"master_id": master_id, "n": c["n"],
                      "mean_l2_error": float(np.sqrt(max(0, squared_error))),
                      "relative_l2_error": float(np.sqrt(max(0, squared_error) / true_squared_norm)),
                      "cosine_to_exact_mean": cosine,
                      "angle_degrees": float(np.degrees(np.arccos(cosine)))}
            for key in ("mean_l2_error", "relative_l2_error", "cosine_to_exact_mean", "angle_degrees"):
                close(f"oracle_{key}/{n}/{master_id}", metric[key], oracle_by_id[master_id][key],
                      3e-6 if key == "angle_degrees" else 2e-10)
            if n == 89:
                position = next(i for i, m in enumerate(state["masters"]) if m["id"] == master_id)
                close(f"sidecar_mean_squared_norm/{master_id}",
                      np.sum((sums[position] / c["n"])**2), approx_squared_norm)
            metrics.append(metric)
        for key, calculated in {
            "cluster_macro_mean_l2_error": np.mean([m["mean_l2_error"] for m in metrics]),
            "member_weighted_mean_l2_error": sum(m["n"] * m["mean_l2_error"] for m in metrics) / n,
            "max_mean_l2_error": max(m["mean_l2_error"] for m in metrics),
            "member_weighted_cosine": sum(m["n"] * m["cosine_to_exact_mean"] for m in metrics) / n,
        }.items():
            close(f"oracle_summary_{key}/{n}", calculated, audit["visual_center_error"][key], 2e-10)
        old, coefficients, old_owners = clusters, new_coefficients, owners
        independent_final_metrics = metrics
    check("final_state_matches_audit", canonical(m["member_ids"] for m in state["masters"]) ==
          canonical(c["member_ids"] for c in audits[-1]["clusters"]))
    payload_bytes = sums.nbytes + basis.nbytes + z.nbytes
    check("storage_numeric_payload", payload_bytes == results["storage"]["visual_numeric_payload_bytes"])
    check("storage_actual_files", sum(p.stat().st_size for p in (OUT / "memory").iterdir()) ==
          results["storage"]["all_memory_files_bytes"])
    check("no_qa_quality_claim", results["semantic_quality_or_qa_accuracy_evaluated"] is False)
    check("explicit_omitted_k_endpoints", "K=1 and K=N" in protocol["limitations"][0])
    representation = load(OUT / "representation_diagnostics.json")
    # Use pairwise distances rather than the diagnostic's trace/centering
    # implementation: centered SSE = sum_{i<j} ||x_i-x_j||^2 / group size.
    full_dist_sq = 2 - 2 * gram
    projected_dist_sq = np.sum((z[:, None, :] - z[None, :, :])**2, axis=2)
    global_sse = np.triu(full_dist_sq, 1).sum() / len(ids)
    global_projected_sse = np.triu(projected_dist_sq, 1).sum() / len(ids)
    close("global_visual_retention", global_projected_sse / global_sse,
          representation["global_centered_visual_variance"]["two_coordinate_variance_retention"])
    full_within = projected_within = 0.
    for master in state["masters"]:
        members = [index[m] for m in master["member_ids"]]
        full_within += np.triu(full_dist_sq[np.ix_(members, members)], 1).sum() / len(members)
        projected_within += np.triu(projected_dist_sq[np.ix_(members, members)], 1).sum() / len(members)
    close("within_cluster_visual_retention", projected_within / full_within,
          representation["within_cluster_visual_variance_summary"]["pooled_two_coordinate_variance_retention"])
    filenames = {r["filename"]: index[r["image_id"]] for r in rows}
    left, right = filenames["D7_IMG_005.jpg"], filenames["D7_IMG_006.jpg"]
    close("q10_pair_distance_retention", projected_dist_sq[left, right] / full_dist_sq[left, right],
          representation["q10_D7_005_vs_D7_006"]["squared_distance_retention"])
    centered_radius = np.sqrt(global_sse / len(ids))
    rms_center_error = np.sqrt(sum(m["n"] * m["mean_l2_error"]**2
                                  for m in independent_final_metrics) / len(ids))
    close("normalized_rms_centroid_error", rms_center_error / centered_radius,
          representation["center_error_summary"]["member_weighted_rms_l2_error_relative_to_global_scale"])
    for item in representation["sources"].values():
        check(f"representation_source_sha/{item['path']}", digest(item["path"]) == item["sha256"])
    for pair in results["important_pairs"]:
        check(f"important_pair/{pair['left']}/{pair['right']}",
              pair["same_cluster"] == (nodes[filenames[pair["left"]]]["master_id"] ==
                                       nodes[filenames[pair["right"]]]["master_id"]))
    output = {
        "passed": True, "check_count": len(checks), "checks": checks,
        "validation_scope": "Current frozen inputs and independent mathematical replay; original source chain is partial",
        "historical_source_chain_complete": not historical_unavailable and not migrated_code,
        "retired_historical_sources": historical_unavailable, "migrated_code": migrated_code,
        "max_absolute_discrepancies": maxima,
        "independent_implementation": {
            "clustering": "SciPy cut_tree exact K, hand-computed Euclidean silhouettes, no core import",
            "oracle": "Own overlap intersections and residual coefficient replay from membership only",
            "image_encodes": 0, "caption_encodes": 89,
            "causality": "Each membership checked against caption prefix only; oracle independent of clustering",
        },
        "final_centroid_metrics": independent_final_metrics,
        "limitations": [
            "K=1 and K=N are generally omitted; n<=2 startup singletons is an explicit convention",
            "No semantic cluster purity, downstream QA accuracy, or visual detail quality is inferred",
            "Independent mathematical replay verifies saved oracle metrics; intermediate high-dimensional "
            "master sums are intentionally not persisted and therefore are not reloaded for per-step comparison",
            "External calibration PCA implementation reviewed; no second calibration-image encoding run",
        ],
        "script_sha256": digest(__file__),
        "audited_files_sha256": {str(p.relative_to(OUT)): digest(p) for p in
                                [OUT / "protocol.json", OUT / "results.json",
                                 OUT / "input_manifest.json", OUT / "insertion_audit.jsonl",
                                 OUT / "memory/graph.json", OUT / "memory/basis.npy",
                                 OUT / "memory/master_sums.npy", OUT / "representation_diagnostics.json"]},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "validation_after_cleanup.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "checks": len(checks),
                      "elapsed_seconds": output["elapsed_seconds"], "maxima": maxima}))


if __name__ == "__main__":
    main()
