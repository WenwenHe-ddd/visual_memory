#!/usr/bin/env python3
"""Audit the saved 2D/4D comparison without encoding images or importing runners."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path("/storage/hewenwen/VisualMemory")
OLD = ROOT / "reports/home_caption_ward_visual_20261020"
OUT = ROOT / "reports/home_caption_visual_2d4d_20261020"
sys.path.insert(0, str(ROOT / ".vendor"))
import numpy as np

checks = []
maxima = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, condition):
    checks.append({"check": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)


def close(name, value, expected, tolerance=1e-10):
    error = float(np.max(np.abs(np.asarray(value) - np.asarray(expected)), initial=0.))
    family = name.split("/")[0]
    maxima[family] = max(maxima.get(family, 0.), error)
    check(name, error <= tolerance)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    started = time.perf_counter()
    protocol = read(OUT / "protocol.json")
    results = read(OUT / "results.json")
    calibration = read(OUT / "calibration.json")
    manifest = read(OLD / "input_manifest.json")
    old_steps = jsonl(OLD / "insertion_audit.jsonl")
    new_steps = jsonl(OUT / "insertion_audit.jsonl")
    old_graph = read(OLD / "memory/graph.json")
    new_graph = read(OUT / "memory/graph.json")
    old_sums = np.load(OLD / "memory/master_sums.npy", allow_pickle=False)
    new_sums = np.load(OUT / "memory/master_sums.npy", allow_pickle=False)
    old_basis = np.load(OLD / "memory/basis.npy", allow_pickle=False)
    new_basis = np.load(OUT / "memory/basis.npy", allow_pickle=False)
    rows = manifest["home"]
    ids = [r["image_id"] for r in rows]
    index = {image_id: i for i, image_id in enumerate(ids)}
    for path, expected in protocol["source_sha256"].items():
        check(f"source_sha/{path}", sha(path) == expected)
    check("original_independent_validation_passed", read(OLD / "validation_independent.json")["passed"])
    check("89_chronological_arrivals", len(rows) == len(old_steps) == len(new_steps) == 89)
    check("date_order", [r["timestamp"] for r in rows] == sorted(r["timestamp"] for r in rows))
    check("344_unique_external_calibration",
          len(manifest["calibration"]) == len({r["image_sha256"] for r in manifest["calibration"]}) == 344)
    check("no_home_calibration_hash_overlap", not
          {r["image_sha256"] for r in rows}.intersection(
              r["image_sha256"] for r in manifest["calibration"]))
    check("same_encoder_metadata", protocol["image_encoder"] == read(OLD / "protocol.json")["image_encoder"])
    check("nested_basis_shape", new_basis.shape == (512, 4) and old_basis.shape == (512, 2))
    close("nested_first_two_basis", new_basis[:, :2], old_basis, 1e-11)
    close("new_basis_orthonormal", new_basis.T @ new_basis, np.eye(4), 1e-11)
    close("calibration_spectrum", calibration["eigenvalues"],
          read(OLD / "calibration.json")["eigenvalues"], 1e-12)
    spectrum = np.array(calibration["eigenvalues"])
    for rank in (2, 4):
        close(f"calibration_variance/{rank}", spectrum[:rank].sum() / spectrum.sum(),
              calibration[f"top{rank}_centered_calibration_variance_ratio"])
    check("no_caption_reclustering", results["caption_partitions_recomputed"] is False)
    check("same_final_cluster_count", len(old_graph["masters"]) == len(new_graph["masters"]) == 25)
    check("same_final_member_ids", [
        (m["id"], m["member_ids"], m["n"], m["approximate"]) for m in old_graph["masters"]
    ] == [(m["id"], m["member_ids"], m["n"], m["approximate"]) for m in new_graph["masters"]])
    expected_keys = {"id", "caption", "dialogue", "timestamp", "visual_coords", "master_id"}
    for old, new, row in zip(old_graph["image_nodes"], new_graph["image_nodes"], rows):
        check(f"leaf_schema/{row['image_id']}", set(new) == expected_keys)
        check(f"leaf_four_coords/{row['image_id']}",
              len(new["visual_coords"]) == 4 and np.isfinite(new["visual_coords"]).all())
        check(f"leaf_preserved_information/{row['image_id']}",
              all(new[k] == old[k] for k in expected_keys - {"visual_coords"}))
        close(f"nested_leaf_coords/{row['image_id']}", new["visual_coords"][:2], old["visual_coords"])
    check("only_expected_memory_files", {p.name for p in (OUT / "memory").iterdir()} ==
          {"basis.npy", "master_sums.npy", "graph.json"})
    check("only_final_master_sums", new_sums.shape == old_sums.shape == (25, 512))
    check("float64_arrays", new_sums.dtype == new_basis.dtype == np.float64)
    z4 = np.array([n["visual_coords"] for n in new_graph["image_nodes"]])
    coords = {2: z4[:, :2], 4: z4}
    gram = np.eye(89)
    seen = set()
    gram_path = ROOT / "reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv"
    for pair in csv.DictReader(gram_path.open()):
        if pair["earlier_id"] in index and pair["later_id"] in index:
            i, j = index[pair["earlier_id"]], index[pair["later_id"]]
            check(f"gram_pair_unique/{i}/{j}", (min(i, j), max(i, j)) not in seen and i != j)
            seen.add((min(i, j), max(i, j)))
            gram[i, j] = gram[j, i] = float(pair["cosine"])
    check("gram_complete", len(seen) == 3916)
    old_clusters = {}
    coefficients = {}
    count = 0
    final_measured = {}
    for n, (old_step, new_step) in enumerate(zip(old_steps, new_steps), 1):
        check(f"step_control/{n}", new_step["step"] == old_step["n_images"] == n and
              new_step["image_id"] == old_step["image_id"] == ids[n-1] and
              new_step["n_clusters"] == old_step["n_clusters"] and
              new_step["split_master_ids"] == old_step["split_master_ids"])
        clusters = {c["master_id"]: c for c in old_step["clusters"]}
        check(f"prefix_only_members/{n}", sorted(m for c in clusters.values() for m in c["member_ids"]) == sorted(ids[:n]))
        next_coefficients = {}
        for mid, cluster in clusters.items():
            a = np.zeros(n)
            for old_mid, old in old_clusters.items():
                overlap = set(cluster["member_ids"]) & set(old["member_ids"])
                if overlap:
                    a[:n-1] += coefficients[old_mid] * (len(overlap) / old["n"])
            if ids[n-1] in cluster["member_ids"]:
                a[-1] += 1
            next_coefficients[mid] = a
        close(f"coefficient_conservation/{n}", sum(next_coefficients.values()), np.ones(n))
        measured = {}
        for rank in (2, 4):
            h = gram[:n, :n] - coords[rank][:n] @ coords[rank][:n].T
            check(f"residual_gram_psd/{rank}/{n}", np.linalg.eigvalsh(h).min() >= -1e-10)
            recorded = {m["master_id"]: m for m in new_step["arms"][str(rank)]["visual_center_error"]["clusters"]}
            check(f"every_master_evaluated/{rank}/{n}", set(recorded) == set(clusters))
            measured[rank] = {}
            for mid, cluster in clusters.items():
                size = cluster["n"]
                a = next_coefficients[mid] / size
                b = np.array([float(i in cluster["member_ids"]) / size for i in ids[:n]])
                delta = a - b
                error_sq = float(delta @ h @ delta)
                true_sq = float(b @ gram[:n, :n] @ b)
                p = b @ coords[rank][:n]
                approx_sq = float(p @ p + a @ h @ a)
                inner = float(p @ p + a @ h @ b)
                cosine = float(np.clip(inner / np.sqrt(true_sq * approx_sq), -1, 1))
                metrics = {
                    "mean_l2_error": float(np.sqrt(max(0, error_sq))),
                    "relative_l2_error": float(np.sqrt(max(0, error_sq) / true_sq)),
                    "cosine_to_exact_mean": cosine,
                    "angle_degrees": float(np.degrees(np.arccos(cosine))),
                    "mean_norm_squared": approx_sq,
                }
                for key in ("mean_l2_error", "relative_l2_error", "cosine_to_exact_mean", "angle_degrees"):
                    close(f"independent_{key}/{rank}/{n}/{mid}", metrics[key], recorded[mid][key],
                          3e-6 if key == "angle_degrees" else 2e-10)
                measured[rank][mid] = metrics
                if rank == 2:
                    old_recorded = next(m for m in old_step["visual_center_error"]["clusters"] if m["master_id"] == mid)
                    for key in ("mean_l2_error", "relative_l2_error", "cosine_to_exact_mean"):
                        close(f"original_2d_reproduction_{key}/{n}/{mid}", metrics[key], old_recorded[key], 2e-10)
            for key, value in {
                "cluster_macro_mean_l2_error": np.mean([m["mean_l2_error"] for m in measured[rank].values()]),
                "member_weighted_mean_l2_error": sum(clusters[mid]["n"] * m["mean_l2_error"] for mid, m in measured[rank].items()) / n,
                "max_mean_l2_error": max(m["mean_l2_error"] for m in measured[rank].values()),
                "member_weighted_cosine": sum(clusters[mid]["n"] * m["cosine_to_exact_mean"] for mid, m in measured[rank].items()) / n,
            }.items():
                close(f"summary_{key}/{rank}/{n}", value,
                      new_step["arms"][str(rank)]["visual_center_error"][key])
            check(f"reported_global_sum_conservation/{rank}/{n}",
                  new_step["arms"][str(rank)]["global_sum_l2_error"] < 1e-10)
            check(f"reported_projection_invariant/{rank}/{n}",
                  new_step["arms"][str(rank)]["projection_l2_error"] < 1e-10)
        for mid, cluster in clusters.items():
            a = next_coefficients[mid] / cluster["n"]
            b = np.array([float(i in cluster["member_ids"]) / cluster["n"] for i in ids[:n]])
            e2, e4 = measured[2][mid]["mean_l2_error"], measured[4][mid]["mean_l2_error"]
            improvement = float(np.sum(((a - b) @ z4[:n, 2:])**2))
            check(f"all_master_l2_nonincrease/{n}/{mid}", e4 <= e2 + 1e-10)
            close(f"squared_error_improvement_identity/{n}/{mid}", e2*e2-e4*e4, improvement, 2e-10)
            count += 1
        old_clusters, coefficients = clusters, next_coefficients
        final_measured = measured
    check("all1029_master_prefix_comparisons", count == 1029)
    for position, master in enumerate(new_graph["masters"]):
        mid = master["id"]
        members = [index[i] for i in master["member_ids"]]
        check(f"sidecar_sum_ref/{mid}", master["visual_sum_ref"] == {"path": "master_sums.npy", "row": position})
        check(f"sidecar_mean_ref/{mid}", master["visual_mean_ref"] ==
              {"path": "master_sums.npy", "row": position, "scale": 1 / master["n"]})
        close(f"actual_four_projection/{mid}", new_basis.T @ new_sums[position], z4[members].sum(axis=0))
        close(f"actual_four_mean_norm/{mid}", np.sum((new_sums[position] / master["n"])**2),
              final_measured[4][mid]["mean_norm_squared"])
        b = np.array([float(image_id in master["member_ids"]) for image_id in ids])
        projected_correction = new_basis[:, 2:] @ ((b - coefficients[mid]) @ z4[:, 2:])
        recovered_two = new_sums[position] - projected_correction
        close(f"recover_original_2d_full_sum/{mid}", recovered_two, old_sums[position], 1e-10)
    close("global_sum_matches_original", new_sums.sum(axis=0), old_sums.sum(axis=0), 1e-10)
    close("global_sum_norm_matches_gram", np.sum(new_sums.sum(axis=0)**2), gram.sum(), 1e-8)
    filename_index = {r["filename"]: i for i, r in enumerate(rows)}
    i, j = [filename_index[name] for name in ("D7_IMG_005.jpg", "D7_IMG_006.jpg")]
    full_sq_dist = 2 - 2 * gram
    for rank in (2, 4):
        z = coords[rank]
        projected_sq_dist = np.sum((z[:, None] - z[None])**2, axis=2)
        global_retention = np.triu(projected_sq_dist, 1).sum() / np.triu(full_sq_dist, 1).sum()
        full_within = projection_within = 0.
        for master in new_graph["masters"]:
            members = [index[m] for m in master["member_ids"]]
            full_within += np.triu(full_sq_dist[np.ix_(members, members)], 1).sum() / len(members)
            projection_within += np.triu(projected_sq_dist[np.ix_(members, members)], 1).sum() / len(members)
        close(f"global_variance_retention/{rank}", global_retention,
              results["arms"][str(rank)]["global_home_centered_variance"]["retained_fraction"])
        close(f"within_variance_retention/{rank}", projection_within / full_within,
              results["arms"][str(rank)]["pooled_final_cluster_within_variance"]["retained_fraction"])
        close(f"q10_squared_distance_retention/{rank}", projected_sq_dist[i, j] / full_sq_dist[i, j],
              results["arms"][str(rank)]["q10_D7_IMG_005_vs_D7_IMG_006_squared_distance_retention"])
        payload = (512 * rank + 25 * 512 + 89 * rank) * 8
        check(f"numeric_payload_bytes/{rank}", results["arms"][str(rank)]["storage"]["visual_numeric_payload_bytes"] == payload)
    check("additional_four_dim_bytes", results["storage_extra_bytes_4d_vs_2d"] == 9616)
    centering = np.eye(89) - np.ones((89, 89)) / 89
    eigenvalues = np.linalg.eigvalsh(centering @ gram @ centering)[::-1]
    posthoc = results["global_home_posthoc_best_linear_pca"]
    check("fullhome_pca_only_evaluation", posthoc["evaluation_only"] and
          posthoc["uses_all_home_images_including_future"] and not posthoc["used_by_stream_algorithm"])
    for rank in (2, 4):
        close(f"posthoc_home_pca_best_fraction/{rank}", eigenvalues[:rank].sum() / eigenvalues.sum(),
              posthoc[f"top{rank}_centered_variance_fraction"])
    check("no_qa_claim", results["qa_accuracy_evaluated"] is False)
    check("fresh_replay_disclosure", "fresh empty-memory replay" in protocol["previous_experiment_images"])
    final_two = sum(old_clusters[mid]["n"] * value["mean_l2_error"]
                    for mid, value in final_measured[2].items()) / 89
    final_four = sum(old_clusters[mid]["n"] * value["mean_l2_error"]
                     for mid, value in final_measured[4].items()) / 89
    close("weighted_relative_improvement", (final_two-final_four)/final_two,
          results["final_weighted_mean_l2_relative_reduction"])
    output = {
        "passed": True, "check_count": len(checks), "checks": checks,
        "master_prefix_comparisons": count,
        "max_absolute_discrepancies": maxima,
        "independent_method": "Membership-only overlap/coefficient replay, historical Gram and final coordinates; no imports from core, comparison runner or its evaluator",
        "auditor_image_encodes": 0, "auditor_caption_encodes": 0,
        "final_weighted_l2": {"2d": final_two, "4d": final_four},
        "limitations": [
            "Intermediate full master sums are intentionally not persisted; their reported per-step invariants are checked, while independent actual-vector comparisons use final sidecars",
            "Calibration covariance accumulation and stream encoder call sites were reviewed statically; no second encoding run is performed by this audit",
            "All Home PCA spectrum is evaluation-only and cannot select the causal stream basis",
            "Nested dimensions mathematically guarantee nonincreasing centroid L2 under this fixed partition and redistribution rule; QA performance is not evaluated",
        ],
        "script_sha256": sha(__file__),
        "audited_files_sha256": {str(path.relative_to(OUT)): sha(path) for path in [
            OUT / "protocol.json", OUT / "results.json", OUT / "insertion_audit.jsonl",
            OUT / "calibration.json", OUT / "memory/graph.json", OUT / "memory/basis.npy",
            OUT / "memory/master_sums.npy"]},
        "elapsed_seconds": time.perf_counter()-started,
    }
    (OUT / "validation_independent.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": True, "checks": len(checks), "comparisons": count,
                      "seconds": output["elapsed_seconds"], "final": output["final_weighted_l2"],
                      "maxima": maxima}))


if __name__ == "__main__":
    main()
