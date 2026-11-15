"""Diagnose saved visual geometry without loading any image or encoder.

All within-domain distances and variances come from the historical CLIP Gram
matrix. Projected geometry comes from the two coordinates saved at each leaf.
Center errors are explicitly reused from the existing run's Gram evaluator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DOMAIN = "Home_Renovation_Interior_Design"
Q10_FILENAMES = [
    "D13_IMG_001.jpg", "D7_IMG_005.jpg", "D7_IMG_002.jpg",
    "D7_IMG_006.jpg", "D13_IMG_002.jpg",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonnegative(value: float, name: str) -> float:
    if value < -1e-10:
        raise ValueError(f"{name} is materially negative: {value}")
    return max(0.0, float(value))


def variance(gram: np.ndarray, coords: np.ndarray) -> dict:
    n = len(gram)
    full_sse = nonnegative(float(np.trace(gram) - gram.sum() / n), "centered SSE")
    projected_sse = float(np.square(coords - coords.mean(axis=0)).sum())
    if projected_sse > full_sse + 1e-9:
        raise ValueError("Saved coordinates retain more energy than the original vectors")
    return {
        "n": n,
        "full_centered_sum_squared_distances": full_sse,
        "two_coordinate_centered_sum_squared_distances": projected_sse,
        "two_coordinate_variance_retention": projected_sse / full_sse if full_sse > 1e-12 else None,
        "full_centered_rms_radius": float(np.sqrt(full_sse / n)),
        "two_coordinate_centered_rms_radius": float(np.sqrt(projected_sse / n)),
        "zero_variance": full_sse <= 1e-12,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = args.report_dir.resolve()
    repo = report.parent.parent
    inputs = {
        "graph": report / "memory/graph.json",
        "basis": report / "memory/basis.npy",
        "master_sums": report / "memory/master_sums.npy",
        "manifest": report / "input_manifest.json",
        "existing_center_errors": report / "results.json",
        "historical_pairwise_gram": repo / "reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv",
    }
    graph = json.loads(inputs["graph"].read_text())
    manifest = json.loads(inputs["manifest"].read_text())["home"]
    results = json.loads(inputs["existing_center_errors"].read_text())
    basis = np.load(inputs["basis"], allow_pickle=False)
    master_sums = np.load(inputs["master_sums"], allow_pickle=False)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)
    ids = [row["image_id"] for row in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs in the input manifest")
    n = len(ids)
    index = {image_id: position for position, image_id in enumerate(ids)}
    nodes = {node["id"]: node for node in graph["image_nodes"]}
    if set(nodes) != set(ids):
        raise ValueError("Graph and manifest image IDs disagree")
    coords = np.asarray([nodes[image_id]["visual_coords"] for image_id in ids], dtype=np.float64)
    if coords.shape != (n, 2):
        raise ValueError("Each leaf must expose exactly two visual coordinates")
    gram = np.eye(n, dtype=np.float64)
    seen_pairs = set()
    with inputs["historical_pairwise_gram"].open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["task"] != DOMAIN:
                continue
            left, right = index[row["earlier_id"]], index[row["later_id"]]
            pair = tuple(sorted((left, right)))
            if left == right or pair in seen_pairs:
                raise ValueError(f"Unexpected duplicate or diagonal Gram pair: {pair}")
            seen_pairs.add(pair)
            gram[left, right] = gram[right, left] = float(row["cosine"])
    if len(seen_pairs) != n * (n - 1) // 2:
        raise ValueError("Historical Gram matrix is incomplete")
    min_gram_eigenvalue = float(np.linalg.eigvalsh(gram).min())
    min_residual_eigenvalue = float(np.linalg.eigvalsh(gram - coords @ coords.T).min())
    if min(min_gram_eigenvalue, min_residual_eigenvalue) < -1e-9:
        raise ValueError("Gram or orthogonal residual Gram is not positive semidefinite")

    global_variance = variance(gram, coords)
    global_scale = global_variance["full_centered_rms_radius"]
    if global_scale <= 0:
        raise ValueError("Global centered visual scale is zero")
    existing_errors = {
        row["master_id"]: row for row in results["final_visual_center_error"]["clusters"]
    }
    clusters = []
    partition_ids = []
    for master in graph["masters"]:
        member_ids = master["member_ids"]
        partition_ids.extend(member_ids)
        positions = [index[image_id] for image_id in member_ids]
        if len(positions) != master["n"]:
            raise ValueError("Master member count mismatch")
        if any(nodes[image_id]["master_id"] != master["id"] for image_id in member_ids):
            raise ValueError("Leaf/master membership mismatch")
        center_error = existing_errors[master["id"]]
        if center_error["n"] != master["n"]:
            raise ValueError("Reused center-error member count mismatch")
        stats = variance(gram[np.ix_(positions, positions)], coords[positions])
        sum_vector = master_sums[master["visual_sum_ref"]["row"]]
        np.testing.assert_allclose(basis.T @ sum_vector, coords[positions].sum(axis=0), atol=1e-11)
        true_mean_norm_squared = float(gram[np.ix_(positions, positions)].sum()) / master["n"] ** 2
        approximate_mean_norm = float(np.linalg.norm(sum_vector / master["n"]))
        error = float(center_error["mean_l2_error"])
        clusters.append({
            "master_id": master["id"],
            "member_ids": member_ids,
            "approximate": master["approximate"],
            **stats,
            "center_error": {
                "l2_error_reused_from_existing_gram_evaluator": error,
                "relative_to_global_centered_rms_radius": error / global_scale,
                "relative_to_own_cluster_centered_rms_radius": (
                    error / stats["full_centered_rms_radius"] if not stats["zero_variance"] else None
                ),
                "cosine_to_exact_mean_reused": center_error["cosine_to_exact_mean"],
                "exact_mean_norm_from_gram": float(np.sqrt(true_mean_norm_squared)),
                "approximate_mean_norm_from_saved_master": approximate_mean_norm,
            },
        })
    if sorted(partition_ids) != sorted(ids):
        raise ValueError("Final clusters do not partition all inputs exactly once")
    if set(existing_errors) != {row["master_id"] for row in clusters}:
        raise ValueError("Existing center errors do not cover exactly the current masters")

    within_full = sum(row["full_centered_sum_squared_distances"] for row in clusters)
    within_two = sum(row["two_coordinate_centered_sum_squared_distances"] for row in clusters)
    nonzero = [row for row in clusters if not row["zero_variance"]]
    errors = [row["center_error"]["l2_error_reused_from_existing_gram_evaluator"] for row in clusters]
    weights = np.array([row["n"] for row in clusters], dtype=np.float64)
    center_summary = {
        "source": "results.json: final_visual_center_error.clusters; no independent center-error recomputation here",
        "global_centered_scale_definition": "sqrt(sum_i ||v_i - global_mean||^2 / N), computed from the historical Gram matrix",
        "global_centered_rms_radius": global_scale,
        "cluster_macro_mean_l2_error_relative_to_global_scale": float(np.mean(errors) / global_scale),
        "member_weighted_mean_l2_error_relative_to_global_scale": float(np.average(errors, weights=weights) / global_scale),
        "member_weighted_rms_l2_error_relative_to_global_scale": float(np.sqrt(np.average(np.square(errors), weights=weights)) / global_scale),
        "max_l2_error_relative_to_global_scale": float(max(errors) / global_scale),
        "member_weighted_cosine_to_exact_mean": float(np.average([
            row["center_error"]["cosine_to_exact_mean_reused"] for row in clusters
        ], weights=weights)),
    }
    by_filename = {}
    for row in manifest:
        if row["filename"] in by_filename:
            raise ValueError("This Q10 lookup requires unique filenames")
        by_filename[row["filename"]] = row
    q10 = []
    for rank, filename in enumerate(Q10_FILENAMES, 1):
        row = by_filename[filename]
        node = nodes[row["image_id"]]
        q10.append({
            "rank_in_190_point_top5": rank,
            "filename": filename,
            "image_id": row["image_id"],
            "memory_id": row["memory_id"],
            "master_id": node["master_id"],
            "visual_coords": node["visual_coords"],
        })
    left_id = by_filename["D7_IMG_005.jpg"]["image_id"]
    right_id = by_filename["D7_IMG_006.jpg"]["image_id"]
    left, right = index[left_id], index[right_id]
    pair_full_sq = nonnegative(
        float(gram[left, left] + gram[right, right] - 2 * gram[left, right]), "pair distance squared",
    )
    pair_two_sq = float(np.square(coords[left] - coords[right]).sum())
    pair = {
        "left_image_id": left_id,
        "right_image_id": right_id,
        "same_final_master": nodes[left_id]["master_id"] == nodes[right_id]["master_id"],
        "original_cosine": float(gram[left, right]),
        "original_squared_distance": pair_full_sq,
        "original_distance": float(np.sqrt(pair_full_sq)),
        "two_coordinate_squared_distance": pair_two_sq,
        "two_coordinate_distance": float(np.sqrt(pair_two_sq)),
        "squared_distance_retention": pair_two_sq / pair_full_sq,
        "distance_ratio": float(np.sqrt(pair_two_sq / pair_full_sq)),
    }
    within_ratio = within_two / within_full
    global_ratio = global_variance["two_coordinate_variance_retention"]
    output = {
        "domain": DOMAIN,
        "n_images": n,
        "n_final_clusters": len(clusters),
        "analysis_kind": "post-hoc saved-geometry diagnostic; does not alter memory or encode inputs",
        "sources": {key: {"path": str(path), "sha256": sha256(path)} for key, path in inputs.items()},
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "definitions": {
            "visual_vectors": "L2-normalized 512-D CLIP image representations; only their historical pairwise Gram matrix is read",
            "coordinates": "Stored absolute projections z_i = U.T @ v_i, using the fixed external-domain PCA basis",
            "retention": "Ratio of sums of squared centered distances, not the ratio of unsquared distances",
            "within_cluster_aggregation": "Sum each final cluster's centered squared distances, then divide projected total by full-space total",
            "singletons": "Within-cluster variance is zero; individual retention is null and no variance mass is added",
            "relation_to_previous_top5_experiment": "This fixed shared two-dimensional basis differs from fitting a separate PCA to each QA's Top5",
            "center_errors": "Reused existing evaluator errors; normalized here by the independently calculated global centered RMS radius",
        },
        "global_centered_visual_variance": global_variance,
        "within_cluster_visual_variance_summary": {
            "nonzero_variance_clusters": len(nonzero),
            "zero_variance_clusters": len(clusters) - len(nonzero),
            "full_within_cluster_sum_squared_distances": within_full,
            "two_coordinate_within_cluster_sum_squared_distances": within_two,
            "pooled_two_coordinate_variance_retention": within_ratio,
            "macro_mean_nonzero_cluster_retention": float(np.mean([
                row["two_coordinate_variance_retention"] for row in nonzero
            ])),
        },
        "center_error_summary": center_summary,
        "clusters": clusters,
        "q10_top5": q10,
        "q10_D7_005_vs_D7_006": pair,
        "validation": {
            "historical_pair_count": len(seen_pairs),
            "minimum_gram_eigenvalue": min_gram_eigenvalue,
            "minimum_orthogonal_residual_gram_eigenvalue": min_residual_eigenvalue,
            "basis_orthonormal": True,
            "graph_and_manifest_coverage_equal": True,
            "master_sum_projections_match_member_coords": True,
            "input_image_reads": 0,
            "encoder_calls": 0,
            "memory_writes": 0,
        },
        "interpretation": {
            "measured": (
                f"The fixed two-coordinate representation retains {global_ratio:.2%} of Home's total centered visual variance, "
                f"{within_ratio:.2%} of the final clusters' pooled within-cluster variance, and "
                f"{pair['squared_distance_retention']:.2%} of the D7_IMG_005 versus D7_IMG_006 squared distance."
            ),
            "center_warning": (
                f"The member-weighted master/exact-center cosine is "
                f"{center_summary['member_weighted_cosine_to_exact_mean']:.6f}, but the member-weighted RMS center error is "
                f"{center_summary['member_weighted_rms_l2_error_relative_to_global_scale']:.2%} of Home's centered visual RMS radius. "
                "High raw center cosine alone does not establish preservation of within-domain differences."
            ),
            "limits": (
                "These are representation-geometry diagnostics, not semantic ground truth, clustering accuracy, or QA accuracy. "
                "The stored master centers retain additional aggregate high-dimensional information; the reported two-coordinate "
                "variance retention measures only leaf-level distinguishability in the fixed plane."
            ),
        },
    }
    destination = report / "representation_diagnostics.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(destination),
        "global_two_coordinate_variance_retention": global_ratio,
        "within_cluster_two_coordinate_variance_retention": within_ratio,
        "center_weighted_rms_error_relative_to_global_scale": center_summary["member_weighted_rms_l2_error_relative_to_global_scale"],
        "q10_005_006_squared_distance_retention": pair["squared_distance_retention"],
    }, indent=2))


if __name__ == "__main__":
    main()
