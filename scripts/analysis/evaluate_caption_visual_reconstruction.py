#!/usr/bin/env python3
"""Evaluate per-image reconstruction from the saved final 2D memory only.

Reconstruction never accesses original vectors, original pixels, or insertion
history. An isolated Gram/history evaluator scores it against original vectors,
with an additional direct check on the already-saved Q10 five-vector artifact.
No original or reconstructed per-image vectors are written to disk.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/visual_reconstruction_matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/home_caption_ward_visual_20261020"
PAIRWISE = ROOT / "reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv"
Q10 = ROOT / "reports/home_q10_top5_embeddings_20261020/embeddings.npz"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def summary(values):
    x = np.array(values, dtype=np.float64)
    return {
        "n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
        "min": float(x.min()), "max": float(x.max()),
        "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
        "std_population": float(x.std()),
    }


def reconstruct(node, master, basis, mean_coords):
    """The deployment reconstruction path: final state only, no oracle input."""
    mean = master["visual_sum"] / master["n"]
    return mean + basis @ (np.asarray(node["visual_coords"]) - mean_coords)


def load_gram(ids):
    index = {key: i for i, key in enumerate(ids)}
    gram = np.eye(len(ids))
    seen = set()
    with PAIRWISE.open() as handle:
        for row in csv.DictReader(handle):
            if row["earlier_id"] not in index or row["later_id"] not in index:
                continue
            i, j = index[row["earlier_id"]], index[row["later_id"]]
            key = tuple(sorted((i, j)))
            assert i != j and key not in seen
            seen.add(key)
            gram[i, j] = gram[j, i] = float(row["cosine"])
    assert len(seen) == len(ids) * (len(ids) - 1) // 2
    return gram


def evaluation_weights(ids, audits):
    """Replay residual coefficients for scoring, never for reconstruction."""
    index = {key: i for i, key in enumerate(ids)}
    old_groups, old_weights = {}, {}
    for step, audit in enumerate(audits, 1):
        assert audit["image_id"] == ids[step-1]
        groups = {c["master_id"]: set(c["member_ids"]) for c in audit["clusters"]}
        assert set().union(*groups.values()) == set(ids[:step])
        weights = {}
        for key, members in groups.items():
            coefficient = np.zeros(len(ids))
            for old_key, old_members in old_groups.items():
                overlap = len(members & old_members)
                if overlap:
                    coefficient += overlap / len(old_members) * old_weights[old_key]
            if audit["image_id"] in members:
                coefficient[index[audit["image_id"]]] += 1
            assert abs(coefficient.sum() - len(members)) < 1e-11
            weights[key] = coefficient
        assert np.max(np.abs(sum(weights.values())[:step]-1)) < 1e-12
        old_groups, old_weights = groups, weights
    return {key: value / len(old_groups[key]) for key, value in old_weights.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/home_image_reconstruction_2d_20261020")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Use a fresh --output directory.")
    out.mkdir(parents=True, exist_ok=True)
    graph = read_json(SOURCE / "memory/graph.json")
    manifest = read_json(SOURCE / "input_manifest.json")["home"]
    audits = [json.loads(line) for line in (SOURCE / "insertion_audit.jsonl").read_text().splitlines()]
    assert graph["leaf_visual_dim"] == 2
    basis = np.load(SOURCE / "memory" / graph["basis_ref"]["path"], allow_pickle=False)
    sidecar = np.load(SOURCE / "memory/master_sums.npy", allow_pickle=False)
    nodes = {node["id"]: node for node in graph["image_nodes"]}
    ids = list(nodes)
    index = {key: i for i, key in enumerate(ids)}
    assert ids == [row["image_id"] for row in manifest] and len(ids) == 89
    inputs = {row["image_id"]: row for row in manifest}
    coords = np.array([nodes[key]["visual_coords"] for key in ids])
    assert basis.shape == (512, 2) and coords.shape == (89, 2)
    assert np.max(np.abs(basis.T @ basis - np.eye(2))) < 1e-12
    masters = {}
    zbar = {}
    for master in graph["masters"]:
        key = master["id"]
        masters[key] = {**master, "visual_sum": sidecar[master["visual_sum_ref"]["row"]]}
        positions = [index[m] for m in master["member_ids"]]
        zbar[key] = coords[positions].mean(axis=0)
        assert np.max(np.abs(basis.T @ (masters[key]["visual_sum"] / master["n"]) - zbar[key])) < 1e-12

    # Original-image geometry stays inside the evaluator.
    gram = load_gram(ids)
    residual_gram = gram - coords @ coords.T
    weights = evaluation_weights(ids, audits)
    master_ids = list(masters)
    w_matrix = np.array([weights[key] for key in master_ids])
    master_residuals = np.array([
        masters[key]["visual_sum"] / masters[key]["n"] - basis @ zbar[key] for key in master_ids])
    residual_inner_error = float(np.max(np.abs(
        master_residuals @ master_residuals.T - w_matrix @ residual_gram @ w_matrix.T)))
    assert residual_inner_error < 1e-11
    results = []
    norm_discrepancies = []
    projection_discrepancies = []
    for key in ids:
        i, node = index[key], nodes[key]
        mid = node["master_id"]
        master = masters[mid]
        rebuilt = reconstruct(node, master, basis, zbar[mid])
        norm = float(np.linalg.norm(rebuilt))
        assert norm > 0
        w, z = weights[mid], coords[i]
        inner = float(z @ z + w @ residual_gram[:, i])
        predicted_norm_squared = float(z @ z + w @ residual_gram @ w)
        norm_discrepancies.append(abs(norm**2 - predicted_norm_squared))
        assert norm_discrepancies[-1] < 1e-11
        projection_discrepancies.append(float(np.max(np.abs(basis.T @ rebuilt - z))))
        assert projection_discrepancies[-1] < 1e-11
        cosine = float(np.clip(inner / norm, -1, 1))  # original norm is exactly one
        delta = w.copy()
        delta[i] -= 1
        squared_l2 = float(delta @ residual_gram @ delta)
        assert squared_l2 >= -1e-11
        assert abs(squared_l2 - (1 + norm**2 - 2 * inner)) < 1e-11
        center = master["visual_sum"] / master["n"]
        center_inner = float(zbar[mid] @ z + w @ residual_gram[:, i])
        center_cosine = float(np.clip(center_inner / np.linalg.norm(center), -1, 1))
        results.append({
            "image_id": key, "filename": inputs[key]["filename"], "master_id": mid,
            "cluster_size": master["n"], "master_approximate": master["approximate"],
            "cosine_original_vs_reconstructed": cosine,
            "cosine_original_vs_master_only": center_cosine,
            "cosine_gain_from_two_coords": cosine - center_cosine,
            "reconstruction_raw_l2_error": float(np.sqrt(max(0, squared_l2))),
            "reconstruction_norm": norm,
            "normalized_reconstruction_l2_error": float(np.sqrt(max(0, 2 - 2*cosine))),
        })
        del rebuilt, center

    # Direct dot products against five preexisting original vectors independently
    # confirm that the Gram evaluator scores the actual saved-state reconstruction.
    direct_checks = []
    row_by_id = {row["image_id"]: row for row in results}
    with np.load(Q10, allow_pickle=False) as cache:
        q10_ids = cache["memory_ids"].tolist()
        q10_vectors = cache["image_embeddings"]
        assert q10_vectors.shape == (5, 512)
        for mid, original in zip(q10_ids, q10_vectors):
            key = f"{mid}#image1"
            node = nodes[key]
            master_id = node["master_id"]
            rebuilt = reconstruct(node, masters[master_id], basis, zbar[master_id])
            direct_cosine = float(original @ rebuilt / (np.linalg.norm(original) * np.linalg.norm(rebuilt)))
            direct_l2 = float(np.linalg.norm(original - rebuilt))
            direct_checks.append({
                "image_id": key,
                "direct_cosine": direct_cosine,
                "gram_cosine": row_by_id[key]["cosine_original_vs_reconstructed"],
                "cosine_abs_difference": abs(direct_cosine-row_by_id[key]["cosine_original_vs_reconstructed"]),
                "l2_abs_difference": abs(direct_l2-row_by_id[key]["reconstruction_raw_l2_error"]),
                "original_projection_max_abs_difference": float(np.max(np.abs(basis.T @ original-coords[index[key]]))),
            })
            assert max(direct_checks[-1][k] for k in (
                "cosine_abs_difference", "l2_abs_difference", "original_projection_max_abs_difference")) < 2e-6
            del rebuilt
        del q10_vectors

    # Same-cluster reconstruction differs only in the two retained directions.
    filenames = {row["filename"]: row["image_id"] for row in manifest}
    pair_ids = [filenames[name] for name in ("D7_IMG_005.jpg", "D7_IMG_006.jpg")]
    first, second = [nodes[key] for key in pair_ids]
    first_rebuilt = reconstruct(first, masters[first["master_id"]], basis, zbar[first["master_id"]])
    second_rebuilt = reconstruct(second, masters[second["master_id"]], basis, zbar[second["master_id"]])
    original_pair_cosine = float(gram[index[pair_ids[0]], index[pair_ids[1]]])
    original_pair_squared_distance = 2 - 2*original_pair_cosine
    reconstructed_pair_squared_distance = float(np.sum((first_rebuilt-second_rebuilt)**2))
    pair = {
        "images": pair_ids, "same_master": first["master_id"] == second["master_id"],
        "original_pair_cosine": original_pair_cosine,
        "reconstructed_pair_cosine": float(first_rebuilt @ second_rebuilt /
                                          (np.linalg.norm(first_rebuilt)*np.linalg.norm(second_rebuilt))),
        "original_squared_distance": original_pair_squared_distance,
        "reconstructed_squared_distance": reconstructed_pair_squared_distance,
        "squared_distance_retention": reconstructed_pair_squared_distance / original_pair_squared_distance,
        "interpretation": "Pairwise distinguishability is separate from own-original reconstruction cosine.",
    }
    del first_rebuilt, second_rebuilt
    ordered = sorted(results, key=lambda row: row["cosine_original_vs_reconstructed"])
    cosine_gains = [row["cosine_gain_from_two_coords"] for row in results]
    summary_result = {
        "n_images": len(ids), "n_masters": len(masters), "leaf_visual_dimensions": 2,
        "reconstruction_formula": "vhat_i = saved_master_mean + U @ (saved_z_i - mean_member_z)",
        "cosine_original_vs_reconstructed": summary([r["cosine_original_vs_reconstructed"] for r in results]),
        "cosine_original_vs_master_only": summary([r["cosine_original_vs_master_only"] for r in results]),
        "cosine_gain_from_two_coords": {
            **summary(cosine_gains),
            "positive_over_1e_12": sum(g > 1e-12 for g in cosine_gains),
            "negative_below_minus_1e_12": sum(g < -1e-12 for g in cosine_gains),
            "numerical_ties": sum(abs(g) <= 1e-12 for g in cosine_gains),
        },
        "reconstruction_raw_l2_error": summary([r["reconstruction_raw_l2_error"] for r in results]),
        "reconstruction_norm": summary([r["reconstruction_norm"] for r in results]),
        "lowest_cosine_5": ordered[:5], "highest_cosine_5": ordered[-5:][::-1],
        "q10_top5": [row_by_id[f"{key}#image1"] for key in q10_ids],
        "q10_D7_005_vs_D7_006": pair,
        "direct_q10_cache_crosschecks": direct_checks,
        "interpretation": "Observed reconstruction geometry, not a measure of semantic-detail or QA accuracy.",
    }
    save_json(out / "results.json", summary_result)
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (out / "per_cluster.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["master_id", "n", "approximate", "mean_cosine", "min_cosine", "max_cosine", "mean_l2"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, master in masters.items():
            members = [row_by_id[m] for m in master["member_ids"]]
            cs = [m["cosine_original_vs_reconstructed"] for m in members]
            writer.writerow({"master_id": key, "n": master["n"], "approximate": master["approximate"],
                             "mean_cosine": np.mean(cs), "min_cosine": min(cs), "max_cosine": max(cs),
                             "mean_l2": np.mean([m["reconstruction_raw_l2_error"] for m in members])})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cosines = np.array([r["cosine_original_vs_reconstructed"] for r in results])
    centers = np.array([r["cosine_original_vs_master_only"] for r in results])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    axes[0].hist(cosines, bins=12, color="#347caa", edgecolor="white")
    axes[0].axvline(cosines.mean(), color="#d76838", label=f"Mean = {cosines.mean():.4f}")
    axes[0].set(xlabel="Cosine: original vs reconstructed", ylabel="Images",
                title="Home: 89 individual reconstructions")
    axes[0].legend()
    axes[1].scatter(centers, cosines, s=25, alpha=0.7, color="#347caa")
    lo = float(min(centers.min(), cosines.min())) - .01
    axes[1].plot([lo, 1.005], [lo, 1.005], "--", color="gray", label="Equal cosine")
    axes[1].set(xlim=(lo, 1.005), ylim=(lo, 1.005), xlabel="Cosine using master mean only",
                ylabel="Cosine using master + 2D offset", title="Effect of the two saved coordinates")
    axes[1].set_aspect("equal")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"reconstruction_similarity.{ext}", dpi=180)
    plt.close(fig)
    sources = [
        Path(__file__), SOURCE / "memory/graph.json", SOURCE / "memory/basis.npy",
        SOURCE / "memory/master_sums.npy", SOURCE / "input_manifest.json",
        SOURCE / "insertion_audit.jsonl", PAIRWISE, Q10,
    ]
    save_json(out / "protocol.json", {
        "domain": "Home_Renovation_Interior_Design",
        "memory_source": str(SOURCE / "memory"), "state": "final 2D state after all 89 arrivals",
        "reconstruction_inputs_only": ["saved master sums/counts", "fixed 512x2 basis", "saved leaf 2D coordinates"],
        "score": "cosine between original unit CLIP image vector and reconstructed vector, normalizing both",
        "reconstruction_vector_normalization": "not renormalized before raw-L2 measurement; cosine handles norms",
        "oracle_inputs": "historical original Gram plus independent residual-weight replay; never fed into reconstruction",
        "oracle_math": "H=G-ZZ.T; w=master residual coefficients/n; original dot vhat_i=||z_i||^2+w.T H[:,i]",
        "direct_crosscheck": "five previously saved original Q10 embeddings; no new image reads or encoding",
        "image_encoder": read_json(SOURCE / "protocol.json")["image_encoder"],
        "image_pixels_read": 0, "image_encoder_calls": 0,
        "original_vectors_persisted": False, "reconstructed_vectors_persisted": False,
        "memory_state_modified": False,
        "source_sha256": {str(path): sha(path) for path in sources},
    })
    validation = {
        "passed": True, "n_images_scored": len(results), "n_masters": len(masters),
        "complete_member_cover": sorted(m for c in masters.values() for m in c["member_ids"]) == sorted(ids),
        "max_master_residual_pairwise_inner_discrepancy": residual_inner_error,
        "max_reconstruction_norm_squared_discrepancy": max(norm_discrepancies),
        "max_reconstruction_projection_discrepancy": max(projection_discrepancies),
        "direct_original_vector_crosschecks": len(direct_checks),
        "direct_cosine_max_abs_difference": max(r["cosine_abs_difference"] for r in direct_checks),
        "direct_l2_max_abs_difference": max(r["l2_abs_difference"] for r in direct_checks),
        "all_cosines_finite_and_in_range": bool(np.isfinite(cosines).all() and np.all(np.abs(cosines) <= 1)),
        "image_pixels_read": 0, "image_encoder_calls": 0, "memory_writes": 0,
        "per_image_vector_output_files": 0,
    }
    assert validation["complete_member_cover"]
    save_json(out / "validation.json", validation)
    print(json.dumps({
        "complete": True, "output": str(out),
        "cosine": summary_result["cosine_original_vs_reconstructed"],
        "mean_master_only_cosine": float(centers.mean()),
        "q10_pair": pair, "validation": validation,
    }), flush=True)


if __name__ == "__main__":
    main()
