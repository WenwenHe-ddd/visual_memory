#!/usr/bin/env python3
"""Controlled 2-D versus 4-D visual-state replay of fixed caption partitions.

Only the arriving image is encoded in the Home stream. The two arms share that
transient vector and retain only leaf coordinates and current master sums.
Historical Gram matrices are used by the evaluator, never by memory updates.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/caption_dimension_matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from scripts.analysis.run_caption_ward_visual_pilot import (
    DOMAIN, GRAM_SOURCE, GramEvaluator, ImageEncoder, jsonl,
    load_evaluation_gram, read_json, sha, unit, write_json,
)

ORIGINAL = ROOT / "reports/home_caption_ward_visual_20261020"
DEFAULT_OUTPUT = ROOT / "reports/home_caption_visual_2d4d_20261020"


def calibrate_four(encoder, calibration):
    """Streaming sufficient statistics; no saved individual calibration vectors."""
    started = time.perf_counter()
    total = np.zeros(encoder.embedding_dim)
    second = np.zeros((encoder.embedding_dim, encoder.embedding_dim))
    for index, row in enumerate(calibration, 1):
        assert sha(row["source_path"]) == row["image_sha256"]
        visual = unit(encoder.encode(row["source_path"]))
        total += visual
        second += np.outer(visual, visual)
        del visual
        if index % 50 == 0 or index == len(calibration):
            print(json.dumps({"stage": "calibration", "done": index}), flush=True)
    covariance = (second - np.outer(total, total) / len(calibration)) / (len(calibration) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    basis = eigenvectors[:, -4:][:, ::-1].copy()
    for col in range(4):
        if basis[np.argmax(np.abs(basis[:, col])), col] < 0:
            basis[:, col] *= -1
    original_basis = np.load(ORIGINAL / "memory/basis.npy")
    original_eigenvalues = np.asarray(read_json(ORIGINAL / "calibration.json")["eigenvalues"])
    basis_delta = float(np.max(np.abs(basis[:, :2] - original_basis)))
    spectrum_delta = float(np.max(np.abs(eigenvalues - original_eigenvalues)))
    assert basis_delta < 1e-10, basis_delta
    assert spectrum_delta < 1e-12, spectrum_delta
    assert np.allclose(basis.T @ basis, np.eye(4), atol=1e-12, rtol=0)
    return basis, {
        "n": len(calibration), "dimension": encoder.embedding_dim,
        "eigenvalues": eigenvalues.tolist(),
        "top2_centered_calibration_variance_ratio": float(eigenvalues[:2].sum() / eigenvalues.sum()),
        "top4_centered_calibration_variance_ratio": float(eigenvalues[:4].sum() / eigenvalues.sum()),
        "top2_basis_max_abs_difference_from_original": basis_delta,
        "spectrum_max_abs_difference_from_original": spectrum_delta,
        "fit": "centered non-Home covariance from running sum and second moment",
        "projection": "z = U.T @ v; absolute projection, calibration mean not subtracted",
        "individual_calibration_vectors_retained": False,
        "seconds": time.perf_counter() - started,
    }


class PartitionReplay:
    """Dimension-generic aggregate update; receives only the current partition."""

    def __init__(self, basis):
        self.basis = basis.copy()
        self.image_nodes = {}
        self.masters = {}

    def add(self, row, visual, audit):
        assert row["image_id"] == audit["image_id"]
        incoming = row["image_id"]
        old_owners = {member: mid for mid, master in self.masters.items()
                      for member in master["member_ids"]}
        assert set(old_owners) == set(self.image_nodes)
        partition = [c["member_ids"] for c in audit["clusters"]]
        flat = [member for members in partition for member in members]
        assert len(flat) == len(set(flat)) == audit["n_images"]
        assert set(flat) == set(self.image_nodes) | {incoming}
        prepared = {}
        for mid, master in self.masters.items():
            coords = np.sum([self.image_nodes[m]["visual_coords"] for m in master["member_ids"]], axis=0)
            prepared[mid] = master["visual_sum"] - self.basis @ coords
        updated = {}
        computed_overlaps = []
        for cluster in audit["clusters"]:
            mid, members = cluster["master_id"], cluster["member_ids"]
            sources = {}
            for member in members:
                if member in old_owners:
                    sources.setdefault(old_owners[member], []).append(member)
            if mid in self.masters:
                assert self.masters[mid]["member_ids"] == members
                master = copy.deepcopy(self.masters[mid])
                assert np.array_equal(master["visual_sum"], self.masters[mid]["visual_sum"])
            else:
                total = np.zeros(self.basis.shape[0])
                approximate = False
                source_records = []
                for source, selected in sources.items():
                    old = self.masters[source]
                    count = len(selected)
                    fraction = count / old["n"]
                    approximate |= old["approximate"]
                    if count == old["n"]:
                        total += old["visual_sum"]
                    else:
                        total += self.basis @ np.sum(
                            [self.image_nodes[m]["visual_coords"] for m in selected], axis=0)
                        total += fraction * prepared[source]
                        approximate = True
                    source_records.append({"old_master_id": source, "count": count,
                                           "old_n": old["n"], "fraction": fraction})
                if incoming in members:
                    total += visual
                master = {"id": mid, "member_ids": members.copy(), "n": len(members),
                          "visual_sum": total, "approximate": approximate,
                          "source_overlaps": source_records}
            assert master["approximate"] == cluster["approximate"]
            updated[mid] = master
            for source, selected in sources.items():
                computed_overlaps.append({
                    "old_master_id": source, "new_master_id": mid, "count": len(selected),
                    "old_n": self.masters[source]["n"],
                    "fraction": len(selected) / self.masters[source]["n"],
                    "member_ids": selected,
                })
        assert computed_overlaps == audit["source_overlaps"]
        self.image_nodes[incoming] = {
            "id": incoming, "caption": row["caption"], "dialogue": row["dialogue"],
            "timestamp": row["timestamp"], "visual_coords": (self.basis.T @ visual).tolist(),
        }
        self.masters = updated
        for mid, master in updated.items():
            for member in master["member_ids"]:
                self.image_nodes[member]["master_id"] = mid

    def save(self, directory):
        directory.mkdir()
        np.save(directory / "basis.npy", self.basis)
        np.save(directory / "master_sums.npy", np.stack([m["visual_sum"] for m in self.masters.values()]))
        masters = []
        for index, master in enumerate(self.masters.values()):
            item = {k: copy.deepcopy(v) for k, v in master.items() if k != "visual_sum"}
            item["visual_sum_ref"] = {"path": "master_sums.npy", "row": index}
            item["visual_mean_ref"] = {"path": "master_sums.npy", "row": index, "scale": 1 / master["n"]}
            masters.append(item)
        write_json(directory / "graph.json", {
            "method": "caption_partition_replay_visual_dimension_comparison",
            "schema_version": 1, "visual_dim": self.basis.shape[0],
            "leaf_visual_dim": self.basis.shape[1], "text_dim": 384,
            "basis_ref": {"path": "basis.npy", "shape": list(self.basis.shape)},
            "masters": masters, "image_nodes": list(self.image_nodes.values()),
            "text_embedding_cache": {"persisted": False, "rebuild_from": "image_nodes[].caption"},
        })


def centered_variance_retention(gram, coords, groups):
    full, projected = 0.0, 0.0
    for group in groups:
        indices = np.array(group)
        g = gram[np.ix_(indices, indices)]
        z = coords[indices]
        full += float(np.trace(g) - g.sum() / len(group))
        projected += float(np.sum(z*z) - np.sum(z.sum(axis=0)**2) / len(group))
    return {"full_centered_sum_squares": full, "projected_centered_sum_squares": projected,
            "retained_fraction": projected / full if full else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Use a fresh --output directory; completed experiments are never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = read_json(ORIGINAL / "input_manifest.json")
    rows, calibration = manifest["home"], manifest["calibration"]
    original_audits = jsonl(ORIGINAL / "insertion_audit.jsonl")
    original_protocol = read_json(ORIGINAL / "protocol.json")
    assert len(rows) == len(original_audits) == 89
    assert len(calibration) == 344
    assert not {r["image_sha256"] for r in rows} & {r["image_sha256"] for r in calibration}
    assert [r["timestamp"] for r in rows] == sorted(r["timestamp"] for r in rows)
    gram = load_evaluation_gram(rows)
    import torch
    torch.set_num_threads(4)
    encoder = ImageEncoder(str(ROOT / "models/clip-vit-base-patch32"),
                           revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                           device="cpu", local_files_only=True)
    assert encoder.metadata == original_protocol["image_encoder"]
    basis, calibration_info = calibrate_four(encoder, calibration)
    memories = {rank: PartitionReplay(basis[:, :rank]) for rank in (2, 4)}
    evaluators = {rank: GramEvaluator(rows, gram) for rank in memories}
    global_sum = np.zeros(encoder.embedding_dim)
    audits = []
    checks = {
        "max_global_sum_l2_error": 0.0, "max_projection_l2_error": 0.0,
        "max_new_vector_gram_row_sum_error": 0.0,
        "max_baseline_2d_l2_difference": 0.0,
        "max_4d_minus_2d_l2_error": 0.0,
        "max_squared_error_improvement_identity_difference": 0.0,
        "max_oracle_norm_sq_discrepancy": 0.0, "cluster_comparisons": 0,
    }
    for index, (row, original_audit) in enumerate(zip(rows, original_audits)):
        assert sha(row["source_path"]) == row["image_sha256"]
        image_started = time.perf_counter()
        visual = unit(encoder.encode(row["source_path"]))
        image_seconds = time.perf_counter() - image_started
        gram_error = abs(float(visual @ global_sum) - float(gram[index, :index].sum()))
        checks["max_new_vector_gram_row_sum_error"] = max(
            checks["max_new_vector_gram_row_sum_error"], gram_error)
        assert gram_error < 1e-8
        global_sum += visual
        arms = {}
        for rank, memory in memories.items():
            update_started = time.perf_counter()
            memory.add(row, visual, original_audit)
            update_seconds = time.perf_counter() - update_started
            global_error = float(np.linalg.norm(
                sum((m["visual_sum"] for m in memory.masters.values()), np.zeros_like(global_sum)) - global_sum))
            projection_error = max(float(np.linalg.norm(
                memory.basis.T @ m["visual_sum"] - np.sum(
                    [memory.image_nodes[x]["visual_coords"] for x in m["member_ids"]], axis=0)))
                for m in memory.masters.values())
            assert global_error < 1e-8 and projection_error < 1e-8
            checks["max_global_sum_l2_error"] = max(checks["max_global_sum_l2_error"], global_error)
            checks["max_projection_l2_error"] = max(checks["max_projection_l2_error"], projection_error)
            error = evaluators[rank].measure(memory, original_audit)
            checks["max_oracle_norm_sq_discrepancy"] = max(
                checks["max_oracle_norm_sq_discrepancy"],
                max(m["oracle_norm_sq_discrepancy"] for m in error["clusters"]))
            arms[str(rank)] = {"update_seconds": update_seconds, "visual_center_error": error,
                               "global_sum_l2_error": global_error, "projection_l2_error": projection_error}
        # The full vector is no longer available after this insertion.
        del visual
        original_errors = {m["master_id"]: m for m in original_audit["visual_center_error"]["clusters"]}
        errors_2 = {m["master_id"]: m for m in arms["2"]["visual_center_error"]["clusters"]}
        errors_4 = {m["master_id"]: m for m in arms["4"]["visual_center_error"]["clusters"]}
        z_extra = np.array([n["visual_coords"][2:] for n in memories[4].image_nodes.values()])
        ids = {r["image_id"]: j for j, r in enumerate(rows[:index+1])}
        for mid, error_2 in errors_2.items():
            error_4 = errors_4[mid]
            e2, e4 = error_2["mean_l2_error"], error_4["mean_l2_error"]
            baseline_delta = abs(e2 - original_errors[mid]["mean_l2_error"])
            checks["max_baseline_2d_l2_difference"] = max(checks["max_baseline_2d_l2_difference"], baseline_delta)
            assert baseline_delta < 1e-7, (index, mid, baseline_delta)
            assert e4 <= e2 + 1e-7, (index, mid, e2, e4)
            checks["max_4d_minus_2d_l2_error"] = max(checks["max_4d_minus_2d_l2_error"], e4-e2)
            a = evaluators[4].coefficients[mid][:index+1]
            b = np.zeros(index+1)
            b[[ids[x] for x in memories[4].masters[mid]["member_ids"]]] = 1
            expected_improvement = float(np.sum(((a-b) @ z_extra)**2)) / error_4["n"]**2
            identity_delta = abs(e2**2-e4**2-expected_improvement)
            assert identity_delta < 1e-10
            checks["max_squared_error_improvement_identity_difference"] = max(
                checks["max_squared_error_improvement_identity_difference"], identity_delta)
            checks["cluster_comparisons"] += 1
        audits.append({"step": index+1, "image_id": row["image_id"],
                       "n_clusters": len(memories[2].masters),
                       "split_master_ids": original_audit["split_master_ids"],
                       "image_encode_seconds": image_seconds, "arms": arms})
        if (index+1) % 10 == 0 or index+1 == len(rows):
            print(json.dumps({"stage": "home", "done": index+1, "weighted_l2":
                              {r: arms[r]["visual_center_error"]["member_weighted_mean_l2_error"]
                               for r in arms}}), flush=True)
    old_graph = read_json(ORIGINAL / "memory/graph.json")
    old_sums = np.load(ORIGINAL / "memory/master_sums.npy")
    old_nodes = {n["id"]: n for n in old_graph["image_nodes"]}
    checks["max_final_2d_coords_difference"] = max(
        float(np.max(np.abs(np.array(n["visual_coords"])-old_nodes[key]["visual_coords"])))
        for key, n in memories[2].image_nodes.items())
    checks["max_final_2d_master_sum_difference"] = max(
        float(np.max(np.abs(memories[2].masters[m["id"]]["visual_sum"]-old_sums[m["visual_sum_ref"]["row"]])))
        for m in old_graph["masters"])
    assert checks["max_final_2d_coords_difference"] < 1e-10
    assert checks["max_final_2d_master_sum_difference"] < 1e-10
    memories[4].save(output / "memory")
    indices = {r["image_id"]: i for i, r in enumerate(rows)}
    final_groups = [[indices[m] for m in c["member_ids"]] for c in memories[4].masters.values()]
    filename_indices = {r["filename"]: i for i, r in enumerate(rows)}
    pair_indices = [filename_indices["D7_IMG_005.jpg"], filename_indices["D7_IMG_006.jpg"]]
    paired = {}
    for rank, memory in memories.items():
        coords = np.array([n["visual_coords"] for n in memory.image_nodes.values()])
        l2 = audits[-1]["arms"][str(rank)]["visual_center_error"]
        pair = centered_variance_retention(gram, coords, [pair_indices])
        visual_bytes = (memory.basis.size + len(memory.masters)*encoder.embedding_dim + coords.size)*8
        paired[str(rank)] = {
            "final_visual_center_error": l2,
            "global_home_centered_variance": centered_variance_retention(gram, coords, [list(range(len(rows)))]),
            "pooled_final_cluster_within_variance": centered_variance_retention(gram, coords, final_groups),
            "q10_D7_IMG_005_vs_D7_IMG_006_squared_distance_retention": pair["retained_fraction"],
            "storage": {"dtype": "float64", "shared_basis_bytes": memory.basis.nbytes,
                        "master_visual_sums_bytes": len(memory.masters)*encoder.embedding_dim*8,
                        "leaf_coords_numeric_bytes": coords.nbytes, "visual_numeric_payload_bytes": visual_bytes},
            "update_seconds": {
                "mean": float(np.mean([a["arms"][str(rank)]["update_seconds"] for a in audits])),
                "total": float(sum(a["arms"][str(rank)]["update_seconds"] for a in audits)),
            },
        }
    centered_gram = gram - gram.mean(axis=0)[None, :] - gram.mean(axis=1)[:, None] + gram.mean()
    home_eigenvalues = np.maximum(np.linalg.eigvalsh(centered_gram)[::-1], 0)
    full_home_oracle = {
        "evaluation_only": True, "uses_all_home_images_including_future": True,
        "used_by_stream_algorithm": False,
        "meaning": "posthoc optimum fraction of global centered variance retained by a single fixed linear PCA basis",
        "positive_rank_tolerance_1e_10": int(np.sum(home_eigenvalues > 1e-10)),
        "top2_centered_variance_fraction": float(home_eigenvalues[:2].sum()/home_eigenvalues.sum()),
        "top4_centered_variance_fraction": float(home_eigenvalues[:4].sum()/home_eigenvalues.sum()),
    }
    e2 = paired["2"]["final_visual_center_error"]["member_weighted_mean_l2_error"]
    e4 = paired["4"]["final_visual_center_error"]["member_weighted_mean_l2_error"]
    results = {
        "n_home_images": len(rows), "n_final_clusters": len(final_groups),
        "caption_partitions_recomputed": False,
        "partition_control": "identical original prefix-only caption memberships at every insertion",
        "arms": paired,
        "final_weighted_mean_l2_absolute_reduction": e2-e4,
        "final_weighted_mean_l2_relative_reduction": (e2-e4)/e2,
        "storage_extra_bytes_4d_vs_2d": paired["4"]["storage"]["visual_numeric_payload_bytes"]-
                                       paired["2"]["storage"]["visual_numeric_payload_bytes"],
        "global_home_posthoc_best_linear_pca": full_home_oracle,
        "qa_accuracy_evaluated": False,
        "original_2d_core_or_results_modified": False,
    }
    write_json(output / "results.json", results)
    write_json(output / "calibration.json", calibration_info)
    with (output / "insertion_audit.jsonl").open("w") as handle:
        for row in audits:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    with (output / "insertion_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "image_id", "n_clusters",
                                "weighted_l2_2d", "weighted_l2_4d", "max_l2_2d", "max_l2_4d"])
        writer.writeheader()
        for row in audits:
            writer.writerow({"step": row["step"], "image_id": row["image_id"], "n_clusters": row["n_clusters"],
                **{f"weighted_l2_{r}d": row["arms"][str(r)]["visual_center_error"]["member_weighted_mean_l2_error"] for r in (2,4)},
                **{f"max_l2_{r}d": row["arms"][str(r)]["visual_center_error"]["max_mean_l2_error"] for r in (2,4)}})
    with (output / "final_cluster_errors.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["master_id", "n", "approximate", "l2_2d", "l2_4d",
                                                   "cosine_2d", "cosine_4d"])
        writer.writeheader()
        two, four = [paired[str(r)]["final_visual_center_error"]["clusters"] for r in (2,4)]
        for a, b in zip(two, four):
            assert a["master_id"] == b["master_id"]
            writer.writerow({"master_id": a["master_id"], "n": a["n"], "approximate": a["approximate"],
                             "l2_2d": a["mean_l2_error"], "l2_4d": b["mean_l2_error"],
                             "cosine_2d": a["cosine_to_exact_mean"], "cosine_4d": b["cosine_to_exact_mean"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for rank, color in ((2, "#d36b37"), (4, "#237ab3")):
        ax.plot([a["step"] for a in audits],
                [a["arms"][str(rank)]["visual_center_error"]["member_weighted_mean_l2_error"] for a in audits],
                label=f"{rank}D fixed external PCA", color=color, linewidth=2)
    ax.set(xlabel="Images inserted", ylabel="Member-weighted mean centroid L2 error",
           title="Home: identical caption partitions, two versus four visual coordinates")
    ax.grid(alpha=0.2); ax.legend()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"center_error_2d_vs_4d.{suffix}", dpi=180)
    plt.close(fig)
    source_paths = [Path(__file__), ORIGINAL/"input_manifest.json", ORIGINAL/"insertion_audit.jsonl",
                    ORIGINAL/"protocol.json", ORIGINAL/"memory/graph.json", ORIGINAL/"memory/basis.npy",
                    ORIGINAL/"memory/master_sums.npy", GRAM_SOURCE,
                    ROOT/"scripts/analysis/run_caption_ward_visual_pilot.py",
                    ROOT/"scripts/graph/visual_image_encoder.py"]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "domain": DOMAIN,
        "controlled_change": "leaf coordinate and fixed basis dimension 2 -> 4",
        "partition_replay": "original prefix partitions reused verbatim, no caption encoder or clustering rerun",
        "visual_update": "same full-cluster sums and proportional omitted-residual redistribution in both arms",
        "calibration": "344 unique non-Home images, zero SHA overlap with Home; centered covariance PCA",
        "chronology": "89 original chronological arrivals; one current image encoded once and shared by two arms",
        "old_home_image_reencodes_during_stream": 0,
        "previous_experiment_images": "This is a fresh empty-memory replay from original source corpus, not an in-place conversion of the saved 2-D state.",
        "image_encoder": encoder.metadata, "torch_threads": torch.get_num_threads(),
        "oracle": "historical full-domain Gram and residual mixing coefficients, evaluator only",
        "memory_policy": "only final 4D leaf coordinates, final master full sums and shared fixed basis persisted; no image copies or per-image full embeddings",
        "source_sha256": {str(p): sha(p) for p in source_paths},
        "limitations": [
            "Four PCs explain all centered variation in five observations by rank, not all domain variation.",
            "External-domain top4 directions may retain little Home variation.",
            "Nested bases guarantee no larger centroid L2 error under fixed replay but do not guarantee cosine or QA improvement.",
            "No new coordinates of previously discarded images can be recovered exactly from a 2-D saved state.",
            "Posthoc Home PCA spectrum is evaluation-only and uses future images; it is not the stream basis.",
        ],
    }
    write_json(output / "protocol.json", protocol)
    validation = {
        "passed": True, **checks, "home_inputs": len(rows), "calibration_inputs": len(calibration),
        "calibration_home_sha_overlap": 0, "ordered_insertions": len(audits),
        "incoming_home_encodes": len(rows), "old_home_reencodes_during_stream": 0,
        "encoder_metadata_exactly_matches_original": True,
        "both_arms_same_partitions_every_step": True,
        "nested_basis_error_monotonicity_all_clusters_all_steps": True,
        "leaf_visual_dimensions": [4],
        "memory_arrays": {"basis.npy": list(basis.shape),
                          "master_sums.npy": [len(memories[4].masters), encoder.embedding_dim]},
        "no_source_paths_in_memory_leaves": all("source_path" not in n and "image_path" not in n
                                                for n in memories[4].image_nodes.values()),
        "copied_original_images": 0, "retained_individual_full_visual_embeddings": 0,
        "wall_seconds": time.perf_counter()-started,
    }
    write_json(output / "validation.json", validation)
    print(json.dumps({"complete": True, "output": str(output), "weighted_l2_2d": e2,
                      "weighted_l2_4d": e4, "relative_reduction": (e2-e4)/e2,
                      "checks": validation}), flush=True)


if __name__ == "__main__":
    main()
