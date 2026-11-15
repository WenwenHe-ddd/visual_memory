#!/usr/bin/env python3
"""Pair-aligned PCA overlays and recovery of stored caption partitions.

One similarity transform per domain is fitted to paired image IDs in PCA-2D.
Separate Ward clustering uses reconstructed 512-D vectors, never aligned 2-D.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/memeye_overlay_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scripts.analysis.plot_memeye_reconstructed_visual_pca import (
    DEFAULT_SOURCE, read_json, sha, unit, write_csv, write_json, save_figure,
)
from scripts.graph.caption_ward_visual import choose_caption_partition

CAPTION_REPORT = ROOT / "reports/memeye_caption_embedding_pca_20261104"


def similarity_align(target, source):
    """Fit source @ (scale*rotation) + translation; no per-cluster adjustment."""
    target_mean, source_mean = target.mean(0), source.mean(0)
    a, b = source - source_mean, target - target_mean
    rotation, nuclear_sum = orthogonal_procrustes(a, b)
    scale = float(nuclear_sum / np.sum(a*a))
    translation = target_mean - scale * source_mean @ rotation
    fitted = scale * source @ rotation + translation
    sse = float(np.sum((target - fitted)**2))
    norm_sse = sse / float(np.sum(b*b))
    _, _, reference_disparity = procrustes(target, source)
    assert abs(norm_sse - reference_disparity) < 1e-12
    distance_error = float(np.max(np.abs(pdist(fitted) - scale*pdist(source))))
    assert distance_error < 1e-12
    return fitted, {
        "rotation_or_reflection": rotation.tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "isotropic_scale": scale, "translation": translation.tolist(),
        "fitted_on": "all paired image IDs within this domain; descriptive in-sample alignment",
        "normalized_sse": norm_sse, "normalized_rmse": float(np.sqrt(norm_sse)),
        "distance_preservation_max_abs_error_after_scale": distance_error,
        "n_fitted_pairs": len(target),
    }


def labels_from_partition(partition, n):
    labels = np.full(n, -1, dtype=int)
    for label, group in enumerate(partition):
        assert np.all(labels[group] == -1)
        labels[group] = label
    assert np.all(labels >= 0)
    return labels


def cut_exact_k(tree, n, k):
    active = {i: [i] for i in range(n)}
    for j, row in enumerate(tree):
        if len(active) == k:
            break
        left, right = int(row[0]), int(row[1])
        active[n+j] = sorted(active.pop(left) + active.pop(right))
    assert len(active) == k
    return sorted(active.values(), key=lambda group: group[0])


def agreement(truth, predicted):
    return {
        "adjusted_rand_index": float(adjusted_rand_score(truth, predicted)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(truth, predicted, average_method="arithmetic")),
        "identical_partition": bool(np.array_equal(
            truth[:, None] == truth[None, :], predicted[:, None] == predicted[None, :])),
    }


def analyze_domain(record, directory, tracked):
    graph_path = directory / "memory/graph.json"
    basis_path, sums_path = directory / "memory/basis.npy", directory / "memory/master_sums.npy"
    for p in (graph_path, basis_path, sums_path):
        tracked[str(p)] = sha(p)
    graph = read_json(graph_path)
    basis = np.load(basis_path, allow_pickle=False)
    sums = np.load(sums_path, allow_pickle=False)
    nodes, masters = graph["image_nodes"], graph["masters"]
    n, k = len(nodes), len(masters)
    assert [node["id"] for node in nodes] == [p["image_id"] for p in record["points"]]
    coords = np.array([node["visual_coords"] for node in nodes])
    index = {node["id"]: i for i, node in enumerate(nodes)}
    original_partition = [[index[key] for key in m["member_ids"]] for m in masters]
    caption_labels = labels_from_partition(original_partition, n)
    rebuilt = np.empty((n, 512))
    projection_error, aggregate_error = 0., 0.
    for master, idx in zip(masters, original_partition):
        total = sums[master["visual_sum_ref"]["row"]]
        mean = total / master["n"]
        rebuilt[idx] = mean + (coords[idx] - mean @ basis) @ basis.T
        projection_error = max(projection_error, float(np.max(np.abs(rebuilt[idx] @ basis - coords[idx]))))
        aggregate_error = max(aggregate_error, float(np.max(np.abs(rebuilt[idx].sum(0) - total))))
    assert max(projection_error, aggregate_error) < 1e-10
    # Normalize exactly as the original caption Ward implementation does.
    candidate = choose_caption_partition(unit(rebuilt))
    visual_labels = labels_from_partition(candidate["partition"], n)
    fixed_labels = labels_from_partition(cut_exact_k(candidate["linkage"], n, k), n)
    auto_metrics, fixed_metrics = agreement(caption_labels, visual_labels), agreement(caption_labels, fixed_labels)

    text_xy = np.array([point["text"] for point in record["points"]])
    visual_xy = np.array([point["visual"] for point in record["points"]])
    aligned, transform = similarity_align(text_xy, visual_xy)
    for point, v, text_label, visual_label, fixed_label in zip(
        record["points"], aligned, caption_labels, visual_labels, fixed_labels,
    ):
        point["visual_aligned"] = v.tolist()
        point["caption_cluster_index"] = int(text_label)
        point["visual_autok_cluster_index"] = int(visual_label)
        point["visual_fixedk_cluster_index"] = int(fixed_label)
    record["alignment"] = transform
    record["clustering_comparison"] = {
        "n_images": n, "caption_k": k, "visual_selected_k": candidate["selected_k"],
        "visual_selected_silhouette": candidate["selected_score"],
        "visual_auto_k": auto_metrics, "visual_caption_k_control": fixed_metrics,
        "visual_candidate_scores": candidate["candidate_scores"],
        "normalization": "L2 reconstructed leaves before Euclidean Ward, as in caption partition selection",
    }
    record["reconstruction_validation"] = {
        "coordinate_max_abs": projection_error, "cluster_aggregate_max_abs": aggregate_error,
    }
    return record


def draw_overlay(ax, record, individual=False):
    text = np.array([p["text"] for p in record["points"]])
    visual = np.array([p["visual_aligned"] for p in record["points"]])
    colors = [p["color"] for p in record["points"]]
    segments = np.stack([text, visual], axis=1)
    ax.add_collection(LineCollection(segments, colors="#666666", alpha=.19, linewidths=.65, zorder=1))
    ax.scatter(*text.T, c=colors, marker="o", s=43 if individual else 26,
               alpha=.8, edgecolors="white", linewidths=.45, zorder=3)
    ax.scatter(*visual.T, facecolors="none", edgecolors=colors, marker="^",
               s=58 if individual else 40, alpha=.92, linewidths=1.1, zorder=4)
    c = record["clustering_comparison"]
    ax.set_title(
        f'{record["short_domain"]} | Caption K={c["caption_k"]}, visual K={c["visual_selected_k"]}\n'
        f'ARI={c["visual_auto_k"]["adjusted_rand_index"]:.3f} | '
        f'PCA alignment nRMSE={record["alignment"]["normalized_rmse"]:.3f}',
        fontsize=12 if individual else 10.5, pad=9)
    ax.set_xlabel("Caption PC1 reference axis")
    ax.set_ylabel("Caption PC2 reference axis")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(.1)
    ax.grid(alpha=.15, linewidth=.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend_handles():
    return [
        Line2D([], [], marker="o", color="#333333", linestyle="", label="Caption PCA"),
        Line2D([], [], marker="^", markerfacecolor="none", color="#333333",
               linestyle="", label="Visual PCA after global similarity alignment"),
        Line2D([], [], color="#999999", linewidth=.7, label="Same image ID"),
    ]


def render(records, output):
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.7))
    for ax, record in zip(axes.flat, records):
        draw_overlay(ax, record)
    fig.suptitle("MemEye | Caption and reconstructed visual PCA overlays", fontsize=19, y=.99)
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(.5, .955),
               ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .025,
             "One rotation/reflection + uniform scale + translation per domain, fitted to paired image IDs. Color: existing caption cluster.\n"
             "ARI compares caption groups with Ward clusters recomputed from normalized 512-D reconstructed visuals; it is not calculated from the overlay.",
             ha="center", fontsize=9.5, color="#444444")
    fig.subplots_adjust(left=.045, right=.985, top=.85, bottom=.11, hspace=.48, wspace=.31)
    save_figure(fig, output / "pca_overlay_by_domain")
    for record in records:
        folder = output / record["domain"]
        folder.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 8))
        draw_overlay(ax, record, individual=True)
        ax.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(.5, -.12),
                  frameon=False, fontsize=9)
        fig.subplots_adjust(bottom=.23, top=.89)
        fig.text(.5, .025,
                 "A single similarity transform preserves the visual PCA shape up to scale.\n"
                 "Gray lines show residual differences between matched image positions.",
                 ha="center", fontsize=9, color="#444444")
        save_figure(fig, folder / "pca_overlay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-report", type=Path, default=CAPTION_REPORT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/memeye_text_visual_overlay_20261104")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Use a fresh --output directory.")
    plot_path = args.caption_report / "plot_data.json"
    tracked = {str(plot_path.resolve()): sha(plot_path)}
    manifest_path = args.source_root / "manifest.json"
    tracked[str(manifest_path.resolve())] = sha(manifest_path)
    expected = {r["domain"]: r for r in read_json(manifest_path)["domains"]}
    records = read_json(plot_path)["domains"]
    rows, memberships, xy_rows = [], [], []
    for record in records:
        directory = args.source_root / record["domain"]
        for name in ("graph.json", "basis.npy", "master_sums.npy"):
            assert sha(directory / "memory" / name) == expected[record["domain"]]["sha256"]["memory/" + name]
        analyze_domain(record, directory, tracked)
        comp, align = record["clustering_comparison"], record["alignment"]
        rows.append({
            "domain": record["domain"], "n_images": comp["n_images"], "caption_k": comp["caption_k"],
            "visual_selected_k": comp["visual_selected_k"],
            "visual_autok_ari": comp["visual_auto_k"]["adjusted_rand_index"],
            "visual_autok_nmi": comp["visual_auto_k"]["normalized_mutual_information"],
            "visual_autok_identical_partition": comp["visual_auto_k"]["identical_partition"],
            "visual_fixedk_ari": comp["visual_caption_k_control"]["adjusted_rand_index"],
            "visual_fixedk_nmi": comp["visual_caption_k_control"]["normalized_mutual_information"],
            "alignment_normalized_rmse": align["normalized_rmse"],
            "alignment_isotropic_scale": align["isotropic_scale"],
            "caption_pca2_explained_ratio": record["metrics"]["text"]["pca2_explained_ratio"],
            "visual_raw_pca2_explained_ratio": record["metrics"]["visual"]["pca2_explained_ratio"],
        })
        for point in record["points"]:
            memberships.append({
                "domain": record["domain"], "image_id": point["image_id"], "filename": point["filename"],
                "stored_master_id": point["master_id"],
                "caption_cluster_index": point["caption_cluster_index"],
                "visual_autok_cluster_index": point["visual_autok_cluster_index"],
                "visual_fixedk_cluster_index": point["visual_fixedk_cluster_index"],
            })
            xy_rows.append({
                "domain": record["domain"], "image_id": point["image_id"],
                "caption_pc1": point["text"][0], "caption_pc2": point["text"][1],
                "visual_raw_pc1": point["visual"][0], "visual_raw_pc2": point["visual"][1],
                "visual_aligned_x": point["visual_aligned"][0], "visual_aligned_y": point["visual_aligned"][1],
            })
        print(f'{record["short_domain"]}: K {comp["caption_k"]}->{comp["visual_selected_k"]}, '
              f'ARI={rows[-1]["visual_autok_ari"]:.6f}, nRMSE={align["normalized_rmse"]:.3f}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for filename, data in (("metrics.csv", rows), ("memberships.csv", memberships), ("aligned_points.csv", xy_rows)):
        write_csv(out / filename, data)
    write_json(out / "plot_data.json", {"domains": records})
    render(records, out)
    assert all(sha(p) == digest for p, digest in tracked.items())
    summary = {
        "completed": True, "n_domains": len(records), "n_images": len(memberships),
        "caption_masters": sum(r["caption_k"] for r in rows),
        "visual_autok_masters": sum(r["visual_selected_k"] for r in rows),
        "macro_visual_autok_ari": float(np.mean([r["visual_autok_ari"] for r in rows])),
        "macro_visual_autok_nmi": float(np.mean([r["visual_autok_nmi"] for r in rows])),
        "macro_visual_fixedk_ari": float(np.mean([r["visual_fixedk_ari"] for r in rows])),
        "macro_visual_fixedk_nmi": float(np.mean([r["visual_fixedk_nmi"] for r in rows])),
        "n_exactly_recovered_domains": sum(r["visual_autok_identical_partition"] for r in rows),
        "domains": rows,
    }
    write_json(out / "summary.json", summary)
    write_json(out / "protocol.json", {
        "source_sha256": tracked, "script": str(Path(__file__).resolve()), "script_sha256": sha(__file__),
        "alignment": "Paired-image-ID least-squares similarity alignment of raw reconstructed visual PCA-2D to caption PCA-2D",
        "alignment_scope": "One transform per domain; reflection allowed; uniform scale; no per-cluster or per-point warp",
        "alignment_labels_used": False, "alignment_fit_and_display_same_points": True,
        "clustering": "L2-normalized reconstructed full 512-D vectors; original Ward + max silhouette rule; K independently selected",
        "same_k_control": "Cut the same visual Ward tree at caption K; reported separately from automatic K",
        "reference_partition": "stored caption clusters, verified in previous caption-only re-encoding run",
        "nmi_average_method": "arithmetic", "aggregation": "unweighted macro average over 8 domains",
        "interpretation": "Recovery of caption-defined grouping in compressed visual memory; reconstruction already conditions on caption clusters",
        "original_images_read": 0, "encoder_calls": 0, "api_calls": 0,
        "persistent_memory_changed": False, "source_files_unchanged": True,
        "original_or_reconstructed_full_visual_embeddings_saved": False,
        "alignment_reference": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html",
    })
    print(f"Complete: {out}; macro ARI={summary['macro_visual_autok_ari']:.6f}", flush=True)


if __name__ == "__main__":
    main()
