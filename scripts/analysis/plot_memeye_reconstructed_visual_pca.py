#!/usr/bin/env python3
"""Plot existing caption_visual_master partitions from saved visual state only.

No images, original image embeddings, encoders, API calls, or reclustering.
Reconstructed 512-D vectors remain transient; only plot coordinates are saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/memeye_reconstructed_pca_matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain"
NAMES = {
    "Brand_Memory_Test": "Brand",
    "Card_Playlog_Test": "Card",
    "Cartoon_Entertainment_Companion": "Cartoon",
    "Home_Renovation_Interior_Design": "Home",
    "Multi-Scene_Visual_Case_Archive_Assistant": "Multi-Scene",
    "Outdoor_Navigation_Route_Memory_Assistant": "Outdoor",
    "Personal_Health_Dashboard_Assistant": "Health",
    "Social_Chat_Memory_Test": "Social",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unit(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.all(norms > 0)
    return x / norms


def pca(x, anchors, labels):
    """Fit centered, unwhitened PCA to leaves; transform anchors in those axes."""
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    for row in vt:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    ratio = s**2 / np.sum(s**2)
    scores = (x - mean) @ vt[:2].T
    anchor_scores = (anchors - mean) @ vt[:2].T
    n_labels = len(np.unique(labels))
    valid = 1 < n_labels < len(x)
    metrics = {
        "pca2_explained_ratio": float(ratio[:2].sum()),
        "explained_ratio": ratio.tolist(),
        "effective_centered_rank": int(np.count_nonzero(s > 1e-12)),
        "silhouette_512d_euclidean": float(silhouette_score(x, labels)) if valid else None,
        "silhouette_512d_cosine": float(silhouette_score(x, labels, metric="cosine")) if valid else None,
        "silhouette_pca2_euclidean": float(silhouette_score(scores, labels)) if valid else None,
    }
    return scores, anchor_scores, metrics


def reconstruct_domain(directory, manifest_record, tracked):
    state = directory / "memory"
    for name in ("graph.json", "basis.npy", "master_sums.npy"):
        path = state / name
        tracked[str(path)] = sha(path)
        assert tracked[str(path)] == manifest_record["sha256"]["memory/" + name], path
    graph = read_json(state / "graph.json")
    assert graph["method"] == "caption_ward_silhouette_visual"
    assert graph["leaf_visual_dim"] == 2 and graph["visual_dim"] == 512
    assert graph["basis_ref"]["path"] == "basis.npy"
    basis = np.load(state / "basis.npy", allow_pickle=False)
    sums = np.load(state / "master_sums.npy", allow_pickle=False)
    nodes, masters = graph["image_nodes"], graph["masters"]
    ids = [node["id"] for node in nodes]
    assert len(ids) == len(set(ids)) == manifest_record["n_leaves"]
    assert len(masters) == manifest_record["n_masters"]
    assert basis.shape == (512, 2) and sums.shape == (len(masters), 512)
    assert np.isfinite(basis).all() and np.isfinite(sums).all()
    coords = np.array([node["visual_coords"] for node in nodes], dtype=np.float64)
    assert coords.shape == (len(nodes), 2) and np.isfinite(coords).all()
    index = {key: i for i, key in enumerate(ids)}
    labels = np.full(len(nodes), -1, dtype=int)
    rebuilt = np.empty((len(nodes), 512))
    centers = np.empty((len(masters), 512))
    errors = {"basis_orthogonality_max_abs": float(np.max(np.abs(basis.T @ basis - np.eye(2))))}
    projection_errors, aggregate_errors, internal_errors = [], [], []
    for j, master in enumerate(masters):
        positions = [index[key] for key in master["member_ids"]]
        assert len(positions) == len(set(positions)) == master["n"] > 0
        assert np.all(labels[positions] == -1)
        assert all(nodes[i]["master_id"] == master["id"] for i in positions)
        ref, mean_ref = master["visual_sum_ref"], master["visual_mean_ref"]
        assert ref["path"] == mean_ref["path"] == "master_sums.npy"
        assert ref["row"] == mean_ref["row"]
        assert abs(mean_ref["scale"] - 1 / master["n"]) < 1e-14
        total = sums[ref["row"]]
        center = total / master["n"]
        centers[j] = center
        labels[positions] = j
        rebuilt[positions] = center + (coords[positions] - center @ basis) @ basis.T
        projection_errors.append(np.max(np.abs(rebuilt[positions] @ basis - coords[positions])))
        aggregate_errors.append(np.max(np.abs(rebuilt[positions].sum(axis=0) - total)))
        # Same-master differences retain exactly the two stored directions.
        deviations = rebuilt[positions] - rebuilt[positions].mean(axis=0)
        expected = (coords[positions] - coords[positions].mean(axis=0)) @ basis.T
        internal_errors.append(np.max(np.abs(deviations - expected)))
    assert np.all(labels >= 0) and np.isfinite(rebuilt).all()
    errors.update(
        coordinate_max_abs=float(max(projection_errors)),
        cluster_aggregate_max_abs=float(max(aggregate_errors)),
        within_cluster_2d_identity_max_abs=float(max(internal_errors)),
    )
    assert max(errors.values()) < 1e-10, errors

    # Filenames are metadata only. No source image paths are opened.
    metadata_path = directory / "input_manifest.json"
    tracked[str(metadata_path)] = sha(metadata_path)
    assert tracked[str(metadata_path)] == manifest_record["sha256"]["input_manifest.json"]
    arrivals = read_json(metadata_path)["arrivals"]
    filename = {row["image_id"]: row["filename"] for row in arrivals}
    assert set(filename) == set(ids)
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    colors = [to_hex(palette[j % len(palette)]) for j in range(len(masters))]
    record = {
        "domain": directory.name, "short_domain": NAMES.get(directory.name, directory.name),
        "n_images": len(nodes), "n_masters": len(masters),
        "n_singleton_masters": sum(m["n"] == 1 for m in masters),
        "n_approximate_masters": sum(bool(m["approximate"]) for m in masters),
        "reconstruction_norm_min": float(np.linalg.norm(rebuilt, axis=1).min()),
        "reconstruction_norm_mean": float(np.linalg.norm(rebuilt, axis=1).mean()),
        "reconstruction_norm_max": float(np.linalg.norm(rebuilt, axis=1).max()),
        "validation": errors, "metrics": {}, "points": [], "masters": [],
    }
    for i, node in enumerate(nodes):
        j = labels[i]
        record["points"].append({
            "image_id": node["id"], "filename": filename[node["id"]],
            "master_id": node["master_id"], "cluster_label": f"C{j+1:02d}",
            "color": colors[j], "caption": node["caption"], "dialogue": node["dialogue"],
        })
    for j, master in enumerate(masters):
        record["masters"].append({
            "master_id": master["id"], "cluster_label": f"C{j+1:02d}",
            "n": master["n"], "approximate": bool(master["approximate"]), "color": colors[j],
        })
    for mode, vectors, anchors in (
        ("raw", rebuilt, centers), ("normalized", unit(rebuilt), unit(centers)),
    ):
        points, anchor_points, metrics = pca(vectors, anchors, labels)
        record["metrics"][mode] = metrics
        for item, point in zip(record["points"], points):
            item[mode] = point.tolist()
        for item, point in zip(record["masters"], anchor_points):
            item[mode] = point.tolist()
    # Neither full-dimensional originals nor reconstructed leaf vectors are saved.
    return record


def plot_panel(ax, record, mode, individual=False):
    points = np.array([point[mode] for point in record["points"]])
    anchors = np.array([master[mode] for master in record["masters"]])
    colors = [point["color"] for point in record["points"]]
    ax.scatter(*points.T, c=colors, s=48 if individual else 27,
               edgecolors="white", linewidths=0.45, alpha=0.87, zorder=3)
    ax.scatter(*anchors.T, c=[m["color"] for m in record["masters"]],
               marker="X", s=110 if individual else 72,
               edgecolors="#222222", linewidths=0.65, zorder=4)
    metrics = record["metrics"][mode]
    ax.set_title(
        f'{record["short_domain"]}  |  {record["n_images"]} images, {record["n_masters"]} masters\n'
        f'PC1+PC2: {metrics["pca2_explained_ratio"]:.1%}  |  '
        f'Silhouette (512-D): {metrics["silhouette_512d_euclidean"]:.3f}',
        fontsize=12 if individual else 11, pad=9)
    ax.set_xlabel(f'PC1 ({metrics["explained_ratio"][0]:.1%})')
    ax.set_ylabel(f'PC2 ({metrics["explained_ratio"][1]:.1%})')
    ax.axhline(0, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.grid(True, alpha=0.16, linewidth=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def place_master_labels(ax, record, mode):
    """Keep labels readable without shifting or jittering any data points."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    occupied = []
    candidates = [(x, y) for radius in (6, 16, 28, 42)
                  for x, y in ((radius, radius), (radius, -radius),
                               (-radius, radius), (-radius, -radius),
                               (0, radius), (0, -radius), (radius, 0), (-radius, 0))]
    for master in record["masters"]:
        for dx, dy in candidates:
            label = ax.annotate(
                master["cluster_label"], master[mode], xytext=(dx, dy),
                textcoords="offset points", fontsize=8, zorder=5,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top")
            box = label.get_window_extent(renderer).expanded(1.12, 1.18)
            within_axes = (ax.bbox.contains(box.x0, box.y0)
                           and ax.bbox.contains(box.x1, box.y1))
            if within_axes and not any(box.overlaps(other) for other in occupied):
                occupied.append(box)
                break
            label.remove()
        else:
            raise ValueError(f"Unable to place label for {record['domain']}/{master['master_id']}")


def save_figure(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(stem) + "." + ext, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all(records, output):
    plt.rcParams.update({
        "font.size": 10, "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    for mode in ("raw", "normalized"):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10.2))
        for ax, record in zip(axes.flat, records):
            plot_panel(ax, record, mode)
        title = "Raw reconstructed visual embeddings" if mode == "raw" else "L2-normalized reconstructed visual embeddings"
        fig.suptitle(f"MemEye | {title} | PCA by domain", fontsize=20, y=0.99)
        fig.text(0.5, 0.023,
                 "Circles: image memories   X: saved visual masters   Color: existing caption cluster\n"
                 "PCA is fit independently per domain; clusters are not recomputed. Shared residuals make same-cluster points closer by construction.",
                 ha="center", fontsize=10, color="#444444")
        fig.subplots_adjust(left=0.045, right=0.985, top=0.89, bottom=0.105, wspace=0.29, hspace=0.45)
        save_figure(fig, output / f"pca_by_domain_{mode}")
        for record in records:
            directory = output / record["domain"]
            directory.mkdir(exist_ok=True)
            fig = plt.figure(figsize=(12.4, 8))
            ax = fig.add_axes([0.08, 0.12, 0.58, 0.76])
            plot_panel(ax, record, mode, individual=True)
            handles = [
                Line2D([], [], marker="o", linestyle="", markerfacecolor=m["color"],
                       markeredgecolor="white", markersize=7,
                       label=f'{m["cluster_label"]}  {m["master_id"]}  (n={m["n"]})')
                for m in record["masters"]
            ]
            ax.legend(handles=handles, title="Existing caption clusters",
                      bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False,
                      fontsize=8, title_fontsize=10, labelspacing=0.6)
            place_master_labels(ax, record, mode)
            fig.text(0.08, 0.025,
                     "Circles = reconstructed image memories; X = stored master transformed by the same PCA.\n"
                     + ("Raw vectors preserve stored cluster means." if mode == "raw"
                        else "Leaf and master vectors are individually L2-normalized before PCA/transform."),
                     fontsize=9, color="#444444")
            save_figure(fig, directory / f"pca_{mode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/memeye_reconstructed_visual_pca_20261104")
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Use a fresh --output directory.")
    manifest = read_json(source / "manifest.json")
    assert manifest["completed"] and manifest["n_domains"] == 8
    tracked = {str(source / "manifest.json"): sha(source / "manifest.json")}
    records = []
    for item in manifest["domains"]:
        record = reconstruct_domain(source / item["domain"], item, tracked)
        records.append(record)
        m = record["metrics"]["raw"]
        print(f'{record["short_domain"]}: N={record["n_images"]}, K={record["n_masters"]}, '
              f'PC1+2={m["pca2_explained_ratio"]:.2%}, '
              f'silhouette512={m["silhouette_512d_euclidean"]:.4f}', flush=True)
    assert sum(r["n_images"] for r in records) == 438
    assert sum(r["n_masters"] for r in records) == 88
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plot_data.json", {"domains": records})
    metrics_rows, point_rows, master_rows, spectrum_rows = [], [], [], []
    for record in records:
        domain = record["domain"]
        for mode, m in record["metrics"].items():
            metrics_rows.append({
                "domain": domain, "mode": mode, "n_images": record["n_images"],
                "n_masters": record["n_masters"], "n_singleton_masters": record["n_singleton_masters"],
                **{k: v for k, v in m.items() if k != "explained_ratio"},
            })
            cumulative = 0.0
            for pc, ratio in enumerate(m["explained_ratio"], 1):
                cumulative += ratio
                spectrum_rows.append({"domain": domain, "mode": mode, "pc": pc,
                                      "explained_ratio": ratio, "cumulative_ratio": cumulative})
            for point in record["points"]:
                point_rows.append({"domain": domain, "mode": mode,
                                   **{k: point[k] for k in ("image_id", "filename", "master_id", "cluster_label")},
                                   "pc1": point[mode][0], "pc2": point[mode][1]})
            for master in record["masters"]:
                master_rows.append({"domain": domain, "mode": mode,
                                    **{k: master[k] for k in ("master_id", "cluster_label", "n", "approximate")},
                                    "pc1": master[mode][0], "pc2": master[mode][1]})
    for name, rows in (("metrics", metrics_rows), ("points", point_rows),
                       ("masters", master_rows), ("pca_spectrum", spectrum_rows)):
        write_csv(output / f"{name}.csv", rows)
    plot_all(records, output)
    assert all(sha(path) == expected for path, expected in tracked.items())
    caption_protocol = ROOT / "results/natural_observation/protocol.json"
    write_json(output / "protocol.json", {
        "dataset": "MemEye", "memory": "caption_visual_master", "caption_version": "original Qwen",
        "caption_model": read_json(caption_protocol)["caption_model_path"],
        "caption_protocol_sha256": sha(caption_protocol),
        "source_root": str(source), "source_sha256": tracked,
        "script": str(Path(__file__).resolve()), "script_sha256": sha(__file__),
        "reconstruction": "vhat_i = (S_m/n_m) + U @ (z_i - U.T @ (S_m/n_m))",
        "master_normalized_before_reconstruction": False,
        "pca": "centered SVD, no whitening; independently fit on leaves in each domain",
        "raw": "PCA on exact saved-state reconstruction output, without row normalization",
        "normalized": "PCA after L2-normalizing each reconstructed leaf; anchors normalized separately",
        "clusters": "existing caption Ward labels; no reclustering",
        "original_images_read": 0, "image_encoder_calls": 0,
        "original_individual_embeddings_read": False, "reconstructed_512d_embeddings_saved": False,
        "input_manifests": "read only for filename labels; never open source_path",
        "source_files_unchanged": True,
        "silhouette": "existing labels, Euclidean and cosine in 512-D; Euclidean in displayed PCA-2D; singletons score zero",
        "interpretation": "Reconstruction shares discarded components within masters. This describes stored geometry, not independent original-image clustering quality.",
    })
    write_json(output / "summary.json", {
        "completed": True, "n_domains": len(records),
        "n_images": sum(r["n_images"] for r in records),
        "n_masters": sum(r["n_masters"] for r in records),
        "domains": [{k: v for k, v in r.items() if k not in ("points", "masters")} for r in records],
    })
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
