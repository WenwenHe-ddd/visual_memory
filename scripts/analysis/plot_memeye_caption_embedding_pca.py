#!/usr/bin/env python3
"""Caption-only BGE PCA, with frozen caption partitions and visual comparison."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/memeye_caption_pca_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from scripts.analysis.plot_memeye_reconstructed_visual_pca import (
    DEFAULT_SOURCE, place_master_labels, read_json, save_figure, sha,
    unit, write_csv, write_json,
)
from scripts.graph.caption_ward_visual import choose_caption_partition

VISUAL = ROOT / "reports/memeye_reconstructed_visual_pca_20261104"


def fit_pca(vectors, centers, labels):
    mean = vectors.mean(axis=0)
    _, singular, vt = np.linalg.svd(vectors - mean, full_matrices=False)
    for row in vt:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    ratios = singular**2 / np.sum(singular**2)
    points = (vectors - mean) @ vt[:2].T
    centroids = (centers - mean) @ vt[:2].T
    return points, centroids, {
        "dimension": int(vectors.shape[1]),
        "pca2_explained_ratio": float(ratios[:2].sum()),
        "explained_ratio": ratios.tolist(),
        "effective_centered_rank": int(np.count_nonzero(singular > 1e-12)),
        "silhouette_full_euclidean": float(silhouette_score(vectors, labels)),
        "silhouette_full_cosine": float(silhouette_score(vectors, labels, metric="cosine")),
        "silhouette_pca2_euclidean": float(silhouette_score(points, labels)),
    }


def analyze_domain(visual, graph, vectors, token_counts):
    record = copy.deepcopy(visual)
    nodes, masters = graph["image_nodes"], graph["masters"]
    index = {node["id"]: i for i, node in enumerate(nodes)}
    assert [p["image_id"] for p in record["points"]] == list(index)
    assert [m["master_id"] for m in record["masters"]] == [m["id"] for m in masters]
    assert all(p["caption"] == node["caption"] for p, node in zip(record["points"], nodes))
    labels = np.full(len(nodes), -1, dtype=int)
    centers = []
    for j, master in enumerate(masters):
        idx = [index[key] for key in master["member_ids"]]
        assert np.all(labels[idx] == -1) and len(idx) == master["n"]
        labels[idx] = j
        centers.append(vectors[idx].mean(axis=0))
    assert np.all(labels >= 0)
    regenerated = choose_caption_partition(vectors)
    predicted = np.full(len(nodes), -1, dtype=int)
    for j, group in enumerate(regenerated["partition"]):
        predicted[group] = j
    old_sets = {frozenset(m["member_ids"]) for m in masters}
    new_sets = {frozenset(nodes[i]["id"] for i in group)
                for group in regenerated["partition"]}
    partition_audit = {
        "stored_k": len(masters), "recomputed_k": regenerated["selected_k"],
        "identical_member_sets": old_sets == new_sets,
        "adjusted_rand_index": float(adjusted_rand_score(labels, predicted)),
        "recomputed_selected_silhouette": regenerated["selected_score"],
        "candidate_scores": regenerated["candidate_scores"],
        "note": "Ward rerun is validation only; plotted colors always use stored master membership.",
    }
    xy, center_xy, text_metrics = fit_pca(vectors, np.asarray(centers), labels)
    raw = visual["metrics"]["raw"]
    record["metrics"] = {
        "text": text_metrics,
        "visual": {
            "dimension": 512, "pca2_explained_ratio": raw["pca2_explained_ratio"],
            "explained_ratio": raw["explained_ratio"],
            "effective_centered_rank": raw["effective_centered_rank"],
            "silhouette_full_euclidean": raw["silhouette_512d_euclidean"],
            "silhouette_full_cosine": raw["silhouette_512d_cosine"],
            "silhouette_pca2_euclidean": raw["silhouette_pca2_euclidean"],
        },
    }
    for p, text_xy in zip(record["points"], xy):
        p["text"] = text_xy.tolist()
        p["visual"] = p.pop("raw")
        p.pop("normalized")
    for p, text_xy in zip(record["masters"], center_xy):
        p["text"] = text_xy.tolist()
        p["visual"] = p.pop("raw")
        p.pop("normalized")
    record["partition_validation"] = partition_audit
    record["caption_encoding"] = {
        "max_tokens": max(token_counts), "truncated_inputs": 0,
        "input": "full caption only", "prefix": "", "dimension": 384,
        "l2_norm_max_abs_error": float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1))),
    }
    return record


def draw_panel(ax, record, mode="text", individual=False):
    points = np.array([p[mode] for p in record["points"]])
    centers = np.array([p[mode] for p in record["masters"]])
    ax.scatter(*points.T, c=[p["color"] for p in record["points"]],
               s=44 if individual else 27, edgecolors="white", linewidths=.45, alpha=.87, zorder=3)
    ax.scatter(*centers.T, c=[p["color"] for p in record["masters"]],
               marker="X", s=100 if individual else 68, edgecolors="#222222", linewidths=.65, zorder=4)
    m = record["metrics"][mode]
    subtitle = "Caption embedding" if mode == "text" else "Reconstructed visual embedding"
    ax.set_title(
        f'{record["short_domain"]} | {subtitle}\n'
        f'PC1+2: {m["pca2_explained_ratio"]:.1%} | '
        f'Silhouette ({m["dimension"]}-D): {m["silhouette_full_euclidean"]:.3f}',
        fontsize=12 if individual else 10.5, pad=9)
    ax.set_xlabel(f'PC1 ({m["explained_ratio"][0]:.1%})')
    ax.set_ylabel(f'PC2 ({m["explained_ratio"][1]:.1%})')
    ax.axhline(0, color="#eeeeee", linewidth=.8, zorder=0)
    ax.axvline(0, color="#eeeeee", linewidth=.8, zorder=0)
    ax.grid(alpha=.16, linewidth=.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(.12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def render(records, out):
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.2))
    for ax, record in zip(axes.flat, records):
        draw_panel(ax, record)
    fig.suptitle("MemEye | Caption-only BGE embeddings | PCA by domain", fontsize=20, y=.99)
    fig.text(.5, .023,
             "438 image captions, 88 existing clusters | Circles: caption embeddings | X: caption cluster means\n"
             "Same membership and colors as the visual reconstruction plots. PCA is independently fit per domain.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=.045, right=.985, top=.89, bottom=.105, wspace=.29, hspace=.45)
    save_figure(fig, out / "pca_by_domain_caption")
    for record in records:
        directory = out / record["domain"]
        directory.mkdir(exist_ok=True)
        fig = plt.figure(figsize=(12.4, 8))
        ax = fig.add_axes([.08, .12, .58, .76])
        draw_panel(ax, record, individual=True)
        handles = [
            Line2D([], [], marker="o", linestyle="", markerfacecolor=m["color"],
                   markeredgecolor="white", markersize=7,
                   label=f'{m["cluster_label"]}  {m["master_id"]}  (n={m["n"]})')
            for m in record["masters"]
        ]
        ax.legend(handles=handles, title="Existing caption clusters",
                  bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False,
                  fontsize=8, title_fontsize=10, labelspacing=.6)
        place_master_labels(ax, record, "text")
        fig.text(.08, .025,
                 "Full caption only; BGE-small-en-v1.5; normalized 384-D embeddings.\n"
                 "X marks the arithmetic mean of caption vectors in each cluster, not the visual master.",
                 fontsize=9, color="#444444")
        save_figure(fig, directory / "pca_caption")
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.7))
        for ax, mode in zip(axes, ("text", "visual")):
            draw_panel(ax, record, mode)
        fig.suptitle(f'{record["short_domain"]}: same {record["n_images"]} memories and '
                     f'{record["n_masters"]} clusters', fontsize=15, y=.99)
        fig.text(.5, .025,
                 "Same colors denote the same clusters. Separate PCA fits: axes are not aligned across modalities.\n"
                 "Caption X: text centroid. Visual X: saved master. Shared visual residuals tighten reconstructed clusters.",
                 ha="center", fontsize=9, color="#444444")
        fig.subplots_adjust(top=.80, bottom=.17, left=.07, right=.98, wspace=.28)
        save_figure(fig, directory / "caption_vs_reconstructed_visual")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--visual-report", type=Path, default=VISUAL)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/memeye_caption_embedding_pca_20261104")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    out, source = args.output.resolve(), args.source_root.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Use a fresh --output directory.")
    manifest_path = source / "manifest.json"
    visual_path = args.visual_report.resolve() / "plot_data.json"
    tracked = {str(p): sha(p) for p in (manifest_path, visual_path)}
    manifest, visual = read_json(manifest_path), read_json(visual_path)
    by_domain = {r["domain"]: r for r in visual["domains"]}
    groups, metadata, captions, image_ids = [], None, [], []
    for item in manifest["domains"]:
        directory = source / item["domain"]
        for name in ("memory/graph.json", "protocol.json"):
            p = directory / name
            tracked[str(p)] = sha(p)
            assert tracked[str(p)] == item["sha256"][name]
        graph = read_json(directory / "memory/graph.json")
        meta = read_json(directory / "protocol.json")["text_encoder"]
        assert metadata is None or metadata == meta
        metadata = meta
        assert len(graph["image_nodes"]) == item["n_leaves"]
        start = len(captions)
        captions.extend(n["caption"] for n in graph["image_nodes"])
        image_ids.extend(n["id"] for n in graph["image_nodes"])
        groups.append((item["domain"], graph, slice(start, len(captions))))
    assert len(captions) == len(set(image_ids)) == 438 and len(groups) == 8
    assert metadata["dimension"] == 384
    model_path = Path(metadata["model_path"])
    for name, expected in metadata["files_sha256"].items():
        assert sha(model_path / name) == expected, name

    import torch
    from sentence_transformers import SentenceTransformer
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU encoding requires host GPU access.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = SentenceTransformer(str(model_path), device=args.device, local_files_only=True).float().eval()
    assert model.max_seq_length == metadata["max_seq_length"] == 512
    lengths = [len(tokens) for tokens in model.tokenizer(captions, truncation=False)["input_ids"]]
    assert max(lengths) <= model.max_seq_length
    print(f"Encoding {len(captions)} full captions on {args.device}; max tokens={max(lengths)}", flush=True)
    with torch.inference_mode():
        vectors = unit(np.asarray(model.encode(
            captions, batch_size=args.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False, prompt="",
        ), dtype=np.float64))
    assert vectors.shape == (438, 384) and np.isfinite(vectors).all()
    gpu_peak = int(torch.cuda.max_memory_allocated(args.device)) if args.device.startswith("cuda") else 0
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    records = []
    for domain, graph, sl in groups:
        record = analyze_domain(by_domain[domain], graph, vectors[sl], lengths[sl])
        records.append(record)
        m, check = record["metrics"]["text"], record["partition_validation"]
        print(f'{record["short_domain"]}: K={record["n_masters"]}; '
              f'PC1+2={m["pca2_explained_ratio"]:.2%}; '
              f'S384={m["silhouette_full_euclidean"]:.4f}; '
              f'S2={m["silhouette_pca2_euclidean"]:.4f}; '
              f'partition_match={check["identical_member_sets"]}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    # This is a text-only analysis cache, separate from persistent memory.
    np.savez_compressed(out / "caption_embeddings.npz",
                        image_ids=np.asarray(image_ids), caption_embeddings=vectors)
    write_json(out / "plot_data.json", {"domains": records})
    metric_rows, point_rows, spectra, comparisons = [], [], [], []
    for record in records:
        for mode, m in record["metrics"].items():
            metric_rows.append({
                "domain": record["domain"], "representation": mode,
                "n_images": record["n_images"], "n_masters": record["n_masters"],
                **{k: v for k, v in m.items() if k != "explained_ratio"},
            })
            cumulative = 0.0
            for pc, ratio in enumerate(m["explained_ratio"], 1):
                cumulative += ratio
                spectra.append({"domain": record["domain"], "representation": mode,
                                "pc": pc, "explained_ratio": ratio, "cumulative_ratio": cumulative})
            for point in record["points"]:
                point_rows.append({
                    "domain": record["domain"], "representation": mode,
                    **{k: point[k] for k in ("image_id", "filename", "master_id", "cluster_label")},
                    "pc1": point[mode][0], "pc2": point[mode][1],
                })
        text, vis = record["metrics"]["text"], record["metrics"]["visual"]
        comparisons.append({
            "domain": record["domain"], "n_images": record["n_images"], "n_masters": record["n_masters"],
            "caption_pc1_pc2_ratio": text["pca2_explained_ratio"],
            "visual_reconstructed_pc1_pc2_ratio": vis["pca2_explained_ratio"],
            "caption_silhouette_384d": text["silhouette_full_euclidean"],
            "visual_reconstructed_silhouette_512d": vis["silhouette_full_euclidean"],
            "caption_silhouette_pca2": text["silhouette_pca2_euclidean"],
            "visual_reconstructed_silhouette_pca2": vis["silhouette_pca2_euclidean"],
            "ward_partition_matches_saved": record["partition_validation"]["identical_member_sets"],
        })
    for name, rows in (("metrics", metric_rows), ("points", point_rows),
                       ("pca_spectrum", spectra), ("comparison_metrics", comparisons)):
        write_csv(out / f"{name}.csv", rows)
    render(records, out)
    assert all(sha(p) == digest for p, digest in tracked.items())
    write_json(out / "protocol.json", {
        "dataset": "MemEye", "caption_version": "original Qwen3-VL-8B-Instruct",
        "source_sha256": tracked, "text_encoder": metadata,
        "encoding_input": "graph.image_nodes[].caption only", "dialogue_encoded": False,
        "query_prefix": "", "prompt": "", "batch_size": args.batch_size,
        "device": args.device, "dtype": "float32 model; float64 L2 renormalization",
        "torch_version": torch.__version__, "cuda_peak_allocated_bytes": gpu_peak,
        "max_tokens": max(lengths), "truncated_inputs": 0,
        "clustering": "same Ward + maximum silhouette rerun for membership verification only",
        "plotted_labels": "saved master membership; same image IDs and colors as previous visual analysis",
        "pca": "centered SVD on normalized 384-D caption vectors; no whitening",
        "caption_centroids": "arithmetic mean of normalized caption embeddings; never the visual master",
        "visual_comparison": "unchanged raw reconstructed visual PCA coordinates from previous report",
        "pca_axes_shared_between_modalities": False,
        "original_images_read": 0, "visual_encoder_calls": 0, "api_calls": 0,
        "source_files_unchanged": True, "memory_modified": False,
        "caption_embeddings_saved": "analysis cache only; not written to persistent memory",
        "script": str(Path(__file__).resolve()), "script_sha256": sha(__file__),
        "plot_helper_sha256": sha(ROOT / "scripts/analysis/plot_memeye_reconstructed_visual_pca.py"),
        "ward_helper_sha256": sha(ROOT / "scripts/graph/caption_ward_visual.py"),
    })
    write_json(out / "summary.json", {
        "completed": True, "n_domains": 8, "n_images": 438, "n_masters": 88,
        "all_ward_partitions_match_saved": all(r["partition_validation"]["identical_member_sets"] for r in records),
        "domains": [{k: v for k, v in r.items() if k not in ("points", "masters")} for r in records],
    })
    print(f"Complete: {out}", flush=True)


if __name__ == "__main__":
    main()
