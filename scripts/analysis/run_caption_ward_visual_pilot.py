#!/usr/bin/env python3
"""Chronological Home pilot; no retained image pixels or individual CLIP vectors.

The evaluator uses an independently recorded pairwise Gram matrix, never passed
to the memory implementation. It can measure visual mean error without storing
or re-encoding old images. Only the current incoming image is encoded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/caption_ward_matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from scripts.graph.caption_ward_visual import CaptionWardVisualMemory
from scripts.graph.visual_image_encoder import ImageEncoder

DOMAIN = "Home_Renovation_Interior_Design"
SOURCE = ROOT / "data/independent_visual_inputs"
REGISTRY = ROOT / "results/natural_observation"
ARRIVALS = SOURCE / "arrivals.jsonl"
DIALOG = ROOT / "data/MemEye/data/dialog" / f"{DOMAIN}_Open.json"
GRAM_SOURCE = ROOT / "reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv"
PREVIOUS_PROTOCOL = ROOT / "reports/home_q10_top5_embeddings_20261020/protocol.json"
TEXT_MODEL = Path("/storage/hewenwen/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/"
                  "snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def prepare_inputs():
    source_protocol = read_json(SOURCE / "protocol.json")
    assert sha(ARRIVALS) == source_protocol["export_hashes"][ARRIVALS.name]
    memories = jsonl(REGISTRY / "memories.jsonl")
    memory_by_id = {row["id"]: row for row in memories}
    images = {row["id"]: row for row in jsonl(REGISTRY / "images.jsonl")}
    reader = {row["id"]: row for row in jsonl(SOURCE / "retrieval_memories.jsonl")}
    caption_by_id = {m["image_id"]: m["caption"] for m in jsonl(ARRIVALS)}
    rows = []
    for session in read_json(DIALOG)["multi_session_dialogues"]:
        for turn in session["dialogues"]:
            paths = turn.get("input_image", [])
            if not paths:
                continue
            memory_id = f"{DOMAIN}/{turn['round']}"
            memory = memory_by_id[memory_id]
            assert len(paths) == len(memory["images"]) == 1
            dialogue, caption = reader[memory_id]["reader_text"].split("\nMemory description: ", 1)
            for image_index, (relative_path, image_sha) in enumerate(zip(paths, memory["images"]), 1):
                image_id = f"{memory_id}#image{image_index}"
                path = ROOT / "data/MemEye/data/image" / relative_path
                assert caption == caption_by_id[image_id]
                assert sha(path) == image_sha
                rows.append({
                    "image_id": image_id, "memory_id": memory_id,
                    "filename": path.name, "source_path": str(path), "image_sha256": image_sha,
                    "caption": caption, "dialogue": dialogue, "timestamp": session["date"],
                    "caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
                })
    assert len(rows) == len({row["image_id"] for row in rows}) == 89
    assert [r["timestamp"] for r in rows] == sorted(r["timestamp"] for r in rows)
    home_hashes = {row["image_sha256"] for row in rows}
    calibration_hashes = sorted({s for m in memories if m["task"] != DOMAIN for s in m["images"]})
    assert not home_hashes.intersection(calibration_hashes)
    calibration = []
    for digest in calibration_hashes:
        path = Path(images[digest]["path"])
        assert sha(path) == digest
        calibration.append({"image_sha256": digest, "source_path": str(path)})
    assert len(calibration) == 344
    return rows, calibration, source_protocol


def load_evaluation_gram(rows):
    indices = {row["image_id"]: i for i, row in enumerate(rows)}
    gram = np.eye(len(rows), dtype=np.float64)
    seen = set()
    with GRAM_SOURCE.open() as handle:
        for pair in csv.DictReader(handle):
            left, right = pair["earlier_id"], pair["later_id"]
            if left not in indices or right not in indices:
                continue
            i, j = indices[left], indices[right]
            key = tuple(sorted((i, j)))
            assert i != j and key not in seen
            seen.add(key)
            gram[i, j] = gram[j, i] = float(pair["cosine"])
    assert len(seen) == len(rows) * (len(rows) - 1) // 2
    assert np.linalg.eigvalsh(gram).min() > -1e-10
    return gram


def calibrate_basis(encoder, calibration):
    start = time.perf_counter()
    dim = encoder.embedding_dim
    total = np.zeros(dim)
    second = np.zeros((dim, dim))
    for index, row in enumerate(calibration, 1):
        vector = unit(encoder.encode(row["source_path"]))
        total += vector
        second += np.outer(vector, vector)
        del vector
        if index % 50 == 0 or index == len(calibration):
            print(json.dumps({"stage": "calibration", "done": index, "total": len(calibration)}),
                  flush=True)
    n = len(calibration)
    covariance = (second - np.outer(total, total) / n) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    basis = eigenvectors[:, -2:][:, ::-1].copy()
    for col in range(2):
        if basis[np.argmax(np.abs(basis[:, col])), col] < 0:
            basis[:, col] *= -1
    return basis, {
        "n": n, "dimension": dim, "eigenvalues": eigenvalues.tolist(),
        "top2_centered_calibration_variance_ratio": float(eigenvalues[:2].sum() / eigenvalues.sum()),
        "elapsed_seconds": time.perf_counter() - start,
        "projection": "z = U.T @ v (absolute projection; calibration mean is NOT subtracted)",
        "fit": "PCA of centered external vectors from running sum and second moment",
        "no_individual_calibration_vectors_retained": True,
    }


class GramEvaluator:
    """Evaluation-only residual mixing; no high-dimensional individual vectors."""

    def __init__(self, rows, gram):
        self.indices = {r["image_id"]: i for i, r in enumerate(rows)}
        self.gram = gram
        self.coefficients = {}

    def measure(self, memory, audit):
        n = audit["n_images"]
        weights = {m: np.zeros(len(self.gram)) for m in memory.masters}
        for overlap in audit["source_overlaps"]:
            weights[overlap["new_master_id"]] += (
                overlap["fraction"] * self.coefficients[overlap["old_master_id"]]
            )
        incoming = audit["image_id"]
        weights[memory.image_nodes[incoming]["master_id"]][self.indices[incoming]] += 1
        self.coefficients = weights
        coefficient_conservation = float(np.max(np.abs(sum(weights.values())[:n] - 1)))
        assert coefficient_conservation < 1e-12
        z = np.array([node["visual_coords"] for node in memory.image_nodes.values()])
        gram = self.gram[:n, :n]
        residual_gram = gram - z @ z.T
        metrics = []
        for master_id, master in memory.masters.items():
            a = weights[master_id][:n]
            b = np.zeros(n)
            members = [self.indices[m] for m in master["member_ids"]]
            b[members] = 1
            size = master["n"]
            difference = a - b
            error_sq = float(difference @ residual_gram @ difference) / size**2
            assert error_sq >= -1e-10
            true_norm_sq = float(b @ gram @ b) / size**2
            projected_sq = float(np.sum(z[members].sum(axis=0)**2)) / size**2
            approx_norm_sq = projected_sq + float(a @ residual_gram @ a) / size**2
            inner = projected_sq + float(a @ residual_gram @ b) / size**2
            cosine = float(np.clip(inner / np.sqrt(true_norm_sq * approx_norm_sq), -1, 1))
            actual_norm_sq = float(np.sum((master["visual_sum"] / size)**2))
            norm_error = abs(actual_norm_sq - approx_norm_sq)
            assert norm_error < 1e-8, (master_id, norm_error)
            metrics.append({
                "master_id": master_id, "n": size, "approximate": master["approximate"],
                "mean_l2_error": np.sqrt(max(0, error_sq)).item(),
                "relative_l2_error": np.sqrt(max(0, error_sq) / true_norm_sq).item(),
                "cosine_to_exact_mean": cosine,
                "angle_degrees": np.degrees(np.arccos(cosine)).item(),
                "oracle_norm_sq_discrepancy": norm_error,
            })
        return {
            "clusters": metrics,
            "cluster_macro_mean_l2_error": float(np.mean([m["mean_l2_error"] for m in metrics])),
            "member_weighted_mean_l2_error": float(sum(m["n"] * m["mean_l2_error"] for m in metrics) / n),
            "max_mean_l2_error": max(m["mean_l2_error"] for m in metrics),
            "member_weighted_cosine": float(sum(m["n"] * m["cosine_to_exact_mean"] for m in metrics) / n),
            "max_angle_degrees": max(m["angle_degrees"] for m in metrics),
            "coefficient_conservation_max_abs": coefficient_conservation,
        }


def write_outputs(output, memory, rows, audits, protocol, calibration_info):
    state = output / "memory"
    state.mkdir()
    snapshot = memory.snapshot()
    master_rows = list(memory.masters.values())
    np.save(state / "basis.npy", memory.basis)
    np.save(state / "master_sums.npy", np.stack([m["visual_sum"] for m in master_rows]))
    snapshot["basis_ref"] = {"path": "basis.npy", "shape": list(memory.basis.shape)}
    for index, master in enumerate(snapshot["masters"]):
        master["visual_sum_ref"] = {"path": "master_sums.npy", "row": index}
        master["visual_mean_ref"] = {"path": "master_sums.npy", "row": index, "scale": 1 / master["n"]}
    snapshot["text_embedding_cache"]["persisted"] = False
    snapshot["text_embedding_cache"]["rebuild_from"] = "image_nodes[].caption"
    write_json(state / "graph.json", snapshot)
    write_json(output / "calibration.json", calibration_info)
    write_json(output / "protocol.json", protocol)
    with (output / "insertion_audit.jsonl").open("w") as handle:
        for row in audits:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    sizes = [master["n"] for master in master_rows]
    final = audits[-1]
    membership = {m["image_id"]: memory.image_nodes[m["image_id"]]["master_id"] for m in rows}
    filename_to_id = {row["filename"]: row["image_id"] for row in rows}
    important_pairs = [
        ("D1_IMG_001.jpg", "D1_IMG_002.jpg"),
        ("D1_IMG_001.jpg", "D1_IMG_003.jpg"),
        ("D7_IMG_005.jpg", "D7_IMG_006.jpg"),
    ]
    numeric_bytes = (memory.basis.size + sum(m["visual_sum"].size for m in master_rows)
                     + 2 * len(rows)) * 8
    individual_bytes = len(rows) * memory.visual_dim * 8
    timings = {}
    for key in ("image_encode", "text_encode", "clustering", "visual_update", "memory_total", "total_without_oracle"):
        values = [a["runtime_seconds"][key] for a in audits]
        timings[key] = {
            "total": sum(values), "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)), "max": max(values),
        }
    results = {
        "n_images": len(rows), "n_clusters": len(sizes),
        "cluster_sizes_descending": sorted(sizes, reverse=True),
        "singleton_clusters": sum(size == 1 for size in sizes),
        "singleton_image_fraction": sum(size == 1 for size in sizes) / len(rows),
        "selected_silhouette": final["selected_score"],
        "approximate_masters": sum(m["approximate"] for m in master_rows),
        "steps_with_partial_splits": sum(bool(a["split_master_ids"]) for a in audits),
        "total_old_cluster_split_events": sum(len(a["split_master_ids"]) for a in audits),
        "steps_with_old_pair_relation_changes": sum(a["old_pair_relation_changes"] > 0 for a in audits),
        "old_pair_relation_changes_total": sum(a["old_pair_relation_changes"] for a in audits),
        "final_visual_center_error": final["visual_center_error"],
        "maximum_center_error_over_all_steps": max(a["visual_center_error"]["max_mean_l2_error"] for a in audits),
        "max_global_sum_l2_error": max(a["global_stream_sum_l2_error"] for a in audits),
        "max_projection_l2_error": max(a["max_projection_consistency_l2_error"] for a in audits),
        "max_new_vector_gram_row_sum_error": max(a["new_vector_gram_row_sum_error"] for a in audits),
        "runtime_seconds": timings,
        "storage": {
            "dtype": "float64", "master_visual_sums_bytes": len(sizes) * memory.visual_dim * 8,
            "shared_basis_bytes": memory.basis.nbytes, "leaf_two_coords_numeric_bytes": len(rows) * 2 * 8,
            "visual_numeric_payload_bytes": numeric_bytes,
            "comparison_all_individual_visual_embeddings_bytes": individual_bytes,
            "visual_numeric_payload_ratio": numeric_bytes / individual_bytes,
            "visual_numeric_payload_reduction_fraction": 1 - numeric_bytes / individual_bytes,
            "graph_json_bytes": (state / "graph.json").stat().st_size,
            "all_memory_files_bytes": sum(p.stat().st_size for p in state.iterdir()),
            "text_embedding_cache_ram_bytes": sum(v.nbytes for v in memory.text_embedding_cache.values()),
            "text_embedding_cache_disk_bytes": 0,
            "individual_visual_embedding_files": 0, "copied_image_files": 0,
            "comparison_note": "Numeric visual payload only; does not claim total graph size reduction. "
                               "JSON text, model weights, evaluation/provenance files are separate.",
        },
        "important_pairs": [{
            "left": left, "right": right,
            "left_master": membership[filename_to_id[left]],
            "right_master": membership[filename_to_id[right]],
            "same_cluster": membership[filename_to_id[left]] == membership[filename_to_id[right]],
        } for left, right in important_pairs],
        "semantic_quality_or_qa_accuracy_evaluated": False,
    }
    write_json(output / "results.json", results)
    with (output / "insertion_metrics.csv").open("w") as handle:
        fields = ["step", "image_id", "n_clusters", "singletons", "silhouette",
                  "splits", "old_pair_relation_changes", "memory_ms", "image_ms", "text_ms",
                  "center_l2_weighted", "center_l2_max"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for a in audits:
            writer.writerow({
                "step": a["n_images"], "image_id": a["image_id"], "n_clusters": a["n_clusters"],
                "singletons": sum(c["n"] == 1 for c in a["clusters"]),
                "silhouette": a["selected_score"], "splits": len(a["split_master_ids"]),
                "old_pair_relation_changes": a["old_pair_relation_changes"],
                "memory_ms": a["runtime_seconds"]["memory_total"] * 1000,
                "image_ms": a["runtime_seconds"]["image_encode"] * 1000,
                "text_ms": a["runtime_seconds"]["text_encode"] * 1000,
                "center_l2_weighted": a["visual_center_error"]["member_weighted_mean_l2_error"],
                "center_l2_max": a["visual_center_error"]["max_mean_l2_error"],
            })
    make_report(output, rows, audits, results, snapshot)
    return results


def make_report(output, rows, audits, results, snapshot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [a["n_images"] for a in audits]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(steps, [a["n_clusters"] for a in audits], label="Clusters")
    axes[0, 0].plot(steps, [sum(c["n"] == 1 for c in a["clusters"]) for a in audits], label="Singletons")
    axes[0, 0].set_title("Partition size"); axes[0, 0].legend()
    axes[0, 1].plot(steps, [a["runtime_seconds"]["memory_total"] * 1000 for a in audits], label="Clustering + update")
    axes[0, 1].set_title("Memory update time (excludes encoders / oracle)")
    axes[0, 1].set_ylabel("Milliseconds")
    axes[1, 0].plot(steps, [a["visual_center_error"]["member_weighted_mean_l2_error"] for a in audits], label="Member-weighted mean")
    axes[1, 0].plot(steps, [a["visual_center_error"]["max_mean_l2_error"] for a in audits], label="Maximum")
    axes[1, 0].set_title("Visual centroid L2 error vs exact mean"); axes[1, 0].legend()
    axes[1, 1].bar(steps, [a["old_pair_relation_changes"] for a in audits], width=0.85)
    axes[1, 1].set_title("Old pairs whose same-cluster relation changes")
    for ax in axes.flat:
        ax.set_xlabel("Images inserted"); ax.grid(alpha=0.2)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"streaming_diagnostics.{suffix}", dpi=180)
    plt.close(fig)
    by_id = {row["image_id"]: row for row in rows}
    error_by_master = {m["master_id"]: m for m in results["final_visual_center_error"]["clusters"]}
    pieces = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Home caption Ward pilot</title>',
        "<style>body{max-width:1100px;margin:40px auto;font:16px/1.6 sans-serif;color:#243047}"
        "details{border:1px solid #ddd;padding:12px;margin:10px 0}summary{cursor:pointer}"
        "pre{white-space:pre-wrap}article{border-top:1px solid #eee;padding:12px}"
        "img{max-width:100%}small{color:#5b6575}</style>",
        "<h1>Home：caption 分簇与动态视觉中心</h1>",
        f"<p>{results['n_images']} 张图片 → {results['n_clusters']} 簇；"
        f"单图簇 {results['singleton_clusters']} 个。轮廓系数 {results['selected_silhouette']:.4f}。</p>",
        "<p>文本 caption 决定分簇；每个叶子仅保留两个视觉坐标。"
        "此页只展示文本和统计图，没有复制原图片。轮廓系数不能替代人工语义质量或 QA 评测。</p>",
        '<img src="streaming_diagnostics.png" alt="Streaming diagnostics">',
        "<p>误差以各簇真实 CLIP 平均向量为基准，由历史两两相似度独立计算；"
        "历史相似度不参与分簇和记忆更新。approximate 表示经历过部分拆分或继承近似状态。</p>",
    ]
    for master in snapshot["masters"]:
        err = error_by_master[master["id"]]
        pieces.append(f"<details><summary>{html.escape(master['id'])} · {master['n']} 张 · "
                      f"L2={err['mean_l2_error']:.4f} · cosine={err['cosine_to_exact_mean']:.4f} · "
                      f"approximate={master['approximate']}</summary>")
        for image_id in master["member_ids"]:
            row = by_id[image_id]
            pieces.append("<article><b>" + html.escape(row["filename"]) + "</b><br><small>"
                          + html.escape(image_id + " · " + row["timestamp"])
                          + "</small><p>" + html.escape(row["caption"]) + "</p><pre>"
                          + html.escape(row["dialogue"]) + "</pre></article>")
        pieces.append("</details>")
    pieces.append("</html>")
    (output / "clusters.html").write_text("\n".join(pieces))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/home_caption_ward_visual_20261020")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists() or (output / "memory").exists():
        raise SystemExit("Output already contains a completed run; use a fresh --output path.")
    start = time.perf_counter()
    rows, calibration, source_protocol = prepare_inputs()
    write_json(output / "input_manifest.json", {"home": rows, "calibration": calibration})
    gram = load_evaluation_gram(rows)
    previous = read_json(PREVIOUS_PROTOCOL)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    encoder = ImageEncoder(str(ROOT / "models/clip-vit-base-patch32"),
                           revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                           device="cpu", local_files_only=True)
    assert encoder.metadata == previous["image_encoder"]
    for name, expected in previous["text_encoder"]["files_sha256"].items():
        assert sha(TEXT_MODEL / name) == expected
    text_model = SentenceTransformer(str(TEXT_MODEL), device="cpu", local_files_only=True)
    assert text_model.max_seq_length == 512
    basis, calibration_info = calibrate_basis(encoder, calibration)
    memory = CaptionWardVisualMemory(basis)
    evaluator = GramEvaluator(rows, gram)
    global_sum = np.zeros(encoder.embedding_dim)
    audits = []
    for index, row in enumerate(rows):
        # No previous image path is passed to the encoder anywhere in this loop.
        t0 = time.perf_counter()
        visual = unit(encoder.encode(row["source_path"]))
        t1 = time.perf_counter()
        tokens = text_model.tokenizer(row["caption"], truncation=False)["input_ids"]
        assert len(tokens) <= text_model.max_seq_length
        text = unit(text_model.encode(row["caption"], normalize_embeddings=True, show_progress_bar=False))
        t2 = time.perf_counter()
        old_owners = {key: node["master_id"] for key, node in memory.image_nodes.items()}
        audit = memory.add(row["image_id"], row["caption"], text, visual,
                           dialogue=row["dialogue"], timestamp=row["timestamp"])
        t3 = time.perf_counter()
        gram_error = abs(float(visual @ global_sum) - float(gram[index, :index].sum()))
        assert gram_error < 1e-8
        global_sum += visual
        del visual, text
        new_sum = sum((m["visual_sum"] for m in memory.masters.values()), np.zeros_like(global_sum))
        global_error = float(np.linalg.norm(new_sum - global_sum))
        assert global_error < 1e-8
        old_ids = list(old_owners)
        relation_changes = 0
        for i, left in enumerate(old_ids):
            for right in old_ids[i+1:]:
                relation_changes += ((old_owners[left] == old_owners[right]) !=
                                     (memory.image_nodes[left]["master_id"] == memory.image_nodes[right]["master_id"]))
        audit.update({
            "caption_token_count": len(tokens), "old_pair_relation_changes": relation_changes,
            "new_vector_gram_row_sum_error": gram_error, "global_stream_sum_l2_error": global_error,
            "runtime_seconds": {
                "image_encode": t1-t0, "text_encode": t2-t1, "memory_total": t3-t2,
                "total_without_oracle": t3-t0,
                "clustering": audit["timings_seconds"]["clustering"],
                "visual_update": audit["timings_seconds"]["visual_update"],
            },
            "visual_center_error": evaluator.measure(memory, audit),
        })
        audits.append(audit)
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(json.dumps({"stage": "home", "done": index + 1, "clusters": len(memory.masters),
                              "error_l2_weighted": audit["visual_center_error"]["member_weighted_mean_l2_error"]}),
                  flush=True)
    source_paths = [SOURCE / "protocol.json", SOURCE / "retrieval_memories.jsonl",
                    REGISTRY / "memories.jsonl", REGISTRY / "images.jsonl",
                    ARRIVALS, DIALOG, GRAM_SOURCE, PREVIOUS_PROTOCOL,
                    Path(__file__), ROOT / "scripts/graph/caption_ward_visual.py",
                    ROOT / "scripts/graph/visual_image_encoder.py"]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "domain": DOMAIN,
        "source_experiment_signature": source_protocol["experiment_signature"],
        "caption_source": "Frozen full captions recovered from independently retained Ward input manifests",
        "clustering_input": "full caption only; no dialogue, timestamps, visual vectors, QA or oracle",
        "chronology": "original session/date, round and image order; captions enter one at a time",
        "method": "normalized caption vectors; Euclidean Ward; exact K cuts; max mean silhouette",
        "candidate_k": "2..N-1, except N<=2 singleton startup and N>=3 exactly identical text => K=1",
        "ties": "score tolerance 1e-12, prefer smaller K, deterministic original insertion order",
        "manual_similarity_threshold": None, "preset_cluster_count": None,
        "scalability": "full prefix Ward rebuild and exhaustive silhouette at every insertion; small-domain reference",
        "visual_approximation": "fixed U; R_C=S_C-U sum(z_i); split allocates |C intersect D|/n_C of R_C",
        "visual_state": "selected masters only; leaf absolute U.T v two coords; no visual-vector history",
        "original_image_policy": "corpus used as stream input, no copies retained; no old Home re-encoding",
        "calibration_home_sha_overlap": 0, "calibration_uses_future_home": False,
        "oracle": "historical pairwise Gram, evaluator-only residual mixing coefficients; never passed to core",
        "oracle_math": "H=G-ZZ^T; exact membership b, residual mixing a; error^2=(a-b)^T H (a-b)/n^2",
        "image_encoder": encoder.metadata, "text_encoder": previous["text_encoder"],
        "text_input_max_tokens": max(a["caption_token_count"] for a in audits),
        "truncated_text_inputs": 0, "torch_threads": torch.get_num_threads(),
        "source_sha256": {str(p): sha(p) for p in source_paths},
        "wall_seconds_before_artifact_rendering": time.perf_counter() - start,
        "limitations": [
            "silhouette omits general K=1 and K=N; no guarantee of semantic or QA quality",
            "two coordinates cannot recover omitted visual directions; split centers are approximate",
            "external-domain PCA may preserve little within-Home variation",
            "full reclustering is not an asymptotically efficient online algorithm",
        ],
    }
    results = write_outputs(output, memory, rows, audits, protocol, calibration_info)
    validation = {
        "passed": True, "home_inputs": len(rows), "calibration_inputs": len(calibration),
        "calibration_home_sha_overlap": 0, "ordered_insertions": len(audits),
        "incoming_image_encodes": len(rows), "old_home_image_reencodes": 0,
        "caption_graph_matches": len(rows), "truncated_text_inputs": 0,
        "gram_pairs": len(rows) * (len(rows) - 1) // 2,
        "encoder_metadata_exactly_matches_previous": True,
        "max_oracle_norm_sq_discrepancy": max(m["oracle_norm_sq_discrepancy"]
            for a in audits for m in a["visual_center_error"]["clusters"]),
        "max_global_sum_l2_error": results["max_global_sum_l2_error"],
        "max_projection_l2_error": results["max_projection_l2_error"],
        "memory_arrays": {"basis.npy": [512, 2], "master_sums.npy": [len(memory.masters), 512]},
        "memory_no_image_paths": all("image_path" not in n and "source_path" not in n
                                     for n in memory.image_nodes.values()),
        "memory_leaf_visual_dimensions": sorted({len(n["visual_coords"]) for n in memory.image_nodes.values()}),
        "wall_seconds_including_artifact_rendering": time.perf_counter() - start,
    }
    write_json(output / "validation.json", validation)
    print(json.dumps({"complete": True, "output": str(output), "images": len(rows),
                      "clusters": len(memory.masters), "validation": validation}), flush=True)


if __name__ == "__main__":
    main()
