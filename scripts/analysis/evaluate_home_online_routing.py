#!/usr/bin/env python3
"""Frozen Home memory: all-question CLIP routing and dialogue+caption BGE scores."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/home_online_routing_matplotlib")

import numpy as np

from scripts.analysis.probe_caption_visual_online import (
    OFFLINE, QA_ROOT, TOKENIZER_HASHES, leaf_retrieval_text, normalize,
    read_json, sha, write_json,
)

DOMAIN = "Home_Renovation_Interior_Design"
SINGLE_RUN = ROOT / "reports/home_q10_master_routing_dialogue_caption_20261020"


def encode_questions_and_leaves(questions, leaves, offline, prefix):
    import torch
    import transformers
    import sentence_transformers
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    image_info = offline["image_encoder"]
    clip_path = Path(image_info["model_name"])
    for name, expected in {**image_info["files_sha256"], **TOKENIZER_HASHES}.items():
        assert sha(clip_path / name) == expected, name
    start = time.perf_counter()
    config = AutoConfig.from_pretrained(str(clip_path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(clip_path), local_files_only=True)
    model, loading = CLIPTextModelWithProjection.from_pretrained(
        str(clip_path), config=config.text_config, local_files_only=True, output_loading_info=True,
    )
    assert not any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs"))
    assert all(k.startswith(("vision_model.", "visual_projection.")) or k == "logit_scale"
               for k in loading.get("unexpected_keys", []))
    model.to(device="cpu", dtype=torch.float32).eval()
    clip_queries, clip_inputs = [], []
    for question in questions:
        inputs = tokenizer(question["question"], return_tensors="pt", truncation=False)
        ids = inputs["input_ids"][0].tolist()
        if len(ids) > config.text_config.max_position_embeddings:
            raise ValueError(f"CLIP input exceeds token limit: {question['id']}")
        with torch.inference_mode():
            vector = model(**inputs, return_dict=True).text_embeds[0].double().numpy()
        clip_queries.append(normalize(vector))
        clip_inputs.append({"id": question["id"], "text": question["question"],
                            "input_ids": ids, "token_count": len(ids)})
    clip_seconds = time.perf_counter() - start
    del model

    start = time.perf_counter()
    text_info = offline["text_encoder"]
    bge_path = Path(text_info["model_path"])
    for name, expected in text_info["files_sha256"].items():
        assert sha(bge_path / name) == expected, name
    text_model = SentenceTransformer(str(bge_path), device="cpu", local_files_only=True)
    texts = [prefix + question["question"] for question in questions]
    texts += [leaf["retrieval_text"] for leaf in leaves]
    lengths = [len(x) for x in text_model.tokenizer(texts, truncation=False)["input_ids"]]
    if max(lengths) > text_model.max_seq_length:
        raise ValueError("BGE input would be truncated")
    vectors = normalize(text_model.encode(
        texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False, prompt="",
    ))
    nq = len(questions)
    assert vectors.shape == (len(texts), text_info["dimension"])
    metadata = {
        "clip": {"checkpoint": image_info, "tokenizer_files_sha256": TOKENIZER_HASHES,
                 "inputs": clip_inputs, "token_limit": config.text_config.max_position_embeddings,
                 "max_tokens": max(x["token_count"] for x in clip_inputs)},
        "bge": {"checkpoint": text_info, "query_prefix": prefix,
                "query_inputs": texts[:nq], "query_token_counts": lengths[:nq],
                "leaf_token_counts": lengths[nq:], "token_limit": text_model.max_seq_length,
                "max_query_tokens": max(lengths[:nq]), "max_leaf_tokens": max(lengths[nq:]),
                "leaf_format": "dialogue + '\\nMemory description: ' + full caption"},
        "truncated_inputs": 0,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__, "numpy": np.__version__},
        "device": "cpu", "torch_threads": 4,
        "timing_seconds": {"clip_load_encode": clip_seconds,
                           "bge_load_hash_encode": time.perf_counter() - start},
    }
    return np.array(clip_queries), vectors[:nq], vectors[nq:], metadata


def evaluate_question(question, masters, leaves, member_indices, clip_scores, leaf_scores, means):
    """Gold is used only here, after scores have been computed."""
    nm = len(masters)
    clip_order = np.argsort(-clip_scores, kind="stable")
    text_order = np.argsort(-means, kind="stable")
    flat_order = np.argsort(-leaf_scores, kind="stable")
    clip_ranks = {int(i): rank for rank, i in enumerate(clip_order, 1)}
    text_ranks = {int(i): rank for rank, i in enumerate(text_order, 1)}
    flat_ranks = {int(i): rank for rank, i in enumerate(flat_order, 1)}
    master_index = {m["id"]: i for i, m in enumerate(masters)}
    gold = set(question["gold_ids"])
    assert gold and len(gold) == len(question["gold_ids"])
    by_memory = {leaf["memory_id"]: i for i, leaf in enumerate(leaves)}
    assert len(by_memory) == len(leaves)
    assert gold <= set(by_memory)
    gold_indices = {by_memory[m] for m in gold}
    within_ranks = {}
    rows = []
    for i in clip_order:
        master = masters[i]
        members = member_indices[i]
        ordered = sorted(members, key=lambda j: (-leaf_scores[j], j))
        within_ranks.update({j: rank for rank, j in enumerate(ordered, 1)})
        rows.append({
            "master_rank": clip_ranks[int(i)], "master_id": master["id"], "n": len(members),
            "clip_similarity": float(clip_scores[i]), "text_mean_similarity": float(means[i]),
            "text_mean_rank": text_ranks[int(i)], "gold_count": len(gold_indices.intersection(members)),
            "leaves": [{"image_id": leaves[j]["image_id"], "filename": leaves[j]["filename"],
                        "score": float(leaf_scores[j]), "within_master_rank": within_ranks[j],
                        "is_gold": j in gold_indices} for j in members],
        })
    targets = []
    for memory_id in question["gold_ids"]:
        j = by_memory[memory_id]
        leaf = leaves[j]
        mi = master_index[leaf["master_id"]]
        targets.append({
            "memory_id": memory_id, "image_id": leaf["image_id"], "filename": leaf["filename"],
            "master_id": leaf["master_id"], "clip_master_rank": clip_ranks[mi],
            "text_mean_master_rank": text_ranks[mi], "within_master_text_rank": within_ranks[j],
            "global_leaf_text_rank": flat_ranks[j],
        })
    diagnostics = []
    for k in range(1, nm + 1):
        candidate = {j for i in clip_order[:k] for j in member_indices[i]}
        text_candidate = {j for i in text_order[:k] for j in member_indices[i]}
        # Filtering the global stable order keeps identical ties for all K.
        selected = [int(j) for j in flat_order if j in candidate][:5]
        ch = len(gold_indices.intersection(candidate))
        th = len(gold_indices.intersection(text_candidate))
        sh = len(gold_indices.intersection(selected))
        best_mean = next(int(i) for i in text_order if i in set(clip_order[:k]))
        diagnostics.append({
            "k": k, "clip_candidate_leaf_count": len(candidate),
            "clip_gold_hits": ch, "clip_any": ch > 0, "clip_all": ch == len(gold),
            "text_mean_candidate_leaf_count": len(text_candidate),
            "text_mean_gold_hits": th, "text_mean_any": th > 0, "text_mean_all": th == len(gold),
            "cascade_top5": [leaves[j]["image_id"] for j in selected],
            "cascade_gold_hits": sh, "cascade_any": sh > 0, "cascade_all": sh == len(gold),
            "mean_rerank_top1_master": masters[best_mean]["id"],
            "mean_rerank_top1_any": bool(gold_indices.intersection(member_indices[best_mean])),
        })
    top5 = [int(j) for j in flat_order[:5]]
    flat_hits = len(gold_indices.intersection(top5))
    assert diagnostics[-1]["cascade_top5"] == [leaves[j]["image_id"] for j in top5]
    return {
        "id": question["id"], "question": question["question"], "gold_targets": targets,
        "masters": rows, "flat_bge_top5": [leaves[j]["image_id"] for j in top5],
        "flat_bge_gold_hits": flat_hits, "flat_bge_any": flat_hits > 0,
        "flat_bge_all": flat_hits == len(gold), "diagnostics": diagnostics,
    }


def aggregate(rows):
    n = len(rows)
    ng = sum(len(row["gold_targets"]) for row in rows)
    coverage = []
    for ki in range(len(rows[0]["diagnostics"])):
        diags = [row["diagnostics"][ki] for row in rows]
        counts = np.array([d["clip_candidate_leaf_count"] for d in diags])
        coverage.append({
            "k": ki + 1,
            "clip_any_count": sum(d["clip_any"] for d in diags),
            "clip_all_count": sum(d["clip_all"] for d in diags),
            "clip_gold_hits": sum(d["clip_gold_hits"] for d in diags),
            "text_mean_any_count": sum(d["text_mean_any"] for d in diags),
            "text_mean_all_count": sum(d["text_mean_all"] for d in diags),
            "text_mean_gold_hits": sum(d["text_mean_gold_hits"] for d in diags),
            "candidate_leaves_mean": float(counts.mean()),
            "candidate_leaves_median": float(np.median(counts)),
            "candidate_leaves_p90": float(np.quantile(counts, 0.9)),
            "candidate_leaves_min": int(counts.min()), "candidate_leaves_max": int(counts.max()),
            "text_mean_candidate_leaves_mean": float(np.mean(
                [d["text_mean_candidate_leaf_count"] for d in diags])),
            "cascade_any_count": sum(d["cascade_any"] for d in diags),
            "cascade_all_count": sum(d["cascade_all"] for d in diags),
            "cascade_gold_hits": sum(d["cascade_gold_hits"] for d in diags),
            "mean_rerank_top1_any_count": sum(d["mean_rerank_top1_any"] for d in diags),
        })
    return {
        "n_questions": n, "n_gold_memories": ng,
        "clip_top1_any_count": coverage[0]["clip_any_count"],
        "text_mean_top1_any_count": coverage[0]["text_mean_any_count"],
        "flat_bge_top5": {"any_count": sum(row["flat_bge_any"] for row in rows),
                          "all_count": sum(row["flat_bge_all"] for row in rows),
                          "gold_hits": sum(row["flat_bge_gold_hits"] for row in rows)},
        "coverage": coverage,
        "rank_statistics": {
            name: {"nearest_gold_master_mrr": float(np.mean([
                       1 / min(t[field] for t in row["gold_targets"]) for row in rows])),
                   "gold_memory_mean_master_rank": float(np.mean([
                       t[field] for row in rows for t in row["gold_targets"]])),
                   "worst_gold_master_rank": max(t[field] for row in rows for t in row["gold_targets"])}
            for name, field in (("clip", "clip_master_rank"), ("text_mean", "text_mean_master_rank"))
        },
    }


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = summary["scopes"]["new_questions"]
    cov = data["coverage"]
    ks = [r["k"] for r in cov]
    n, ng = data["n_questions"], data["n_gold_memories"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(ks, [r["clip_all_count"] / n for r in cov], "o-", label="CLIP master rank")
    ax.plot(ks, [r["text_mean_all_count"] / n for r in cov], "s-", label="BGE mean master rank")
    ax.set_ylabel("Questions with ALL target memories covered")
    ax.set_title(f"Candidate coverage ({n} new questions)")
    ax.legend()
    ax = axes[1]
    ax.plot(ks, [r["cascade_gold_hits"] / ng for r in cov], "o-", label="CLIP Top-K -> BGE leaf Top-5")
    ax.axhline(data["flat_bge_top5"]["gold_hits"] / ng, color="#e17829", linestyle="--",
               label="Direct BGE leaf Top-5")
    ax.set_ylabel(f"Target-memory recall (denominator = {ng})")
    ax.set_title("Evidence retrieval diagnostic")
    ax.legend()
    ax = axes[2]
    ax.plot(ks, [r["candidate_leaves_mean"] for r in cov], "o-")
    ax.axhline(89, color="gray", linestyle="--")
    ax.set_ylabel("Mean number of candidate leaves")
    ax.set_title("Candidate size (not measured speedup)")
    for ax in axes:
        ax.set_xlabel("Number of candidate masters K")
        ax.set_xticks([1, 3, 5, 10, 15, 20, 25])
        ax.grid(alpha=0.2)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / f"routing_diagnostics.{ext}", dpi=180)
    plt.close(fig)


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/home_online_routing_dialogue_caption_20261020")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Use a fresh output directory.")
    start = time.perf_counter()
    graph_path = OFFLINE / "memory/graph.json"
    graph = read_json(graph_path)
    offline = read_json(OFFLINE / "protocol.json")
    source_protocol = read_json(QA_ROOT / "protocol.json")
    qa = [json.loads(line) for line in (QA_ROOT / "qa_manifest.jsonl").read_text().splitlines() if line.strip()]
    questions = [{key: q[key] for key in ("id", "question", "gold_ids", "query_image_paths")}
                 for q in qa if q["task"] == DOMAIN]
    assert len(questions) == len({q["id"] for q in questions}) == 52
    assert not any(q["query_image_paths"] for q in questions)
    names = {x["image_id"]: x["filename"] for x in read_json(OFFLINE / "input_manifest.json")["home"]}
    nodes = graph["image_nodes"]
    leaves = [{
        "image_id": node["id"], "memory_id": node["id"].rsplit("#image", 1)[0],
        "filename": names[node["id"]], "master_id": node["master_id"],
        "dialogue": node["dialogue"], "caption": node["caption"],
        "retrieval_text": leaf_retrieval_text(node),
    } for node in nodes]
    masters = graph["masters"]
    node_index = {node["id"]: i for i, node in enumerate(nodes)}
    member_indices = [[node_index[key] for key in m["member_ids"]] for m in masters]
    assert len(nodes) == len(node_index) == 89 and len(masters) == 25
    assert sorted(j for members in member_indices for j in members) == list(range(len(nodes)))
    assert all(nodes[j]["master_id"] == master["id"]
               for master, members in zip(masters, member_indices) for j in members)
    sums_path = graph_path.parent / "master_sums.npy"
    sums = np.load(sums_path, allow_pickle=False)
    centers = []
    for master in masters:
        ref = master["visual_mean_ref"]
        assert ref["path"] == sums_path.name and master["n"] == len(master["member_ids"])
        assert np.isclose(ref["scale"], 1 / master["n"], atol=1e-15, rtol=0)
        centers.append(sums[ref["row"]] * ref["scale"])
    sources = [graph_path, sums_path, graph_path.parent / "basis.npy", OFFLINE / "protocol.json",
               OFFLINE / "input_manifest.json", QA_ROOT / "qa_manifest.jsonl", QA_ROOT / "protocol.json",
               QA_ROOT / "retrieval_memories.jsonl", SINGLE_RUN / "results.json", Path(__file__),
               ROOT / "scripts/analysis/probe_caption_visual_online.py"]
    source_hashes = {str(path.resolve()): sha(path) for path in sources}
    reader = {r["id"]: r["reader_text"] for r in (
        json.loads(line) for line in (QA_ROOT / "retrieval_memories.jsonl").read_text().splitlines() if line.strip())}
    assert all(leaf["retrieval_text"] == reader[leaf["memory_id"]] for leaf in leaves)
    clip_q, bge_q, bge_leaf, encoder_info = encode_questions_and_leaves(
        questions, leaves, offline, source_protocol["query_prefix"])
    clip_scores = clip_q @ normalize(centers).T
    leaf_scores = bge_q @ bge_leaf.T
    means = np.column_stack([leaf_scores[:, members].mean(axis=1) for members in member_indices])
    assert clip_scores.shape == means.shape == (52, 25) and leaf_scores.shape == (52, 89)
    assert all(np.isfinite(x).all() for x in (clip_scores, leaf_scores, means))
    rows = [evaluate_question(q, masters, leaves, member_indices, clip_scores[i], leaf_scores[i], means[i])
            for i, q in enumerate(questions)]
    new_rows = [row for row in rows if row["id"] != f"{DOMAIN}/Q10"]
    summary = {"scopes": {"new_questions": aggregate(new_rows), "all_questions": aggregate(rows)}}
    # Reproduce the already inspected single question before reporting more questions.
    previous = read_json(SINGLE_RUN / "results.json")
    q10 = next(row for row in rows if row["id"] == previous["question_id"])
    assert [r["master_id"] for r in q10["masters"]] == [r["master_id"] for r in previous["masters"]]
    clip_error = max(abs(a["clip_similarity"] - b["clip_query_master_cosine"])
                     for a, b in zip(q10["masters"], previous["masters"]))
    text_error = max(abs(a["text_mean_similarity"] - b["leaf_text_similarity_mean"])
                     for a, b in zip(q10["masters"], previous["masters"]))
    assert max(clip_error, text_error) < 1e-6
    assert all(sha(path) == expected for path, expected in source_hashes.items())
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "inputs.json", {
        "questions": questions, "leaves": leaves,
        "masters": [{"id": m["id"], "n": m["n"], "member_ids": m["member_ids"]} for m in masters],
    })
    write_json(output / "results.json", {"questions": rows, "qa_accuracy_evaluated": False})
    write_json(output / "summary.json", summary)
    # These are similarity matrices, not retained image or text embeddings.
    np.savez_compressed(output / "score_matrices.npz",
                        clip_master=clip_scores, bge_leaf=leaf_scores, bge_master_mean=means)
    write_csv(output / "question_summary.csv", [{
        "question_id": row["id"], "question": row["question"],
        "gold_filenames": " | ".join(t["filename"] for t in row["gold_targets"]),
        "gold_masters": " | ".join(t["master_id"] for t in row["gold_targets"]),
        "gold_master_clip_ranks": " | ".join(str(t["clip_master_rank"]) for t in row["gold_targets"]),
        "gold_master_text_mean_ranks": " | ".join(str(t["text_mean_master_rank"]) for t in row["gold_targets"]),
        "clip_top5_masters": " | ".join(m["master_id"] for m in row["masters"][:5]),
        "flat_bge_top5_gold_hits": row["flat_bge_gold_hits"],
    } for row in rows])
    write_csv(output / "master_rankings.csv", [{
        "question_id": row["id"], **{k: v for k, v in master.items() if k != "leaves"},
    } for row in rows for master in row["masters"]])
    write_csv(output / "gold_locations.csv", [{
        "question_id": row["id"], "question": row["question"], **target,
    } for row in rows for target in row["gold_targets"]])
    write_csv(output / "leaf_scores.csv", [{
        "question_id": row["id"], "master_id": master["master_id"],
        "clip_master_rank": master["master_rank"], **leaf,
    } for row in rows for master in row["masters"] for leaf in master["leaves"]])
    write_csv(output / "coverage_sweep.csv", [{
        "scope": name, "n_questions": scope["n_questions"], "n_gold_memories": scope["n_gold_memories"], **r,
    } for name, scope in summary["scopes"].items() for r in scope["coverage"]])
    write_json(output / "validation.json", {
        "passed": True, "n_questions": 52, "n_new_questions": 51,
        "n_gold_occurrences": 68, "n_new_gold_occurrences": 67,
        "masters_per_question": 25, "leaves_per_question": 89,
        "all_gold_memories_mapped": True, "leaf_reader_text_exact_matches": 89,
        "q10_max_clip_error": clip_error, "q10_max_text_mean_error": text_error,
        "q10_clip_order_equal": True, "k25_cascade_equals_flat_bge_all_questions": True,
        "all_source_hashes_unchanged": True, "truncated_inputs": 0,
        "original_images_read": 0, "image_encodings": 0, "reader_calls": 0,
    })
    write_json(output / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "domain": DOMAIN,
        "question_source": "52 eligible Home questions in fixed prior manifest; Q10 kept as regression control",
        "main_scope": "51 other questions, 67 gold-memory occurrences",
        "stage1": "CLIP question text cosine to actual saved master visual means; all 25 masters",
        "stage2": "BGE question to original dialogue + full caption; arithmetic mean of all members",
        "core_order": "CLIP order preserved; global text-mean rank is a separately labeled diagnostic",
        "leaf_input": "dialogue + '\\nMemory description: ' + caption",
        "gold_policy": "only used after encoding/scoring for evaluation; no options or answers encoded",
        "diagnostics": {
            "candidate_K": list(range(1, 26)), "chosen_default_K": None,
            "cluster_reference": "global BGE member-score means; no normalized text centroid",
            "leaf_reference": "global BGE ranking of all 89 leaves, top5",
            "cascade": "CLIP topK masters, union of leaves, BGE leaf top min(5,n_candidates)",
            "mean_rerank": "diagnostic top1 by BGE mean among CLIP topK masters",
            "ties": "master and leaf original saved order; stable across all K",
            "denominators": "any/all per question; evidence recall per gold-memory occurrence, including same-cluster targets",
            "candidate_cost": "leaf counts only; actual run scores all 89; not a measured speedup",
        },
        "manual_similarity_threshold": None, "score_fusion": None, "memory_changed": False,
        "qa_accuracy_evaluated": False, "encoders": encoder_info,
        "source_sha256": source_hashes,
        "wall_seconds_before_plotting": time.perf_counter() - start,
    })
    plot_summary(output, summary)
    print(json.dumps({"output": str(output), "new_questions": summary["scopes"]["new_questions"]}))


if __name__ == "__main__":
    main()
