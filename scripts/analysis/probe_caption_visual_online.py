#!/usr/bin/env python3
"""One text query: CLIP master ranking, then mean BGE dialogue+caption similarity.

Reads a frozen caption-Ward memory. No image pixels, per-image visual vectors,
reader calls, thresholds, score fusion, or second-stage reordering are used.
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
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

OFFLINE = ROOT / "reports/home_caption_ward_visual_20261020"
QA_ROOT = ROOT / "data/independent_visual_inputs"
TOKENIZER_HASHES = {
    "vocab.json": "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
    "merges.txt": "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
    "tokenizer_config.json": "34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad",
    "special_tokens_map.json": "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def normalize(vectors):
    x = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (norms <= 0).any():
        raise ValueError("Non-finite or zero embedding")
    return x / norms


def encode_clip_question(question, metadata):
    import torch
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection

    source = Path(metadata["model_name"])
    for name, expected in {**metadata["files_sha256"], **TOKENIZER_HASHES}.items():
        if sha(source / name) != expected:
            raise ValueError(f"CLIP checkpoint/tokenizer changed: {name}")
    config = AutoConfig.from_pretrained(str(source), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    inputs = tokenizer(question, return_tensors="pt", truncation=False)
    n_tokens = inputs["input_ids"].shape[1]
    limit = config.text_config.max_position_embeddings
    if n_tokens > limit:
        raise ValueError(f"CLIP question has {n_tokens} tokens; limit is {limit}")
    model, loading = CLIPTextModelWithProjection.from_pretrained(
        str(source), config=config.text_config, local_files_only=True, output_loading_info=True,
    )
    for key in ("missing_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise ValueError(f"Incomplete CLIP text/projection weights: {key}={loading[key]}")
    unexpected = loading.get("unexpected_keys", [])
    if any(not (k.startswith(("vision_model.", "visual_projection.")) or k == "logit_scale")
           for k in unexpected):
        raise ValueError("Unexpected non-vision CLIP weights")
    model.to(device="cpu", dtype=torch.float32).eval()
    with torch.inference_mode():
        vector = model(**inputs, return_dict=True).text_embeds[0].double().cpu().numpy()
    assert vector.shape == (metadata["embedding_dim"],)
    return normalize(vector), {
        "checkpoint": metadata,
        "tokenizer_files_sha256": TOKENIZER_HASHES,
        "tokenizer_source": "https://huggingface.co/openai/clip-vit-base-patch32/tree/" + metadata["revision"],
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "pooling": "trained text projection of CLIP pooled EOT output (text_embeds)",
        "input": question,
        "input_ids": inputs["input_ids"][0].tolist(),
        "token_count": n_tokens,
        "token_limit": limit,
        "truncated_inputs": 0,
        "missing_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
        "expected_unused_vision_keys": len(unexpected),
    }


def leaf_retrieval_text(node):
    """Use the original user/assistant dialogue followed by the full caption."""
    for key in ("dialogue", "caption"):
        if not isinstance(node[key], str) or not node[key].strip():
            raise ValueError(f"Leaf {node['id']} has no {key}")
    return node["dialogue"] + "\nMemory description: " + node["caption"]


def encode_bge(question, leaf_texts, metadata, query_prefix):
    from sentence_transformers import SentenceTransformer

    source = Path(metadata["model_path"])
    for name, expected in metadata["files_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"BGE checkpoint/tokenizer changed: {name}")
    model = SentenceTransformer(str(source), device="cpu", local_files_only=True)
    inputs = [query_prefix + question, *leaf_texts]
    token_lengths = [len(t) for t in model.tokenizer(inputs, truncation=False)["input_ids"]]
    if max(token_lengths) > model.max_seq_length:
        raise ValueError("BGE input would be silently truncated")
    vectors = normalize(model.encode(
        inputs, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False, prompt="",
    ))
    assert vectors.shape == (len(inputs), metadata["dimension"])
    return vectors[0], vectors[1:], {
        **metadata,
        "query_prefix": query_prefix,
        "query_input": inputs[0],
        "leaf_text_prefix": "",
        "leaf_text_format": "original dialogue + '\\nMemory description: ' + full caption",
        "token_limit": model.max_seq_length,
        "query_token_count": token_lengths[0],
        "leaf_text_max_token_count": max(token_lengths[1:]),
        "leaf_text_token_counts": token_lengths[1:],
        "truncated_inputs": 0,
        "normalization": "float32 SentenceTransformer output; float64 L2 renormalization",
    }


def rank_masters(masters, centers, clip_query, nodes, text_vectors, text_query):
    by_id = {node["id"]: i for i, node in enumerate(nodes)}
    assert len(by_id) == len(nodes)
    member_ids = [i for master in masters for i in master["member_ids"]]
    assert len(member_ids) == len(set(member_ids)) == len(nodes)
    assert set(member_ids) == set(by_id)
    visual_scores = normalize(centers) @ clip_query
    leaf_scores = text_vectors @ text_query
    order = np.argsort(-visual_scores, kind="stable")
    rows = []
    for rank, index in enumerate(order, 1):
        master = masters[index]
        assert master["n"] == len(master["member_ids"])
        leaves = []
        for image_id in master["member_ids"]:
            i = by_id[image_id]
            node = nodes[i]
            assert node["master_id"] == master["id"]
            leaves.append({
                "image_id": image_id,
                "leaf_text_similarity": float(leaf_scores[i]),
                "dialogue": node["dialogue"],
                "caption": node["caption"],
                "retrieval_text": leaf_retrieval_text(node),
            })
        scores = [leaf["leaf_text_similarity"] for leaf in leaves]
        rows.append({
            "master_rank": rank,
            "master_id": master["id"],
            "n": master["n"],
            "approximate_visual_center": master["approximate"],
            "clip_query_master_cosine": float(visual_scores[index]),
            "leaf_text_similarity_mean": float(np.mean(scores)),
            "leaves": leaves,
        })
    return rows


def render_report(output, question, rows):
    esc = lambda x: html.escape(str(x))
    parts = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Home Q10 master 检索</title>',
        "<style>body{max-width:1150px;margin:36px auto;font:16px/1.6 sans-serif;color:#263348}"
        "table{border-collapse:collapse;width:100%}th,td{padding:7px;border-bottom:1px solid #ddd}"
        "th,td{text-align:left}details{padding:12px;border:1px solid #ddd;margin:14px 0}"
        "summary{cursor:pointer}article{border-top:1px solid #ddd;padding:10px}"
        "pre{white-space:pre-wrap;font:inherit}small{color:#617086}</style>",
        "<h1>Home Q10：master 定位与对话 + 完整 caption 均值</h1>",
        f"<p>{esc(question)}</p>",
        "<p>按 CLIP 问题文本—master 视觉中心余弦降序排列。"
        "BGE 分数为问题与每个叶子的「原始对话 + 完整 caption」的余弦，"
        "两部分拼接后一起编码，再对簇内所有成员取算术平均。"
        "两种分数分别报告。未筛选、融合、重排或生成答案。</p>",
        "<p>文件名与目标标记仅在完成评分后添加，用于检查。查询未使用选项或答案。</p>",
        "<table><tr><th>顺序</th><th>Master</th><th>成员数</th><th>CLIP 余弦</th>"
        "<th>对话 + 完整 caption 相似度均值</th><th>目标证据</th></tr>",
    ]
    for row in rows:
        parts.append(f"<tr><td>M{row['master_rank']}</td><td>{esc(row['master_id'])}</td>"
                     f"<td>{row['n']}</td><td>{row['clip_query_master_cosine']:.6f}</td>"
                     f"<td>{row['leaf_text_similarity_mean']:.6f}</td>"
                     f"<td>{'含目标图片' if row['contains_gold_memory'] else ''}</td></tr>")
    parts.append("</table>")
    for row in rows:
        parts.append(f"<details><summary>M{row['master_rank']} · {esc(row['master_id'])}"
                     f" · {row['n']} 张 · 叶子文本均值 {row['leaf_text_similarity_mean']:.6f}</summary>")
        for leaf in row["leaves"]:
            parts.append(f"<article><b>{esc(leaf['filename'])}</b> · "
                         f"{leaf['leaf_text_similarity']:.6f}"
                         f"{' · 目标证据' if leaf['is_gold_memory'] else ''}<br>"
                         f"<small>{esc(leaf['image_id'])}</small>"
                         f"<pre>{esc(leaf['retrieval_text'])}</pre></article>")
        parts.append("</details>")
    parts.append("</html>")
    (output / "report.html").write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-run", type=Path, default=OFFLINE)
    parser.add_argument("--qa-manifest", type=Path, default=QA_ROOT / "qa_manifest.jsonl")
    parser.add_argument("--query-protocol", type=Path, default=QA_ROOT / "protocol.json")
    parser.add_argument("--question-id", default="Home_Renovation_Interior_Design/Q10")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/home_q10_master_routing_dialogue_caption_20261020")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Use a fresh output directory; existing results will not be overwritten.")
    start = time.perf_counter()
    run = args.offline_run.resolve()
    graph_path = run / "memory/graph.json"
    graph = read_json(graph_path)
    offline = read_json(run / "protocol.json")
    source_protocol = read_json(args.query_protocol)
    matches = [json.loads(line) for line in args.qa_manifest.read_text().splitlines()
               if line.strip() and json.loads(line)["id"] == args.question_id]
    if len(matches) != 1:
        raise ValueError("Question ID missing or duplicated")
    record = matches[0]
    question = record["question"]
    if record["query_image_paths"]:
        raise ValueError("This text-only probe requires a query without attached images.")
    assert all(node["id"].startswith(record["task"] + "/") for node in graph["image_nodes"])
    sidecars = {}
    centers = []
    for master in graph["masters"]:
        ref = master["visual_mean_ref"]
        path = (graph_path.parent / ref["path"]).resolve()
        if path.parent != graph_path.parent:
            raise ValueError("Visual sidecar must be in the memory directory")
        if path not in sidecars:
            sidecars[path] = np.load(path, allow_pickle=False)
        assert np.isclose(ref["scale"], 1 / master["n"], rtol=0, atol=1e-15)
        centers.append(sidecars[path][ref["row"]] * ref["scale"])
    sources = [graph_path, graph_path.parent / graph["basis_ref"]["path"], *sidecars,
               run / "protocol.json", run / "input_manifest.json",
               args.qa_manifest, args.query_protocol, Path(__file__)]
    source_hashes = {str(p.resolve()): sha(p) for p in sources}
    assert np.asarray(centers).shape == (len(graph["masters"]), graph["visual_dim"])

    import torch
    import transformers
    import sentence_transformers
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    clip_query, clip_metadata = encode_clip_question(question, offline["image_encoder"])
    clip_seconds = time.perf_counter() - t0
    print(json.dumps({"stage": "clip_query_encoded", "tokens": clip_metadata["token_count"]}), flush=True)
    t0 = time.perf_counter()
    leaf_texts = [leaf_retrieval_text(n) for n in graph["image_nodes"]]
    text_query, leaf_vectors, bge_metadata = encode_bge(
        question, leaf_texts,
        offline["text_encoder"], source_protocol["query_prefix"],
    )
    bge_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows = rank_masters(graph["masters"], centers, clip_query, graph["image_nodes"],
                        leaf_vectors, text_query)
    ranking_seconds = time.perf_counter() - t0

    # Evaluation annotations are deliberately attached only after scoring.
    manifest = read_json(run / "input_manifest.json")
    filenames = {r["image_id"]: r["filename"] for r in manifest["home"]}
    gold = set(record["gold_ids"])
    for row in rows:
        for leaf in row["leaves"]:
            leaf["filename"] = filenames[leaf["image_id"]]
            leaf["is_gold_memory"] = leaf["image_id"].rsplit("#image", 1)[0] in gold
        row["contains_gold_memory"] = any(x["is_gold_memory"] for x in row["leaves"])
    assert all(sha(p) == expected for p, expected in source_hashes.items())
    validation = {
        "all_source_files_unchanged": True,
        "masters_scored": len(rows),
        "leaves_scored_once": sum(r["n"] for r in rows),
        "all_leaves_used_in_means": True,
        "strict_mean_not_normalized_text_centroid": True,
        "output_order_is_clip_descending": all(
            a["clip_query_master_cosine"] >= b["clip_query_master_cosine"]
            for a, b in zip(rows, rows[1:])),
        "max_mean_recomputation_error": max(
            abs(r["leaf_text_similarity_mean"] - sum(x["leaf_text_similarity"] for x in r["leaves"]) / r["n"])
            for r in rows),
        "clip_dimension": len(clip_query),
        "bge_dimension": len(text_query),
        "truncated_inputs": 0,
        "original_images_read": 0,
        "image_encodings": 0,
        "persisted_leaf_visual_embeddings": 0,
        "gold_options_used_in_encoder_inputs": False,
        "dialogue_in_all_leaf_inputs": True,
        "timestamps_used_in_encoder_inputs": False,
        "reader_calls": 0,
    }
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question_id": record["id"],
        "question": question,
        "query_image_paths": [],
        "stage1": "cos(CLIP_text(question), saved_visual_sum/n); rank all masters descending",
        "stage2": "arithmetic mean of cos(BGE(prefix+question), BGE(dialogue + '\\nMemory description: ' + full_caption)) over ALL members",
        "stage2_execution": "combined leaf text encoding batched once; means traversed in CLIP master order",
        "leaf_text_mode": "dialogue_full_caption",
        "ties": "stable saved master order",
        "threshold": None,
        "score_fusion": None,
        "stage2_rerank": False,
        "early_stopping": False,
        "retrieval_encoder_inputs": "question only; BGE adds its fixed query instruction; each leaf uses original user/assistant dialogue + full caption",
        "evaluation_annotations": "gold IDs and filenames added after ranking; options/answer not used",
        "memory_is_frozen": True,
        "visual_centers": "actual saved 2D-stream state; approximate after partial reassignment; no oracle center",
        "clip": clip_metadata,
        "bge": bge_metadata,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__, "numpy": np.__version__},
        "device": "cpu",
        "torch_threads": 4,
        "timing_seconds": {"clip_load_hash_encode": clip_seconds, "bge_load_hash_encode": bge_seconds,
                           "ranking_and_means": ranking_seconds,
                           "total_before_artifact_write": time.perf_counter() - start},
        "source_sha256": source_hashes,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "results.json", {
        "question_id": record["id"], "question": question,
        "leaf_text_mode": "dialogue_full_caption",
        "masters": rows, "gold_master_ranks": [r["master_rank"] for r in rows if r["contains_gold_memory"]],
        "qa_accuracy_evaluated": False,
    })
    write_json(output / "protocol.json", protocol)
    write_json(output / "validation.json", validation)
    master_fields = ["master_rank", "master_id", "n", "clip_query_master_cosine",
                     "leaf_text_similarity_mean", "approximate_visual_center", "contains_gold_memory"]
    with (output / "master_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=master_fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in master_fields} for row in rows)
    leaf_fields = ["master_rank", "master_id", "image_id", "filename",
                   "leaf_text_similarity", "is_gold_memory", "dialogue", "caption", "retrieval_text"]
    with (output / "leaf_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=leaf_fields)
        writer.writeheader()
        for row in rows:
            writer.writerows({"master_rank": row["master_rank"], "master_id": row["master_id"], **leaf}
                             for leaf in row["leaves"])
    render_report(output, question, rows)
    print(json.dumps({"output": str(output), "validation": validation,
                      "top3": [{k: r[k] for k in master_fields} for r in rows[:3]]}), flush=True)


if __name__ == "__main__":
    main()
