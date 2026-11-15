#!/usr/bin/env python3
"""Independent CLIP full-model and raw BGE CLS verification for the Q10 probe."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "reports/home_caption_ward_visual_20261020"
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel

checks = []
errors = {}


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def check(name, value):
    checks.append({"check": name, "passed": bool(value)})
    if not value:
        raise AssertionError(name)


def close(name, actual, expected, tolerance=1e-11):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0.0))
    group = name.split("/")[0]
    errors[group] = max(errors.get(group, 0.0), error)
    check(name, error <= tolerance)


def unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def main():
    torch.set_num_threads(4)
    protocol = load(OUT / "protocol.json")
    result = load(OUT / "results.json")
    graph = load(SOURCE / "memory/graph.json")
    manifest = load(SOURCE / "input_manifest.json")["home"]
    nodes = graph["image_nodes"]
    masters = graph["masters"]
    source_sha = protocol["source_sha256"]
    for path, expected in source_sha.items():
        check(f"source_hash_before/{path}", digest(path) == expected)
    qa_path = ROOT / "data/independent_visual_inputs/qa_manifest.jsonl"
    records = [json.loads(line) for line in qa_path.read_text().splitlines()]
    question_rows = [row for row in records if row["id"] == protocol["question_id"]]
    check("unique_original_question", len(question_rows) == 1)
    record = question_rows[0]
    question = record["question"]
    check("question_exact", result["question"] == protocol["question"] == protocol["clip"]["input"] == question)
    check("no_query_images", not record["query_image_paths"] and not protocol["query_image_paths"])
    query_prefix = load(qa_path.parent / "protocol.json")["query_prefix"]
    check("bge_query_exact", protocol["bge"]["query_input"] == query_prefix + question)
    check("empty_caption_prefix", protocol["bge"]["caption_prefix"] == "")
    check("25_masters_89_leaves", len(masters) == len(result["masters"]) == 25 and len(nodes) == 89)
    node_index = {node["id"]: i for i, node in enumerate(nodes)}
    master_index = {master["id"]: i for i, master in enumerate(masters)}
    check("unique_leaf_ids", len(node_index) == 89)
    names = {row["image_id"]: row["filename"] for row in manifest}

    clip_info = protocol["clip"]
    clip_path = Path(clip_info["checkpoint"]["model_name"])
    for name, expected in {**clip_info["checkpoint"]["files_sha256"], **clip_info["tokenizer_files_sha256"]}.items():
        check(f"clip_file_hash/{name}", digest(clip_path / name) == expected)
    tokenizer = AutoTokenizer.from_pretrained(str(clip_path), local_files_only=True)
    clip_inputs = tokenizer(question, return_tensors="pt", truncation=False)
    check("clip_tokenizer_vocab", len(tokenizer) == 49408)
    check("clip_tokenizer_special_ids", tokenizer.bos_token_id == 49406 and tokenizer.eos_token_id == 49407)
    check("clip_input_ids_exact", clip_inputs["input_ids"][0].tolist() == clip_info["input_ids"])
    check("clip_eot_pooling_position",
          clip_inputs["input_ids"][0, -1].item() == 49407
          and clip_inputs["input_ids"][0].argmax().item() == len(clip_info["input_ids"]) - 1)
    model, loading = CLIPModel.from_pretrained(str(clip_path), local_files_only=True, output_loading_info=True)
    check("clip_complete_loading", not any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")))
    model.to(device="cpu", dtype=torch.float32).eval()
    check("clip_no_truncation", clip_inputs["input_ids"].shape[1] == clip_info["token_count"] == 18
          and clip_inputs["input_ids"].shape[1] <= model.config.text_config.max_position_embeddings == 77)
    with torch.inference_mode():
        pooled = model.text_model(**clip_inputs).pooler_output
        explicit = model.text_projection(pooled)[0].double().cpu().numpy()
        public = model.get_text_features(**clip_inputs).pooler_output[0].double().cpu().numpy()
    close("full_clip_projection_api_agreement", explicit, public, 1e-7)
    clip_query = unit(explicit)
    check("clip_projected_dimension", clip_query.shape == (512,))
    del model, pooled, explicit, public

    bge_info = protocol["bge"]
    bge_path = Path(bge_info["model_path"])
    for name, expected in bge_info["files_sha256"].items():
        check(f"bge_file_hash/{name}", digest(bge_path / name) == expected)
    pooling = load(bge_path / "1_Pooling/config.json")
    check("bge_CLS_pooling", pooling["pooling_mode_cls_token"] is True and pooling["pooling_mode_mean_tokens"] is False)
    bge_tokenizer = AutoTokenizer.from_pretrained(str(bge_path), local_files_only=True)
    bge_model = AutoModel.from_pretrained(str(bge_path), local_files_only=True).to(device="cpu", dtype=torch.float32).eval()
    texts = [query_prefix + question] + [node["caption"] for node in nodes]
    token_lengths = [len(ids) for ids in bge_tokenizer(texts, truncation=False)["input_ids"]]
    check("bge_query_tokens", token_lengths[0] == bge_info["query_token_count"])
    check("bge_caption_token_counts", token_lengths[1:] == bge_info["caption_token_counts"])
    check("bge_no_truncation", max(token_lengths) == bge_info["caption_max_token_count"] == 129
          and max(token_lengths) <= bge_model.config.max_position_embeddings == 512)
    embeddings = []
    # Different batch size and raw AutoModel CLS pooling avoid the production
    # SentenceTransformer encoding path.
    with torch.inference_mode():
        for begin in range(0, len(texts), 9):
            batch = bge_tokenizer(texts[begin:begin + 9], padding=True, truncation=False, return_tensors="pt")
            hidden = bge_model(**batch).last_hidden_state[:, 0].double().cpu().numpy()
            embeddings.extend(unit(hidden))
    embeddings = np.asarray(embeddings)
    check("bge_embedding_shape", embeddings.shape == (90, 384))
    scores = embeddings[1:] @ embeddings[0]
    del bge_model, embeddings

    sums = np.load(SOURCE / "memory/master_sums.npy", allow_pickle=False)
    centers = np.array([sums[m["visual_sum_ref"]["row"]] / len(m["member_ids"]) for m in masters])
    clip_scores = unit(centers) @ clip_query
    order = sorted(range(len(masters)), key=lambda i: (-clip_scores[i], i))
    check("stable_clip_order", [row["master_id"] for row in result["masters"]] == [masters[i]["id"] for i in order])
    seen = []
    group_means = []
    for rank, row in enumerate(result["masters"], 1):
        master = masters[master_index[row["master_id"]]]
        check(f"rank/{rank}", row["master_rank"] == rank)
        check(f"member_count/{rank}", row["n"] == len(row["leaves"]) == master["n"])
        check(f"exact_member_order/{rank}", [leaf["image_id"] for leaf in row["leaves"]] == master["member_ids"])
        close(f"CLIP_score/{rank}", row["clip_query_master_cosine"], clip_scores[master_index[master["id"]]], 1e-6)
        members = [node_index[key] for key in master["member_ids"]]
        mean = float(scores[members].sum() / len(members))
        group_means.append(mean)
        close(f"BGE_mean/{rank}", row["caption_similarity_mean"], mean, 1e-6)
        close(f"reported_arithmetic_mean/{rank}", row["caption_similarity_mean"],
              sum(leaf["caption_similarity"] for leaf in row["leaves"]) / len(row["leaves"]))
        check(f"mean_within_member_range/{rank}", min(scores[members]) - 1e-6 <= mean <= max(scores[members]) + 1e-6)
        for leaf in row["leaves"]:
            key = leaf["image_id"]
            node = nodes[node_index[key]]
            seen.append(key)
            check(f"leaf_caption/{key}", leaf["caption"] == node["caption"])
            check(f"leaf_master/{key}", node["master_id"] == master["id"])
            check(f"leaf_filename/{key}", leaf["filename"] == names[key])
            close(f"BGE_leaf/{key}", leaf["caption_similarity"], scores[node_index[key]], 1e-6)
    check("all_leaves_scored_exactly_once", len(seen) == len(set(seen)) == 89 and set(seen) == set(node_index))
    close("weighted_means_equal_all_leaf_mean", sum(row["n"] * mean for row, mean in zip(result["masters"], group_means)) / 89,
          scores.mean())
    close("reported_weighted_means_equal_all_leaf_mean",
          sum(row["n"] * row["caption_similarity_mean"] for row in result["masters"]) / 89, scores.mean(), 1e-6)
    for file, expected_rows in (
        ("master_scores.csv", result["masters"]),
        ("leaf_scores.csv", [dict(leaf, master_rank=row["master_rank"], master_id=row["master_id"])
                             for row in result["masters"] for leaf in row["leaves"]]),
    ):
        with (OUT / file).open(encoding="utf-8-sig") as handle:
            csv_rows = list(csv.DictReader(handle))
        check(f"csv_count/{file}", len(csv_rows) == len(expected_rows))
        for i, (csv_row, expected) in enumerate(zip(csv_rows, expected_rows)):
            for key, value in csv_row.items():
                check(f"csv_field/{file}/{i}/{key}", value == str(expected[key]))
    check("no_secondary_reordering_or_filtering", protocol["stage2_rerank"] is False
          and protocol["early_stopping"] is False and protocol["threshold"] is None and protocol["score_fusion"] is None)
    for path, expected in source_sha.items():
        check(f"source_hash_after/{path}", digest(path) == expected)
    validation = {
        "passed": True,
        "checks_passed": len(checks),
        "masters_scored": 25,
        "leaves_scored": 89,
        "independent_CLIP": "Full CLIPModel text_model plus text_projection; verified against get_text_features",
        "independent_BGE": "Raw AutoModel last_hidden_state CLS, different batch size 9, float64 normalization",
        "max_absolute_errors": errors,
        "first_master_id": result["masters"][0]["master_id"],
        "first_CLIP_cosine": float(clip_scores[order[0]]),
        "first_BGE_caption_mean": group_means[0],
        "image_encoder_calls": 0,
        "original_image_pixels_read": 0,
        "memory_modified": False,
        "checks": checks,
    }
    (OUT / "validation_independent.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in validation.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
