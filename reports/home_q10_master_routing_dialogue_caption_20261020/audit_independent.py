#!/usr/bin/env python3
"""Verify dialogue+caption routing with independent raw BGE CLS inference."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "reports/home_q10_master_routing_20261020"
OFFLINE = ROOT / "reports/home_caption_ward_visual_20261020"
QA = ROOT / "data/independent_visual_inputs"
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

checks = []
errors = {}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def check(name, passed):
    checks.append({"check": name, "passed": bool(passed)})
    if not passed:
        raise AssertionError(name)


def close(name, actual, expected, tolerance=1e-11):
    value = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0.0))
    group = name.split("/")[0]
    errors[group] = max(errors.get(group, 0.0), value)
    check(name, value <= tolerance)


def normalize(value):
    value = np.asarray(value, dtype=np.float64)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def main():
    torch.set_num_threads(4)
    protocol = load(OUT / "protocol.json")
    result = load(OUT / "results.json")
    previous_protocol = load(OLD / "protocol.json")
    previous = load(OLD / "results.json")
    previous_validation = load(OLD / "validation_independent.json")
    graph = load(OFFLINE / "memory/graph.json")
    manifest = load(OFFLINE / "input_manifest.json")["home"]
    nodes = graph["image_nodes"]
    masters = {master["id"]: master for master in graph["masters"]}
    node_index = {node["id"]: i for i, node in enumerate(nodes)}
    filenames = {row["image_id"]: row["filename"] for row in manifest}
    provenance_paths = [QA / "retrieval_memories.jsonl", OLD / "results.json",
                        OLD / "protocol.json", OLD / "validation_independent.json"]
    extra_hashes = {str(path): sha(path) for path in provenance_paths}
    for path, expected in protocol["source_sha256"].items():
        check(f"source_hash_before/{path}", sha(path) == expected)
    check("prior_CLIP_independent_validation_passed", previous_validation["passed"])
    check("CLIP_protocol_unchanged", protocol["clip"] == previous_protocol["clip"])
    check("question_unchanged", result["question"] == previous["question"] == protocol["question"])
    check("question_id_unchanged", result["question_id"] == previous["question_id"] == protocol["question_id"])
    check("query_prefix_unchanged", protocol["bge"]["query_prefix"] == previous_protocol["bge"]["query_prefix"])
    check("query_input_unchanged", protocol["bge"]["query_input"] == previous_protocol["bge"]["query_input"])
    check("leaf_mode", result["leaf_text_mode"] == protocol["leaf_text_mode"] == "dialogue_full_caption")
    check("no_leaf_prefix", protocol["bge"]["leaf_text_prefix"] == "")
    check("all_25_masters", len(result["masters"]) == len(previous["masters"]) == len(masters) == 25)
    check("all_89_nodes", len(nodes) == len(node_index) == 89)
    check("stage1_order_identical",
          [row["master_id"] for row in result["masters"]] == [row["master_id"] for row in previous["masters"]])
    for i, (current, prior) in enumerate(zip(result["masters"], previous["masters"])):
        for key in ("master_rank", "master_id", "n", "approximate_visual_center", "clip_query_master_cosine"):
            check(f"stage1_exact/{i}/{key}", current[key] == prior[key])
        check(f"leaf_order_unchanged/{i}", [leaf["image_id"] for leaf in current["leaves"]]
              == [leaf["image_id"] for leaf in prior["leaves"]])

    original_records = {row["id"]: row for row in map(json.loads, (QA / "retrieval_memories.jsonl").read_text().splitlines())}
    question_record = next(row for row in map(json.loads, (QA / "qa_manifest.jsonl").read_text().splitlines())
                           if row["id"] == result["question_id"])
    check("question_from_original_manifest", result["question"] == question_record["question"])
    prefix = load(QA / "protocol.json")["query_prefix"]
    check("query_prefix_from_original_protocol", protocol["bge"]["query_prefix"] == prefix)
    check("query_without_extra_content", protocol["bge"]["query_input"] == prefix + question_record["question"])
    leaf_texts = []
    for node in nodes:
        text = node["dialogue"] + "\nMemory description: " + node["caption"]
        memory_id = node["id"].rsplit("#image", 1)[0]
        check(f"exact_original_reader_text/{node['id']}", text == original_records[memory_id]["reader_text"])
        leaf_texts.append(text)
    bge = protocol["bge"]
    model_path = Path(bge["model_path"])
    for name, expected in bge["files_sha256"].items():
        check(f"model_file_hash/{name}", sha(model_path / name) == expected)
    pooling = load(model_path / "1_Pooling/config.json")
    check("raw_CLS_pooling_is_equivalent", pooling["pooling_mode_cls_token"] is True
          and pooling["pooling_mode_mean_tokens"] is False)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device="cpu", dtype=torch.float32).eval()
    texts = [prefix + question_record["question"], *leaf_texts]
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
    check("query_token_count", lengths[0] == bge["query_token_count"] == 27)
    check("leaf_token_counts_exact", lengths[1:] == bge["leaf_text_token_counts"])
    check("no_truncation", max(lengths) == bge["leaf_text_max_token_count"] == 156
          and max(lengths) <= model.config.max_position_embeddings == bge["token_limit"] == 512)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), 9):
            batch = tokenizer(texts[start:start + 9], padding=True, truncation=False, return_tensors="pt")
            hidden = model(**batch).last_hidden_state[:, 0].double().cpu().numpy()
            vectors.extend(normalize(hidden))
    vectors = np.asarray(vectors)
    check("embedding_shape", vectors.shape == (90, 384))
    scores = vectors[1:] @ vectors[0]
    del model, vectors
    used = []
    for rank, row in enumerate(result["masters"], 1):
        master = masters[row["master_id"]]
        check(f"rank/{rank}", row["master_rank"] == rank)
        check(f"all_members/{rank}", [leaf["image_id"] for leaf in row["leaves"]] == master["member_ids"])
        check(f"member_count/{rank}", len(row["leaves"]) == row["n"] == master["n"])
        member_scores = []
        for leaf in row["leaves"]:
            key = leaf["image_id"]
            i = node_index[key]
            node = nodes[i]
            used.append(key)
            member_scores.append(scores[i])
            check(f"leaf_master/{key}", node["master_id"] == master["id"])
            check(f"leaf_dialogue/{key}", leaf["dialogue"] == node["dialogue"])
            check(f"leaf_caption/{key}", leaf["caption"] == node["caption"])
            check(f"leaf_retrieval_text/{key}", leaf["retrieval_text"] == leaf_texts[i])
            check(f"leaf_filename/{key}", leaf["filename"] == filenames[key])
            check(f"leaf_gold_annotation/{key}",
                  leaf["is_gold_memory"] == (key.rsplit("#image", 1)[0] in question_record["gold_ids"]))
            close(f"raw_BGE_leaf/{key}", leaf["leaf_text_similarity"], scores[i], 1e-6)
        close(f"raw_BGE_mean/{rank}", row["leaf_text_similarity_mean"], np.mean(member_scores), 1e-6)
        close(f"reported_mean/{rank}", row["leaf_text_similarity_mean"],
              sum(leaf["leaf_text_similarity"] for leaf in row["leaves"]) / row["n"])
    check("89_unique_leaves_scored_once", len(used) == len(set(used)) == 89 and set(used) == set(node_index))
    close("weighted_mean_identity", sum(row["n"] * row["leaf_text_similarity_mean"] for row in result["masters"]) / 89,
          scores.mean(), 1e-6)
    for name, expected_rows in (
        ("master_scores.csv", result["masters"]),
        ("leaf_scores.csv", [dict(leaf, master_rank=row["master_rank"], master_id=row["master_id"])
                             for row in result["masters"] for leaf in row["leaves"]]),
    ):
        with (OUT / name).open(encoding="utf-8-sig") as handle:
            actual_rows = list(csv.DictReader(handle))
        check(f"csv_count/{name}", len(actual_rows) == len(expected_rows))
        for i, (actual, expected) in enumerate(zip(actual_rows, expected_rows)):
            for key, value in actual.items():
                check(f"csv_field/{name}/{i}/{key}", value == str(expected[key]))
    check("no_new_selection_or_reader", protocol["stage2_rerank"] is False
          and protocol["early_stopping"] is False and protocol["threshold"] is None
          and protocol["score_fusion"] is None and result["qa_accuracy_evaluated"] is False)
    for path, expected in {**protocol["source_sha256"], **extra_hashes}.items():
        check(f"source_hash_after/{path}", sha(path) == expected)
    validation = {
        "passed": True,
        "checks_passed": len(checks),
        "masters_scored": 25,
        "leaves_scored": 89,
        "leaf_reader_text_exact_matches": 89,
        "CLIP_stage1_exactly_identical_to_prior_audited_run": True,
        "independent_BGE_method": "Raw AutoModel CLS pooling; batch size 9; float64 normalization; no production helper imports",
        "max_absolute_errors": errors,
        "max_leaf_tokens": max(lengths[1:]),
        "first_master_id": result["masters"][0]["master_id"],
        "first_master_mean": result["masters"][0]["leaf_text_similarity_mean"],
        "extra_provenance_sha256": extra_hashes,
        "timestamps_in_input": False,
        "original_images_read": 0,
        "image_encoder_calls": 0,
        "memory_modified": False,
        "checks": checks,
    }
    (OUT / "validation_independent.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in validation.items() if key not in ("checks", "extra_provenance_sha256")}, indent=2))


if __name__ == "__main__":
    main()
