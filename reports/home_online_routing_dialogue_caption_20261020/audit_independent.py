#!/usr/bin/env python3
"""Independent full-domain text encoding, rank, coverage, and denominator audit."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OFFLINE = ROOT / "reports/home_caption_ward_visual_20261020"
QA = ROOT / "data/independent_visual_inputs"
DOMAIN = "Home_Renovation_Interior_Design"
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel

checks, errors = [], {}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(name, passed):
    checks.append({"check": name, "passed": bool(passed)})
    if not passed:
        raise AssertionError(name)


def close(name, actual, expected, tolerance=1e-11):
    delta = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0.0))
    family = name.split("/")[0]
    errors[family] = max(errors.get(family, 0.0), delta)
    check(name, delta <= tolerance)


def unit(value):
    value = np.asarray(value, dtype=np.float64)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def order(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def main():
    torch.set_num_threads(4)
    inputs = load(OUT / "inputs.json")
    result = load(OUT / "results.json")
    summary = load(OUT / "summary.json")
    protocol = load(OUT / "protocol.json")
    graph = load(OFFLINE / "memory/graph.json")
    matrix_file = np.load(OUT / "score_matrices.npz", allow_pickle=False)
    clip_scores, leaf_scores, means = [matrix_file[name] for name in ("clip_master", "bge_leaf", "bge_master_mean")]
    questions = [row for row in map(json.loads, (QA / "qa_manifest.jsonl").read_text().splitlines()) if row["task"] == DOMAIN]
    question_inputs = [{key: row[key] for key in ("id", "question", "gold_ids", "query_image_paths")} for row in questions]
    check("questions_exact_source", inputs["questions"] == question_inputs)
    check("question_count_52", len(questions) == len({q["id"] for q in questions}) == 52)
    check("gold_occurrences_68", sum(len(q["gold_ids"]) for q in questions) == 68)
    check("no_empty_or_duplicate_gold", all(q["gold_ids"] and len(q["gold_ids"]) == len(set(q["gold_ids"])) for q in questions))
    check("no_query_images", not any(q["query_image_paths"] for q in questions))
    check("matrix_shapes", clip_scores.shape == means.shape == (52, 25) and leaf_scores.shape == (52, 89))
    check("finite_cosine_matrices", all(np.isfinite(x).all() and np.max(np.abs(x)) <= 1 + 1e-12
                                      for x in (clip_scores, leaf_scores, means)))
    for path, expected in protocol["source_sha256"].items():
        check(f"source_hash_before/{path}", sha(path) == expected)
    nodes, masters = graph["image_nodes"], graph["masters"]
    leaves = inputs["leaves"]
    image_ids = [node["id"] for node in nodes]
    node_index = {key: i for i, key in enumerate(image_ids)}
    memory_index = {key.rsplit("#image", 1)[0]: i for i, key in enumerate(image_ids)}
    master_index = {master["id"]: i for i, master in enumerate(masters)}
    membership = [{node_index[key] for key in master["member_ids"]} for master in masters]
    names = {row["image_id"]: row["filename"] for row in load(OFFLINE / "input_manifest.json")["home"]}
    check("master_schema_exact", inputs["masters"] == [{key: master[key] for key in ("id", "n", "member_ids")} for master in masters])
    check("leaf_partition", sorted(j for group in membership for j in group) == list(range(89)))
    check("all_gold_mapped", all(set(q["gold_ids"]) <= set(memory_index) for q in questions))
    reader = {row["id"]: row["reader_text"] for row in map(json.loads, (QA / "retrieval_memories.jsonl").read_text().splitlines())}
    leaf_texts = []
    for i, (leaf, node) in enumerate(zip(leaves, nodes)):
        expected = {
            "image_id": node["id"], "memory_id": node["id"].rsplit("#image", 1)[0],
            "filename": names[node["id"]], "master_id": node["master_id"],
            "dialogue": node["dialogue"], "caption": node["caption"],
            "retrieval_text": node["dialogue"] + "\nMemory description: " + node["caption"],
        }
        check(f"exact_leaf/{i}", leaf == expected)
        check(f"original_reader_text/{i}", leaf["retrieval_text"] == reader[leaf["memory_id"]])
        leaf_texts.append(expected["retrieval_text"])
    for m, members in enumerate(membership):
        close(f"member_mean/{m}", means[:, m], leaf_scores[:, sorted(members)].sum(axis=1) / len(members))

    # Full CLIPModel is independent of the production text-tower wrapper.
    cm = protocol["encoders"]["clip"]
    cp = Path(cm["checkpoint"]["model_name"])
    for name, expected in {**cm["checkpoint"]["files_sha256"], **cm["tokenizer_files_sha256"]}.items():
        check(f"CLIP_hash/{name}", sha(cp / name) == expected)
    ct = AutoTokenizer.from_pretrained(str(cp), local_files_only=True)
    clip, loading = CLIPModel.from_pretrained(str(cp), local_files_only=True, output_loading_info=True)
    check("CLIP_complete_loading", not any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")))
    clip.to(device="cpu", dtype=torch.float32).eval()
    clip_queries = []
    with torch.inference_mode():
        for i, question in enumerate(questions):
            tokens = ct(question["question"], return_tensors="pt", truncation=False)
            token_ids = tokens["input_ids"][0].tolist()
            check(f"CLIP_input/{i}", cm["inputs"][i] ==
                  {"id": question["id"], "text": question["question"], "input_ids": token_ids, "token_count": len(token_ids)})
            check(f"CLIP_no_truncation/{i}", len(token_ids) <= clip.config.text_config.max_position_embeddings == 77)
            check(f"CLIP_EOT/{i}", token_ids[-1] == 49407 and token_ids.index(max(token_ids)) == len(token_ids) - 1)
            projected = clip.text_projection(clip.text_model(**tokens).pooler_output)
            clip_queries.append(unit(projected[0].double().cpu().numpy()))
    sums = np.load(OFFLINE / "memory/master_sums.npy", allow_pickle=False)
    centers = np.array([sums[m["visual_sum_ref"]["row"]] / m["n"] for m in masters])
    independent_clip = np.asarray(clip_queries) @ unit(centers).T
    close("independent_CLIP_matrix", clip_scores, independent_clip, 1e-6)
    check("independent_CLIP_orders", all(order(a) == order(b) for a, b in zip(clip_scores, independent_clip)))
    del clip, clip_queries

    # Direct AutoModel CLS extraction avoids production SentenceTransformer.
    bm = protocol["encoders"]["bge"]
    bp = Path(bm["checkpoint"]["model_path"])
    for name, expected in bm["checkpoint"]["files_sha256"].items():
        check(f"BGE_hash/{name}", sha(bp / name) == expected)
    pooling = load(bp / "1_Pooling/config.json")
    check("BGE_CLS_pooling", pooling["pooling_mode_cls_token"] and not pooling["pooling_mode_mean_tokens"])
    prefix = load(QA / "protocol.json")["query_prefix"]
    query_texts = [prefix + question["question"] for question in questions]
    check("BGE_query_inputs_exact", bm["query_inputs"] == query_texts and bm["query_prefix"] == prefix)
    bt = AutoTokenizer.from_pretrained(str(bp), local_files_only=True)
    bge = AutoModel.from_pretrained(str(bp), local_files_only=True).to(device="cpu", dtype=torch.float32).eval()
    texts = query_texts + leaf_texts
    lengths = [len(ids) for ids in bt(texts, truncation=False)["input_ids"]]
    check("BGE_token_counts", bm["query_token_counts"] == lengths[:52] and bm["leaf_token_counts"] == lengths[52:])
    check("BGE_no_truncation", max(lengths) <= bm["token_limit"] == bge.config.max_position_embeddings == 512)
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(texts), 11):
            batch = bt(texts[start:start + 11], padding=True, truncation=False, return_tensors="pt")
            encoded.extend(unit(bge(**batch).last_hidden_state[:, 0].double().cpu().numpy()))
    encoded = np.asarray(encoded)
    check("BGE_vector_dimensions", encoded.shape == (141, 384))
    independent_leaf = encoded[:52] @ encoded[52:].T
    independent_means = np.column_stack([independent_leaf[:, sorted(group)].mean(axis=1) for group in membership])
    close("independent_BGE_leaf_matrix", leaf_scores, independent_leaf, 1e-6)
    close("independent_BGE_mean_matrix", means, independent_means, 1e-6)
    check("independent_BGE_leaf_orders", all(order(a) == order(b) for a, b in zip(leaf_scores, independent_leaf)))
    check("independent_BGE_master_orders", all(order(a) == order(b) for a, b in zip(means, independent_means)))
    del bge, encoded

    # Reconstruct every rank and diagnostic from source gold and score matrices.
    reconstructed = []
    for qi, (question, output) in enumerate(zip(questions, result["questions"])):
        qid = question["id"]
        co, to, lo = order(clip_scores[qi]), order(means[qi]), order(leaf_scores[qi])
        cr = {m: i + 1 for i, m in enumerate(co)}
        tr = {m: i + 1 for i, m in enumerate(to)}
        lr = {j: i + 1 for i, j in enumerate(lo)}
        gold = {memory_index[key] for key in question["gold_ids"]}
        within = {j: r + 1 for group in membership
                  for r, j in enumerate(sorted(group, key=lambda j: (-leaf_scores[qi, j], j)))}
        check(f"question_identity/{qid}", output["id"] == qid and output["question"] == question["question"])
        check(f"all_CLIP_order/{qid}", [m["master_id"] for m in output["masters"]] == [masters[m]["id"] for m in co])
        expected_targets = []
        for memory_id in question["gold_ids"]:
            j = memory_index[memory_id]
            m = master_index[nodes[j]["master_id"]]
            expected_targets.append({
                "memory_id": memory_id, "image_id": image_ids[j], "filename": names[image_ids[j]],
                "master_id": masters[m]["id"], "clip_master_rank": cr[m], "text_mean_master_rank": tr[m],
                "within_master_text_rank": within[j], "global_leaf_text_rank": lr[j],
            })
        check(f"gold_locations/{qid}", output["gold_targets"] == expected_targets)
        for m, row in zip(co, output["masters"]):
            expected_row = {
                "master_rank": cr[m], "master_id": masters[m]["id"], "n": len(membership[m]),
                "clip_similarity": float(clip_scores[qi, m]), "text_mean_similarity": float(means[qi, m]),
                "text_mean_rank": tr[m], "gold_count": len(gold & membership[m]),
                "leaves": [{"image_id": key, "filename": names[key], "score": float(leaf_scores[qi, node_index[key]]),
                            "within_master_rank": within[node_index[key]], "is_gold": node_index[key] in gold}
                           for key in masters[m]["member_ids"]],
            }
            check(f"all_master_fields/{qid}/{m}", row == expected_row)
        flat = [image_ids[j] for j in lo[:5]]
        fh = len(gold & set(lo[:5]))
        for key, value in {"flat_bge_top5": flat, "flat_bge_gold_hits": fh, "flat_bge_any": fh > 0,
                           "flat_bge_all": fh == len(gold)}.items():
            check(f"flat/{qid}/{key}", output[key] == value)
        independent_diagnostics = []
        for k in range(1, 26):
            candidates = set().union(*(membership[m] for m in co[:k]))
            tcandidates = set().union(*(membership[m] for m in to[:k]))
            selected = sorted(candidates, key=lambda j: (-leaf_scores[qi, j], j))[:5]
            reranked = min(co[:k], key=lambda m: (-means[qi, m], m))
            ch, th, sh = len(gold & candidates), len(gold & tcandidates), len(gold & set(selected))
            diag = {
                "k": k, "clip_candidate_leaf_count": len(candidates), "clip_gold_hits": ch,
                "clip_any": ch > 0, "clip_all": ch == len(gold),
                "text_mean_candidate_leaf_count": len(tcandidates), "text_mean_gold_hits": th,
                "text_mean_any": th > 0, "text_mean_all": th == len(gold),
                "cascade_top5": [image_ids[j] for j in selected], "cascade_gold_hits": sh,
                "cascade_any": sh > 0, "cascade_all": sh == len(gold),
                "mean_rerank_top1_master": masters[reranked]["id"],
                "mean_rerank_top1_any": bool(gold & membership[reranked]),
            }
            check(f"all_K_diagnostics/{qid}/{k}", output["diagnostics"][k - 1] == diag)
            independent_diagnostics.append(diag)
        check(f"K25_boundary/{qid}", independent_diagnostics[-1]["cascade_top5"] == flat
              and independent_diagnostics[-1]["clip_candidate_leaf_count"] == 89)
        reconstructed.append(dict(id=qid, targets=expected_targets, gold_n=len(gold), flat_hits=fh,
                                  diagnostics=independent_diagnostics))

    # Aggregate independently with explicit question and evidence denominators.
    for scope, expected_n, expected_gold in (("new_questions", 51, 67), ("all_questions", 52, 68)):
        subset = [row for row in reconstructed if scope == "all_questions" or row["id"] != DOMAIN + "/Q10"]
        saved = summary["scopes"][scope]
        check(f"scope_denominators/{scope}", len(subset) == saved["n_questions"] == expected_n
              and sum(row["gold_n"] for row in subset) == saved["n_gold_memories"] == expected_gold)
        flat = {"any_count": sum(row["flat_hits"] > 0 for row in subset),
                "all_count": sum(row["flat_hits"] == row["gold_n"] for row in subset),
                "gold_hits": sum(row["flat_hits"] for row in subset)}
        check(f"flat_summary/{scope}", saved["flat_bge_top5"] == flat)
        for ki in range(25):
            ds = [row["diagnostics"][ki] for row in subset]
            counts = [d["clip_candidate_leaf_count"] for d in ds]
            expected = {
                "k": ki + 1,
                "clip_any_count": sum(d["clip_any"] for d in ds),
                "clip_all_count": sum(d["clip_all"] for d in ds),
                "clip_gold_hits": sum(d["clip_gold_hits"] for d in ds),
                "text_mean_any_count": sum(d["text_mean_any"] for d in ds),
                "text_mean_all_count": sum(d["text_mean_all"] for d in ds),
                "text_mean_gold_hits": sum(d["text_mean_gold_hits"] for d in ds),
                "candidate_leaves_mean": sum(counts) / len(counts),
                "candidate_leaves_median": float(np.median(counts)),
                "candidate_leaves_p90": float(np.percentile(counts, 90)),
                "candidate_leaves_min": min(counts), "candidate_leaves_max": max(counts),
                "text_mean_candidate_leaves_mean": np.mean([d["text_mean_candidate_leaf_count"] for d in ds]),
                "cascade_any_count": sum(d["cascade_any"] for d in ds),
                "cascade_all_count": sum(d["cascade_all"] for d in ds),
                "cascade_gold_hits": sum(d["cascade_gold_hits"] for d in ds),
                "mean_rerank_top1_any_count": sum(d["mean_rerank_top1_any"] for d in ds),
            }
            for key, value in expected.items():
                close(f"aggregate/{scope}/{ki + 1}/{key}", saved["coverage"][ki][key], value)
        check(f"top1_aliases/{scope}", saved["clip_top1_any_count"] == saved["coverage"][0]["clip_any_count"]
              and saved["text_mean_top1_any_count"] == saved["coverage"][0]["text_mean_any_count"])
        for name, field in (("clip", "clip_master_rank"), ("text_mean", "text_mean_master_rank")):
            ranks = [target[field] for row in subset for target in row["targets"]]
            expected = {
                "nearest_gold_master_mrr": sum(1 / min(target[field] for target in row["targets"]) for row in subset) / len(subset),
                "gold_memory_mean_master_rank": sum(ranks) / len(ranks),
                "worst_gold_master_rank": max(ranks),
            }
            for key, value in expected.items():
                close(f"rank_statistics/{scope}/{name}/{key}", saved["rank_statistics"][name][key], value)
    for filename, count in (("master_rankings.csv", 1300), ("leaf_scores.csv", 4628),
                            ("gold_locations.csv", 68), ("question_summary.csv", 52), ("coverage_sweep.csv", 50)):
        with (OUT / filename).open(encoding="utf-8-sig") as handle:
            csv_rows = list(csv.DictReader(handle))
        check(f"CSV_coverage/{filename}", len(csv_rows) == count)
    previous = load(ROOT / "reports/home_q10_master_routing_dialogue_caption_20261020/results.json")
    q10 = next(row for row in result["questions"] if row["id"] == previous["question_id"])
    check("Q10_CLIP_order", [row["master_id"] for row in q10["masters"]] == [row["master_id"] for row in previous["masters"]])
    close("Q10_CLIP_scores", [row["clip_similarity"] for row in q10["masters"]],
          [row["clip_query_master_cosine"] for row in previous["masters"]], 1e-6)
    close("Q10_BGE_means", [row["text_mean_similarity"] for row in q10["masters"]],
          [row["leaf_text_similarity_mean"] for row in previous["masters"]], 1e-6)
    check("diagnostic_K_unselected", protocol["diagnostics"]["candidate_K"] == list(range(1, 26))
          and protocol["diagnostics"]["chosen_default_K"] is None)
    check("no_QA_claim", result["qa_accuracy_evaluated"] is False and protocol["qa_accuracy_evaluated"] is False)
    paired = {"more": [], "fewer": [], "same": []}
    deltas = []
    for row in reconstructed:
        if row["id"] == DOMAIN + "/Q10":
            continue
        delta = row["diagnostics"][2]["cascade_gold_hits"] - row["flat_hits"]
        paired["more" if delta > 0 else "fewer" if delta < 0 else "same"].append(row["id"].split("/")[-1])
        deltas.append(delta)
    check("K3_more_target_questions", set(paired["more"]) == {"Q4", "Q53", "Q54", "Q56"})
    check("K3_fewer_target_questions", set(paired["fewer"]) == {"Q40", "Q45", "Q46", "Q60", "Q61"})
    check("K3_same_target_count_questions", len(paired["same"]) == 42)
    check("K3_paired_delta", sum(deltas) == -1 and set(deltas) == {-1, 0, 1})
    for path, expected in protocol["source_sha256"].items():
        check(f"source_hash_after/{path}", sha(path) == expected)
    validation = {
        "passed": True, "checks_passed": len(checks),
        "n_questions": 52, "n_new_questions": 51, "gold_occurrences": 68, "new_gold_occurrences": 67,
        "independent_CLIP_queries_encoded": 52,
        "independent_BGE_queries_encoded": 52, "independent_BGE_leaf_texts_encoded": 89,
        "independent_encoding": "Full CLIPModel text tower and raw AutoModel BGE CLS; no production encoder imports",
        "all_1300_question_K_diagnostics_exact": True,
        "all_52_K25_top5_boundaries_exact": True,
        "all_68_gold_locations_exact": True,
        "scope_denominators_verified": True,
        "K3_vs_flat_paired_gold_counts": paired,
        "max_absolute_errors": errors,
        "original_images_read": 0, "image_encoder_calls": 0, "memory_modified": False,
        "checks": checks,
    }
    (OUT / "validation_independent.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in validation.items() if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
