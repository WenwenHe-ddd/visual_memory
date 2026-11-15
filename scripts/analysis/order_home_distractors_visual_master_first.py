"""Custom visual-master-first ablation, NOT pipeline caption_visual_master.

Stage 1: CLIP question text versus every saved visual master center.
Stage 2: within each master, CLIP chooses a candidate answer statement and BGE
scores its leaves using the expanded query. The complete 15-image order is
lexicographic: master rank first, leaf BGE cosine second. The production pipeline
instead globally ranks the BGE leaf scores across all masters. All masters remain
available so every reviewed candidate receives a position.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/home_15_distractor_review_20261031"
FEATURES = (ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain"
            / "Home_Renovation_Interior_Design")
CVM = (ROOT / "results/online_memory_path_validation_20261023"
       / "Home_Renovation_Interior_Design/caption_visual_master/retrieval_results.json")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    plans = read_jsonl(SOURCE / "candidate_order.jsonl")
    report = json.loads(CVM.read_text())
    inputs = json.loads((FEATURES / "inputs.json").read_text())
    graph = json.loads((FEATURES / "memory/graph.json").read_text())
    if not len(plans) == len(report["questions"]) == len(inputs["questions"]):
        raise ValueError("Home question counts disagree")
    if len(plans) != 52 or len(graph["masters"]) != 25:
        raise ValueError("Expected 52 questions and 25 saved visual masters")
    if report["memory"] != "caption_visual_master" or report["retrieval"] != "multi_path":
        raise ValueError("Saved leaf scores are not from caption_visual_master multi_path")
    saved_graph_sha = next(expected for name, expected in report["source_sha256"].items()
                           if name.endswith("caption_visual_master/graph.json"))
    if sha(FEATURES / "memory/graph.json") != saved_graph_sha:
        raise ValueError("Visual-first master graph differs from expanded-query CVM graph")
    with np.load(FEATURES / "features.npz", allow_pickle=False) as features:
        clip = features["clip_master"].astype(np.float64)
        clip_order = features["clip_order"]
    if clip.shape != (52, 25) or clip_order.shape != (52, 25):
        raise ValueError("Incorrect question-to-master CLIP score matrix shape")
    if not np.array_equal(clip_order, np.argsort(-clip, axis=1, kind="stable")):
        raise ValueError("Saved visual master ordering does not match CLIP scores")
    masters = [m["id"] for m in graph["masters"]]
    qindices = {q["id"]: i for i, q in enumerate(inputs["questions"])}
    questions = {q["question_id"]: q for q in report["questions"]}
    if set(qindices) != set(questions) or set(qindices) != {p["question_id"] for p in plans}:
        raise ValueError("Question cohorts differ")
    leaf_master = {n["id"].rsplit("#image", 1)[0]: n["master_id"]
                   for n in graph["image_nodes"]}
    if len(leaf_master) != 89:
        raise ValueError("Expected 89 distinct Home memory leaves")
    orders, details = [], []
    for plan in plans:
        qid = plan["question_id"]
        qi = qindices[qid]
        q = questions[qid]
        if q["question"] != inputs["questions"][qi]["question"]:
            raise ValueError(f"Question text differs: {qid}")
        rank_by_master = {masters[int(mi)]: rank for rank, mi in enumerate(clip_order[qi], 1)}
        visual_by_master = {mid: float(clip[qi, i]) for i, mid in enumerate(masters)}
        text_by_leaf = {leaf["memory_id"]: leaf for leaf in q["leaf_scores"]}
        choice_by_master = {group["group_id"]: group["choice"] for group in q["groups"]}
        if set(text_by_leaf) != set(leaf_master) or set(choice_by_master) != set(masters):
            raise ValueError(f"CVM leaf/master coverage differs: {qid}")
        if any(text_by_leaf[mid]["group_id"] != leaf_master[mid] for mid in leaf_master):
            raise ValueError(f"CVM master membership differs: {qid}")
        for condition in ("similar", "unrelated"):
            candidate_ids = plan[condition + "_ids"]
            if len(candidate_ids) != 15 or len(set(candidate_ids)) != 15:
                raise ValueError(f"Expected 15 unique candidates: {qid} {condition}")
            if set(candidate_ids) & set(plan["gold_memory_ids"]):
                raise ValueError(f"Gold image selected as distractor: {qid}")
            if condition == "similar":
                sort_key = lambda mid: (rank_by_master[leaf_master[mid]],
                                        -text_by_leaf[mid]["raw_score"], mid)
            else:
                sort_key = lambda mid: (-rank_by_master[leaf_master[mid]],
                                        text_by_leaf[mid]["raw_score"], mid)
            ordered = sorted(candidate_ids, key=sort_key)
            orders.append({
                "question_id": qid,
                "condition": condition,
                "similarity_method": "caption_visual_master_visual_first",
                "candidate_ids": ordered,
                "candidate_query_similarities": [text_by_leaf[mid]["raw_score"] for mid in ordered],
                "candidate_master_ranks": [rank_by_master[leaf_master[mid]] for mid in ordered],
            })
            visual_original_rank = {mid: i for i, mid in enumerate(candidate_ids, 1)}
            for position, mid in enumerate(ordered, 1):
                master = leaf_master[mid]
                details.append({
                    "question_id": qid,
                    "condition": condition,
                    "insertion_rank": position,
                    "candidate_memory_id": mid,
                    "original_visual_group_rank": visual_original_rank[mid],
                    "master_id": master,
                    "question_master_clip_rank": rank_by_master[master],
                    "question_master_clip_cosine": visual_by_master[master],
                    "master_selected_answer_choice": choice_by_master[master],
                    "expanded_query_leaf_bge_cosine": text_by_leaf[mid]["raw_score"],
                })
    order_path = SOURCE / "caption_visual_master_visual_first_orders.jsonl"
    with order_path.open("w", encoding="utf-8") as stream:
        for row in orders:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    scores_path = SOURCE / "caption_visual_master_visual_first_scores.csv"
    with scores_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    protocol = {
        "matches_pipeline_caption_visual_master": False,
        "variant_name": "custom question-to-master CLIP rank then within-master expanded BGE",
        "stage_1": "CLIP cosine of question text and saved visual master center; all 25 masters ranked",
        "stage_2": "per master choose candidate answer statement by CLIP; append it to question; BGE cosine with each leaf's dialogue plus full caption",
        "total_order": "similar: visual master rank ascending then expanded-query BGE cosine descending; unrelated: reverse both",
        "master_cutoff": None,
        "candidate_scope": "same 15 similar and 15 unrelated candidate IDs per question as prior experiment",
        "zscore": "unused; saved raw_score is the uncorrected expanded-query BGE cosine",
        "questions": 52,
        "candidate_score_rows": len(details),
        "review_status": "new original visual ranks 11-15 are pending human review",
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            SOURCE / "candidate_order.jsonl", CVM, FEATURES / "inputs.json",
            FEATURES / "features.npz", FEATURES / "memory/graph.json")},
    }
    (SOURCE / "caption_visual_master_visual_first_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"orders": len(orders), "candidate_score_rows": len(details)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
