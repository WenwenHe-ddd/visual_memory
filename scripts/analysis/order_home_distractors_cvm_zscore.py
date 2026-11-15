"""Reorder Home distractors by pipeline CVM plus within-master z-score.

Uses frozen query/leaf vectors and saved CLIP master choices. For each master,
applies the production ZScoreWeighting to all of its original Home leaves,
then globally orders the resulting weighted leaf scores. The answer reader and
the later small-pool original-caption Top-3 retrieval are not called here.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.memory.online_zscore import ZScoreWeighting


SOURCE = ROOT / "results/home_15_distractor_review_20261031"
FROZEN = (ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain"
          / "Home_Renovation_Interior_Design")
SCORES = (ROOT / "reports/all_master_answer_query_20261021/domains"
          / "Home_Renovation_Interior_Design/score_matrices.npz")
RIDGE = 0.03
ALPHA = 1.0


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    inputs = json.loads((FROZEN / "inputs.json").read_text())
    graph = json.loads((FROZEN / "memory/graph.json").read_text())
    plans = read_jsonl(SOURCE / "candidate_order.jsonl")
    with np.load(FROZEN / "features.npz", allow_pickle=False) as data:
        leaf_vectors = data["bge_leaf"].astype(np.float64)
    with np.load(SCORES, allow_pickle=False) as data:
        queries = data["bge_augmented_query_vectors"].astype(np.float64)
        choices = data["selected_choice_indices"].copy()
        master_indices = data["leaf_master_indices"].copy()
        reference_raw = data["selected_leaf_scores"].copy()
    questions, leaves, masters = inputs["questions"], inputs["leaves"], graph["masters"]
    if not len(questions) == len(plans) == 52:
        raise ValueError("Expected the frozen 52 Home questions")
    if leaf_vectors.shape != (89, 384) or queries.shape != (52, 4, 384):
        raise ValueError("Frozen BGE embedding shapes changed")
    if choices.shape != (52, 25) or master_indices.shape != (89,) or len(masters) != 25:
        raise ValueError("Frozen master choice/membership shapes changed")
    if any(leaves[i]["master_id"] != masters[int(mi)]["id"]
           for i, mi in enumerate(master_indices)):
        raise ValueError("Leaf/master membership order changed")
    q_index = {q["id"]: i for i, q in enumerate(questions)}
    leaf_index = {leaf["memory_id"]: i for i, leaf in enumerate(leaves)}
    if len(q_index) != 52 or len(leaf_index) != 89:
        raise ValueError("Duplicate question or memory ID")
    if {p["question_id"] for p in plans} != set(q_index):
        raise ValueError("Candidate plan questions differ from frozen questions")
    weighting = ZScoreWeighting(RIDGE, ALPHA)
    scores_by_question = {}
    largest_raw_error = 0.0
    for qid, qi in q_index.items():
        raw, white, z, correction, weighted = (np.empty(89, dtype=np.float64)
                                               for _ in range(5))
        for mi in range(25):
            indices = np.flatnonzero(master_indices == mi)
            if len(indices) != masters[mi]["n"]:
                raise ValueError("Master member count changed")
            query = queries[qi, int(choices[qi, mi])]
            result = weighting.apply(query, leaf_vectors[indices])
            for target, name in ((raw, "raw"), (white, "white"), (z, "z"),
                                 (correction, "correction"), (weighted, "weighted")):
                target[indices] = result[name]
        largest_raw_error = max(largest_raw_error,
                                float(np.max(np.abs(raw - reference_raw[qi]))))
        scores_by_question[qid] = dict(raw=raw, white=white, z=z,
                                       correction=correction, weighted=weighted)
    if largest_raw_error > 1e-6:
        raise ValueError(f"Raw expanded-query scores do not reproduce frozen reference: {largest_raw_error}")

    orders, details = [], []
    for plan in plans:
        qid = plan["question_id"]
        qscore = scores_by_question[qid]
        for condition in ("similar", "unrelated"):
            candidates = plan[condition + "_ids"]
            if len(candidates) != 15 or len(set(candidates)) != 15:
                raise ValueError(f"Expected 15 unique candidates: {qid} {condition}")
            if set(candidates) & set(plan["gold_memory_ids"]):
                raise ValueError("Gold memory is a candidate distractor")
            ordered = sorted(candidates, key=lambda mid: (
                -qscore["weighted"][leaf_index[mid]] if condition == "similar"
                else qscore["weighted"][leaf_index[mid]], mid))
            orders.append({
                "question_id": qid, "condition": condition,
                "similarity_method": "caption_visual_master_zscore_l0p03_a1",
                "candidate_ids": ordered,
                "candidate_query_similarities": [float(qscore["weighted"][leaf_index[mid]])
                                                 for mid in ordered],
            })
            original_rank = {mid: i for i, mid in enumerate(candidates, 1)}
            for rank, mid in enumerate(ordered, 1):
                i = leaf_index[mid]
                details.append({
                    "question_id": qid, "condition": condition,
                    "insertion_rank": rank,
                    "candidate_memory_id": mid,
                    "original_visual_group_rank": original_rank[mid],
                    "master_id": leaves[i]["master_id"],
                    "selected_answer_choice": "ABCD"[int(choices[q_index[qid], master_indices[i]])],
                    "raw_bge_cosine": float(qscore["raw"][i]),
                    "whitened_score": float(qscore["white"][i]),
                    "white_z": float(qscore["z"][i]),
                    "correction": float(qscore["correction"][i]),
                    "weighted_score": float(qscore["weighted"][i]),
                })
    order_path = SOURCE / "caption_visual_master_zscore_l0p03_a1_orders.jsonl"
    with order_path.open("w", encoding="utf-8") as stream:
        for row in orders:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    score_path = SOURCE / "caption_visual_master_zscore_l0p03_a1_scores.csv"
    with score_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    protocol = {
        "method": "pipeline caption_visual_master multi_path plus ZScoreWeighting",
        "lambda": RIDGE, "alpha": ALPHA,
        "query_choice": "frozen per-master CLIP-selected candidate answer statement",
        "zscore_scope": "all original leaves within each saved master, before selecting distractors",
        "score": "raw BGE cosine + alpha * population_std(raw within master) * tanh(whitened_z)",
        "global_order": "similar descending, unrelated ascending weighted leaf score within each 15-candidate group",
        "max_abs_raw_score_error_vs_frozen_reference": largest_raw_error,
        "questions": 52, "orders": len(orders), "candidate_rows": len(details),
        "review_status": "new original visual ranks 11-15 are pending human review",
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            SOURCE / "candidate_order.jsonl", FROZEN / "inputs.json", FROZEN / "features.npz",
            FROZEN / "memory/graph.json", SCORES)},
    }
    (SOURCE / "caption_visual_master_zscore_l0p03_a1_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"questions": 52, "orders": len(orders),
                      "candidate_rows": len(details),
                      "max_abs_raw_score_error": largest_raw_error}, ensure_ascii=False))


if __name__ == "__main__":
    main()
