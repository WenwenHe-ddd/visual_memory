"""Order a prepared domain distractor pool by pipeline CVM raw and z-score scores.

Both methods use the frozen per-master CLIP-selected answer statement and the
resulting BGE query-to-leaf cosine. Z-score applies to all leaves in each
master before ranking candidates; it does not recompute scores on the small pool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.memory.online_zscore import ZScoreWeighting


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(task: str, source: Path) -> None:
    frozen = ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain" / task
    scores_path = ROOT / "reports/all_master_answer_query_20261021/domains" / task / "score_matrices.npz"
    report_path = (ROOT / "results/online_memory_path_validation_20261023" / task
                   / "caption_visual_master/retrieval_results.json")
    plans_path = source / "candidate_order.jsonl"
    plans = read_jsonl(plans_path)
    report = json.loads(report_path.read_text())
    if report["memory"] != "caption_visual_master" or report["retrieval"] != "multi_path":
        raise ValueError("Saved result is not the pipeline CVM multi_path run")
    for name, expected in report["source_sha256"].items():
        path = Path(name)
        if path.name in {"memory.json", "leaves.json", "graph.json", "master_sums.npy",
                         "candidates.jsonl", "inputs.json"} and sha(path) != expected:
            raise ValueError(f"Frozen CVM input changed: {path}")
    inputs = json.loads((frozen / "inputs.json").read_text())
    graph = json.loads((frozen / "memory/graph.json").read_text())
    questions, leaves, masters = inputs["questions"], inputs["leaves"], graph["masters"]
    if not plans or len(questions) != len(report["questions"]):
        raise ValueError("Frozen CVM question cohorts differ or candidate plans are empty")
    if len(leaves) != report["images"] or len({leaf["memory_id"] for leaf in leaves}) != len(leaves):
        raise ValueError("Leaf cohort differs")
    qi_by_id = {row["id"]: i for i, row in enumerate(questions)}
    li_by_id = {row["memory_id"]: i for i, row in enumerate(leaves)}
    plan_qids = [p["question_id"] for p in plans]
    if (len(qi_by_id) != len(questions) or len(set(plan_qids)) != len(plans)
            or not set(plan_qids) <= set(qi_by_id)):
        raise ValueError("Candidate question IDs are duplicated or absent from frozen CVM questions")
    with np.load(frozen / "features.npz", allow_pickle=False) as data:
        leaf_vectors = data["bge_leaf"].astype(np.float64)
    with np.load(scores_path, allow_pickle=False) as data:
        augmented_queries = data["bge_augmented_query_vectors"].astype(np.float64)
        choices = data["selected_choice_indices"].copy()
        master_indices = data["leaf_master_indices"].copy()
        reference_raw = data["selected_leaf_scores"].copy()
    nq, nl, nm = len(questions), len(leaves), len(masters)
    if (leaf_vectors.shape != (nl, 384) or augmented_queries.shape != (nq, 4, 384)
            or choices.shape != (nq, nm) or master_indices.shape != (nl,)
            or reference_raw.shape != (nq, nl)):
        raise ValueError("Frozen feature dimensions changed")
    if any(leaves[i]["master_id"] != masters[int(mi)]["id"]
           for i, mi in enumerate(master_indices)):
        raise ValueError("Leaf/master membership changed")
    by_qid = {row["question_id"]: row for row in report["questions"]}
    if set(by_qid) != set(qi_by_id):
        raise ValueError("CVM report questions differ")
    weighting = ZScoreWeighting(.03, 1.0)
    scored = {}
    max_reference_error = 0.0
    max_report_error = 0.0
    for qid, qi in qi_by_id.items():
        raw = np.empty(nl, dtype=np.float64)
        weighted = np.empty(nl, dtype=np.float64)
        for mi in range(nm):
            indices = np.flatnonzero(master_indices == mi)
            if len(indices) != masters[mi]["n"]:
                raise ValueError("Saved master member count differs")
            query = augmented_queries[qi, int(choices[qi, mi])]
            result = weighting.apply(query, leaf_vectors[indices])
            raw[indices] = result["raw"]
            weighted[indices] = result["weighted"]
        max_reference_error = max(max_reference_error,
                                  float(np.max(np.abs(raw - reference_raw[qi]))))
        report_raw = {row["memory_id"]: float(row["raw_score"])
                      for row in by_qid[qid]["leaf_scores"]}
        if set(report_raw) != set(li_by_id):
            raise ValueError(f"CVM report leaves differ for {qid}")
        max_report_error = max(max_report_error, max(
            abs(raw[li_by_id[mid]] - value) for mid, value in report_raw.items()))
        scored[qid] = (raw, weighted)
    if max_reference_error > 1e-6 or max_report_error > 1e-6:
        raise ValueError("Raw scores do not reproduce frozen CVM reference")
    orders = []
    details = []
    for plan in plans:
        qid = plan["question_id"]
        for condition in ("similar", "unrelated"):
            candidates = plan[condition + "_ids"]
            if len(candidates) != 15 or len(set(candidates)) != 15:
                raise ValueError(f"Expected 15 unique candidates: {qid}, {condition}")
            if set(candidates) & set(plan["gold_memory_ids"]):
                raise ValueError("Gold memory appears among distractors")
            for method, values in zip(("caption_visual_master",
                                       "caption_visual_master_zscore_l0p03_a1"), scored[qid]):
                ordered = sorted(candidates, key=lambda mid: (
                    -float(values[li_by_id[mid]]) if condition == "similar"
                    else float(values[li_by_id[mid]]), mid))
                orders.append({"question_id": qid, "condition": condition,
                               "similarity_method": method, "candidate_ids": ordered,
                               "candidate_query_similarities": [float(values[li_by_id[mid]])
                                                                for mid in ordered]})
                for rank, mid in enumerate(ordered, 1):
                    details.append({"question_id": qid, "condition": condition,
                                    "method": method, "rank": rank,
                                    "candidate_memory_id": mid,
                                    "master_id": leaves[li_by_id[mid]]["master_id"],
                                    "score": float(values[li_by_id[mid]])})
    order_path = source / "caption_visual_master_and_zscore_orders.jsonl"
    with order_path.open("w", encoding="utf-8") as stream:
        for row in orders:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (source / "caption_visual_master_and_zscore_scores.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    protocol = {
        "task": task, "questions": len(plans), "frozen_questions": nq,
        "leaves": nl, "masters": nm,
        "candidate_groups": "same prepared 15-image similar and unrelated groups; no new candidate selection",
        "original_cvm": "frozen CLIP-selected answer per master, expanded-query BGE cosine with full dialogue+caption leaf, global text-score order",
        "zscore": "ZScoreWeighting(lambda=0.03, alpha=1) within each full original master; global weighted-score order",
        "candidate_review_status": "not human reviewed in new domain",
        "max_abs_raw_error_vs_score_matrix": max_reference_error,
        "max_abs_raw_error_vs_pipeline_report": max_report_error,
        "orders": len(orders), "score_rows": len(details),
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
            plans_path, report_path, frozen / "inputs.json", frozen / "features.npz",
            frozen / "memory/graph.json", scores_path)},
    }
    (source / "caption_visual_master_and_zscore_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"orders": len(orders), "score_rows": len(details),
                      "max_reference_error": max_reference_error,
                      "max_report_error": max_report_error}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    main(args.task, args.source.resolve())
