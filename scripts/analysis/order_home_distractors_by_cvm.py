"""Order reviewed Home candidates by frozen caption_visual_master raw text scores.

The saved full-domain multi_path retrieval contains one BGE cosine per question
and leaf. CLIP only selected a master-specific candidate statement; it is not
fused into the leaf similarity. The saved z-score run also records the raw
cosine, which is independent of the z-score correction.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/home_15_distractor_review_20261031"
REPORT = (ROOT / "results/online_memory_path_validation_20261023"
          / "Home_Renovation_Interior_Design/caption_visual_master/retrieval_results.json")


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
    report = json.loads(REPORT.read_text())
    if (report["memory"] != "caption_visual_master" or report["retrieval"] != "multi_path"
            or report["images"] != 89 or len(report["questions"]) != 52):
        raise ValueError("The saved report is not the full Home CVM multi_path run")
    for name, expected in report["source_sha256"].items():
        path = Path(name)
        if path.name in {"memory.json", "leaves.json", "graph.json", "master_sums.npy",
                         "candidates.jsonl", "inputs.json"} and sha(path) != expected:
            raise ValueError(f"Frozen CVM input changed: {name}")
    plans = read_jsonl(SOURCE / "candidate_order.jsonl")
    if len(plans) != 52:
        raise ValueError("Expected 52 Home question candidate plans")
    scores = {}
    master_choices = {}
    for question in report["questions"]:
        qid = question["question_id"]
        leaves = question["leaf_scores"]
        if len(leaves) != 89 or len({leaf["memory_id"] for leaf in leaves}) != 89:
            raise ValueError(f"Missing CVM leaf scores: {qid}")
        scores[qid] = {leaf["memory_id"]: leaf for leaf in leaves}
        master_choices[qid] = {group["group_id"]: group["choice"]
                               for group in question["groups"]}
    if {plan["question_id"] for plan in plans} != set(scores):
        raise ValueError("CVM question cohort differs from the reviewed cohort")

    orders, details = [], []
    for plan in plans:
        qid = plan["question_id"]
        gold = set(plan["gold_memory_ids"])
        for condition in ("similar", "unrelated"):
            members = plan[condition + "_ids"]
            if len(members) != 15 or len(set(members)) != 15 or gold & set(members):
                raise ValueError(f"Invalid candidate set: {qid} {condition}")
            ordered = sorted(members, key=lambda mid: (
                -scores[qid][mid]["raw_score"] if condition == "similar"
                else scores[qid][mid]["raw_score"], mid))
            orders.append({
                "question_id": qid, "condition": condition,
                "similarity_method": "caption_visual_master",
                "candidate_ids": ordered,
                "candidate_query_similarities": [scores[qid][mid]["raw_score"] for mid in ordered],
            })
            original_pos = {mid: i for i, mid in enumerate(members, 1)}
            for position, mid in enumerate(ordered, 1):
                leaf = scores[qid][mid]
                details.append({
                    "question_id": qid, "condition": condition,
                    "cvm_insertion_rank": position,
                    "candidate_memory_id": mid,
                    "original_visual_group_rank": original_pos[mid],
                    "cvm_raw_text_cosine": leaf["raw_score"],
                    "master_id": leaf["group_id"],
                    "master_selected_choice": master_choices[qid][leaf["group_id"]],
                })
    orders_path = SOURCE / "caption_visual_master_query_orders.jsonl"
    with orders_path.open("w", encoding="utf-8") as stream:
        for row in orders:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    details_path = SOURCE / "caption_visual_master_candidate_scores.csv"
    with details_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    protocol = {
        "method": "caption_visual_master multi_path, online_weighting=none raw BGE leaf cosine",
        "saved_report_online_weighting": report["online_weighting"],
        "raw_score_independent_of_saved_zscore_correction": True,
        "master_choice": "CLIP cosine selects one candidate statement per master",
        "leaf_text": "original dialogue + newline + Memory description: + full original caption",
        "leaf_similarity": "BGE cosine of selected master-specific expanded query and leaf text",
        "selection": "similar descending, unrelated ascending within each 15-image candidate group",
        "questions": len(plans), "orders": len(orders), "candidate_score_rows": len(details),
        "review_status": "new original visual ranks 11-15 are pending human review",
        "input_sha256": {
            str(REPORT.relative_to(ROOT)): sha(REPORT),
            str((SOURCE / "candidate_order.jsonl").relative_to(ROOT)): sha(SOURCE / "candidate_order.jsonl"),
        },
    }
    (SOURCE / "caption_visual_master_order_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"questions": len(plans), "orders": len(orders),
                      "candidate_score_rows": len(details)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
