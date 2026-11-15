"""Evaluate frozen-caption Top-3 retrieval as reviewed Home distractors enter.

The two similarity methods determine insertion order only. Retrieval always uses
the original question-to-caption BGE cosine. No model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/home_all_questions_distractor_review_20261031"
OUTPUT = ROOT / "results/home_top3_interference_20261031"
TASK = "Home_Renovation_Interior_Design"
ORDERS: Path | None = None
NAT = ROOT / "results/natural_observation/embeddings.npz"
QUERY = ROOT / "results/mcq_fifth_non_target_rank/query_embeddings.npz"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    orders_path = ORDERS or SOURCE / "proposed_query_similarity_orders.jsonl"
    candidates_path = SOURCE / "candidate_order.jsonl"
    manifest_path = ROOT / "results/mcq_top15_caption_qa/qa_manifest.jsonl"
    orders = read_jsonl(orders_path)
    candidates = {row["question_id"]: row for row in read_jsonl(candidates_path)}
    questions = {row["id"]: row for row in read_jsonl(manifest_path)
                 if row["task"] == TASK and row["id"] in candidates}
    methods = sorted({row["similarity_method"] for row in orders})
    if not questions or len(candidates) != len(questions) or len(orders) != len(questions) * 2 * len(methods):
        raise ValueError(f"Expected complete {TASK} question cohort and two conditions per insertion method")
    per_group = len(orders[0]["candidate_ids"])
    if per_group < 1 or any(len(order["candidate_ids"]) != per_group for order in orders):
        raise ValueError("Candidate counts differ across question-condition-method orders")

    with np.load(NAT, allow_pickle=False) as data:
        memory = {str(mid): data["memory"][i].astype(np.float64)
                  for i, mid in enumerate(data["memory_ids"])}
    with np.load(QUERY, allow_pickle=False) as data:
        query = {str(qid): data["query"][i].astype(np.float64)
                 for i, qid in enumerate(data["query_ids"])}
    for mid, vector in memory.items():
        memory[mid] = vector / np.linalg.norm(vector)
    for qid, vector in query.items():
        query[qid] = vector / np.linalg.norm(vector)

    full_pool_exact_matches = 0
    full_pool_score_equiv_matches = 0
    for qid, question in questions.items():
        domain_ids = [mid for mid in memory if mid.startswith(TASK + "/")]
        full_pool = sorted(domain_ids, key=lambda mid: (-float(query[qid] @ memory[mid]), mid))
        historical = question["top15_memory_ids"]
        full_pool_exact_matches += full_pool[:15] == historical
        full_pool_score_equiv_matches += all(
            abs(float(query[qid] @ memory[actual]) -
                float(query[qid] @ memory[saved])) <= 1e-6
            for actual, saved in zip(full_pool[:15], historical))
    if full_pool_score_equiv_matches != len(questions):
        raise ValueError(f"Frozen caption scores differ from baseline: {full_pool_score_equiv_matches}/{len(questions)}")

    detail: list[dict] = []
    seen = set()
    for order in orders:
        qid = order["question_id"]
        condition = order["condition"]
        method = order["similarity_method"]
        key = (qid, condition, method)
        if key in seen:
            raise ValueError(f"Duplicate insertion order: {key}")
        seen.add(key)
        if condition not in {"similar", "unrelated"} or method not in {"text", "image", "caption_visual_master", "caption_visual_master_visual_first", "caption_visual_master_zscore_l0p03_a1", "vamo_reconstruction_clip", "vamo_target_text"}:
            raise ValueError(f"Unknown condition or method: {key}")
        gold = candidates[qid]["gold_memory_ids"]
        if gold != questions[qid]["gold_ids"]:
            raise ValueError(f"Gold images differ from frozen MCQ manifest: {qid}")
        distractors = order["candidate_ids"]
        source_ids = candidates[qid][condition + "_ids"]
        if len(gold) < 1 or len(distractors) != per_group or len(set(distractors)) != per_group:
            raise ValueError(f"Invalid gold or candidate count: {key}")
        if set(distractors) != set(source_ids) or set(distractors) & set(gold):
            raise ValueError(f"Insertion order changed reviewed candidate set: {key}")
        for n in range(per_group + 1):
            pool = list(dict.fromkeys(gold + distractors[:n]))
            scored = sorted(((mid, float(query[qid] @ memory[mid])) for mid in pool),
                            key=lambda pair: (-pair[1], pair[0]))
            ranked = [mid for mid, _ in scored]
            best_gold_rank = min(ranked.index(mid) + 1 for mid in gold)
            top3 = scored[:3]
            score = max(0.0, (4 - best_gold_rank) / 3) if best_gold_rank <= 3 else 0.0
            detail.append({
                "question_id": qid,
                "condition": condition,
                "insertion_method": method,
                "distractors_added": n,
                "gold_count": len(gold),
                "pool_count": len(pool),
                "best_gold_rank": best_gold_rank,
                "rank_score": score,
                "gold_in_top3": int(best_gold_rank <= 3),
                "top3_count": len(top3),
                "top3_ids": " | ".join(mid for mid, _ in top3),
                "top3_caption_cosines": " | ".join(f"{value:.8f}" for _, value in top3),
                "added_candidate_id": distractors[n - 1] if n else "",
                "gold_ids": " | ".join(gold),
            })

    expected = {(qid, condition, method) for qid in candidates
                for condition in ("similar", "unrelated") for method in methods}
    if seen != expected or len(detail) != len(questions) * 2 * len(methods) * (per_group + 1):
        raise ValueError("Missing or duplicated question-condition-method-N rows")
    by_curve: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in detail:
        by_curve[(row["question_id"], row["condition"], row["insertion_method"])].append(row)
        if row["top3_count"] != min(3, row["pool_count"]):
            raise ValueError("Top-3 was padded or truncated incorrectly")
    for key, rows in by_curve.items():
        rows.sort(key=lambda row: row["distractors_added"])
        if any(rows[i]["best_gold_rank"] > rows[i + 1]["best_gold_rank"]
               for i in range(per_group)):
            raise ValueError(f"Best gold rank improved after adding a distractor: {key}")
        if any(rows[i]["rank_score"] < rows[i + 1]["rank_score"]
               for i in range(per_group)):
            raise ValueError(f"Rank score improved after adding a distractor: {key}")

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in detail:
        grouped[(row["condition"], row["insertion_method"], row["distractors_added"])].append(row)
    summary = []
    for (condition, method, n), rows in sorted(grouped.items()):
        summary.append({
            "condition": condition,
            "insertion_method": method,
            "distractors_added": n,
            "questions": len(rows),
            "mean_rank_score": sum(row["rank_score"] for row in rows) / len(rows),
            "gold_in_top3_rate": sum(row["gold_in_top3"] for row in rows) / len(rows),
            "mean_best_gold_rank": sum(row["best_gold_rank"] for row in rows) / len(rows),
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "per_question.csv", detail)
    write_csv(OUTPUT / "summary.csv", summary)
    protocol = {
        "scope": f"{len(questions)} {TASK} MCQs; all gold images plus N of {per_group} candidates per condition",
        "review_status": ("new VAMO-selected candidates have not been individually reviewed" if "vamo_target_text" in methods else
                         (("old 10 candidate groups verbally reviewed; newly added 11–15 "
                           "candidates have not been individually reviewed") if per_group > 10 else
                          "user says review completed; individual acceptance fields in source workbook are blank")
                         if TASK == "Home_Renovation_Interior_Design" else
                         "candidate images in this new domain have not been human reviewed"),
        "candidate_scope": (f"VAMO-reconstructed image-to-image cosine selects {per_group} candidates from full task pool" if "vamo_target_text" in methods else
                            f"reorder within each complete {per_group}-image similar/unrelated candidate group"),
        "insertion_order": {
            "text": "question-to-original-caption frozen BGE cosine",
            "image": "question-text-to-original-image frozen CLIP cosine",
            "caption_visual_master": "saved raw multi_path BGE leaf cosine; master selects candidate statement using CLIP",
            "caption_visual_master_visual_first": "question-to-master CLIP rank first, then master-specific answer-expanded BGE leaf cosine",
            "caption_visual_master_zscore_l0p03_a1": "pipeline CVM answer-expanded BGE leaf cosine plus within-master z-score correction; lambda=.03 alpha=1",
            "vamo_reconstruction_clip": "question CLIP text cosine with CLIP image embedding of saved VAMO reconstruction",
            "vamo_target_text": "VAMO latent target-to-candidate cosine selects candidate group; question-to-original-caption frozen BGE cosine orders it",
            "similar": "descending query-candidate similarity",
            "unrelated": "ascending query-candidate similarity",
        },
        "insertion_methods_present": methods,
        "retrieval": "question-to-original-caption frozen BGE cosine, Top-3 within gold-plus-N pool",
        "target_rank": "minimum rank over all annotated gold images; ties broken by memory ID",
        "rank_score": {"1": 1.0, "2": 2 / 3, "3": 1 / 3, "4+": 0.0},
        "no_top3_padding": True,
        "n_values": list(range(per_group + 1)),
        "question_count": len(questions),
        "full_domain_pool_top15_exact_id_matches_original_caption_baseline": full_pool_exact_matches,
        "full_domain_pool_top15_score_equivalent_to_original_caption_baseline": full_pool_score_equiv_matches,
        "baseline_score_equivalence_tolerance": 1e-6,
        "rows": len(detail),
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                         (orders_path, candidates_path, manifest_path, NAT, QUERY)},
        "qa_reader_run": False,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": len(detail), "summary_rows": len(summary),
                      "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--orders", type=Path, default=None)
    parser.add_argument("--task", default=TASK)
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    OUTPUT = args.output.resolve()
    ORDERS = args.orders.resolve() if args.orders else None
    TASK = args.task
    main()
