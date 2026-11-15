"""Select distractors by target-to-candidate VAMO latent cosine, then BGE-order.

The retrieval evaluator subsequently uses the same frozen original-caption BGE
vectors for Top-3 scoring. No new model inference or QA reader is involved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/vamo_target_selection_top3_20261114"
NAT = ROOT / "results/natural_observation"
QUERY = ROOT / "results/mcq_fifth_non_target_rank/query_embeddings.npz"
METHOD = "vamo_target_text"
DOMAINS = (
    ("Brand_Memory_Test", "brand", 25),
    ("Card_Playlog_Test", "card", 48),
    ("Cartoon_Entertainment_Companion", "cartoon", 76),
    ("Home_Renovation_Interior_Design", "home", 52),
    ("Multi-Scene_Visual_Case_Archive_Assistant", "multiscene", 50),
    ("Outdoor_Navigation_Route_Memory_Assistant", "outdoor", 28),
    ("Personal_Health_Dashboard_Assistant", "health", 51),
    ("Social_Chat_Memory_Test", "social", 36),
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid embedding norm")
    return vector / norm


def main() -> None:
    memories = read_jsonl(NAT / "memories.jsonl")
    memory_by_id = {row["id"]: row for row in memories}
    with np.load(NAT / "embeddings.npz", allow_pickle=False) as data:
        bge = {str(mid): normalized(v) for mid, v in
               zip(data["memory_ids"], data["memory"])}
    with np.load(QUERY, allow_pickle=False) as data:
        queries = {str(qid): normalized(v) for qid, v in
                   zip(data["query_ids"], data["query"])}

    OUT.mkdir(parents=True, exist_ok=True)
    for task, slug, expected_questions in DOMAINS:
        original_plans = (ROOT / "results/home_15_distractor_review_20261031" if slug == "home"
                          else ROOT / "results" / f"{slug}_all_questions_distractor_review_20261101") / "candidate_order.jsonl"
        plans = read_jsonl(original_plans)
        if len(plans) != expected_questions:
            raise ValueError(f"Question count changed for {task}: {len(plans)}")
        vamo_dir = (ROOT / "reports/brand_vae_vamo_20261105" if slug == "brand"
                    else ROOT / "reports/other_domains_vae_vamo_20261105" / task)
        metadata_path = vamo_dir / "image_metadata.json"
        latent_path = vamo_dir / "evaluation_only/reconstructed_latents.npy"
        metadata = json.loads(metadata_path.read_text())
        latents = np.load(latent_path, mmap_mode="r", allow_pickle=False)
        if latents.shape != (len(metadata), 4096):
            raise ValueError(f"VAMO latent inventory mismatch: {task}")
        vamo = {row["image_id"].removesuffix("#image1"):
                normalized(latents[row["index"] - 1]) for row in metadata}
        task_memories = sorted((row for row in memories if row["task"] == task),
                               key=lambda row: row["id"])
        if set(vamo) != {row["id"] for row in task_memories}:
            raise ValueError(f"VAMO image IDs differ from natural memory inventory: {task}")
        folder = OUT / slug
        folder.mkdir(exist_ok=True)
        new_plans, orders, scores = [], [], []
        for plan in plans:
            qid = plan["question_id"]
            gold = plan["gold_memory_ids"]
            target_hashes = {memory_by_id[mid]["images"][0] for mid in gold}
            if not gold or any(mid not in vamo for mid in gold):
                raise ValueError(f"Missing VAMO target: {qid}")
            unique_candidates = {}
            for memory in task_memories:
                mid = memory["id"]
                image_hash = memory["images"][0]
                if mid in gold or image_hash in target_hashes:
                    continue
                unique_candidates.setdefault(image_hash, mid)
            if len(unique_candidates) < 15:
                raise ValueError(f"Insufficient distinct candidates: {qid}")
            scored_visual = [(mid, max(float(vamo[mid] @ vamo[target]) for target in gold))
                             for mid in unique_candidates.values()]
            scored_visual.sort(key=lambda item: (-item[1], item[0]))
            similar = scored_visual[:15]
            unrelated = sorted(scored_visual[-15:], key=lambda item: (item[1], item[0]))
            new_plans.append({"question_id": qid, "question": plan["question"],
                              "gold_memory_ids": gold,
                              "similar_ids": [mid for mid, _ in similar],
                              "unrelated_ids": [mid for mid, _ in unrelated]})
            for condition, group, reverse in (("similar", similar, True),
                                               ("unrelated", unrelated, False)):
                by_bge = [(mid, float(queries[qid] @ bge[mid]), visual)
                          for mid, visual in group]
                by_bge.sort(key=lambda row: ((-row[1] if reverse else row[1]), row[0]))
                orders.append({"question_id": qid, "condition": condition,
                               "similarity_method": METHOD,
                               "candidate_ids": [mid for mid, _, _ in by_bge]})
                for rank, (mid, text_score, visual_score) in enumerate(by_bge, 1):
                    scores.append({"question_id": qid, "condition": condition,
                                   "insertion_rank": rank, "candidate_id": mid,
                                   "candidate_image_sha256": memory_by_id[mid]["images"][0],
                                   "vamo_target_cosine": visual_score,
                                   "original_caption_bge_query_cosine": text_score})
        write_jsonl(folder / "candidate_order.jsonl", new_plans)
        write_jsonl(folder / "orders.jsonl", orders)
        with (folder / "candidate_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
            writer.writeheader()
            writer.writerows(scores)
        (folder / "protocol.json").write_text(json.dumps({
            "task": task, "questions": len(plans), "candidate_count": 15,
            "candidate_selection": "maximum target-to-candidate cosine of L2-normalized saved 4096D VAMO reconstructed VAE latents; deduplicate original image SHA; exclude all targets",
            "similar_condition": "15 highest VAMO image-image similarities, then BGE question-to-original-caption descending",
            "unrelated_condition": "15 lowest VAMO image-image similarities, then BGE question-to-original-caption ascending; diagnostic only, not plotted",
            "retrieval": "frozen original-caption BGE cosine Top-3 within all gold images plus first N candidates",
            "human_review": "newly selected VAMO candidates have not been individually reviewed",
            "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                             (original_plans, metadata_path, latent_path,
                              NAT / "memories.jsonl", NAT / "embeddings.npz", QUERY)},
        }, ensure_ascii=False, indent=2) + "\n")
        cmd = [sys.executable, str(ROOT / "scripts/analysis/evaluate_home_top3_interference.py"),
               "--source", str(folder), "--orders", str(folder / "orders.jsonl"),
               "--output", str(folder / "evaluation"), "--task", task]
        subprocess.run(cmd, check=True)
        print(f"{slug}: {len(plans)} questions, {len(scores)} candidate rows", flush=True)


if __name__ == "__main__":
    main()
