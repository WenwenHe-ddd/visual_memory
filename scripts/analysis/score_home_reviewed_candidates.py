"""Score Home review candidates by question-to-caption BGE and question-to-image CLIP.

The script scores candidates but does not run retrieval or reader evaluation.
No raw image vectors are persisted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from scripts.graph.visual_image_encoder import ImageEncoder
from scripts.memory.caption_visual_master import load_clip_text_encoder, normalize


TASK = "Home_Renovation_Interior_Design"
DEVICE = "cpu"
SOURCE = ROOT / "results/home_all_questions_distractor_review_20261031"
NAT = ROOT / "results/natural_observation"
MCQ = ROOT / "results/mcq_fifth_non_target_rank"
ENCODER_PROTOCOL = (
    ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain"
    / TASK / "protocol.json"
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    encoder_protocol = (ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain"
                        / TASK / "protocol.json")
    memory_data = np.load(NAT / "embeddings.npz", allow_pickle=False)
    query_data = np.load(MCQ / "query_embeddings.npz", allow_pickle=False)
    memory_vectors = {str(mid): memory_data["memory"][i] for i, mid in
                      enumerate(memory_data["memory_ids"])}
    query_vectors = {str(qid): query_data["query"][i] for i, qid in
                     enumerate(query_data["query_ids"])}
    memories = {row["id"]: row for row in read_jsonl(NAT / "memories.jsonl")
                if row["task"] == TASK}
    images = {row["id"]: Path(row["path"]) for row in read_jsonl(NAT / "images.jsonl")}
    all_questions = {row["id"]: row["question"] for row in read_jsonl(MCQ / "mcq_queries.jsonl")
                     if row["task"] == TASK}
    reviewed = list(csv.DictReader((SOURCE / "review_candidates.csv").open(encoding="utf-8-sig")))
    question_ids = {row["question_id"] for row in reviewed}
    questions = {qid: all_questions[qid] for qid in question_ids}
    if not questions or len(reviewed) % (len(questions) * 2):
        raise ValueError(f"{TASK} candidate cohort changed")
    per_group = len(reviewed) // (len(questions) * 2)
    config = json.loads(encoder_protocol.read_text())["image_encoder"]
    import torch
    torch.set_num_threads(4)
    encoder = ImageEncoder(config["model_name"], revision=config["revision"],
                           device=DEVICE, local_files_only=True)
    image_vectors = {}
    for index, mid in enumerate(sorted(memories), 1):
        image_id = memories[mid]["images"][0]
        path = images[image_id]
        if sha(path) != image_id:
            raise ValueError(f"Image changed: {mid}")
        image_vectors[mid] = np.asarray(encoder.encode(path), dtype=np.float64)
        if index % 20 == 0 or index == len(memories):
            print(f"Encoded {index}/{len(memories)} original images", flush=True)
    del encoder
    model, tokenizer, token_limit = load_clip_text_encoder(config, DEVICE)
    query_ids = sorted(questions)
    clip_queries = {}
    with torch.inference_mode():
        for start in range(0, len(query_ids), 16):
            ids = query_ids[start:start + 16]
            tokens = tokenizer([questions[qid] for qid in ids], padding=True,
                               truncation=True, max_length=token_limit,
                               return_tensors="pt")
            if tokens["input_ids"].shape[1] > token_limit:
                raise ValueError("A question exceeds the frozen CLIP text length")
            tokens = tokens.to(DEVICE)
            vectors = normalize(model(**tokens).text_embeds.double().cpu().numpy())
            clip_queries.update(zip(ids, vectors))
    del model
    rows = []
    for item in reviewed:
        qid, mid = item["question_id"], item["candidate_memory_id"]
        rows.append({
            "question_id": qid, "condition": item["condition"],
            "original_image_similarity_order": int(item["add_order"]),
            "candidate_memory_id": mid,
            "question_caption_bge_cosine": round(float(query_vectors[qid] @ memory_vectors[mid]), 8),
            "question_image_clip_cosine": round(float(clip_queries[qid] @ image_vectors[mid]), 8),
            "target_image_clip_cosine": item["clip_visual_cosine_to_nearest_gold"],
        })
    write_csv(SOURCE / "reviewed_candidate_query_scores.csv", rows)
    orders = []
    for qid in query_ids:
        for condition in ("similar", "unrelated"):
            group = [row for row in rows if row["question_id"] == qid and row["condition"] == condition]
            if len(group) != per_group:
                raise ValueError(f"Question group does not have {per_group} candidates: {qid}")
            for method, key in (("text", "question_caption_bge_cosine"),
                                ("image", "question_image_clip_cosine")):
                ordered = sorted(group, key=lambda row: (
                    -float(row[key]) if condition == "similar" else float(row[key]),
                    row["candidate_memory_id"]))
                orders.append({"question_id": qid, "condition": condition,
                               "similarity_method": method,
                               "candidate_ids": [row["candidate_memory_id"] for row in ordered],
                               "candidate_query_similarities": [row[key] for row in ordered]})
    with (SOURCE / "proposed_query_similarity_orders.jsonl").open("w") as handle:
        for row in orders:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    protocol = {
        "status": "scoring and ordering only; QA not run",
        "scope": {"questions": len(query_ids), "candidate_rows": len(rows),
                  "proposed_orders": len(orders), "candidates_per_group": per_group},
        "ordering_scope": f"within each prepared {per_group}-image group only",
        "text_similarity": "frozen BGE question embedding cosine original caption embedding",
        "image_similarity": "frozen CLIP question text embedding cosine original image embedding",
        "direction": "similar descending; unrelated ascending",
        "clip_query_tokenization": f"CLIP tokenizer truncates to {token_limit} tokens when needed",
        "image_embeddings_persisted": False,
        "device": DEVICE,
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            SOURCE / "review_candidates.csv", NAT / "embeddings.npz",
            MCQ / "query_embeddings.npz", NAT / "memories.jsonl",
            NAT / "images.jsonl", MCQ / "mcq_queries.jsonl", encoder_protocol)},
    }
    (SOURCE / "query_similarity_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(protocol["scope"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    TASK = args.task
    DEVICE = args.device
    main()
