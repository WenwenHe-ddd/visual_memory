"""Order frozen distractor groups by query-to-VAMO-reconstruction CLIP cosine.

The VAMO decoder outputs are fixed. This script re-embeds those decoded images
with the same CLIP checkpoint used for the original visual-order baseline.
It never changes the target images, candidate groups, caption retriever, or QA.
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
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from PIL import Image

from scripts.graph.visual_image_encoder import ImageEncoder
from scripts.memory.caption_visual_master import load_clip_text_encoder, normalize

DOMAINS = (
    ("Brand_Memory_Test", "brand"),
    ("Card_Playlog_Test", "card"),
    ("Cartoon_Entertainment_Companion", "cartoon"),
    ("Home_Renovation_Interior_Design", "home"),
    ("Multi-Scene_Visual_Case_Archive_Assistant", "multiscene"),
    ("Outdoor_Navigation_Route_Memory_Assistant", "outdoor"),
    ("Personal_Health_Dashboard_Assistant", "health"),
    ("Social_Chat_Memory_Test", "social"),
)
OUT = ROOT / "results/vamo_top3_interference_20261114"
METHOD = "vamo_reconstruction_clip"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_plans(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(device: str, batch_size: int) -> None:
    import torch

    torch.set_num_threads(4)
    metadata_path = (ROOT / "reports/adaptive_retrieval_comparison_20261020"
                     / "inputs_by_domain/Brand_Memory_Test/protocol.json")
    metadata = json.loads(metadata_path.read_text())["image_encoder"]
    encoder = ImageEncoder(metadata["model_name"], revision=metadata["revision"],
                           device=device, local_files_only=True)
    if {k: v for k, v in encoder.metadata.items() if k != "device"} != {
            k: v for k, v in metadata.items() if k != "device"}:
        raise ValueError("CLIP encoder differs from frozen visual baseline")

    domain_data: dict[str, dict] = {}
    all_image_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {str(metadata_path): digest(metadata_path)}
    for task, slug in DOMAINS:
        source = (ROOT / "results/home_15_distractor_review_20261031" if slug == "home"
                  else ROOT / "results" / f"{slug}_all_questions_distractor_review_20261101")
        plans_path = source / "candidate_order.jsonl"
        plans = load_plans(plans_path)
        vamo_dir = (ROOT / "reports/brand_vae_vamo_20261105" if slug == "brand" else
                    ROOT / "reports/other_domains_vae_vamo_20261105" / task)
        metrics_path = vamo_dir / "per_image_metrics.csv"
        with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        image_index = {row["image_id"]: int(row["index"])
                       for row in rows if row["comparison"] == "vamo_vs_original"}
        if len(image_index) != len(rows) // 3:
            raise ValueError(f"Incomplete VAMO image inventory: {task}")
        relevant = {mid + "#image1" for plan in plans
                    for key in ("gold_memory_ids", "similar_ids", "unrelated_ids")
                    for mid in plan[key]}
        if not relevant <= image_index.keys():
            raise ValueError(f"Candidate missing a VAMO reconstruction: {task}")
        width = 2 if slug == "brand" else 3
        paths = {mid: vamo_dir / "vamo" / f"{image_index[mid]:0{width}d}.png"
                 for mid in relevant}
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"VAMO decoded image missing: {task}")
        all_image_paths.update(paths)
        source_hashes[str(plans_path)] = digest(plans_path)
        source_hashes[str(metrics_path)] = digest(metrics_path)
        domain_data[slug] = {"task": task, "source": source, "plans": plans,
                             "paths": paths, "vamo_dir": vamo_dir}
    if len(all_image_paths) != sum(len(d["paths"]) for d in domain_data.values()):
        raise ValueError("VAMO image ID collision across domains")

    visual_vectors: dict[str, np.ndarray] = {}
    ids = sorted(all_image_paths)
    for start in range(0, len(ids), batch_size):
        group = ids[start:start + batch_size]
        with_images = []
        try:
            for mid in group:
                with Image.open(all_image_paths[mid]) as image:
                    with_images.append(image.convert("RGB"))
            pixels = encoder.processor(images=with_images, return_tensors="pt")[
                "pixel_values"].to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                vectors = encoder.model(pixel_values=pixels, return_dict=True).image_embeds
            vectors = normalize(vectors.double().cpu().numpy())
            visual_vectors.update(zip(group, vectors))
        finally:
            for image in with_images:
                image.close()
        print(f"VAMO visual CLIP {min(start + batch_size, len(ids))}/{len(ids)}", flush=True)
    del encoder

    model, tokenizer, token_limit = load_clip_text_encoder(metadata, device)
    query_vectors: dict[str, np.ndarray] = {}
    all_plans = [plan for data in domain_data.values() for plan in data["plans"]]
    if len({plan["question_id"] for plan in all_plans}) != len(all_plans):
        raise ValueError("Duplicate question ID")
    for start in range(0, len(all_plans), batch_size):
        group = all_plans[start:start + batch_size]
        tokens = tokenizer([plan["question"] for plan in group], padding=True,
                           truncation=True, max_length=token_limit, return_tensors="pt")
        tokens = tokens.to(device)
        with torch.inference_mode():
            vectors = normalize(model(**tokens).text_embeds.double().cpu().numpy())
        query_vectors.update((plan["question_id"], vector)
                             for plan, vector in zip(group, vectors))
    del model

    OUT.mkdir(parents=True, exist_ok=True)
    for task, slug in DOMAINS:
        data = domain_data[slug]
        folder = OUT / slug
        folder.mkdir(exist_ok=True)
        orders, scores = [], []
        for plan in data["plans"]:
            qid = plan["question_id"]
            for condition in ("similar", "unrelated"):
                candidates = plan[condition + "_ids"]
                if len(candidates) != 15 or len(set(candidates)) != 15:
                    raise ValueError(f"Expected 15 unique candidates: {qid} {condition}")
                scored = {mid: float(query_vectors[qid] @ visual_vectors[mid + "#image1"])
                          for mid in candidates}
                ordered = sorted(candidates, key=lambda mid: (
                    -scored[mid] if condition == "similar" else scored[mid], mid))
                orders.append({"question_id": qid, "condition": condition,
                               "similarity_method": METHOD, "candidate_ids": ordered,
                               "candidate_query_similarities": [scored[mid] for mid in ordered]})
                scores.extend({"question_id": qid, "condition": condition,
                               "candidate_memory_id": mid, "vamo_clip_cosine": scored[mid],
                               "rank": rank}
                              for rank, mid in enumerate(ordered, 1))
        with (folder / "orders.jsonl").open("w", encoding="utf-8") as handle:
            for row in orders:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (folder / "scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
            writer.writeheader()
            writer.writerows(scores)
        (folder / "protocol.json").write_text(json.dumps({
            "task": task, "questions": len(data["plans"]), "candidate_groups": 2,
            "candidates_per_group": 15,
            "representation": "Decoded VAMO reconstruction, re-embedded by frozen CLIP image encoder",
            "ordering_score": "cosine(frozen CLIP question-text, frozen CLIP VAMO reconstructed image)",
            "ordering": "similar descending; unrelated ascending; within fixed candidate groups",
            "downstream_retrieval": "unchanged original-caption BGE Top-3 in target-plus-N pool",
            "qa_reader_run": False, "device": device,
            "encoder": metadata, "source_hashes": source_hashes,
            "reconstructed_images": {str(path): digest(path) for path in data["paths"].values()},
        }, ensure_ascii=False, indent=2) + "\n")
        print(f"{slug}: {len(data['plans'])} questions, {len(orders)} orders", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    main(args.device, args.batch_size)
