"""Prepare original-image distractor review for all Home MemEye questions.

Ranking uses frozen CLIP image-to-image cosine only. No retrieval results,
question answers, or option text enter candidate selection. The output is a
review queue; it does not certify question-cue overlap or run QA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image, ImageOps

from scripts.graph.visual_image_encoder import ImageEncoder


TASK = "Home_Renovation_Interior_Design"
N = 10
DEVICE = "cpu"
OUT = ROOT / "results/home_all_questions_distractor_review_20261031"
NAT = ROOT / "results/natural_observation"
MCQ = ROOT / "results/mcq_fifth_non_target_rank"
DIALOG = ROOT / f"data/MemEye/data/dialog/{TASK}.json"
ENCODER_PROTOCOL = (
    ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain"
    / TASK / "protocol.json"
)
OLD_REVIEW = ROOT / "results/home_cue_review_20261031/review_candidates.csv"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cue_words(question: str) -> set[str]:
    # A transparent lexical hint for the reviewer, never an acceptance label.
    stop = {"what", "which", "where", "when", "who", "how", "the", "and", "for", "with",
            "from", "that", "this", "then", "there", "same", "image", "photo", "picture",
            "view", "shown", "shows", "does", "did", "are", "was", "has", "have", "into",
            "about", "after", "before", "during", "later", "earlier", "most", "more"}
    return {word for word in re.findall(r"[a-z0-9]+", question.lower())
            if len(word) >= 3 and word not in stop}


def page_header(title: str) -> str:
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        '<style>body{font:16px/1.5 Arial,sans-serif;max-width:1500px;margin:auto;padding:24px}'
        'a{color:#125d9a}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}'
        '.card{border:1px solid #ccc;padding:12px;border-radius:6px;background:#fafafa}'
        '.card img{width:100%;height:250px;object-fit:contain;background:#eee}'
        '.meta{font-size:13px;color:#555}h2{margin-top:36px}p{overflow-wrap:anywhere}'
        '.score{font-weight:bold}details{margin-top:8px}</style>'
    )


def card_html(mid: str, role: str, rank: int | None, score: float | None,
              memories: dict[str, dict], images: dict[str, dict], captions: dict[str, str],
              dialogs: dict[str, dict], source_paths: dict[str, Path],
              prior: bool = False, lexical: str = "") -> str:
    image_id = memories[mid]["images"][0]
    caption = captions[image_id]
    source = source_paths[image_id]
    dialogue = dialogs[mid]
    heading = f"{role} {rank}" if rank is not None else role
    metric = f"CLIP cosine {score:.4f}" if score is not None else "original annotated evidence"
    prior_text = " · passed previous small pilot" if prior else ""
    return (
        '<div class="card">'
        f'<h3>{html.escape(heading)} · {html.escape(source.name)}</h3>'
        f'<a href="{html.escape(source.as_uri())}"><img src="../thumbs/{image_id}.jpg" '
        f'alt="{html.escape(mid)}"></a>'
        f'<p class="meta">{html.escape(mid)} · {html.escape(image_id[:16])}</p>'
        f'<p class="score">{html.escape(metric)}{html.escape(prior_text)}</p>'
        f'<p>Question words also in caption: {html.escape(lexical or "none")}</p>'
        f'<details><summary>Caption and original dialogue</summary><p>{html.escape(caption)}</p>'
        f'<p><b>User:</b> {html.escape(dialogue["user"])}<br>'
        f'<b>Assistant:</b> {html.escape(dialogue["assistant"])}</p>'
        f'<p class="meta">{html.escape(dialogue["date"])} · '
        f'{html.escape(str(source))}</p></details></div>'
    )


def main() -> None:
    global DIALOG, ENCODER_PROTOCOL, OLD_REVIEW
    DIALOG = ROOT / f"data/MemEye/data/dialog/{TASK}.json"
    ENCODER_PROTOCOL = (ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain"
                        / TASK / "protocol.json")
    if TASK != "Home_Renovation_Interior_Design":
        OLD_REVIEW = OUT / "nonexistent_prior_review.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thumbs").mkdir(exist_ok=True)
    (OUT / "questions").mkdir(exist_ok=True)
    memories_list = [row for row in read_jsonl(NAT / "memories.jsonl") if row["task"] == TASK]
    memories = {row["id"]: row for row in memories_list}
    if len(memories) != len(memories_list):
        raise ValueError("Duplicate Home memory ID")
    images = {row["id"]: row for row in read_jsonl(NAT / "images.jsonl")}
    captions = {row["id"]: row["caption"] for row in read_jsonl(NAT / "captions.jsonl")}
    all_questions = [row for row in read_jsonl(MCQ / "mcq_queries.jsonl") if row["task"] == TASK]
    gold = {row["id"]: row["gold_ids"] for row in read_jsonl(MCQ / "retrieval_details.jsonl")}
    unique_image_ids = {row["images"][0] for row in memories_list}
    questions = [row for row in all_questions if len(unique_image_ids - {
        memories[mid]["images"][0] for mid in gold[row["id"]]}) >= N]
    excluded = [row["id"] for row in all_questions if row not in questions]
    if not questions or len(unique_image_ids) < N + 1:
        raise ValueError(f"Insufficient questions or memories for {TASK}")
    original = json.loads(DIALOG.read_text())
    dialogs = {}
    for session in original["multi_session_dialogues"]:
        for turn in session["dialogues"]:
            if "input_image" not in turn:
                continue
            mid = f"{TASK}/{turn['round']}"
            dialogs[mid] = {"user": turn.get("user", ""),
                            "assistant": turn.get("assistant", ""),
                            "date": session["date"]}
    assert set(memories) <= set(dialogs)
    source_paths = {}
    for mid, memory in memories.items():
        image_id = memory["images"][0]
        source = Path(images[image_id]["path"])
        if sha(source) != image_id:
            raise ValueError(f"Original image changed: {mid}")
        source_paths[image_id] = source
        preview = OUT / "thumbs" / f"{image_id}.jpg"
        if not preview.exists():
            with Image.open(source) as original_image:
                thumbnail = ImageOps.contain(original_image.convert("RGB"), (450, 330))
                thumbnail.save(preview, quality=85, optimize=True)
    if set(source_paths) != unique_image_ids:
        raise ValueError(f"{TASK} image hash inventory changed")
    prior_passed = set()
    if OLD_REVIEW.exists():
        with OLD_REVIEW.open(encoding="utf-8-sig", newline="") as handle:
            prior_passed = {(row["question_id"], row["candidate_id"])
                            for row in csv.DictReader(handle)}
    encoder_info = json.loads(ENCODER_PROTOCOL.read_text())["image_encoder"]
    import torch
    torch.set_num_threads(4)
    encoder = ImageEncoder(encoder_info["model_name"], revision=encoder_info["revision"],
                           device=DEVICE, local_files_only=True)
    vectors = {}
    for index, image_id in enumerate(sorted(source_paths), 1):
        vectors[image_id] = np.asarray(encoder.encode(source_paths[image_id]), dtype=np.float64)
        if index % 10 == 0 or index == len(source_paths):
            print(f"Encoded {index}/{len(source_paths)} {TASK} images", flush=True)
    model_metadata = encoder.metadata
    del encoder

    rows = []
    plans = []
    summary = []
    index_rows = []
    for question in questions:
        qid = question["id"]
        target_ids = gold[qid]
        if not target_ids or not set(target_ids) <= set(memories):
            raise ValueError(f"Missing gold memory: {qid}")
        target_images = {memories[mid]["images"][0] for mid in target_ids}
        candidates = []
        seen_images = set()
        for memory in sorted(memories_list, key=lambda row: row["id"]):
            mid = memory["id"]
            image_id = memory["images"][0]
            if mid in target_ids or image_id in target_images:
                continue
            if image_id in seen_images:
                continue
            seen_images.add(image_id)
            similarity = max(float(vectors[image_id] @ vectors[target])
                             for target in target_images)
            candidates.append((mid, image_id, similarity))
        if len(candidates) < N:
            raise ValueError(f"Too few distinct non-gold images: {qid}")
        candidates.sort(key=lambda item: (-item[2], item[0]))
        similar = candidates[:N]
        unrelated = sorted(candidates[-N:], key=lambda item: (item[2], item[0]))
        group_overlap = {item[1] for item in similar} & {item[1] for item in unrelated}
        if len(candidates) >= 2 * N and group_overlap:
            raise ValueError(f"Candidate groups overlap despite sufficient candidates: {qid}")
        qwords = cue_words(question["question"])
        page = [page_header(qid), '<p><a href="../index.html">All domain questions</a></p>',
                f'<h1>{html.escape(qid)}</h1><p>{html.escape(question["question"])}</p>',
                '<p>Ranking uses image-to-image CLIP cosine against the most similar gold image. '
                'Shared question words are a literal hint only. Please check visual cue overlap, '
                'whether each image could also answer the question, and caption support.</p>',
                '<h2>Original annotated evidence</h2><div class="grid">']
        page.extend(card_html(mid, "Target", None, None, memories, images, captions,
                              dialogs, source_paths) for mid in target_ids)
        page.append('</div><h2>Similar images · highest to lower</h2><div class="grid">')
        for condition, selected in (("similar", similar), ("unrelated", unrelated)):
            if condition == "unrelated":
                page.append('</div><h2>Unrelated images · lowest to higher</h2><div class="grid">')
            for position, (mid, image_id, similarity) in enumerate(selected, 1):
                caption = captions[image_id]
                lexical_overlap = ", ".join(sorted(qwords & cue_words(caption)))
                prior = (qid, mid) in prior_passed
                rows.append({
                    "question_id": qid, "question": question["question"],
                    "gold_memory_ids": " | ".join(target_ids),
                    "gold_image_paths": " | ".join(str(source_paths[memories[g]["images"][0]])
                                                   for g in target_ids),
                    "condition": condition, "add_order": position,
                    "candidate_memory_id": mid, "candidate_image_path": str(source_paths[image_id]),
                    "candidate_image_sha256": image_id, "candidate_caption": caption,
                    "candidate_user_dialogue": dialogs[mid]["user"],
                    "candidate_assistant_dialogue": dialogs[mid]["assistant"],
                    "clip_visual_cosine_to_nearest_gold": round(similarity, 8),
                    "exact_question_word_overlap_hint": lexical_overlap,
                    "previous_small_pilot_passed": "yes" if prior else "",
                    "human_accept_for_this_question": "",
                    "human_shared_question_cues": "",
                    "human_is_alternate_valid_evidence": "",
                    "human_caption_matches_image": "",
                    "human_notes": "",
                })
                page.append(card_html(mid, condition.capitalize(), position, similarity,
                                      memories, images, captions, dialogs, source_paths,
                                      prior=prior, lexical=lexical_overlap))
        page.append('</div></html>')
        (OUT / "questions" / f"{qid.split('/')[-1]}.html").write_text(
            "".join(page), encoding="utf-8")
        plans.append({"question_id": qid, "question": question["question"],
                      "gold_memory_ids": target_ids,
                      "similar_ids": [mid for mid, _, _ in similar],
                      "unrelated_ids": [mid for mid, _, _ in unrelated]})
        summary.append({"question_id": qid, "gold_count": len(target_ids),
                        "pool_count": len(candidates),
                        "similar_unrelated_overlap_count": len(group_overlap),
                        "most_similar_cosine": round(similar[0][2], 8),
                        "last_similar_cosine": round(similar[-1][2], 8),
                        "least_similar_cosine": round(unrelated[0][2], 8),
                        "last_unrelated_cosine": round(unrelated[-1][2], 8)})
        index_rows.append(f'<li><a href="questions/{html.escape(qid.split("/")[-1])}.html">'
                          f'{html.escape(qid.split("/")[-1])}</a> · '
                          f'{html.escape(question["question"])}</li>')
    write_csv(OUT / "review_candidates.csv", rows)
    write_csv(OUT / "question_summary.csv", summary)
    with (OUT / "candidate_order.jsonl").open("w", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(json.dumps(plan, ensure_ascii=False) + "\n")
    (OUT / "index.html").write_text(
        page_header(f"{TASK} distractor review · all questions")
        + f'<h1>{html.escape(TASK)} · {len(questions)}-question distractor review</h1>'
        + f'<p>Each question has {N} visually similar and {N} visually unrelated original images. '
        'They are candidates, not validated distractors. Review original images, captions and '
        'dialogue before any answer test. No answer or option text appears in this pack.</p><ol>'
        + "".join(index_rows) + "</ol></html>", encoding="utf-8")
    protocol = {
        "study": f"{TASK} original-image candidate review before 1-to-{N} retrieval test",
        "scope": {"questions": len(questions), "total_domain_questions": len(all_questions),
                  "excluded_question_ids_with_fewer_than_N_non_targets": excluded,
                  "questions_with_similar_unrelated_overlap": sum(
                      row["similar_unrelated_overlap_count"] > 0 for row in summary),
                  "memory_nodes": len(memories), "memory_images": len(source_paths),
                  "candidate_rows": len(rows), "similar_per_question": N,
                  "unrelated_per_question": N},
        "ranking": "similar descending, unrelated ascending by max CLIP image cosine against all annotated gold images; tie by memory ID",
        "repeated_original_image_policy": "one candidate memory ID per unique image SHA-256, choosing lexicographically first ID; all gold memory IDs retained",
        "selection_inputs": f"original {TASK} images and annotated gold IDs only",
        "answers_options_retrieval_scores_used": False,
        "caption_word_overlap": "literal reviewer hint, not a selection or validity criterion",
        "review_status": "pending; blank human columns; previous pilot passes flagged as metadata only",
        "device": DEVICE,
        "after_review_test": f"original caption only; add distractors in frozen order at N=1..{N}; compare weighted Top-3 target rank on the same {TASK} question cohort",
        "visual_encoder": model_metadata,
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            NAT / "memories.jsonl", NAT / "images.jsonl", NAT / "captions.jsonl",
            MCQ / "mcq_queries.jsonl", MCQ / "retrieval_details.jsonl", DIALOG,
            ENCODER_PROTOCOL)},
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(protocol["scope"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    N = args.n
    TASK = args.task
    DEVICE = args.device
    OUT = args.out.resolve()
    main()
