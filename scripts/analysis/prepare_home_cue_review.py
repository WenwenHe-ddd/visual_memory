"""Prepare a small, answer-free Home distractor review pack from original images.

Caption similarity is used only to surface nearby scenes for human review. It is
not a cue-overlap label, retrieval result, or acceptance decision.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/natural_observation"
QUESTIONS = ROOT / "results/mcq_fifth_non_target_rank"
OUT = ROOT / "results/home_cue_review_20261031"
TASK = "Home_Renovation_Interior_Design"
# Single-target questions covering several repeated visual scenes in this task.
QUESTION_NUMBERS = (10, 17, 22, 49, 55)
NEAR_COUNT = 4
CONTROL_COUNT = 2


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keywords(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z]+", text.lower())
        if len(word) > 2 and word not in ENGLISH_STOP_WORDS
    }


def make_sheet(question: str, cards: list[dict], path: Path) -> None:
    width, height = 590, 410
    rows = (len(cards) + 2) // 3
    canvas = Image.new("RGB", (width * 3, height * rows + 96), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 17)
    small = ImageFont.truetype(font_path, 14)
    draw.text((18, 15), question[:90], fill="black", font=font)
    if len(question) > 90:
        draw.text((18, 43), question[90:180], fill="black", font=small)
    for index, card in enumerate(cards):
        x, y = (index % 3) * width, 96 + (index // 3) * height
        with Image.open(card["image_path"]) as original:
            img = ImageOps.contain(original.convert("RGB"), (width - 20, height - 70))
        canvas.paste(img, (x + (width - img.width) // 2, y + 8))
        label = f'{card["role"]}: {card["memory_id"].split("/")[-1]}  {Path(card["image_path"]).name}'
        draw.text((x + 10, y + height - 54), label, fill="black", font=small)
        draw.text((x + 10, y + height - 32), f'SHA256 {card["image_sha256"][:12]}', fill="#555555", font=small)
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline="#bbbbbb", width=2)
    canvas.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    memories = [row for row in read_jsonl(SOURCE / "memories.jsonl") if row["task"] == TASK]
    images = {row["id"]: row for row in read_jsonl(SOURCE / "images.jsonl")}
    captions = {row["id"]: row["caption"] for row in read_jsonl(SOURCE / "captions.jsonl")}
    questions = {row["id"]: row["question"] for row in read_jsonl(QUESTIONS / "mcq_queries.jsonl")}
    gold = {row["id"]: row["gold_ids"] for row in read_jsonl(QUESTIONS / "retrieval_details.jsonl")}
    memory_by_id = {row["id"]: row for row in memories}
    texts = [captions[row["images"][0]] for row in memories]
    vectors = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
    location = {row["id"]: index for index, row in enumerate(memories)}
    rows: list[dict] = []
    pages: list[str] = []
    for number in QUESTION_NUMBERS:
        qid = f"{TASK}/Q{number}"
        target_ids = gold[qid]
        if len(target_ids) != 1:
            raise ValueError(f"Pilot requires one annotated target: {qid}")
        target_id = target_ids[0]
        target = memory_by_id[target_id]
        target_image = target["images"][0]
        target_caption = captions[target_image]
        target_path = Path(images[target_image]["path"])
        if sha(target_path) != target_image:
            raise ValueError(f"Target bytes changed: {target_id}")
        scores = (vectors @ vectors[location[target_id]].T).toarray().ravel()
        candidates = [index for index, row in enumerate(memories)
                      if row["id"] not in target_ids and row["images"][0] != target_image]
        candidates.sort(key=lambda index: (-float(scores[index]), memories[index]["id"]))
        near = candidates[:NEAR_COUNT]
        q_words = keywords(questions[qid])
        remaining = [index for index in candidates if index not in near]
        low_pool = [index for index in remaining
                    if not q_words.intersection(keywords(texts[index]))]
        if len(low_pool) < CONTROL_COUNT:
            low_pool = remaining
        low_pool.sort(key=lambda index: (
            float(scores[index]), abs(len(texts[index].split()) - len(target_caption.split())),
            memories[index]["id"]))
        controls = low_pool[:CONTROL_COUNT]
        cards = [{"role": "TARGET", "memory_id": target_id,
                  "image_path": str(target_path), "image_sha256": target_image}]
        details = []
        for role, indices in (("near_candidate", near), ("low_scene_control", controls)):
            for index in indices:
                memory = memories[index]
                image_id = memory["images"][0]
                image_path = Path(images[image_id]["path"])
                if sha(image_path) != image_id:
                    raise ValueError(f"Candidate bytes changed: {memory['id']}")
                shared_words = sorted(q_words & keywords(texts[index]))
                record = {
                    "question_id": qid, "question": questions[qid], "target_id": target_id,
                    "target_image_path": str(target_path), "target_caption": target_caption,
                    "candidate_id": memory["id"], "candidate_image_path": str(image_path),
                    "candidate_caption": texts[index], "discovery_role": role,
                    "target_caption_tfidf_cosine": round(float(scores[index]), 6),
                    "question_caption_shared_words": ", ".join(shared_words),
                    "human_shared_locator_cues": "", "human_distinguishing_locator": "",
                    "human_cue_overlap_level": "", "human_is_valid_non_target": "",
                    "human_caption_supported_by_image": "", "human_accept": "",
                    "human_notes": "",
                }
                rows.append(record)
                details.append(record)
                cards.append({"role": role, "memory_id": memory["id"],
                              "image_path": str(image_path), "image_sha256": image_id})
        make_sheet(questions[qid], cards, OUT / f"Q{number}_contact_sheet.png")
        entries = []
        for record in details:
            entries.append(
                f'<tr><td>{html.escape(record["discovery_role"])}</td>'
                f'<td>{html.escape(record["candidate_id"])}</td>'
                f'<td>{record["target_caption_tfidf_cosine"]:.3f}</td>'
                f'<td>{html.escape(record["question_caption_shared_words"])}</td>'
                f'<td>{html.escape(record["candidate_caption"])}</td></tr>')
        pages.append(
            f'<h2>Q{number}</h2><p><b>Question:</b> {html.escape(questions[qid])}</p>'
            f'<p><b>Target:</b> {html.escape(target_id)}<br><b>Target caption:</b> '
            f'{html.escape(target_caption)}</p>'
            f'<img src="Q{number}_contact_sheet.png" alt="Q{number} contact sheet">'
            '<table><tr><th>Discovery role</th><th>Candidate ID</th><th>Target-caption similarity</th>'
            '<th>Exact question-word overlap</th><th>Candidate caption</th></tr>'
            + "".join(entries) + "</table>")
    with (OUT / "review_candidates.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Home cue review</title>'
        '<style>body{font:16px/1.5 Arial,sans-serif;max-width:1800px;margin:auto;padding:24px}'
        'img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}'
        'td,th{border:1px solid #bbb;padding:8px;vertical-align:top}td:last-child{max-width:600px}'
        'h2{margin-top:48px}</style><h1>Home · distractor candidate review</h1>'
        '<p>All candidate labels are unverified. Similarity screens scenes only; human review must '
        'confirm shared query locators and exclude alternate valid evidence. No answers, options, '
        'or retrieval ranking scores were used to select candidates.</p>'
        + "".join(pages) + "</html>", encoding="utf-8")
    provenance = {
        "task": TASK, "questions": [f"{TASK}/Q{n}" for n in QUESTION_NUMBERS],
        "candidates_per_question": {"near_scene": NEAR_COUNT, "low_scene_control": CONTROL_COUNT},
        "selection": "Qwen-caption TF-IDF nearest scenes; low controls have no exact non-stop question word where available. These are discovery labels only.",
        "model_calls": 0, "human_accepted": 0, "answers_or_options_used": False,
        "retrieval_rank_or_query_embedding_used": False,
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            SOURCE / "memories.jsonl", SOURCE / "images.jsonl", SOURCE / "captions.jsonl",
            QUESTIONS / "mcq_queries.jsonl", QUESTIONS / "retrieval_details.jsonl")},
    }
    (OUT / "protocol.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"questions": len(QUESTION_NUMBERS), "candidate_rows": len(rows),
                      "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
