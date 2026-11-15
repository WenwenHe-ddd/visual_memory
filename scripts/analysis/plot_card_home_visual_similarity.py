"""Show Card Q18 examples and matched Card/Home visual-similarity heatmaps.

The similarity is the frozen image-to-image CLIP cosine to the nearest
annotated target image in each question, as stored in review_candidates.csv.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import Normalize
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/card_home_visual_similarity_20261103"
SOURCES = {
    "Card": ROOT / "results/card_all_questions_distractor_review_20261101/review_candidates.csv",
    "Home": ROOT / "results/home_15_distractor_review_20261031/review_candidates.csv",
}
EXAMPLE = "Card_Playlog_Test/Q18"
N = 15
NORM = Normalize(vmin=0.40, vmax=1.00)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def install_times() -> None:
    font_dir = Path.home() / ".local/share/fonts/TimesNewRoman"
    for name in ("Times.TTF", "Timesbd.TTF", "Timesi.TTF", "Timesbi.TTF"):
        path = font_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required Times New Roman font: {path}")
        font_manager.fontManager.addfont(str(path))
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 12,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_image(path: str, expected_sha256: str) -> Image.Image:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Image hash mismatch: {source}")
    return Image.open(source).convert("RGB")


def plot_examples(card_rows: list[dict[str, str]]) -> list[dict]:
    rows = [row for row in card_rows if row["question_id"] == EXAMPLE]
    if len(rows) != 2 * N:
        raise ValueError(f"Expected {2*N} candidate rows for {EXAMPLE}")
    by_group = {}
    for condition in ("similar", "unrelated"):
        group = sorted((row for row in rows if row["condition"] == condition),
                       key=lambda row: int(row["add_order"]))
        if len(group) != N:
            raise ValueError(f"Incomplete {condition} group")
        by_group[condition] = group[:3]

    gold_paths = rows[0]["gold_image_paths"].split(" | ")
    gold_ids = rows[0]["gold_memory_ids"].split(" | ")
    if len(gold_paths) != 2 or len(gold_ids) != 2:
        raise ValueError("Card Q18 gold images changed")
    fig, axes = plt.subplots(3, 3, figsize=(13.4, 10.5), dpi=200)
    fig.patch.set_facecolor("white")
    output_rows = []
    for col, (memory_id, path) in enumerate(zip(gold_ids, gold_paths)):
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        axes[0, col].imshow(Image.open(source).convert("RGB"))
        axes[0, col].set_title(f"Target {col+1}: {memory_id.split('/')[-1]}",
                               weight="bold", fontsize=13)
    axes[0, 2].axis("off")
    axes[0, 2].text(.06, .66, "Card Q18", weight="bold", fontsize=20,
                    transform=axes[0, 2].transAxes)
    axes[0, 2].text(.06, .38, "Nearest-target image cosine\n(two target images)",
                    fontsize=12, transform=axes[0, 2].transAxes)

    for row_index, (condition, label) in enumerate((
        ("similar", "Most similar"), ("unrelated", "Least similar")
    ), start=1):
        for col, row in enumerate(by_group[condition]):
            ax = axes[row_index, col]
            ax.imshow(check_image(row["candidate_image_path"], row["candidate_image_sha256"]))
            ax.set_title(f"{label} {col+1}  |  {float(row['clip_visual_cosine_to_nearest_gold']):.3f}",
                         weight="bold", fontsize=12.5, color="#185C40" if row_index == 1 else "#A18E0C")
            ax.text(.02, .03, row["candidate_memory_id"].split("/")[-1],
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
                    color="white", weight="bold",
                    bbox={"facecolor": "black", "alpha": .68, "edgecolor": "none", "pad": 3})
            output_rows.append({
                "question_id": EXAMPLE, "group": label, "rank": col + 1,
                "memory_id": row["candidate_memory_id"],
                "image_path": row["candidate_image_path"],
                "image_to_nearest_target_clip_cosine": row["clip_visual_cosine_to_nearest_gold"],
            })
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=.03, right=.97, top=.95, bottom=.045, hspace=.22, wspace=.09)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"card_q18_six_candidates.{ext}", dpi=250, facecolor="white")
    plt.close(fig)
    return output_rows


def matrices() -> tuple[dict[str, np.ndarray], dict[str, list[str]], dict[str, dict]]:
    result = {}
    ids = {}
    stats = {}
    for domain, path in SOURCES.items():
        rows = read_rows(path)
        by_question = {}
        for row in rows:
            key = row["question_id"]
            by_question.setdefault(key, {"similar": {}, "unrelated": {}})
            group = by_question[key][row["condition"]]
            rank = int(row["add_order"])
            if rank in group:
                raise ValueError(f"Duplicate rank in {domain}, {key}, {row['condition']}")
            group[rank] = float(row["clip_visual_cosine_to_nearest_gold"])
        question_ids = sorted(by_question, key=lambda qid: int(qid.split("/Q")[-1]))
        ids[domain] = question_ids
        matrix = []
        for qid in question_ids:
            groups = by_question[qid]
            if any(set(groups[condition]) != set(range(1, N + 1)) for condition in groups):
                raise ValueError(f"Incomplete rank sequence: {qid}")
            matrix.append([groups["similar"][rank] for rank in range(1, N + 1)] +
                          [groups["unrelated"][rank] for rank in range(1, N + 1)])
        result[domain] = np.array(matrix)
        stats[domain] = {
            "question_count": len(question_ids), "cells": int(result[domain].size),
            "minimum": float(result[domain].min()),
            "maximum": float(result[domain].max()),
            "similar_median": float(np.median(result[domain][:, :N])),
            "dissimilar_median": float(np.median(result[domain][:, N:])),
            "similar_first3_median": float(np.median(result[domain][:, :3])),
            "dissimilar_first3_median": float(np.median(result[domain][:, N:N+3])),
        }
    return result, ids, stats


def style_heatmap(ax, domain: str, matrix: np.ndarray, question_ids: list[str]) -> None:
    ax.imshow(matrix, cmap="cividis", norm=NORM, aspect="auto", interpolation="nearest")
    ax.axvline(N - .5, color="white", linewidth=2.0)
    ax.set_title(domain, weight="bold", fontsize=16, pad=12)
    ax.set_xticks([0, 4, 9, 14, 15, 19, 24, 29], [1, 5, 10, 15, 1, 5, 10, 15])
    ax.tick_params(axis="both", length=0, labelsize=10.5)
    ax.set_xlabel("Most similar ranks 1–15     |     Least similar ranks 1–15",
                  weight="bold", labelpad=9)
    ax.set_ylabel("Question", weight="bold")
    ticks = list(range(0, matrix.shape[0], 10))
    if ticks[-1] != matrix.shape[0] - 1:
        ticks.append(matrix.shape[0] - 1)
    ax.set_yticks(ticks, [question_ids[i].split("/")[-1] for i in ticks])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_heatmaps(data: dict[str, np.ndarray], ids: dict[str, list[str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 9.0), dpi=200)
    for ax, domain in zip(axes, ("Card", "Home")):
        style_heatmap(ax, domain, data[domain], ids[domain])
    fig.subplots_adjust(left=.065, right=.9, top=.91, bottom=.105, wspace=.26)
    bar_ax = fig.add_axes([.925, .105, .017, .805])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap="cividis"), cax=bar_ax)
    cb.set_label("Image-to-nearest-target CLIP cosine", weight="bold", labelpad=11)
    cb.ax.tick_params(labelsize=11)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"card_home_similarity_heatmaps.{ext}", dpi=250, facecolor="white")
    plt.close(fig)

    for domain in ("Card", "Home"):
        fig, ax = plt.subplots(figsize=(7.8, 8.5), dpi=200)
        style_heatmap(ax, domain, data[domain], ids[domain])
        fig.subplots_adjust(left=.13, right=.82, top=.92, bottom=.11)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap="cividis"), ax=ax,
                          fraction=.055, pad=.035)
        cb.set_label("Image-to-nearest-target CLIP cosine", weight="bold")
        fig.savefig(OUT / f"{domain.lower()}_similarity_heatmap.png", dpi=250,
                    facecolor="white")
        plt.close(fig)


def main() -> None:
    install_times()
    OUT.mkdir(parents=True, exist_ok=True)
    card_rows = read_rows(SOURCES["Card"])
    examples = plot_examples(card_rows)
    write_csv(OUT / "card_q18_six_candidates.csv", examples)
    data, ids, stats = matrices()
    plot_heatmaps(data, ids)
    (OUT / "similarity_summary.json").write_text(
        json.dumps({"metric": "image-to-image CLIP cosine to nearest annotated target image",
                    "question": EXAMPLE, "common_color_scale": [NORM.vmin, NORM.vmax],
                    "sources": {name: str(path) for name, path in SOURCES.items()},
                    "domains": stats}, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
