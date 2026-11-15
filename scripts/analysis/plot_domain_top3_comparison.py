"""Compare frozen-caption Top-3 distractor curves for Home and Cartoon."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "Home (52 questions)": ROOT / "results/home_top3_interference_15_20261031/summary.csv",
    "Cartoon (76 questions)": ROOT / "results/cartoon_top3_interference_15_20261101/summary.csv",
}
SERIES = (
    ("similar", "text", "SemRAG (Disruption)", "#185C40", "o"),
    ("similar", "image", "Modality-Preserving (Disruption)", "#B7A626", "s"),
    ("unrelated", "text", "SemRAG (Irrelevant)", "#63869A", "^"),
)
OUT = ROOT / "reports/home_cartoon_top3_comparison_20261101"


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.linewidth": 1.4, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=True, dpi=200)
    for ax, (domain, source) in zip(axes, SOURCES.items()):
        with source.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for condition, method, label, color, marker in SERIES:
            selected = sorted((row for row in rows if row["condition"] == condition
                               and row["insertion_method"] == method
                               and 1 <= int(row["distractors_added"]) <= 15),
                              key=lambda row: int(row["distractors_added"]))
            if len(selected) != 15:
                raise ValueError(f"Incomplete curve: {domain}, {label}")
            ax.plot([int(row["distractors_added"]) for row in selected],
                    [float(row["mean_rank_score"]) * 100 for row in selected],
                    color=color, marker=marker, markersize=4.4, linewidth=2,
                    label=label)
        ax.set_title(domain, fontsize=12, pad=11)
        ax.set_xlim(.7, 15.3)
        ax.set_ylim(65, 101)
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        ax.set_yticks([70, 80, 90, 100])
        ax.grid(axis="y", color="#BABDBB", linestyle=(0, (4, 3)), alpha=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Number of distractor images")
    axes[0].set_ylabel("Mean weighted target-rank score")
    axes[1].legend(loc="upper right", fontsize=8.6, frameon=True,
                   facecolor="white", edgecolor="#AAAAAA")
    fig.subplots_adjust(left=.075, right=.985, bottom=.17, top=.88, wspace=.14)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"home_cartoon_top3_comparison.{ext}", dpi=300,
                    facecolor="white")
    plt.close(fig)
    print(OUT / "home_cartoon_top3_comparison.png")


if __name__ == "__main__":
    main()
