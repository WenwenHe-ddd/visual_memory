"""Plot three Home Top-3 interference curves with distinct markers."""

from __future__ import annotations

import csv
from collections import defaultdict
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/home_top3_interference_15_20261031/per_question.csv"
OUT = ROOT / "reports/home_top3_three_curves_20261031"

SERIES = (
    ("similar", "text", "SemRAG (Disruption)", "#185C40", "o"),
    ("similar", "image", "Modality-Preserving (Disruption)", "#B7A626", "s"),
    ("unrelated", "text", "SemRAG (Irrelevant)", "#63869A", "^"),
)


class LineKey(Artist):
    def __init__(self, color: str, marker: str) -> None:
        super().__init__()
        self.color = color
        self.marker = marker


class LineKeyHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        color = orig_handle.color
        line = Line2D([xdescent + width * .12, xdescent + width * .88],
                      [ydescent + height * .5] * 2, color=color, linewidth=2.3,
                      marker=orig_handle.marker, markersize=4.8, markerfacecolor=color,
                      markeredgewidth=0, transform=trans)
        return [line]


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 52 * 2 * 2 * 16:
        raise ValueError("Expected the full 52-question, N=0..15 retrieval record")
    grouped = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["insertion_method"], int(row["distractors_added"]))
        grouped[key].append(float(row["rank_score"]) * 100)
    data = []
    for condition, method, label, color, marker in SERIES:
        for n in range(1, 16):
            values = grouped[(condition, method, n)]
            if len(values) != 52:
                raise ValueError(f"Expected 52 question scores: {condition} {method} N={n}")
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            sd = math.sqrt(variance)
            data.append({"series": label, "distractors": n, "questions": len(values),
                         "minimum": min(values), "mean": mean, "maximum": max(values),
                         "population_variance": variance, "population_sd": sd,
                         "band_lower": mean - sd, "band_upper": mean + sd})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "plot_data.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 1.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(12.8, 5.0), dpi=200)
    fig.subplots_adjust(left=.09, right=.64, bottom=.17, top=.94)
    for condition, method, label, color, marker in SERIES:
        selected = [row for row in data if row["series"] == label]
        x = [row["distractors"] for row in selected]
        mean = [row["mean"] for row in selected]
        ax.plot(x, mean, color=color, linewidth=2.5, marker=marker, markersize=5.2,
                markeredgewidth=0, zorder=3, clip_on=False)
    ax.set_xlim(.65, 15.35)
    ax.set_ylim(70, 100)
    ax.set_xticks(range(1, 16, 2))
    ax.set_yticks([70, 80, 90, 100])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.grid(axis="y", color="#B5B8B6", linestyle=(0, (4, 3)), linewidth=1,
            alpha=.82, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=5, width=1.2)
    ax.set_xlabel("Number of distractor images", fontsize=12, labelpad=8)
    ax.set_ylabel("Mean weighted target-rank score", fontsize=12, labelpad=8)
    handles = [LineKey(color, marker) for _, _, _, color, marker in SERIES]
    labels = [label for _, _, label, _, _ in SERIES]
    legend = ax.legend(handles, labels, title="Retrieval condition",
                       handler_map={LineKey: LineKeyHandler()},
                       loc="upper left", bbox_to_anchor=(1.035, .96),
                       borderaxespad=0, frameon=False, handlelength=2.25,
                       handleheight=1.2, labelspacing=1.15,
                       fontsize=10.2, title_fontsize=10.3)
    legend._legend_box.align = "left"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"home_top3_three_curves.{extension}", dpi=300,
                    facecolor="white")
    plt.close(fig)
    print(f"{OUT / 'home_top3_three_curves.png'} · {len(data)} plotted points")


if __name__ == "__main__":
    main()
