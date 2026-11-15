"""Home-only plots: workbook reference and separate VAMO candidate-selection run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "results/home_top3_interference_15_comparison_20261031.xlsx"
VAMO = ROOT / "results/vamo_target_selection_top3_20261114/home/evaluation/summary.csv"
OUT = ROOT / "reports/home_vamo_target_comparison_20261114"
SERIES = (
    ("Text-Similar Disruption Selection", "#B7A625", "o", "-"),
    ("Visul-Similar Disruption Selection", "#185C40", "s", "-"),
    ("Text-Dissimilar", "#A6A6A6", "^", "--"),
    ("VAMO Disruption Selection", "#B13C3C", "*", "-"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    workbook = load_workbook(BOOK, read_only=True, data_only=True)
    sheet = workbook["Home_1至15汇总"]
    headers = [cell.value for cell in sheet[1]]
    expected = {1: "相似组·文本顺序", 2: "相似组·图片顺序", 5: "无关组·文本顺序"}
    for column, name in expected.items():
        if headers[column] != name:
            raise ValueError(f"Home workbook column changed: {column + 1}")
    summary = {}
    with VAMO.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["condition"] == "similar" and row["insertion_method"] == "vamo_target_text":
                if int(row["questions"]) != 52:
                    raise ValueError("Home VAMO cohort changed")
                summary[int(row["distractors_added"])] = float(row["mean_rank_score"]) * 100
    if set(summary) != set(range(16)) or summary[0] != 100:
        raise ValueError("Incomplete VAMO Home curve")
    records = []
    for row in sheet.iter_rows(min_row=2, max_row=16, values_only=True):
        n = row[0]
        if n != len(records) + 1:
            raise ValueError("Workbook distractor sequence changed")
        records.append({"distractors": n,
                        SERIES[0][0]: row[1] * 100,
                        SERIES[1][0]: row[2] * 100,
                        SERIES[2][0]: row[5] * 100,
                        SERIES[3][0]: summary[n],
                        "Original CVM (reference red line)": row[3] * 100})
    workbook.close()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "home_curve_data.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    font_dir = Path(os.environ.get("MEMEYE_TIMES_FONT_DIR") or
                    Path.home() / ".local/share/fonts/TimesNewRoman")
    for filename in ("Times.TTF", "Timesbd.TTF", "Timesi.TTF", "Timesbi.TTF"):
        font_manager.fontManager.addfont(str(font_dir / filename))
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12,
                         "axes.linewidth": 1.75, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10.1, 5.0), dpi=200)
    for label, color, marker, line in SERIES:
        options = {"color": color, "marker": marker, "linestyle": line,
                   "linewidth": 2.3, "markersize": 9.3 if marker == "*" else 5.3,
                   "label": label}
        if marker == "^":
            options.update(markerfacecolor="#7F807F", markeredgecolor="#7F807F")
        ax.plot([row["distractors"] for row in records],
                [row[label] for row in records], **options)
    all_values = [row[label] for row in records for label, *_ in SERIES]
    lower = math.floor(min(all_values) / 10) * 10
    upper = math.ceil(max(all_values) / 10) * 10
    ax.set_xlim(.7, 15.3)
    ax.set_ylim(lower - .5, upper + .5)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.set_yticks(range(lower, upper + 1, 10))
    ax.set_xlabel("Number of distractors")
    ax.set_ylabel("Target-Rank Score")
    ax.grid(axis="y", color="#D7D7D7", linestyle=(0, (4, 3)), linewidth=1.3, alpha=.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#888888")
    ax.tick_params(axis="both", width=1.0, length=5)
    ax.legend(loc="upper right", frameon=False, prop={"size": 10, "weight": "bold"})
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"home_vamo_target_comparison.{extension}", dpi=300,
                    facecolor="white")
    plt.close(fig)
    (OUT / "protocol.json").write_text(json.dumps({
        "scope": "Home only; 52 questions; N=1..15",
        "workbook_baselines": "Home_1至15汇总 columns B, C, F (multiplied by 100)",
        "reference_red_line": "workbook column D is Original CVM, not VAMO; included in CSV for audit only",
        "vamo_red_line": "VAMO reconstructed latent selects 15 candidates from Home full pool; original-caption BGE orders candidates and scores Top-3",
        "workbook_sha256": digest(BOOK), "vamo_summary_sha256": digest(VAMO),
        "human_review": "VAMO-selected candidates have not been individually reviewed",
    }, ensure_ascii=False, indent=2) + "\n")
    print(OUT / "home_vamo_target_comparison.png")

    # The user-provided reference red line matches workbook column D exactly.
    # Render it as CVM so its source is unambiguous; this is not a VAMO result.
    fig, ax = plt.subplots(figsize=(10.1, 5.0), dpi=200)
    reference_series = SERIES[:3] + (("Original CVM", "#B13C3C", "*", "-"),)
    for label, color, marker, line in reference_series:
        values = ([row["Original CVM (reference red line)"] for row in records]
                  if label == "Original CVM" else [row[label] for row in records])
        options = {"color": color, "marker": marker, "linestyle": line,
                   "linewidth": 2.3, "markersize": 9.3 if marker == "*" else 5.3,
                   "label": label}
        if marker == "^":
            options.update(markerfacecolor="#7F807F", markeredgecolor="#7F807F")
        ax.plot([row["distractors"] for row in records], values, **options)
    ax.set_xlim(.7, 15.3)
    ax.set_ylim(70 - .5, 100 + .5)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.set_yticks([70, 80, 90, 100])
    ax.set_xlabel("Number of distractors")
    ax.set_ylabel("Target-Rank Score")
    ax.grid(axis="y", color="#D7D7D7", linestyle=(0, (4, 3)), linewidth=1.3, alpha=.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#888888")
    ax.tick_params(axis="both", width=1.0, length=5)
    ax.legend(loc="upper right", frameon=False, prop={"size": 10, "weight": "bold"})
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"home_workbook_reference.{extension}", dpi=300,
                    facecolor="white")
    plt.close(fig)
    print(OUT / "home_workbook_reference.png")


if __name__ == "__main__":
    main()
