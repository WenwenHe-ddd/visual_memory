"""Four-curve eight-domain plot using the existing Original CVM-Similar data."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/all_eight_domain_top3_interference_20261101/all_eight_curves.csv"
OUT = ROOT / "reports/motivation_original_cvm_top3_20261114"
DOMAINS = ("Brand", "Card", "Cartoon", "Home", "Cross-Scene", "Outdoor", "Health", "Social")
SERIES = (
    ("Text-Similar", "Text-Similar Disruption Selection", "#B7A625", "o", "-"),
    ("Visual-Similar", "Visul-Similar Disruption Selection", "#185C40", "s", "-"),
    ("Text-Dissimilar", "No Disruption", "#A6A6A6", "^", "--"),
    ("CVM-Similar", "VAMO-Similar Disruption Selection", "#B13C3C", "*", "-"),
)


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r["domain"], r["series"], int(r["distractors"])): r for r in rows}
    records = []
    for domain in DOMAINS:
        for source, label, _, _, _ in SERIES:
            for n in range(16):
                row = lookup[domain, source, n]
                records.append({**row, "series": label})
    if len(records) != 8 * 4 * 16:
        raise ValueError("Incomplete eight-domain source data")
    if abs(float(lookup["Home", "CVM-Similar", 1]["mean_weighted_rank_score_x100"])
           - 88.46153846153845) > 1e-8:
        raise ValueError("Home reference CVM curve changed")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "motivation_curves.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    font_dir = Path(os.environ.get("MEMEYE_TIMES_FONT_DIR") or
                    Path.home() / ".local/share/fonts/TimesNewRoman")
    for filename in ("Times.TTF", "Timesbd.TTF", "Timesi.TTF", "Timesbi.TTF"):
        font_manager.fontManager.addfont(str(font_dir / filename))
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12,
                         "axes.linewidth": 1.75, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axs = plt.subplots(2, 4, figsize=(18.6, 8.2), sharex=True, dpi=200)
    for ax, domain in zip(axs.flat, DOMAINS):
        values = [float(r["mean_weighted_rank_score_x100"]) for r in records
                  if r["domain"] == domain and int(r["distractors"]) > 0]
        lower = math.floor(min(values) / 10) * 10
        upper = math.ceil(max(values) / 10) * 10
        if lower == upper:
            lower = max(0, lower - 10)
        for _, label, color, marker, style in SERIES:
            points = sorted((r for r in records if r["domain"] == domain
                             and r["series"] == label and int(r["distractors"]) > 0),
                            key=lambda r: int(r["distractors"]))
            opts = {"color": color, "marker": marker, "linestyle": style,
                    "linewidth": 2.3, "markersize": 9.3 if marker == "*" else 5.1,
                    "label": label}
            if marker == "^":
                opts.update(markerfacecolor="#7F807F", markeredgecolor="#7F807F")
            ax.plot([int(r["distractors"]) for r in points],
                    [float(r["mean_weighted_rank_score_x100"]) for r in points], **opts)
        ax.set_title(domain, fontsize=14.5, pad=8)
        ax.set_xlim(.7, 15.3)
        ax.set_ylim(lower - .5, upper + .5)
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        ax.set_yticks(range(lower, upper + 1, 10))
        ax.grid(axis="y", color="#D7D7D7", linestyle=(0, (4, 3)), linewidth=1.3, alpha=.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#888888")
        ax.tick_params(axis="both", width=1.0, length=5)
    for ax in axs[1]:
        ax.set_xlabel("Number of distractors")
    for ax in axs[:, 0]:
        ax.set_ylabel("Target-Rank Score")
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .006),
               ncol=2, frameon=False, handlelength=2.2, columnspacing=2.2,
               prop={"size": 11, "weight": "bold"})
    fig.subplots_adjust(left=.065, right=.99, top=.93, bottom=.20,
                        wspace=.18, hspace=.26)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"motivation_original_cvm_top3.{ext}", dpi=300,
                    facecolor="white")
    plt.close(fig)
    (OUT / "protocol.json").write_text(json.dumps({
        "data_source": str(SOURCE), "domains": DOMAINS,
        "red_series": "Display label: VAMO-Similar Disruption Selection; source series: CVM-Similar, existing caption_visual_master raw results",
        "home_reference": "matches home_top3_interference_15_comparison_20261031.xlsx column D",
        "measure": "mean weighted target-rank score multiplied by 100",
    }, ensure_ascii=False, indent=2) + "\n")
    print(OUT / "motivation_original_cvm_top3.png")


if __name__ == "__main__":
    main()
