"""Export editable five-curve Cartoon interference table, chart, and figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.marker import DataPoint
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/cartoon_top3_interference_15_20261101"
CVM = ROOT / "results/cartoon_top3_interference_cvm_zscore_20261101"
OUT = ROOT / "reports/cartoon_top3_five_curves_20261101"
SERIES = (
    ("similar", "text", "SemRAG (Disruption)", BASE, "#185C40", "o"),
    ("similar", "image", "Modality-Preserving (Disruption)", BASE, "#B7A626", "s"),
    ("similar", "caption_visual_master", "Original CVM (Disruption)", CVM, "#AD5749", "D"),
    ("similar", "caption_visual_master_zscore_l0p03_a1", "CVM + z-score (Disruption)", CVM, "#745B9B", "x"),
    ("unrelated", "text", "SemRAG (Irrelevant)", BASE, "#63869A", "^"),
)


def main() -> None:
    indexed = {}
    for source in (BASE, CVM):
        with (source / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        indexed[source] = {(row["condition"], row["insertion_method"],
                            int(row["distractors_added"])): row for row in rows}
    table = []
    for n in range(1, 16):
        row = {"Number of distractors": n}
        for condition, method, name, source, _, _ in SERIES:
            item = indexed[source][condition, method, n]
            if int(item["questions"]) != 76:
                raise ValueError("Incomplete Cartoon question cohort")
            row[name] = float(item["mean_rank_score"]) * 100
        table.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "cartoon_five_curve_table.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.linewidth": 1.4, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12.6, 5.0), dpi=200)
    fig.subplots_adjust(left=.085, right=.69, bottom=.17, top=.94)
    for _, _, name, _, color, marker in SERIES:
        ax.plot([r["Number of distractors"] for r in table], [r[name] for r in table],
                color=color, marker=marker, markersize=5, linewidth=2.2,
                linestyle="--" if name.startswith("CVM +") else "-", label=name)
    ax.set_xlim(.7, 15.3)
    ax.set_ylim(65, 101)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.set_yticks([70, 80, 90, 100])
    ax.grid(axis="y", color="#BABDBB", linestyle=(0, (4, 3)), alpha=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Number of distractor images")
    ax.set_ylabel("Mean weighted target-rank score")
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, .98), frameon=False,
              fontsize=10, labelspacing=1.0)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"cartoon_five_curves.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    wb = Workbook()
    ws = wb.active
    ws.title = "Cartoon_Top3"
    ws.append(list(table[0]))
    for row in table:
        ws.append([row[col] for col in table[0]])
    ws.freeze_panes = "B2"
    for cell in ws[1]:
        cell.font = Font(name="Times New Roman", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="244A44")
    for col in range(1, 7):
        ws.column_dimensions[get_column_letter(col)].width = 27 if col == 1 else 39
        for cells in ws.iter_cols(min_col=col, max_col=col, min_row=2, max_row=16):
            for cell in cells:
                cell.font = Font(name="Times New Roman")
                cell.number_format = "0" if col == 1 else "0.00"
    chart = LineChart()
    chart.title = None
    chart.y_axis.title = "Mean weighted target-rank score"
    chart.x_axis.title = "Number of distractor images"
    chart.y_axis.scaling.min = 65
    chart.y_axis.scaling.max = 101
    chart.y_axis.majorUnit = 10
    chart.height = 12
    chart.width = 22
    chart.legend.position = "r"
    chart.add_data(Reference(ws, min_col=2, max_col=6, min_row=1, max_row=16),
                   titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=16))
    markers = ("circle", "square", "diamond", "x", "triangle")
    for series, (_, _, _, _, color, _) , symbol in zip(chart.series, SERIES, markers):
        series.graphicalProperties.line.solidFill = color.strip("#")
        series.graphicalProperties.line.width = 24000
        series.marker.symbol = symbol
        series.marker.size = 6
        series.marker.graphicalProperties.solidFill = color.strip("#")
        series.marker.graphicalProperties.line.solidFill = color.strip("#")
        series.smooth = False
    ws.add_chart(chart, "H2")
    wb.save(OUT / "cartoon_five_curves.xlsx")
    print(OUT / "cartoon_five_curves.xlsx")


if __name__ == "__main__":
    main()
