"""Export either matched Top-3 interference variant to an editable workbook."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[2]
OUT_RAW = ROOT / "reports/all_eight_domain_top3_interference_20261101"
OUT_ZSCORE = ROOT / "reports/all_eight_domain_top3_interference_zscore_20261103"
DOMAINS = ("Brand", "Card", "Cartoon", "Home", "Cross-Scene",
           "Outdoor", "Health", "Social")
BASE_SERIES = ("Text-Similar", "Visual-Similar", "Text-Dissimilar",
               "Visual-Dissimilar")
RAW_SERIES = (*BASE_SERIES, "CVM-Similar", "CVM-Dissimilar")
ZSCORE_SERIES = (*BASE_SERIES, "CVM+z-score Similar", "CVM+z-score Dissimilar")
COLORS = {
    "Text-Similar": "B7A625", "Visual-Similar": "185C40",
    "Text-Dissimilar": "B7A625", "Visual-Dissimilar": "185C40",
    "CVM-Similar": "B13C3C", "CVM-Dissimilar": "B13C3C",
    "CVM+z-score Similar": "B13C3C", "CVM+z-score Dissimilar": "B13C3C",
}


def header(ws, values: list[str]) -> None:
    ws.append(values)
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor=COLORS.get(str(cell.value), "263348"))
        cell.font = Font(name="Times New Roman", size=11, bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=Side(style="medium", color="9099A3"))
    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "D2" if ws.title == "All_Domains" else "B2"


def style_body(ws, widths: list[int], *, percent_free: bool = True) -> None:
    for col, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Times New Roman", size=10.5)
            cell.alignment = Alignment(vertical="center")
            if isinstance(cell.value, float) and percent_free:
                cell.number_format = "0.00"
        if row[0].row % 2 == 0:
            for cell in row:
                cell.fill = PatternFill("solid", fgColor="F5F7F8")
    ws.auto_filter.ref = ws.dimensions


def main(variant: str = "raw") -> None:
    if variant == "zscore":
        out, series_order = OUT_ZSCORE, ZSCORE_SERIES
        target = out / "eight_domain_top3_interference_zscore_20261103.xlsx"
    elif variant == "raw":
        out, series_order = OUT_RAW, RAW_SERIES
        target = out / "eight_domain_top3_interference_20261103.xlsx"
    else:
        raise ValueError(f"Unknown CVM variant: {variant}")
    with (out / "all_eight_curves.csv").open(encoding="utf-8-sig", newline="") as stream:
        curves = list(csv.DictReader(stream))
    if len(curves) != 8 * 6 * 16:
        raise ValueError("Expected eight domains, six series, and N=0..15")
    by_key = {}
    counts = {}
    tasks = {}
    for row in curves:
        domain, series, n = row["domain"], row["series"], int(row["distractors"])
        key = (domain, series, n)
        if key in by_key or domain not in DOMAINS or series not in series_order or n not in range(16):
            raise ValueError(f"Unexpected or duplicate row: {key}")
        by_key[key] = row
        if domain in counts and counts[domain] != int(row["questions"]):
            raise ValueError(f"Question cohort changed in {domain}")
        counts[domain] = int(row["questions"])
        tasks[domain] = row["task"]
    if sum(counts.values()) != 366:
        raise ValueError("Eight-domain question count changed")

    wb = Workbook()
    ws = wb.active
    ws.title = "All_Domains"
    header(ws, ["Domain", "Task", "Questions", "Distractors N", *series_order])
    for domain in DOMAINS:
        for n in range(16):
            ws.append([domain, tasks[domain], counts[domain], n, *[
                float(by_key[domain, series, n]["mean_weighted_rank_score_x100"])
                for series in series_order
            ]])
    style_body(ws, [18, 45, 13, 16, 19, 19, 21, 23, 18, 21])

    for domain in DOMAINS:
        domain_ws = wb.create_sheet(domain)
        header(domain_ws, ["Distractors N", *series_order])
        for n in range(16):
            domain_ws.append([n, *[
                float(by_key[domain, series, n]["mean_weighted_rank_score_x100"])
                for series in series_order
            ]])
        style_body(domain_ws, [18, 19, 19, 21, 23, 18, 21])

    detail = wb.create_sheet("Curve_Data")
    header(detail, ["Domain", "Task", "Questions", "Condition", "Insertion method",
                    "Series", "Distractors N", "Target-rank score x100", "Target in Top-3 x100"])
    for domain in DOMAINS:
        for series in series_order:
            for n in range(16):
                row = by_key[domain, series, n]
                detail.append([domain, tasks[domain], counts[domain], row["condition"],
                               row["insertion_method"], series, n,
                               float(row["mean_weighted_rank_score_x100"]),
                               float(row["gold_in_top3_rate_x100"])])
    style_body(detail, [18, 45, 13, 16, 27, 22, 16, 24, 23])

    notes = wb.create_sheet("Protocol")
    header(notes, ["Item", "Meaning"])
    notes.append(["Variant", "CVM + z-score (lambda=0.03, alpha=1)" if variant == "zscore" else "Raw CVM"])
    notes.append(["Question pool", "366 questions across eight domains; fixed per-domain cohorts."])
    notes.append(["Memory pool", "All annotated target images for the question plus the first N candidate distractors."])
    notes.append(["Score", "Best target rank: 1 -> 100, 2 -> 66.67, 3 -> 33.33, 4+ -> 0; mean across questions."])
    notes.append(["Retrieval", "Original-caption BGE cosine, Top-3, used for all six curves."])
    notes.append(["Text", "Question-to-original-caption BGE cosine orders candidate distractors."])
    notes.append(["Visual", "Question-to-original-image CLIP cosine orders candidate distractors."])
    notes.append(["CVM", "Frozen pipeline CVM leaf score with within-master z-score (lambda=0.03, alpha=1) orders candidate distractors."
                  if variant == "zscore" else
                  "Frozen pipeline CVM raw leaf score orders candidate distractors; no z-score correction."])
    notes.append(["Similar / Dissimilar", "Within each fixed candidate group, descending / ascending score order."])
    notes.append(["Units", "All plotted score values are multiplied by 100 and have no percent suffix."])
    notes.append(["Reader", "No QA reader run in this experiment; these are retrieval-rank scores."])
    style_body(notes, [26, 105], percent_free=False)
    notes.auto_filter.ref = notes.dimensions

    wb.save(target)
    check = load_workbook(target, read_only=True, data_only=True)
    if (len(check.sheetnames) != 11 or check["All_Domains"].max_row != 129
            or check["Curve_Data"].max_row != 769):
        raise ValueError("Saved workbook dimensions changed")
    for i, domain in enumerate(DOMAINS):
        if check[domain].max_row != 17:
            raise ValueError(f"Incomplete domain sheet: {domain}")
        for n in range(16):
            for j, series in enumerate(series_order, 2):
                actual = check[domain].cell(n + 2, j).value
                expected = float(by_key[domain, series, n]["mean_weighted_rank_score_x100"])
                if abs(actual - expected) > 1e-9:
                    raise ValueError(f"Workbook value changed: {domain} {series} N={n}")
    check.close()
    print(f"{target} · {len(curves)} curve rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvm-variant", choices=("raw", "zscore"), default="raw")
    main(parser.parse_args().cvm_variant)
