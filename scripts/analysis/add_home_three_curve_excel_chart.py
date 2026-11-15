"""Add an editable native Excel chart to the Home interference workbook."""

from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook
from openpyxl.chart import ScatterChart, Series, Reference
from openpyxl.chart.layout import Layout, ManualLayout
from openpyxl.styles import Alignment, Font, PatternFill


ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "results/home_top3_interference_15_comparison_20261031.xlsx"


def main() -> None:
    wb = load_workbook(WORKBOOK)
    summary = wb["Home_1至15汇总"]
    for name in ("折线图", "图表数据"):
        if name in wb:
            del wb[name]
    chart_sheet = wb.create_sheet("折线图", 0)
    data_sheet = wb.create_sheet("图表数据", 1)
    data_sheet.append([
        "干扰图数", "SemRAG (Disruption)",
        "Modality-Preserving (Disruption)", "SemRAG (Irrelevant)",
    ])
    for n in range(1, 16):
        source_row = n + 1
        if summary.cell(source_row, 1).value != n:
            raise ValueError(f"Unexpected distractor count in source row {source_row}")
        values = [summary.cell(source_row, column).value for column in (2, 3, 6)]
        if any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in values):
            raise ValueError(f"Invalid source retrieval score: N={n}")
        data_sheet.append([n, *(value * 100 for value in values)])
    data_sheet.freeze_panes = "B2"
    data_sheet.auto_filter.ref = "A1:D16"
    for col, width in {"A": 16, "B": 27, "C": 42, "D": 27}.items():
        data_sheet.column_dimensions[col].width = width
    for cell in data_sheet[1]:
        cell.font = Font(name="Times New Roman", color="FFFFFF", bold=True)
        cell.fill = PatternFill("solid", fgColor="254E70")
        cell.alignment = Alignment(wrap_text=True, horizontal="center")
    data_sheet.row_dimensions[1].height = 34
    for row in data_sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Times New Roman", size=11)
            if cell.column > 1:
                cell.number_format = "0.00"

    chart = ScatterChart()
    chart.scatterStyle = "lineMarker"
    chart.title = None
    chart.x_axis.title = "Number of distractor images"
    chart.x_axis.scaling.min = 1
    chart.x_axis.scaling.max = 15
    chart.x_axis.majorUnit = 2
    chart.x_axis.numFmt = "0"
    chart.y_axis.title = "Mean weighted target-rank score"
    chart.y_axis.scaling.min = 70
    chart.y_axis.scaling.max = 100
    chart.y_axis.majorUnit = 10
    chart.y_axis.numFmt = "0"
    chart.width = 28.5
    chart.height = 13.8
    chart.style = 13

    xvalues = Reference(data_sheet, min_col=1, min_row=2, max_row=16)
    series_specs = (
        (2, "185C40", "circle"),
        (3, "B7A626", "square"),
        (4, "63869A", "triangle"),
    )
    for column, color, marker in series_specs:
        yvalues = Reference(data_sheet, min_col=column, min_row=2, max_row=16)
        item = Series(yvalues, xvalues, title=data_sheet.cell(1, column).value)
        item.graphicalProperties.line.solidFill = color
        item.graphicalProperties.line.width = 28575
        item.marker.symbol = marker
        item.marker.size = 7
        item.marker.graphicalProperties.solidFill = color
        item.marker.graphicalProperties.line.solidFill = color
        item.smooth = False
        chart.series.append(item)

    chart.legend.position = "r"
    chart.legend.overlay = True
    chart.legend.layout = Layout(manualLayout=ManualLayout(
        xMode="factor", yMode="factor", x=.63, y=.48, w=.35, h=.32))
    chart_sheet.add_chart(chart, "A1")
    chart_sheet.sheet_view.showGridLines = False
    chart_sheet.column_dimensions["A"].width = 18
    chart_sheet.sheet_properties.pageSetUpPr.fitToPage = True
    chart_sheet.page_setup.fitToWidth = 1
    chart_sheet.page_setup.fitToHeight = 1
    wb.active = 0
    wb.save(WORKBOOK)
    print(WORKBOOK)


if __name__ == "__main__":
    main()
