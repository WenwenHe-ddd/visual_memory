"""Add the paired visual-master plus z-score curve to the native Excel charts."""
from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.chart import Reference, Series
from openpyxl.chart.data_source import NumData, NumVal
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'reports/MemEye_观测实验汇总_20261030_含视觉Master干扰.xlsx'
GROUPS = ROOT / 'results/random_task_injection_visual_master_zscore/group_means.csv'
OUTPUT = ROOT / 'reports/MemEye_观测实验汇总_20261030_含视觉Master_Zscore干扰.xlsx'
PURPLE = 'C65875'


def cache(values, fmt):
    return NumData(formatCode=fmt, ptCount=len(values),
                   pt=[NumVal(idx=i, v=float(value)) for i, value in enumerate(values)])


def main():
    measurements = {(r['task'], int(r['N'])): float(r['mean_recall_at_5'])
                    for r in csv.DictReader(GROUPS.open(encoding='utf-8-sig'))}
    workbook = load_workbook(SOURCE)
    sheet = workbook['实验2_分任务']
    sheet['L1'] = '视觉 Master + z-score 同任务 Recall@5'
    sheet['L1'].font = Font(name='Times New Roman', bold=True, color='FFFFFF')
    sheet['L1'].fill = PatternFill('solid', fgColor=PURPLE)
    sheet.column_dimensions['L'].width = 38
    for row in range(2, 30):
        key = (sheet.cell(row, 2).value, int(sheet.cell(row, 3).value))
        if key not in measurements:
            raise ValueError(f'Missing paired result: {key}')
        cell = sheet.cell(row, 12, measurements[key])
        cell.number_format = '0.0000'
        cell.font = Font(name='Times New Roman', color=PURPLE)
        cell.alignment = Alignment(horizontal='center')
    if len(sheet._charts) != 8:
        raise ValueError('Expected eight original task charts')
    for chart in sheet._charts:
        if len(chart.series) != 3:
            raise ValueError('Previous chart no longer has three curves')
        x_formula = chart.series[0].xVal.numRef.f
        import re
        match = re.search(r'\$C\$(\d+):\$C\$(\d+)', x_formula)
        if not match:
            raise ValueError(f'Unexpected chart x range: {x_formula}')
        start, end = map(int, match.groups())
        x = [sheet.cell(row, 3).value for row in range(start, end + 1)]
        y = [sheet.cell(row, 12).value for row in range(start, end + 1)]
        new = Series(Reference(sheet, min_col=12, min_row=start, max_row=end),
                     Reference(sheet, min_col=3, min_row=start, max_row=end),
                     title='视觉 Master + z-score · 同任务干扰')
        new.xVal.numRef.numCache = cache(x, 'General')
        new.yVal.numRef.numCache = cache(y, '0.0000')
        new.graphicalProperties.line.solidFill = PURPLE
        new.graphicalProperties.line.width = 24000
        new.marker.symbol = 'triangle'
        new.marker.size = 6
        new.marker.graphicalProperties.solidFill = PURPLE
        new.marker.graphicalProperties.line.solidFill = PURPLE
        chart.series.append(new)
    sheet['E31'] = '粉色：视觉 Master + z-score · 同任务干扰'
    sheet['E31'].font = Font(name='Times New Roman', bold=True, color=PURPLE)
    sheet['A32'] = 'Excel 原生 XY 折线图：横轴为干扰数 N，纵轴为目标 Recall@5；粉线与紫线使用同一图、查询和样本，仅对簇内叶子分数应用 z-score 修正。'
    sheet['A100'] = '粉线使用 ZScoreWeighting（ridge λ=0.03，α=1）：原余弦分数 + α × 簇内原分数标准差 × tanh(白化分数 z 值)。'
    workbook.save(OUTPUT)
    check = load_workbook(OUTPUT, read_only=False)
    assert len(check['实验2_分任务']._charts) == 8
    assert all(len(chart.series) == 4 for chart in check['实验2_分任务']._charts)
    for row in range(2, 30):
        assert abs(check['实验2_分任务'].cell(row, 12).value - measurements[
            (sheet.cell(row, 2).value, int(sheet.cell(row, 3).value))]) < 1e-12
    print(OUTPUT)


if __name__ == '__main__':
    main()
