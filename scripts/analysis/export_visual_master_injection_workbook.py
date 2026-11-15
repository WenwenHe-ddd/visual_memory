"""Add the paired visual-master same-task curve to the existing native Excel charts."""
from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.chart import Reference, Series
from openpyxl.chart.data_source import NumData, NumVal
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'reports/MemEye_观测实验汇总_20261025.xlsx'
GROUPS = ROOT / 'results/random_task_injection_visual_master/group_means.csv'
OUTPUT = ROOT / 'reports/MemEye_观测实验汇总_20261030_含视觉Master干扰.xlsx'
PURPLE = '7455A9'


def cache(values, fmt):
    return NumData(formatCode=fmt, ptCount=len(values),
                   pt=[NumVal(idx=i, v=float(value)) for i, value in enumerate(values)])


def main():
    measurements = {(r['task'], int(r['N'])): float(r['mean_recall_at_5'])
                    for r in csv.DictReader(GROUPS.open(encoding='utf-8-sig'))}
    workbook = load_workbook(SOURCE)
    sheet = workbook['实验2_分任务']
    sheet['K1'] = '视觉 Master 同任务 Recall@5'
    sheet['K1'].font = Font(name='Times New Roman', bold=True, color='FFFFFF')
    sheet['K1'].fill = PatternFill('solid', fgColor=PURPLE)
    sheet.column_dimensions['K'].width = 27
    for row in range(2, 30):
        key = (sheet.cell(row, 2).value, int(sheet.cell(row, 3).value))
        if key not in measurements:
            raise ValueError(f'Missing paired result: {key}')
        cell = sheet.cell(row, 11, measurements[key])
        cell.number_format = '0.0000'
        cell.font = Font(name='Times New Roman', color=PURPLE)
        cell.alignment = Alignment(horizontal='center')
    if len(sheet._charts) != 8:
        raise ValueError('Expected eight original task charts')
    for chart in sheet._charts:
        if len(chart.series) != 2:
            raise ValueError('Original chart no longer has two curves')
        x_formula = chart.series[0].xVal.numRef.f
        import re
        match = re.search(r'\$C\$(\d+):\$C\$(\d+)', x_formula)
        if not match:
            raise ValueError(f'Unexpected chart x range: {x_formula}')
        start, end = map(int, match.groups())
        x = [sheet.cell(row, 3).value for row in range(start, end + 1)]
        y = [sheet.cell(row, 11).value for row in range(start, end + 1)]
        new = Series(Reference(sheet, min_col=11, min_row=start, max_row=end),
                     Reference(sheet, min_col=3, min_row=start, max_row=end),
                     title='视觉 Master · 同任务干扰')
        new.xVal.numRef.numCache = cache(x, 'General')
        new.yVal.numRef.numCache = cache(y, '0.0000')
        new.graphicalProperties.line.solidFill = PURPLE
        new.graphicalProperties.line.width = 24000
        new.marker.symbol = 'square'
        new.marker.size = 6
        new.marker.graphicalProperties.solidFill = PURPLE
        new.marker.graphicalProperties.line.solidFill = PURPLE
        chart.series.append(new)
    sheet['B31'] = '绿色：原 caption · 同任务干扰'
    sheet['C31'] = '黄色：原 caption · 跨任务干扰'
    sheet['D31'] = '紫色：视觉 Master · 同任务干扰'
    for ref, color in [('B31', '286A3D'), ('C31', 'B9A52E'), ('D31', PURPLE)]:
        sheet[ref].font = Font(name='Times New Roman', bold=True, color=color)
    sheet['A32'] = 'Excel 原生 XY 折线图：横轴为干扰数 N，纵轴为目标 Recall@5；紫线沿用绿色曲线的目标与五次抽样，仅改变图组织及 master 引导检索。'
    sheet['A99'] = '紫线按每次目标 + 干扰图序列重新构建 caption_visual_master；叶子使用与原曲线完全相同的 caption，master 选择使用候选陈述。此对比同时改变图组织和查询扩展。'
    workbook.save(OUTPUT)
    check = load_workbook(OUTPUT, read_only=False)
    assert len(check['实验2_分任务']._charts) == 8
    assert all(len(chart.series) == 3 for chart in check['实验2_分任务']._charts)
    for row in range(2, 30):
        assert abs(check['实验2_分任务'].cell(row, 11).value - measurements[
            (sheet.cell(row, 2).value, int(sheet.cell(row, 3).value))]) < 1e-12
    print(OUTPUT)


if __name__ == '__main__':
    main()
