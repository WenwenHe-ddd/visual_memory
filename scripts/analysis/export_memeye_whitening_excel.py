"""Package the verified largest-master spectrum as Excel with native charts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.data_source import NumData, NumVal
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'reports/memeye_largest_master_whitening_20261103'
OUTPUT = SOURCE / 'MemEye_最大簇_白化前后对比.xlsx'
BLUE, ORANGE, NAVY = '466FA3', 'DB8455', '18334F'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(filename):
    with (SOURCE / filename).open(newline='') as handle:
        return list(csv.DictReader(handle))


def base_sheet(wb, name, widths):
    ws = wb.create_sheet(name)
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 85
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.outlinePr.summaryRight = False
    for col, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    return ws


def banner(ws, title, end_col):
    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=end_col)
    cell = ws.cell(1, 1, title)
    cell.font = Font(name='Microsoft YaHei', size=18, bold=True, color='FFFFFF')
    cell.fill = PatternFill('solid', fgColor=NAVY)
    cell.alignment = Alignment(vertical='center')
    ws.row_dimensions[1].height = 24
    ws.row_dimensions[2].height = 18


def data_table(ws, name, headers, rows):
    ws.append([]) if ws.max_row < 3 else None
    for j, label in enumerate(headers, 1):
        cell = ws.cell(5, j, label)
        cell.font = Font(name='Microsoft YaHei', bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor=NAVY)
        cell.alignment = Alignment(wrap_text=True, vertical='center')
    ws.row_dimensions[5].height = 32
    for i, row in enumerate(rows, 6):
        for j, value in enumerate(row, 1):
            cell = ws.cell(i, j, value)
            cell.font = Font(name='Microsoft YaHei', size=10)
            cell.alignment = Alignment(vertical='top')
    ref = f'A5:{get_column_letter(len(headers))}{5+len(rows)}'
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
    ws.add_table(table)
    ws.freeze_panes = 'A6'
    ws.print_title_rows = '1:5'
    ws.print_options.horizontalCentered = True


def cached_numbers(values, number_format):
    return NumData(formatCode=number_format, ptCount=len(values),
                   pt=[NumVal(idx=i, v=float(value)) for i, value in enumerate(values)])


def make_chart(data, rows, columns, title, ylabel, percent=False):
    chart = BarChart()
    chart.type, chart.grouping = 'col', 'clustered'
    chart.title, chart.y_axis.title = title, ylabel
    chart.x_axis.title = '主成分（按原始方差降序）'
    chart.style = 10
    chart.width, chart.height = 32, 11.2
    chart.gapWidth, chart.overlap = 55, 0
    chart.legend.position = 'b'
    chart.y_axis.scaling.min = 0
    chart.y_axis.numFmt = '0%' if percent else '0.0'
    if percent:
        chart.y_axis.majorUnit = .02
    chart.x_axis.tickLblPos = 'low'
    chart.x_axis.tickLblSkip = 1
    chart.x_axis.tickMarkSkip = 1
    chart.add_data(Reference(data, min_col=columns[0], max_col=columns[1], min_row=5,
                             max_row=5+len(rows)), titles_from_data=True)
    chart.set_categories(Reference(data, min_col=1, min_row=6, max_row=5+len(rows)))
    # Cache all 43 values as well as references for spreadsheet preview clients.
    for series, column, color in zip(chart.series, columns, [BLUE, ORANGE]):
        series.graphicalProperties.solidFill = color
        series.graphicalProperties.line.solidFill = color
        series.val.numRef.numCache = cached_numbers([r[column-1] for r in rows], '0.00%' if percent else '0.000000')
        series.cat.numRef.numCache = cached_numbers([r[0] for r in rows], '0')
    return chart


def main():
    metadata = json.loads((SOURCE / 'provenance.json').read_text())
    for filename, expected in metadata['artifact_sha256'].items():
        assert sha(filename) == expected, filename
    spectrum = read_csv('spectrum.csv')
    keys = list(spectrum[0])
    rows = [[int(row[k]) if k == 'pc' else float(row[k]) for k in keys] for row in spectrum]
    members = json.loads((SOURCE / 'selected_members.json').read_text())
    clusters = read_csv('all_master_sizes.csv')
    assert len(rows) == 43 and len(members) == 44 and len(clusters) == 88
    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = 'MemEye 最大簇：白化前后对比'
    wb.properties.subject = 'λ=0.03；44 条记忆；43 个主成分'
    wb.properties.creator = 'VisualMemory'

    overview = base_sheet(wb, '概览与图表', [11]*16)
    banner(overview, 'MemEye 最大簇 · 白化前后对比', 16)
    fields = [('数据集 / Caption', 'MemEye / 原始 Qwen caption'),
              ('Domain', metadata['selection']['domain']), ('Master', metadata['selection']['master_id']),
              ('簇大小 / 主成分数', '44 条记忆 / 43 个主成分（全部方向）'),
              ('选簇规则', '8 个 domain、88 个 master 中，叶子数最多；唯一最大簇'),
              ('λ', metadata['lambda']), ('α', '不影响本步骤的奇异值与方差分布'),
              ('编码', '原实验 BGE-small-en-v1.5，对话＋完整 caption，384 维；复用冻结向量')]
    for row, (key, value) in enumerate(fields, 4):
        overview.merge_cells(start_row=row, start_column=1, end_row=row, end_column=3)
        overview.merge_cells(start_row=row, start_column=4, end_row=row, end_column=16)
        overview.cell(row, 1, key).font = Font(name='Microsoft YaHei', bold=True, color=NAVY)
        overview.cell(row, 4, value).font = Font(name='Microsoft YaHei', size=11)
        overview.row_dimensions[row].height = 23
    summary = [('指标', '白化前', '白化后'),
               ('PC1 奇异值', rows[0][1], rows[0][2]),
               ('PC1 方差占比', metadata['first_pc_variance_share_before'], metadata['first_pc_variance_share_after']),
               ('前 5 个主成分累计方差占比', metadata['top5_pc_variance_share_before'], metadata['top5_pc_variance_share_after']),
               ('最大 / 最小主成分方差比', metadata['largest_smallest_positive_variance_ratio_before'],
                metadata['largest_smallest_positive_variance_ratio_after'])]
    for row, values in enumerate(summary, 14):
        for (start, end), value in zip([(1, 5), (6, 8), (9, 11)], values):
            overview.merge_cells(start_row=row, start_column=start, end_row=row, end_column=end)
            cell = overview.cell(row, start, value)
            cell.font = Font(name='Microsoft YaHei', bold=row==14, color='FFFFFF' if row==14 else NAVY)
            cell.fill = PatternFill('solid', fgColor=NAVY if row==14 else 'F0F4F8')
            cell.alignment = Alignment(vertical='center')
            if start > 1 and row > 14:
                cell.number_format = '0.00%' if row in [16, 17] else '0.000'
        overview.row_dimensions[row].height = 25
    overview.merge_cells('A20:P21')
    overview['A20'] = '图表为 Excel 原生柱状图，关联「主成分数据」工作表，可编辑。下列结果固定于 λ=0.03；仅修改上方参数文字不会重新计算白化。'
    overview['A20'].alignment = Alignment(wrap_text=True, vertical='center')
    overview['A20'].font = Font(name='Microsoft YaHei', size=10, color='526174')
    overview.freeze_panes = 'D4'

    data = base_sheet(wb, '主成分数据', [12, 21, 21, 21, 21, 22, 22, 21, 24])
    banner(data, '全部 43 个主成分 · 完整数值', 9)
    data['A3'] = '按白化前方差从大到小排列；百分比列以数值保存。'
    headers = ['主成分', '奇异值：白化前', '奇异值：白化后', '方差：白化前', '方差：白化后',
               '方差占比：白化前', '方差占比：白化后', '向量方向缩放系数', '匹配贡献保留比例']
    data_table(data, 'SpectrumData', headers, rows)
    for row in data.iter_rows(min_row=6, max_row=48, min_col=2, max_col=9):
        for cell in row:
            cell.number_format = '0.000000' if cell.column <= 5 else '0.00%'
    overview.add_chart(make_chart(data, rows, (2, 3), '奇异值：白化前后对比', '奇异值'), 'A24')
    overview.add_chart(make_chart(data, rows, (6, 7), '方差占比：白化前后对比', '方差占比', True), 'A49')
    overview.print_area = 'A1:P72'

    preview = base_sheet(wb, '高清图预览', [12]*18)
    banner(preview, '高清图预览（静态图片）', 18)
    preview['A3'] = '可编辑图表请见「概览与图表」；此页保留科研绘图的原始排版。'
    picture = Image(SOURCE / 'whitening_spectrum.png')
    picture.width, picture.height = 1485, 851.4
    preview.add_image(picture, 'A5')

    member_sheet = base_sheet(wb, '簇成员', [9, 24, 68, 80, 100])
    banner(member_sheet, '最大簇的 44 条记忆', 5)
    data_table(member_sheet, 'ClusterMembers', ['序号', '图像文件', 'Memory ID', '原始对话', '完整 Caption'],
               [[i, m['filename'], m['memory_id'], m['dialogue'], m['caption']] for i, m in enumerate(members, 1)])
    for row in member_sheet.iter_rows(min_row=6):
        member_sheet.row_dimensions[row[0].row].height = 112
        for cell in row:
            cell.alignment = Alignment(vertical='top', wrap_text=True)
    sizes = base_sheet(wb, '全部簇大小', [10, 66, 24, 16, 18])
    banner(sizes, '全部 88 个 master · 按簇大小降序', 5)
    data_table(sizes, 'MasterSizes', ['排名', 'Domain', 'Master ID', '记忆条数', '本次选中'],
               [[i, c['domain'], c['master_id'], int(c['size']), '是' if i==1 else ''] for i,c in enumerate(clusters, 1)])

    notes = base_sheet(wb, '方法与来源', [31, 145])
    banner(notes, '方法、参数与来源', 2)
    note_rows = [
        ['主成分方差', 'v_j = σ_j² / (k − 1)，k = 44'],
        ['等价白化后奇异值', 'σ_j_after = σ_j × sqrt(λ / (v_j + λ))'],
        ['等价白化后方差', 'v_j_after = λ × v_j / (v_j + λ)'],
        ['匹配贡献保留比例', 'λ / (v_j + λ)；这是双线性匹配贡献的比例，与奇异值的缩放系数不同。'],
        ['变换含义', '使用与现有打分严格等价的对称变换 W=[λ(C+λI)^(-1)]^(1/2)。代码直接计算分数。'],
        ['中心化与归一化', '中心化用于拟合协方差；输入已 L2 归一化；等价变换后不再次 L2 归一化。'],
        ['结果范围', '带 ridge 正则的白化使方差分布更均衡，但不要求每个方向方差相等。α 只影响后续赋权。'],
        ['数据来源', str(SOURCE)], ['向量来源', metadata['embedding_source']],
        ['代码一致性检查最大误差', metadata['audit']['score_equivalence_max_abs_error']],
        ['原实验与 z-score 簇结构', '已逐个核验全部 8 个 domain 的 graph 字节一致。'],
        ['本次行为', '仅整理已核验数据；未重新编码、未运行问答、未修改记忆库。'],
        ['数据校验', '以下 SHA-256 用于核验源文件。'],
    ] + [[filename, sha(SOURCE/filename)] for filename in ['spectrum.csv','all_master_sizes.csv','selected_members.json','provenance.json','whitening_spectrum.png']]
    data_table(notes, 'MethodNotes', ['项目 / 文件', '说明 / 数值'], note_rows)
    for row in notes.iter_rows(min_row=6):
        notes.row_dimensions[row[0].row].height = 38
        row[1].alignment = Alignment(wrap_text=True, vertical='top')
    wb.active = 0
    wb.save(OUTPUT)

    # Verify the workbook data, native chart links and all cached chart values.
    reopened = load_workbook(OUTPUT)
    for i, row in enumerate(rows, 6):
        for j, value in enumerate(row, 1):
            actual = reopened['主成分数据'].cell(i, j).value
            assert abs(actual-value) <= 1e-14 * max(1, abs(value))
    assert len(reopened['概览与图表']._charts) == 2
    assert len(reopened['高清图预览']._images) == 1
    assert reopened['簇成员'].max_row == 49
    assert reopened['全部簇大小'].max_row == 93
    ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
    with zipfile.ZipFile(OUTPUT) as package:
        assert package.testzip() is None
        charts = sorted(n for n in package.namelist() if n.startswith('xl/charts/chart') and n.endswith('.xml'))
        assert len(charts) == 2
        for name, columns in zip(charts, [(2,3), (6,7)]):
            chart = ET.fromstring(package.read(name))
            series = chart.findall('.//c:barChart/c:ser', ns)
            assert len(series) == 2
            for series_item, column in zip(series, columns):
                cache = series_item.findall('c:val/c:numRef/c:numCache/c:pt/c:v', ns)
                assert len(cache) == 43
                for item, source_row in zip(cache, rows):
                    assert abs(float(item.text)-source_row[column-1]) < 1e-14
                formula = series_item.find('c:val/c:numRef/c:f', ns).text
                assert '主成分数据' in formula and '$6:' in formula and '$48' in formula
        assert hashlib.sha256(package.read('xl/media/image1.png')).hexdigest() == sha(SOURCE/'whitening_spectrum.png')
    audit = {'file': str(OUTPUT), 'sha256': sha(OUTPUT), 'worksheets': reopened.sheetnames,
             'native_charts': 2, 'embedded_pngs': 1, 'pc_rows': 43, 'member_rows': 44, 'master_rows': 88,
             'source_values_unchanged': True, 'chart_references_and_caches_verified': True,
             'script_sha256': sha(__file__)}
    (SOURCE/'excel_export_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
