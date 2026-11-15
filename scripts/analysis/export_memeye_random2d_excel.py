#!/usr/bin/env python3
"""Export the verified three-backbone basis ablation as numeric Excel tables."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/memeye_api_random2d_ablation_20261105'
DEST = BASE / 'MemEye_PCA2D_vs_Random2D_3Backbones.xlsx'
LABELS = {'qwen': 'Qwen3-VL-8B-Instruct', 'gpt5': 'GPT-5',
          'deepseek': 'DeepSeek-V4.1-Flash'}
DOMAINS = {'Brand_Memory_Test': 'Brand', 'Card_Playlog_Test': 'Card',
           'Cartoon_Entertainment_Companion': 'Cartoon',
           'Home_Renovation_Interior_Design': 'Home',
           'Multi-Scene_Visual_Case_Archive_Assistant': 'Multi-Scene',
           'Outdoor_Navigation_Route_Memory_Assistant': 'Outdoor',
           'Personal_Health_Dashboard_Assistant': 'Health',
           'Social_Chat_Memory_Test': 'Social'}


def read_csv(name):
    with (BASE / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_table(wb, name, headers, rows, widths, percent_cols=(), decimal_cols=()):
    ws = wb.create_sheet(name)
    ws.append(headers)
    for row in rows:
        ws.append(row)
    ws.freeze_panes = 'B2'
    ws.sheet_view.showGridLines = False
    ws.row_dimensions[1].height = 44
    for cell in ws[1]:
        cell.font = Font(name='Calibri', bold=True, color='FFFFFF', size=11)
        cell.fill = PatternFill('solid', fgColor='24476B')
        cell.alignment = Alignment(vertical='center', wrap_text=True)
    for index, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(index)].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name='Calibri', size=11)
            cell.alignment = Alignment(vertical='center')
        for index in percent_cols:
            row[index - 1].number_format = '0.00%'
        for index in decimal_cols:
            row[index - 1].number_format = '0.00'
    table = Table(displayName=f'Results{len(wb.worksheets)}',
                  ref=f'A1:{get_column_letter(len(headers))}{len(rows) + 1}')
    table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
    ws.add_table(table)
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.print_title_rows = '1:1'
    return ws


def main():
    summaries = read_csv('comparison.csv')
    seeds = read_csv('seed_metrics.csv')
    domains = read_csv('domain_comparison.csv')
    questions = read_csv('per_question.csv')
    assert [len(summaries), len(seeds), len(domains), len(questions)] == [3, 12, 24, 4452]
    expected_conditions = {'pca_2d', 'random_2d_seed_0', 'random_2d_seed_1', 'random_2d_seed_2'}
    assert {r['condition'] for r in seeds} == expected_conditions
    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = 'MemEye: PCA-2D vs Random-2D — Three Backbones'
    wb.properties.subject = 'Seeds 0, 1, 2; full 371-question MemEye evaluation'
    wb.properties.description = 'Numeric results exported from verified CSV artifacts; SD is across three random basis seeds.'

    headers = ['Backbone', 'PCA Recall@5', 'Random Recall@5 均值',
               'Random Recall SD（百分点）', 'PCA 准确率', 'Random 准确率均值',
               'Random 准确率 SD（百分点）', 'Recall 变化（百分点）',
               '准确率变化（百分点）', '问题数', '标注记忆总数']

    def comparison_row(r):
        pca_r, random_r = float(r['pca_recall_percent']), float(r['random_recall_mean_percent'])
        pca_a, random_a = float(r['pca_accuracy_percent']), float(r['random_accuracy_mean_percent'])
        return [LABELS[r['backend']], pca_r / 100, random_r / 100,
                float(r['random_recall_sd_pp']), pca_a / 100, random_a / 100,
                float(r['random_accuracy_sd_pp']), random_r - pca_r, random_a - pca_a,
                int(r['questions']), int(r['gold_references'])]

    add_table(wb, '总体汇总', headers, [comparison_row(r) for r in summaries],
              [27, 18, 23, 23, 18, 23, 25, 22, 22, 12, 16],
              percent_cols=(2, 3, 5, 6), decimal_cols=(4, 7, 8, 9))
    seed_rows = [[LABELS[r['backend']], 'PCA-2D' if r['condition'] == 'pca_2d' else 'Random-2D',
                  None if r['condition'] == 'pca_2d' else int(r['condition'].rsplit('_', 1)[1]),
                  int(r['questions']), int(r['gold_hits']), int(r['gold_references']),
                  float(r['recall_percent']) / 100, int(r['correct']), float(r['accuracy_percent']) / 100]
                 for r in seeds]
    add_table(wb, '逐种子', ['Backbone', '方法', '基底随机种子', '问题数', '命中标注记忆数',
                            '标注记忆总数', 'Recall@5', '答对题数', '准确率'],
              seed_rows, [27, 18, 17, 12, 20, 18, 17, 15, 17], percent_cols=(7, 9))
    domain_rows = [[LABELS[r['backend']], DOMAINS[r['domain']], r['domain'],
                    *comparison_row(r)[1:], int(r['pca_gold_hits']), int(r['pca_correct']),
                    float(r['random_correct_mean'])] for r in domains]
    add_table(wb, '逐领域', ['Backbone', '领域简称', '领域 ID', *headers[1:],
                            'PCA 命中标注记忆数', 'PCA 答对题数', 'Random 平均答对题数'],
              domain_rows, [27, 18, 55, 18, 23, 23, 18, 23, 25, 22, 22, 12, 16, 23, 18, 25],
              percent_cols=(4, 5, 7, 8), decimal_cols=(6, 9, 10, 11, 16))
    add_table(wb, '逐题结果', ['Backbone', '问题 ID', '领域 ID', '条件', '是否答对',
                              'Top5 命中标注记忆数', '该题标注记忆数'],
              [[LABELS[r['backend']], r['question_id'], r['domain'], r['condition'],
                r['correct'] == 'True', int(r['gold_hits']), int(r['gold_references'])] for r in questions],
              [27, 68, 55, 25, 15, 25, 20])
    notes = [
        ['数据集', 'MemEye，全 8 个领域；每条件 371 道题，438 条图片记忆，1041 个标注记忆引用。'],
        ['对比', 'PCA-2D 与 Random-2D。随机基底仅使用种子 0、1、2。'],
        ['均值与标准差', 'Random 为三个基底种子的均值和样本标准差（ddof=1）；不是题目层面的置信区间。'],
        ['数值格式', 'Recall 与准确率以 0–1 数值存储，显示为百分比；SD 与变化列直接存储百分点。所有数值保留来源精度。'],
        ['Recall@5', 'micro Recall@5 = 各题 Top5 命中的标注记忆数之和 / 各题标注记忆数之和。总体分母为 1041，不是 371。'],
        ['准确率', '答对题数 / 371。领域结果使用该领域问题数。'],
        ['控制变量', '每个 backbone 内固定其自身生成的 caption、文本分簇、CLIP/BGE、候选陈述、到达顺序和回答设置，仅改变固定二维基底。'],
        ['在线配置', 'caption_visual_master + multi_path + online_weighting=none；Top5。'],
        ['动态记忆', '每张新图编码一次；四条件共享当前视觉向量后释放；叶子仅保留两个视觉坐标，master 保留完整视觉聚合量。'],
        ['Qwen', 'Qwen3-VL-8B-Instruct；引用此前完成实验中的种子 0、1、2。本次 Excel 导出未重新推理。'],
        ['GPT-5', '官方 OpenAI API，请求 gpt-5，实际返回 gpt-5-2025-08-07；reader seed42，minimal reasoning，2048 completion tokens，temperature 省略。'],
        ['DeepSeek', 'DashScope International 的 deepseek-v4.1-flash；底层权重版本未确认。reader seed42，temperature=0，enable_thinking=false，max_tokens=16。'],
        ['API 回答', 'GPT-5 和 DeepSeek 的 PCA 与 Random 回答均在本轮重新生成：764 次新 API 调用，2968 条逻辑记录；仅同次运行完整请求相同时共享答案。'],
        ['有效性', '无格式错误、拒绝或截断；独立审计通过。完整审计见 independent_audit.json。'],
        ['历史比较', 'DeepSeek 本次 PCA 答对 218 题；历史运行答对 209 题。这里比较本次 fresh PCA，不把跨次 API 回答变化归因于基底。'],
        ['结论范围', '当前三个 backbone 的 MemEye 测试效果接近；不能解释为统计显著提升或其他数据上的等价保证。'],
    ]
    for name in ['comparison.csv', 'seed_metrics.csv', 'domain_comparison.csv', 'per_question.csv']:
        notes.append(['数据来源', str(BASE / name)])
        notes.append([name + ' SHA256', sha(BASE / name)])
    add_table(wb, '实验说明', ['项目', '说明'], notes, [30, 118])
    for row in wb['实验说明'].iter_rows(min_row=2):
        row[1].alignment = Alignment(vertical='top', wrap_text=True)
        wb['实验说明'].row_dimensions[row[0].row].height = 44
    wb.save(DEST)

    with zipfile.ZipFile(DEST) as archive:
        assert archive.testzip() is None
    check = load_workbook(DEST, data_only=True)
    expected_rows = {'总体汇总': 3, '逐种子': 12, '逐领域': 24, '逐题结果': 4452}
    for name, count in expected_rows.items():
        assert check[name].max_row == count + 1
    for index, source in enumerate(summaries, 2):
        for col, expected in enumerate(comparison_row(source), 1):
            cell = check['总体汇总'].cell(index, col)
            if isinstance(expected, (int, float)):
                assert cell.data_type == 'n' and abs(cell.value - expected) < 1e-10
            else:
                assert cell.value == expected
    assert all(check['逐题结果'].cell(row, 5).data_type == 'b' for row in range(2, 4454))
    validation = {'passed': True, 'workbook': str(DEST), 'workbook_sha256': sha(DEST),
                  'data_rows': expected_rows, 'seeds': [0, 1, 2],
                  'numeric_cells_verified': True, 'zip_integrity_verified': True,
                  'source_sha256': {name: sha(BASE / name) for name in
                                    ('comparison.csv', 'seed_metrics.csv', 'domain_comparison.csv', 'per_question.csv')}}
    (BASE / 'excel_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
