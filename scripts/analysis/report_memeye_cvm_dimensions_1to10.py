#!/usr/bin/env python3
"""Export the completed, independently audited dimension ablation."""
import csv
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from scripts.analysis.evaluate_adaptive_changed_qa import read, lines, dump, sha


def report(output):
    summary = read(output / 'summary.json')
    audit = read(output / 'independent_audit.json')
    assert summary['completed'] and audit['passed'] and not audit['inputs_only']
    conditions = summary['conditions']
    retrieved = {(r['question_id'], r['condition']): r for r in lines(output / 'retrieval_results.jsonl')}
    responses = {(r['question_id'], r['condition']): r for r in lines(output / 'reader/responses.jsonl')}
    paired = list(csv.DictReader((output / 'paired_results.csv').open(encoding='utf-8-sig')))
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Summary'
    sheet.append(['PCA dimension', 'Correct', 'Questions', 'Accuracy', 'Delta vs fresh 2D (pp)',
                  'Gold hits', 'Gold links', 'Recall@5', 'Changed ordered Top5',
                  'Changed master choices', 'Wrong to correct', 'Correct to wrong',
                  'Changed answers', 'McNemar exact p', 'Visual numeric bytes'])
    for i, r in enumerate(conditions, 2):
        sheet.append([r['dimension'], r['correct'], r['questions'], f'=B{i}/C{i}', f'=(D{i}-$D$3)*100',
                      r['gold_hits_top5'], r['gold_references'], f'=F{i}/G{i}', r['changed_top5_order'],
                      r['changed_master_choices'], r['wrong_to_correct'], r['correct_to_wrong'],
                      r['answer_changes'], r['mcnemar_exact_two_sided_p'], r['visual_numeric_bytes']])
        sheet.cell(i, 4).number_format = sheet.cell(i, 8).number_format = '0.0000%'
        sheet.cell(i, 5).number_format = '0.0000'
    detail = workbook.create_sheet('Per_question')
    detail.append(['Dimension', 'Question', 'Domain', 'Gold', 'Prediction', 'Correct', '2D prediction',
                   '2D correct', 'Same payload', 'Gold hits', 'Gold links', 'Same ordered Top5', 'Top5 memory IDs'])
    for r in paired:
        d = int(r['dimension'])
        retrieval = retrieved[r['question_id'], f'pca_{d}d']
        detail.append([d, r['question_id'], retrieval['domain'], r['gold_choice'], r['prediction'],
                       r['correct'] == 'True', r['pca2_prediction'], r['pca2_correct'] == 'True',
                       r['same_payload_as_2d'] == 'True', retrieval['gold_hits_top5'], len(retrieval['gold_ids']),
                       retrieval['same_top5_order_as_2d'], '\n'.join(retrieval['top5_memory_ids'])])
    domains = workbook.create_sheet('Domains')
    domains.append(['Dimension', 'Domain', 'Correct', 'Questions', 'Accuracy', 'Hits', 'Gold links', 'Recall@5'])
    for d in range(1, 11):
        for domain in sorted({r['domain'] for r in retrieved.values()}):
            items = [r for r in retrieved.values() if r['dimension'] == d and r['domain'] == domain]
            correct = sum(responses[r['question_id'], r['condition']]['correct'] for r in items)
            hits, gold = sum(r['gold_hits_top5'] for r in items), sum(len(r['gold_ids']) for r in items)
            domains.append([d, domain, correct, len(items), correct/len(items), hits, gold, hits/gold])
    protocol = workbook.create_sheet('Protocol')
    protocol.append(['Setting', 'Value'])
    settings = {'Dataset': 'MemEye, 371 questions, 438 leaves, 88 masters, 8 domains',
                'Reader': 'Qwen3-VL-8B-Instruct; seed=42; greedy; BF16; SDPA; batch=1',
                'Fixed retrieval': 'CVM multi_path + zscore; lambda=.03; alpha=1; Top5',
                'Changed variable': 'Nested leading PCA dimensions 1 through 10',
                'Calibration': 'Original external-domain data; target image hash overlap=0; CPU CLIP',
                'Frozen controls': 'Caption, dialogue, all prefix partitions, arrival order, candidates, BGE vectors, prompts, question/options/images',
                'Fresh inference count': summary['fresh_inferences'],
                'Logical records': summary['logical_records'],
                'Historical answer reuse': False,
                'Historical 2D accuracy': '199/371 = 0.5363881401617251',
                'Fresh 2D answer changes vs history': summary['fresh_2d_vs_historical_answer_changes'],
                'Historical retrieval score max abs error': read(output/'retrieval_manifest.json')['max_historical_top5_score_error'],
                'Historical 2D ordered Top5 / payload parity': '371/371; all exact',
                'Storage definition': 'float64 visual numeric state = 8*(512*d + 88*512 + 438*d); excludes text/JSON/cache',
                'Interpretation': 'Single fixed-option-order paired run; descriptive differences, not independent significance or four-rotation paper EM',
                'Independent audit': str(output/'independent_audit.json'),
                'Source': str(output), 'Run command': 'python3 scripts/analysis/run_memeye_cvm_dimensions_1to10.py --stage all --gpu 0'}
    for k, v in settings.items():
        protocol.append([k, v])
    variance = workbook.create_sheet('PCA_variance')
    variance.append(['Domain', 'Dimension', 'Calibration images', 'Retained calibration variance'])
    calibration = read(output / 'calibration/manifest.json')
    for domain, entry in calibration['domains'].items():
        for d in range(1, 11):
            variance.append([domain, d, entry['calibration_images'], entry['explained_variance_ratio'][str(d)]])
            variance.cell(variance.max_row, 4).number_format = '0.0000%'
    for ws in workbook:
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='234E70')
        for col in ws.columns:
            letter = col[0].column_letter
            ws.column_dimensions[letter].width = min(65, max(14, max(len(str(c.value or '')) for c in col[:50])+2))
    target = output/'CVM_PCA_dimensions_1_to_10.xlsx'
    workbook.save(target)
    check = load_workbook(target, data_only=False)
    assert check['Per_question'].max_row == 3711 and check['Summary'].max_row == 11
    for i, r in enumerate(conditions, 2):
        assert check['Summary'].cell(i, 2).value == audit['correct_counts'][r['condition']]
    base = next(r['correct'] for r in conditions if r['dimension'] == 2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [r['dimension'] for r in conditions]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    for ax, metric, numerator, denominator, label, color in [
        (axes[0], 'accuracy', 'correct', 371, 'QA accuracy (%)', '#245a81'),
        (axes[1], 'recall_at_5', 'gold_hits_top5', 1041, 'Recall@5 (%)', '#a65c23')]:
        y = [100*r[metric] for r in conditions]
        baseline = next(100*r[metric] for r in conditions if r['dimension'] == 2)
        ax.plot(x, y, 'o-', color=color, linewidth=1.8)
        ax.axhline(baseline, color='gray', linestyle='--', linewidth=1, label='2D reference')
        for r, value in zip(conditions, y):
            ax.annotate(str(r[numerator]), (r['dimension'], value), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
        ax.set(xlabel='PCA visual coordinate dimension', ylabel=label, xticks=x,
               ylim=(min(y)-.8, max(y)+1.0), title=f'Labels: numerator / {denominator}')
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='lower right', frameon=False, fontsize=9)
    fig.suptitle('CVM + z-score | Qwen | lambda=0.03, alpha=1, Top-5 | single paired run')
    fig.savefig(output/'dimension_ablation.png', dpi=180)
    fig.savefig(output/'dimension_ablation.pdf')
    plt.close(fig)
    table = ''.join(f"<tr><td>{r['dimension']}D</td><td>{r['correct']}/371</td><td>{100*r['accuracy']:.4f}%</td>"
                    f"<td>{100*(r['correct']-base)/371:+.4f}</td><td>{r['gold_hits_top5']}/1041</td>"
                    f"<td>{r['changed_top5_order']}</td><td>{r['wrong_to_correct']}</td><td>{r['correct_to_wrong']}</td></tr>"
                    for r in conditions)
    report_html = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CVM PCA dimension ablation</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#183040}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#edf3f7}}p{{line-height:1.7}}</style>
<h1>CVM + z-score：PCA 坐标维度消融</h1><p>Qwen3-VL-8B · λ=0.03 · α=1 · Top-5 · 371 题。仅改变视觉 PCA 坐标数量；分簇、caption、候选陈述、文本向量、提示词及评测设置固定。</p>
<img src="dimension_ablation.png" alt="PCA dimension ablation accuracy and recall" style="width:100%">
<table><tr><th>维度</th><th>答对</th><th>ACC</th><th>相对二维（百分点）</th><th>目标命中</th><th>Top-5 变化题数</th><th>错→对</th><th>对→错</th></tr>{table}</table>
<p>二维历史成绩为 199/371；本轮二维有 {summary['fresh_2d_vs_historical_answer_changes']} 题答案与历史不同。十组共 3,710 条记录，{summary['fresh_inferences']} 次新推理，仅本轮相同完整输入共享回答，没有复用历史答案。独立审计通过。</p>
<p>结果为一次固定选项顺序的配对比较，不表示增维效果单调或具有统计显著性。视觉状态字节数仅包含基底、master 聚合向量及叶子坐标。</p>
<p><a href="CVM_PCA_dimensions_1_to_10.xlsx">Excel（汇总、逐题、领域与协议）</a> · <a href="summary.csv">CSV</a> · <a href="independent_audit.json">独立审计</a></p></html>'''
    (output/'report.html').write_text(report_html)
    completion = read(output/'completion_manifest.json')
    completion['independent_audit_passed'] = True
    for p in [target, output/'report.html', output/'independent_audit.json', output/'dimension_ablation.png', output/'dimension_ablation.pdf', Path(__file__)]:
        completion['artifact_sha256'][str(p)] = sha(p)
    dump(output/'completion_manifest.json', completion)
    print(json.dumps({'workbook': str(target), 'verified_data_rows': 3710, 'report': str(output/'report.html')}))


if __name__ == '__main__':
    report(ROOT / 'results/memeye_qwen_cvm_pca_dimensions_1to10_zscore_l0p03_a1_20261114')
