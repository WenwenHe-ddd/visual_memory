#!/usr/bin/env python3
"""Report three-seed basis ablations for GPT-5, DeepSeek and the Qwen reference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/memeye_api_random2d_report_mpl')

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'reports/memeye_api_random2d_ablation_20261105'
QWEN = ROOT / 'reports/memeye_qwen_random2d_ablation_20261105'
SEEDS = [0, 1, 2]
CONDITIONS = ['pca_2d', *[f'random_2d_seed_{s}' for s in SEEDS]]
SOURCES = {
    'gpt5': ROOT / 'results/memeye_gpt5_base_seed42_full_20261024',
    'deepseek': ROOT / 'results/memeye_deepseek_flash_seed42_full_20261025',
}
LABELS = {'qwen': 'Qwen3-VL-8B-Instruct', 'gpt5': 'GPT-5',
          'deepseek': 'DeepSeek-V4.1-Flash (DashScope alias)'}


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reduce_rows(rows):
    assert rows
    hits = sum(r['gold_hits'] for r in rows)
    gold = sum(r['gold_references'] for r in rows)
    correct = sum(r['correct'] for r in rows)
    return {'questions': len(rows), 'gold_hits': hits, 'gold_references': gold,
            'recall_percent': 100 * hits / gold,
            'correct': correct, 'accuracy_percent': 100 * correct / len(rows)}


def collect_api(base, backend, tracked):
    folder = base / backend
    source = SOURCES[backend]
    paths = [folder / 'retrieval_protocol.json', folder / 'retrieval_results.jsonl',
             folder / 'reader/summary.json', folder / 'reader/responses.jsonl',
             folder / 'reader/protocol.json', folder / 'reader/validation_output.json',
             source / 'inputs/evaluation.jsonl']
    tracked.update({str(p): sha(p) for p in paths})
    reader = read(folder / 'reader/summary.json')
    assert reader['status'] == 'complete'
    validation = read(folder / 'reader/validation_output.json')
    assert validation['passed'] and validation['logical_records'] == 1484
    for name in ['requests', 'fresh_responses', 'responses', 'api_attempts']:
        assert sha(folder / 'reader' / (name + '.jsonl')) == validation[name + '_sha256']
    gold = {r['id']: r for r in lines(source / 'inputs/evaluation.jsonl')}
    raw = lines(folder / 'retrieval_results.jsonl')
    assert len(raw) == len(gold) == 371
    predictions = lines(folder / 'reader/responses.jsonl')
    pred = {(r['question_id'], r['condition']): r for r in predictions}
    assert len(pred) == len(predictions) == 1484
    out = []
    for item in raw:
        qid = item['question_id']
        assert {c['name'] for c in item['conditions']} == set(CONDITIONS)
        for condition in item['conditions']:
            c = condition['name']
            answer = pred[qid, c]
            ids = condition['top5_memory_ids']
            assert len(ids) == len(set(ids)) == 5
            # Report scoring deliberately occurs after all retrieval/API work.
            choice = answer.get('predicted_choice')
            if 'predicted_choice' not in answer:
                from scripts.evaluation.memeye_api_reader import parse_choice
                choice = parse_choice(answer.get('raw_response', ''))
            out.append({'backend': backend, 'question_id': qid, 'domain': item['task'],
                        'condition': c, 'gold_hits': len(set(ids) & set(gold[qid]['gold_ids'])),
                        'gold_references': len(gold[qid]['gold_ids']),
                        'correct': answer.get('status') == 'ok' and choice == gold[qid]['gold_choice']})
    for c in CONDITIONS:
        metrics = reduce_rows([r for r in out if r['condition'] == c])
        assert metrics['questions'] == reader['overall'][c]['questions'] == 371
        assert metrics['correct'] == reader['overall'][c]['correct']
        assert metrics['gold_references'] == 1041
    return out, reader


def collect_qwen(tracked):
    rpath = QWEN / 'retrieval_results.jsonl'
    apath = QWEN / 'reader/per_question.jsonl'
    tracked.update({str(p): sha(p) for p in (rpath, apath, QWEN / 'summary.json')})
    retrieval = {(r['question_id'], r['condition']): r for r in lines(rpath)
                 if r['condition'] in CONDITIONS}
    answers = [r for r in lines(apath) if r['condition'] in CONDITIONS]
    assert len(retrieval) == len(answers) == 1484
    return [{'backend': 'qwen', 'question_id': a['question_id'], 'domain': a['domain'],
             'condition': a['condition'], 'correct': a['correct'],
             'gold_hits': retrieval[a['question_id'], a['condition']]['gold_hits_top5'],
             'gold_references': len(retrieval[a['question_id'], a['condition']]['gold_ids'])}
            for a in answers]


def comparison(backend, domain, rows):
    values = {c: reduce_rows([r for r in rows if r['condition'] == c]) for c in CONDITIONS}
    ref = values['pca_2d']
    random = [values[c] for c in CONDITIONS[1:]]
    return {'backend': backend, 'domain': domain, 'questions': ref['questions'],
            'gold_references': ref['gold_references'], 'pca_gold_hits': ref['gold_hits'],
            'pca_recall_percent': ref['recall_percent'],
            'random_recall_mean_percent': statistics.mean(r['recall_percent'] for r in random),
            'random_recall_sd_pp': statistics.stdev(r['recall_percent'] for r in random),
            'pca_correct': ref['correct'], 'pca_accuracy_percent': ref['accuracy_percent'],
            'random_correct_mean': statistics.mean(r['correct'] for r in random),
            'random_accuracy_mean_percent': statistics.mean(r['accuracy_percent'] for r in random),
            'random_accuracy_sd_pp': statistics.stdev(r['accuracy_percent'] for r in random)}


def run(base):
    import sys
    sys.path.insert(0, str(ROOT))
    tracked = {}
    all_rows = collect_qwen(tracked)
    readers = {}
    for backend in SOURCES:
        rows, readers[backend] = collect_api(base, backend, tracked)
        all_rows.extend(rows)
    result, domains, seeds = [], [], []
    for backend in LABELS:
        rows = [r for r in all_rows if r['backend'] == backend]
        result.append(comparison(backend, 'all', rows))
        for domain in sorted({r['domain'] for r in rows}):
            domains.append(comparison(backend, domain, [r for r in rows if r['domain'] == domain]))
        for c in CONDITIONS:
            seeds.append({'backend': backend, 'condition': c,
                          **reduce_rows([r for r in rows if r['condition'] == c])})
    csv_write(base / 'comparison.csv', result)
    csv_write(base / 'domain_comparison.csv', domains)
    csv_write(base / 'seed_metrics.csv', seeds)
    csv_write(base / 'per_question.csv', all_rows)
    summary = {'completed': True, 'dataset': 'MemEye', 'random_seeds': SEEDS,
               'questions_per_condition': 371, 'domains': 8, 'online_weighting': 'none',
               'comparison': result, 'reader_runs': readers,
               'qwen_reference': str(QWEN),
               'std_definition': 'Sample SD across three basis seeds 0,1,2, in percentage points; not a question confidence interval.',
               'controlled_variable': 'Fixed two-dimensional visual basis; captions and text partition are fixed within each backbone.',
               'caption_policy': 'Reuse each backbone\'s own completed caption set; not Qwen captions for API backbones.',
               'reader_policy': 'Fresh paired answers for PCA and Random conditions; exact same-run requests share one response.',
               'limitations': ['Only this fixed MemEye protocol is tested.',
                              'Three random seeds measure basis variability, not API sampling variability.',
                              'DeepSeek identifier is a provider alias with no verified weight revision.',
                              'Differences versus historical PCA answer counts also include a fresh API run.']}
    dump(base / 'summary.json', summary)
    text = ['MemEye / PCA-2D vs Random-2D：三个种子（0、1、2）', '',
            'GPT-5 与 DeepSeek 均固定各自已生成的 caption，并重放动态记忆加入过程。',
            '每张新图编码一次，四条件共享当前向量后释放；每叶只保存两个视觉坐标。',
            '同一 backbone 内只有二维基底改变；文本分簇、CLIP、BGE、候选陈述、题目及回答参数一致。',
            '无 z-score；每种方法遍历相同 domain 的全部候选叶子并取 Top5。',
            'GPT-5 和 DeepSeek 的 PCA 与三个 Random 条件均在本次重新回答，同一完整请求只调用一次。',
            'Qwen 行引用此前已完成的三个种子实验；本次没有重跑 Qwen。', '',
            '总体结果：± 为三种子的样本标准差，单位为百分点。']
    for r in result:
        text += [f"{LABELS[r['backend']]}:",
                 f"  PCA: Recall@5 {r['pca_gold_hits']}/1041 = {r['pca_recall_percent']:.4f}%; QA {r['pca_correct']}/371 = {r['pca_accuracy_percent']:.4f}%.",
                 f"  Random: Recall@5 {r['random_recall_mean_percent']:.4f} ± {r['random_recall_sd_pp']:.4f}%; QA {r['random_accuracy_mean_percent']:.4f} ± {r['random_accuracy_sd_pp']:.4f}%."]
    text += ['', '本次 API 回答覆盖和异常：']
    for backend, r in readers.items():
        text.append(f"{LABELS[backend]}: {r['logical_records']} 条逻辑记录，{r['fresh_unique_requests']} 个不同完整请求，"
                    f"{r['api_calls']} 次 API 调用；格式错误 {r['format_errors']}，拒绝 {r['content_rejections']}，"
                    f"输出截断 {r['token_limit_hits']}（异常数均按逻辑记录统计）。实际返回模型：{', '.join(r['returned_models'])}。")
    text += ['', 'Recall@5 = 全题 Top5 命中的标注记忆数 / 全题标注记忆数（1041）。',
             '准确率 = 正确回答数 / 371；API拒绝、格式错误等终态不从分母中删除。',
             '不能把微小变化解读为统计显著提升，也不能将跨 backbone 的差异只归因于二维基底。',
             'DeepSeek 为 DashScope 的 deepseek-v4.1-flash API 型号；底层权重版本未获确认。',
             'API配置参考：GPT-5官方模型文档 https://developers.openai.com/api/docs/models/gpt-5；实际请求参数和返回版本以各 reader/protocol.json 与 responses.jsonl 为准。',
             '', '复现和核验：',
             'scripts/analysis/build_retrieve_memeye_api_random2d.py：重放建库和无权重检索。',
             'scripts/analysis/evaluate_memeye_api_random2d.py：准备请求、API回答、评分。',
             'scripts/analysis/report_memeye_api_random2d.py：合并三个backbone的三个种子统计。']
    (base / 'report.txt').write_text('\n'.join(text) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    labels = ['Qwen3-VL-8B', 'GPT-5', 'DeepSeek Flash']
    for ax, metric, title in zip(axes, ['recall', 'accuracy'], ['Recall@5', 'QA accuracy']):
        for i, r in enumerate(result):
            pca, mean, sd = r[f'pca_{metric}_percent'], r[f'random_{metric}_mean_percent'], r[f'random_{metric}_sd_pp']
            ax.bar(i - .18, pca, width=.34, color='#3274A1', label='PCA-2D' if i == 0 else None)
            ax.bar(i + .18, mean, width=.34, color='#3C9A77', yerr=sd, capsize=4,
                   label='Random-2D' if i == 0 else None)
            ax.text(i - .18, pca + 2, f'{pca:.2f}', ha='center', fontsize=9)
            ax.text(i + .18, mean + 2, f'{mean:.2f}', ha='center', fontsize=9)
        ax.set_xticks(range(3), labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percent')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['right', 'top']].set_visible(False)
    axes[0].legend(loc='upper left')
    fig.suptitle('MemEye | Fixed-basis ablation | Caption visual master')
    fig.text(.5, .03, '371 questions / backbone · Random seeds 0, 1, 2 · Error bars: sample SD', ha='center')
    fig.tight_layout(rect=(0, .07, 1, .94))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(base / f'comparison.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    paths = ['summary.json', 'comparison.csv', 'domain_comparison.csv', 'seed_metrics.csv',
             'per_question.csv', 'report.txt', 'comparison.png', 'comparison.pdf', 'comparison.svg']
    dump(base / 'report_manifest.json', {'completed': True,
         'source_sha256': tracked, 'script_sha256': sha(Path(__file__)),
         'output_sha256': {p: sha(base / p) for p in paths}})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    run(parser.parse_args().output.resolve())
