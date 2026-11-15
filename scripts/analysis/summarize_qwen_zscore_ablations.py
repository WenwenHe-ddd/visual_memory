"""Regrade saved Qwen ablations and export provenance-aware tables and curves.

Read-only with respect to the experiment inputs; never loads a reader or calls an API.
The joint search takes precedence for overlapping settings, irrespective of score.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.analysis.evaluate_adaptive_changed_qa import reader_text

JOINT = ROOT / 'results/memeye_qwen_zscore_joint_search_20261024'
DEFAULT = ROOT / 'reports/qwen_zscore_ablation_summary_20261025'
METHODS = ('baseline_question_only', 'master_answer_query')
SOURCE_NAMES = (
    'baseline_question_only_lambda_ablation_20261023',
    'baseline_question_only_alpha_ablation_20261023',
    'baseline_question_only_lambda0p1_alpha_ablation_20261024',
    'master_answer_query_alpha_ablation_20261023/alpha_0p01',
    'master_answer_query_alpha_ablation_20261023/alpha_0p1',
    'master_answer_query_alpha10_20261023',
    'master_answer_query_lambda_ablation_20261023/lambda_0p1',
    'master_answer_query_lambda_ablation_20261023/lambda_1p0',
    'master_answer_query_lambda0p1_alpha_ablation_20261024/alpha_0p01',
    'master_answer_query_lambda0p1_alpha_ablation_20261024/alpha_0p1',
    'master_answer_query_lambda0p1_alpha_ablation_20261024/alpha_10',
)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


HASHES = {}


def sha(path):
    path = str(Path(path).resolve())
    if path not in HASHES:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        HASHES[path] = h.hexdigest()
    return HASHES[path]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_key(protocol, name, source):
    if 'condition_specs' in protocol:
        spec = protocol['condition_specs'][name]
        return spec['method'], spec['lambda'], spec['alpha']
    mode = METHODS[0] if 'baseline_question_only' in str(source) else METHODS[1]
    spec = protocol['method_annotations'][name]['3. 检索方式']
    return mode, spec['lambda'], (spec['alpha'] or 0)


def collect():
    original = ROOT / 'data/independent_visual_inputs'
    questions = {q['id']: q for q in lines(original / 'qa_manifest.jsonl')}
    memories = {m['id']: m['reader_text'] for m in lines(original / 'retrieval_memories.jsonl')}
    assert len(questions) == 371 and len(memories) == 438
    source_dirs = [JOINT] + [ROOT / 'reports' / n for n in SOURCE_NAMES]
    reference_protocol = read(JOINT / 'reader/protocol.json')
    chosen, all_runs, source_audits, overlaps = {}, [], [], []
    all_payload_answers = {}
    payload_repeats = 0
    # All source records are checked, including controls and settings not needed
    # for the two principal curves. Referenced-only columns are not counted twice.
    for source in source_dirs:
        folder = source / 'reader'
        p = read(folder / 'protocol.json')
        signed = dict(p)
        signature = signed.pop('experiment_signature')
        assert digest(signed) == signature, source
        for key in ('reader_settings', 'weight_files', 'model_source_sha256', 'attention_implementation'):
            assert p[key] == reference_protocol[key], (source, key)
        data_hash_checks = 0
        for path, expected in p['source_sha256'].items():
            # Historical code may have evolved; verify the frozen data used here,
            # without claiming that every current code file matches the old run.
            if Path(path).suffix not in ('.json', '.jsonl', '.npz', '.npy'):
                continue
            assert sha(path) == expected, path
            data_hash_checks += 1
        assert str(folder / 'qa_manifest.jsonl') in p['source_sha256']
        audit_path = source / 'independent_audit.json' if source == JOINT else folder / 'independent_audit.json'
        prior_audit = read(audit_path)
        assert prior_audit['passed'], source
        if source == JOINT:
            for path, expected in prior_audit['artifact_sha256'].items():
                assert sha(source / path) == expected, path
        summary_path = source / 'summary.json' if source == JOINT else folder / 'summary.json'
        summary = read(summary_path)
        overall = summary['overall'].get('conditions', summary['overall'])
        manifest = {r['record_id']: r for r in lines(folder / 'qa_manifest.jsonl')}
        responses = list(lines(folder / 'responses.jsonl'))
        by_id = {r['record_id']: r for r in responses}
        assert len(manifest) == len(responses) == len(by_id) and set(manifest) == set(by_id)
        per_condition = defaultdict(dict)
        fresh_count = 0
        for response in responses:
            row = manifest[response['record_id']]
            q = questions[row['id']]
            for key in ('task', 'question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths'):
                assert row[key] == q[key], (source, row['record_id'], key)
            assert len(row['top_memory_ids']) == len(set(row['top_memory_ids'])) == 5
            assert all(mid.split('/', 1)[0] == q['task'] for mid in row['top_memory_ids'])
            assert row['reader_text'] == reader_text(q, row['top_memory_ids'], memories)
            assert [i['path'] for i in row['query_images']] == q['query_image_paths']
            for image in row['query_images']:
                assert sha(image['path']) == image['sha256']
            payload = {'system': p['reader_settings']['reader_system_prompt'],
                       'user_text': row['reader_text'], 'query_images': row['query_images']}
            assert digest(payload) == row['payload_sha256'] == response['payload_sha256']
            match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', response['raw_response'].strip(), re.I)
            answer = match.group(1).upper() if match else None
            assert answer is not None and response['status'] == 'ok' and not response['hit_token_limit']
            assert response['predicted_choice'] == answer
            assert response['gold_choice'] == q['gold_choice']
            assert response['correct'] == (answer == q['gold_choice'])
            assert response['condition'] == row['condition'] and response['question_id'] == q['id']
            assert response['experiment_signature'] == signature and response['run_id'] == p['experiment_run_id']
            assert not response['historical_response_reuse']
            if response['fresh_inference_executed_for_this_record']:
                fresh_count += 1
            else:
                origin = by_id[response['same_run_shared_inference_record_id']]
                assert origin['fresh_inference_executed_for_this_record']
                for k in ('payload_sha256', 'raw_response', 'generated_token_ids', 'execution_id'):
                    assert response[k] == origin[k]
            payload_hash = row['payload_sha256']
            if payload_hash in all_payload_answers:
                assert all_payload_answers[payload_hash] == answer, (source, row['record_id'], 'answer drift')
                payload_repeats += 1
            else:
                all_payload_answers[payload_hash] = answer
            per_condition[row['condition']][q['id']] = {
                'answer': answer, 'correct': answer == q['gold_choice'],
                'payload_sha256': payload_hash, 'top_memory_ids': row['top_memory_ids'],
                'gold_hits_top5': len(set(row['top_memory_ids']) & set(q['gold_ids'])),
            }
        assert fresh_count == p['unique_reader_payloads']
        for name, answers in per_condition.items():
            assert set(answers) == set(questions)
            key = condition_key(p, name, source)
            correct = sum(r['correct'] for r in answers.values())
            hits = sum(r['gold_hits_top5'] for r in answers.values())
            assert correct == overall[name]['correct'] and hits == overall[name]['gold_hits_top5']
            annotation = copy.deepcopy(p['method_annotations'][name])
            annotation['5. 回答方式'].update({
                'reader_called': True, 'correct': correct, 'questions': 371,
                'measured_qa_accuracy': correct / 371, 'new_inference_in_this_consolidation': False,
                'answer_source_reader_directory': str(folder),
            })
            item = {'method': key[0], 'lambda': key[1], 'alpha': key[2],
                    'condition': name, 'correct': correct, 'questions': 371, 'accuracy': correct / 371,
                    'gold_hits_top5': hits, 'gold_references': 1041,
                    'source_kind': 'joint_25' if source == JOINT else 'historical_supplement',
                    'reader_directory': str(folder), 'source_run_id': p['experiment_run_id'],
                    'source_signature': signature, 'method_annotations': annotation, 'answers': answers}
            all_runs.append({k: v for k, v in item.items() if k not in ('answers', 'method_annotations')})
            if key in chosen:
                first = chosen[key]
                mismatches = [qid for qid in questions if answers[qid] != first['answers'][qid]]
                assert not mismatches, (source, name, mismatches[:5])
                overlaps.append({'method': key[0], 'lambda': key[1], 'alpha': key[2],
                                 'first_source': first['reader_directory'], 'repeated_source': str(folder),
                                 'questions': 371, 'payload_answer_and_top5_disagreements': 0})
            else:
                chosen[key] = item
        source_audits.append({'reader_directory': str(folder), 'records_regraded': len(responses),
                              'actual_conditions': list(per_condition), 'protocol_signature_verified': True,
                              'data_hash_checks': data_hash_checks, 'source_audit_path': str(audit_path),
                              'source_audit_reported_passed': True,
                              'source_sha256': {str(x): sha(x) for x in (
                                  folder / 'protocol.json', folder / 'qa_manifest.jsonl',
                                  folder / 'responses.jsonl', summary_path, audit_path)}})
        print(json.dumps({'verified': str(source.relative_to(ROOT)), 'records': len(responses)}), flush=True)
    audit = {'passed': True, 'scope': 'Saved data, settings, payload reconstruction, raw-answer regrading, overlap parity; no inference or whitening rerun.',
             'new_model_calls': 0, 'questions': 371, 'memory_leaves': 438,
             'source_runs': source_audits, 'records_regraded': sum(r['records_regraded'] for r in source_audits),
             'source_conditions': len(all_runs), 'unique_conditions': len(chosen),
             'identical_payload_repeat_checks': payload_repeats, 'identical_payload_answer_disagreements': 0,
             'overlapping_conditions': overlaps, 'unverified': ['Model weight bytes were not rehashed; frozen weight checksums match.',
                 'No new token decoding or numerical whitening rerun; previous independent audits are retained as provenance.']}
    return chosen, all_runs, audit, questions


def plot_curves(output, pairs, controls):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter, NullLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = ('#2474B5', '#C65D21')
    panel_specs = [('lambda', 'alpha', 1., r'(a) Vary $\lambda$, fixed $\alpha=1$'),
                   ('alpha', 'lambda', .01, r'(b) Vary $\alpha$, fixed $\lambda=0.01$'),
                   ('alpha', 'lambda', .1, r'Vary $\alpha$, fixed $\lambda=0.1$')]

    def draw(ax, spec):
        varying, fixed, value, title = spec
        data = sorted([r for r in pairs if r[fixed] == value], key=lambda r: r[varying])
        xs = [r[varying] for r in data]
        for i, (short, method) in enumerate(zip(('baseline', 'master'), METHODS)):
            ys = [r[short + '_correct'] for r in data]
            ax.plot(xs, ys, color=colors[i], lw=2, zorder=2)
            for x, y, row in zip(xs, ys, data):
                joint = row[short + '_source_kind'] == 'joint_25'
                ax.scatter(x, y, s=53, facecolor=colors[i] if joint else 'white',
                           edgecolor=colors[i], linewidth=1.7, zorder=3)
                ax.annotate(str(y), (x, y), xytext=(0, 9 if i == 1 else -17),
                            textcoords='offset points', ha='center', color=colors[i], fontsize=10)
            ax.axhline(controls[method], color=colors[i], lw=1.15, linestyle='--', alpha=.7)
        ax.set_xscale('log'); ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xticklabels([format(x, 'g') for x in xs], rotation=25 if len(xs) > 6 else 0)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(176, 207); ax.set_yticks([180, 185, 190, 195, 200, 205])
        ax.set_title(title, loc='left', fontsize=13, pad=15)
        ax.set_xlabel('$\\' + varying + '$'); ax.set_ylabel('Correct answers / 371')
        ax.grid(axis='y', color='#e5e7eb'); ax.set_axisbelow(True)
        return data

    legend = [Line2D([], [], color=colors[0], lw=2, label='baseline_question_only + z-score'),
              Line2D([], [], color=colors[1], lw=2, label='master_answer_query + z-score'),
              Line2D([], [], color='#777777', ls='--', label='Unweighted controls: 184 / 195')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, spec in zip(axes, panel_specs):
        draw(ax, spec)
    fig.suptitle('Qwen3-VL-8B-Instruct | MemEye hyperparameter ablations', x=.065, ha='left', fontsize=16)
    fig.legend(handles=legend, loc='lower center', ncol=3, bbox_to_anchor=(.5, .055), frameon=False, fontsize=10)
    fig.text(.065, .025, 'Filled markers: 25-pair joint search. Hollow markers: earlier matched runs. No new model calls.', fontsize=10, color='#4b5563')
    fig.subplots_adjust(top=.81, bottom=.24, left=.065, right=.98, wspace=.22)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / ('ablation_curves.' + ext), dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    draw(ax, panel_specs[2])
    fig.legend(handles=legend, loc='lower center', ncol=1, bbox_to_anchor=(.5, .015), frameon=False, fontsize=9)
    fig.subplots_adjust(top=.88, bottom=.29, left=.12, right=.96)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / ('alpha_lambda0p1.' + ext), dpi=200)
    plt.close(fig)

    # Plot only the complete 25-pair grid; supplementary pairs remain in tables.
    import numpy as np
    from matplotlib.patches import Rectangle
    grid = [r for r in pairs if r['baseline_source_kind'] == r['master_source_kind'] == 'joint_25']
    lambdas, alphas = sorted({r['lambda'] for r in grid}), sorted({r['alpha'] for r in grid})
    assert len(grid) == len(lambdas) * len(alphas) == 25
    lookup = {(r['lambda'], r['alpha']): r for r in grid}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    for ax, short, control in zip(axes, ('baseline', 'master'), (controls[METHODS[0]], controls[METHODS[1]])):
        delta = np.array([[lookup[l, a][short + '_correct'] - control for a in alphas] for l in lambdas])
        im = ax.imshow(delta, cmap='RdBu', vmin=-12, vmax=12)
        for i, l in enumerate(lambdas):
            for j, a in enumerate(alphas):
                r = lookup[l, a]; d = int(delta[i, j])
                ax.text(j, i, f"{r[short + '_correct']}\n({d:+d})", ha='center', va='center',
                        color='white' if abs(d) >= 8 else '#17212e', fontsize=11)
                if r['passes_all_three']:
                    ax.add_patch(Rectangle((j-.46, i-.46), .92, .92, fill=False, lw=2.2, edgecolor='#38a169'))
        ax.set_xticks(range(len(alphas)), [format(a, 'g') for a in alphas]); ax.set_xlabel(r'$\alpha$')
        ax.set_yticks(range(len(lambdas)), [format(l, 'g') for l in lambdas]); ax.set_ylabel(r'$\lambda$')
        ax.set_title(f'{short} + z-score | unweighted = {control}/371', fontsize=12, pad=12)
    fig.suptitle('Qwen joint search: correct answers (change vs. own unweighted control)', fontsize=14)
    fig.text(.08, .025, 'Green outlines: both methods improve and master exceeds baseline. All cells are from the same joint-search run.', fontsize=10)
    fig.subplots_adjust(left=.075, right=.85, top=.85, bottom=.15, wspace=.32)
    fig.colorbar(im, cax=fig.add_axes((.89, .2, .018, .6)), label='Change in correct answers')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / ('joint_search_heatmap.' + ext), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    chosen, runs, audit, questions = collect()
    controls = {m: chosen[m, None, 0]['correct'] for m in METHODS}
    pairs = []
    settings = sorted({(l, a) for m, l, a in chosen if a > 0})
    for ridge, alpha in settings:
        b, m = (chosen[method, ridge, alpha] for method in METHODS)
        bc, mc = b['correct'], m['correct']
        pairs.append({'lambda': ridge, 'alpha': alpha, 'baseline_correct': bc, 'master_correct': mc,
                      'baseline_accuracy': bc / 371, 'master_accuracy': mc / 371,
                      'baseline_gain': bc - controls[METHODS[0]], 'master_gain': mc - controls[METHODS[1]],
                      'master_minus_baseline': mc - bc,
                      'passes_all_three': bc > controls[METHODS[0]] and mc > controls[METHODS[1]] and mc > bc,
                      'baseline_source_kind': b['source_kind'], 'master_source_kind': m['source_kind'],
                      'baseline_reader_directory': b['reader_directory'], 'master_reader_directory': m['reader_directory'],
                      'baseline_condition': b['condition'], 'master_condition': m['condition']})
    assert len(settings) == 30 and len(chosen) == 62
    eligible = sorted([r for r in pairs if r['passes_all_three']],
                      key=lambda r: (-r['master_correct'], -r['baseline_correct'], r['alpha'], r['lambda']))
    assert len(eligible) == 5
    all_conditions = []
    per_question = []
    for (method, ridge, alpha), item in chosen.items():
        entry = {k: v for k, v in item.items() if k != 'answers'}
        entry['gain_vs_unweighted'] = item['correct'] - controls[method]
        all_conditions.append(entry)
        for qid, answer in item['answers'].items():
            per_question.append({'question_id': qid, 'domain': questions[qid]['task'], 'method': method,
                                 'lambda': ridge, 'alpha': alpha, 'gold_choice': questions[qid]['gold_choice'],
                                 'answer': answer['answer'], 'correct': answer['correct'],
                                 'gold_hits_top5': answer['gold_hits_top5'], 'payload_sha256': answer['payload_sha256'],
                                 'top5_memory_ids': '|'.join(answer['top_memory_ids']),
                                 'reader_directory': item['reader_directory'], 'source_condition': item['condition']})
    slices = {'lambda_at_alpha1': [r for r in pairs if r['alpha'] == 1],
              'alpha_at_lambda0p01': [r for r in pairs if r['lambda'] == .01],
              'alpha_at_lambda0p1': [r for r in pairs if r['lambda'] == .1]}
    for name, rows in slices.items():
        write_csv(output / (name + '.csv'), rows)
    write_csv(output / 'all_pairs.csv', pairs)
    write_csv(output / 'source_conditions.csv', runs)
    write_csv(output / 'per_question.csv', per_question)
    plot_curves(output, pairs, controls)
    summary = {'analysis_type': 'Existing Qwen results consolidated and regraded; no new experiment.',
               'new_model_calls': 0, 'questions': 371, 'source_run_count': len(audit['source_runs']),
               'source_selection': 'Prefer the 25-pair joint run for overlapping settings; supplement only absent settings. Never pick a run by its accuracy.',
               'unweighted_controls': controls, 'unique_weighted_parameter_pairs': 30,
               'joint_search_pairs': 25, 'additional_historical_pairs': 5,
               'conditions': all_conditions, 'pairs': pairs, 'eligible_pairs': eligible, 'slices': slices,
               'interpretation': [
                   'At fixed alpha=1, master peaks at lambda=.1 (202/371), but baseline is 183/371, below its unweighted 184/371.',
                   'Among settings meeting all three strict comparisons, lambda=.01, alpha=1 gives the highest master accuracy: baseline188, master201.',
                   'At lambda=.01, alpha10 reduces master to180 while baseline reaches192; stronger weighting is not uniformly better.',
                   'These are exploratory single-seed results on the same 371 questions, not independent held-out evidence or repeated-run means.',
               ]}
    dump(output / 'summary.json', summary)
    audit['consolidator_sha256'] = sha(Path(__file__))
    dump(output / 'audit.json', audit)

    def table(rows):
        keys = ('lambda', 'alpha', 'baseline_correct', 'master_correct', 'baseline_gain', 'master_gain', 'passes_all_three')
        labels = ('λ', 'α', 'baseline 正确题数', 'master 正确题数', 'baseline 增益', 'master 增益', '三项均通过')
        parts = ['<table><thead><tr>' + ''.join(f'<th>{x}</th>' for x in labels) + '</tr></thead><tbody>']
        for row in rows:
            parts.append('<tr>' + ''.join('<td>' + html.escape(str(row[k])) + '</td>' for k in keys) + '</tr>')
        return ''.join(parts) + '</tbody></table>'

    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>Qwen λ/α 消融汇总</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#18212e}img{max-width:100%}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:8px;text-align:right;border-bottom:1px solid #d5dce5}th{background:#f0f4f8}a{color:#1767ad}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:16px}details{margin:12px 0}</style>',
            '<h1>Qwen：λ / α 消融汇总</h1>',
            '<p>整理已有实测结果，新增模型调用为 0。固定 MemEye 371 道题、438 条 Qwen caption 记忆、88 个 master；回答模型 Qwen3-VL-8B-Instruct，seed=42，贪心解码。未加权对照：baseline 184/371（49.60%），master 195/371（52.56%）。</p>',
            '<p>覆盖 25 组同批联合搜索和 5 组历史补充设置，共 30 组 λ/α。重叠设置统一采用联合搜索批次；没有按分数挑批次。每组的两种检索使用相同参数，历史补充结果不冒充本轮重跑。</p>',
            '<p>所有来源均重新解析原始答案，并核对题目、选项、标准答案、完整回答请求、Top-5 与来源哈希；重叠设置的逐题答案和请求完全一致。五项方法标注保存在 <a href="summary.json">summary.json</a>，核验范围与来源在 <a href="audit.json">audit.json</a>。</p>',
            '<h2>主消融曲线</h2><img src="ablation_curves.png" alt="固定 alpha=1 的 lambda 曲线与固定 lambda=.01 的 alpha 曲线">',
            '<p>实心点来自 25 组联合搜索；空心点来自先前同口径实验。虚线为各自未加权对照；横轴为对数刻度，纵轴为正确题数（分母均为 371）。连线仅引导阅读，不表示未测参数的预测结果。</p>',
            '<p><a href="ablation_curves.pdf">PDF</a> · <a href="ablation_curves.svg">SVG</a> · <a href="all_pairs.csv">全部参数 CSV</a> · <a href="per_question.csv">逐题 CSV</a></p>',
            '<h2>固定 α=1：λ 消融</h2>', table(slices['lambda_at_alpha1']),
            '<h2>固定 λ=0.01：α 消融</h2>', table(slices['alpha_at_lambda0p01']),
            '<h2>固定 λ=0.1：补充 α 消融</h2><img src="alpha_lambda0p1.png" alt="固定 lambda=.1 的 alpha 消融">', table(slices['alpha_at_lambda0p1']),
            '<h2>完整 25 组联合搜索</h2><img src="joint_search_heatmap.png" alt="两种检索的25组联合搜索热力图">',
            '<p>单元格为正确题数及相对各自未加权对照的变化。绿框表示 baseline 加权 &gt; baseline 未加权、master 加权 &gt; master 未加权、master 加权 &gt; baseline 加权全部成立。颜色采用两图一致的增益范围。</p>',
            '<h2>满足三项条件的 5 组</h2>', table(eligible),
            '<p>按你的共同参数约束，λ=0.01、α=1 仍是满足全部条件且 master 正确题数最高的一组：188/371 与 201/371，分别净增加 4 题、6 题。master 单独最高的 λ=0.1、α=1 是 202/371，但 baseline 为 183/371，未通过第一项。</p>',
            '<p>在 λ=0.01 时，α 从 1 增至 10，master 从 201 降到 180，baseline 从 188 升到 192；大权重对两种检索的影响不同。全部结果属于同一测试集上的探索性消融，未计算重复运行均值，不能据此声称独立测试集泛化或统计显著提升。</p>',
            '<h2>全部 30 组与来源</h2>', table(pairs),
            '<p><a href="source_conditions.csv">所有来源条件（含重叠）</a> · <a href="audit.json">核验与哈希</a></p>',
            '<h2>五项方法标注</h2>']
    for item in all_conditions:
        body.append('<details><summary>' + html.escape(f"{item['method']} | λ={item['lambda']}, α={item['alpha']} | {item['correct']}/371") + '</summary><pre>' + html.escape(json.dumps(item['method_annotations'], ensure_ascii=False, indent=2)) + '</pre></details>')
    body.append('</html>')
    (output / 'report.html').write_text('\n'.join(body))
    print(json.dumps({'output': str(output), 'controls': controls, 'pairs': len(pairs),
                      'eligible': len(eligible), 'records_regraded': audit['records_regraded'],
                      'overlap_checks': len(audit['overlapping_conditions']), 'slices': slices}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
