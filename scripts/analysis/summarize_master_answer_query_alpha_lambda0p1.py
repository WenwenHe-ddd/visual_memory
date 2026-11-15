"""Verify and combine completed master-answer-query alpha experiments at lambda=.1."""
from __future__ import annotations
import copy
import html
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.analysis.evaluate_master_answer_query_alpha_lambda0p1 import (
    check_sources, csv_write, digest, dump, lines, read, sha, verify,
)

BASE = ROOT / 'reports/master_answer_query_lambda0p1_alpha_ablation_20261024'
RUNS = {
    'alpha_0p01': BASE / 'alpha_0p01',
    'alpha_0p1': BASE / 'alpha_0p1',
    'alpha_1': ROOT / 'reports/master_answer_query_lambda_ablation_20261023/lambda_0p1',
    'alpha_10': BASE / 'alpha_10',
}
ALPHAS = {'raw': None, 'alpha_0p01': .01, 'alpha_0p1': .1, 'alpha_1': 1., 'alpha_10': 10.}


def compare(rows, control, treatment):
    wins = [r['question_id'] for r in rows if not r[control + '_correct'] and r[treatment + '_correct']]
    losses = [r['question_id'] for r in rows if r[control + '_correct'] and not r[treatment + '_correct']]
    m = len(wins) + len(losses)
    p = min(1., 2 * sum(math.comb(m, i) for i in range(min(len(wins), len(losses)) + 1)) / 2**m) if m else 1.
    return {'wrong_to_right': len(wins), 'right_to_wrong': len(losses),
            'net_correct': len(wins) - len(losses), 'delta_percentage_points': 100 * (len(wins) - len(losses)) / len(rows),
            'changed_top5_order': sum(r[control + '_top5'] != r[treatment + '_top5'] for r in rows),
            'changed_top5_set': sum(set(r[control + '_top5']) != set(r[treatment + '_top5']) for r in rows),
            'mcnemar_exact_two_sided_p_unadjusted': p, 'wins': wins, 'losses': losses}


def main():
    original = lines(ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl')
    ids = [q['id'] for q in original]
    combined = {q['id']: {'question_id': q['id'], 'domain': q['task'], 'gold_choice': q['gold_choice']} for q in original}
    annotations = {}; sources = {str(Path(__file__)): sha(__file__)}; provenance = {}; raw_predictions = None
    settings = None
    payload_predictions = {}; repeated_payload_checks = 0
    for label, directory in RUNS.items():
        method = read(directory / 'protocol.json')
        check_sources(method)
        protocol = read(directory / 'reader/protocol.json'); verify(protocol)
        signed = dict(protocol); signature = signed.pop('experiment_signature')
        assert digest(signed) == signature
        if settings is None: settings = protocol['reader_settings']
        assert protocol['reader_settings'] == settings
        assert method['lambda'] == .1 and method['alpha'] == ALPHAS[label]
        audit = read(directory / 'reader/independent_audit.json'); assert audit['passed']
        summary = read(directory / 'reader/summary.json')
        assert not summary['format_errors'] and not summary['token_limit_hits']
        retrieved = {r['question_id']: r for r in lines(directory / 'retrieval_results.jsonl')}
        responses = lines(directory / 'reader/responses.jsonl')
        answers = {r['record_id']: r for r in responses}
        manifest = {r['record_id']: r for r in lines(directory / 'reader/qa_manifest.jsonl')}
        assert len(answers) == len(responses) == len(manifest) == 742 and set(answers) == set(manifest)
        assert len(retrieved) == 371 and set(retrieved) == set(ids)
        baseline_name, treatment_name = method['conditions']
        assert baseline_name == 'master_answer_query'
        current_raw = []
        for q in original:
            qid = q['id']; row = retrieved[qid]
            assert row['evaluation'] == {k: q[k] for k in ('gold_ids', 'gold_choice')}
            for condition in row['conditions']:
                name = condition['name']; rid = qid + '::' + name
                response, item = answers[rid], manifest[rid]
                for key in ('id', 'task', 'question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths'):
                    assert item[key] == q[key]
                assert response['experiment_signature'] == signature and response['run_id'] == protocol['experiment_run_id']
                assert response['payload_sha256'] == item['payload_sha256']
                payload_hash = item['payload_sha256']
                if payload_hash in payload_predictions:
                    assert response['predicted_choice'] == payload_predictions[payload_hash], rid
                    repeated_payload_checks += 1
                else:
                    payload_predictions[payload_hash] = response['predicted_choice']
                assert not response['historical_response_reuse']
                assert response['correct'] == (response['predicted_choice'] == q['gold_choice'])
                assert item['top_memory_ids'] == condition['top5_memory_ids']
                hits = len(set(condition['top5_memory_ids']) & set(q['gold_ids']))
                assert hits == condition['gold_hits_top5']
                target = 'raw' if name == baseline_name else label
                combined[qid].update({target + '_answer': response['predicted_choice'],
                    target + '_correct': response['correct'], target + '_top5': condition['top5_memory_ids'],
                    target + '_gold_hits': hits, target + '_top1_gold': condition['top5_memory_ids'][0] in q['gold_ids']})
                if target == 'raw':
                    current_raw.append((qid, response['predicted_choice'], response['payload_sha256'], item['top_memory_ids']))
        if raw_predictions is None: raw_predictions = current_raw
        assert current_raw == raw_predictions
        annotation = copy.deepcopy(summary['method_annotations'][treatment_name])
        annotation['result_origin'] = str(directory)
        annotation['new_in_this_ablation'] = label in ('alpha_0p01', 'alpha_0p1', 'alpha_10')
        annotations[label] = annotation
        if label == 'alpha_0p01': annotations['raw'] = copy.deepcopy(summary['method_annotations'][baseline_name])
        provenance[label] = {'directory': str(directory), 'new_in_this_ablation': annotation['new_in_this_ablation'],
            'logical_records': summary['logical_records'], 'fresh_inferences': summary['fresh_inferences'],
            'same_run_shared_records': summary['same_run_shared_records'], 'independent_audit_passed': True,
            'device': 'cuda:' + str(protocol['placement']['gpu_index'])}
        for filename in ('protocol.json', 'retrieval_results.jsonl', 'reader/protocol.json', 'reader/qa_manifest.jsonl',
                         'reader/responses.jsonl', 'reader/summary.json', 'reader/independent_audit.json'):
            path = directory / filename; sources[str(path)] = sha(path)
    rows = [combined[qid] for qid in ids]; gold_total = sum(len(q['gold_ids']) for q in original)
    assert gold_total == 1041

    def stats(part):
        return {name: {'correct': sum(r[name + '_correct'] for r in part),
            'accuracy': sum(r[name + '_correct'] for r in part) / len(part),
            'gold_hits_top5': sum(r[name + '_gold_hits'] for r in part),
            'questions_with_any_gold': sum(r[name + '_gold_hits'] > 0 for r in part),
            'top1_gold': sum(r[name + '_top1_gold'] for r in part)} for name in ALPHAS}

    summary = {'questions': 371, 'gold_references': gold_total, 'lambda': .1, 'alphas': ALPHAS,
        'overall': stats(rows), 'method_annotations': annotations, 'result_provenance': provenance,
        'comparison_vs_raw': {name: compare(rows, 'raw', name) for name in RUNS},
        'comparison_vs_alpha1': {name: compare(rows, 'alpha_1', name) for name in RUNS if name != 'alpha_1'},
        'by_domain': {domain: {'questions': sum(r['domain'] == domain for r in rows),
            'conditions': stats([r for r in rows if r['domain'] == domain])} for domain in sorted({r['domain'] for r in rows})},
        'source_sha256': sources, 'validation': {'passed': True, 'identical_question_ids_and_gold': True,
            'all_raw_predictions_payloads_and_rankings_identical': True, 'all_independent_audits_passed': True,
            'all_identical_payload_answers_match': True, 'repeated_payload_answer_checks': repeated_payload_checks},
        'interpretation': 'Same-test-set fixed-option-order comparison. Alpha .01/.1/10 are fresh on GPU 0, alpha 1 is the verified previous result on GPU 1; raw baseline predictions match across all runs. All p-values unadjusted.'}
    for path, expected in sources.items(): assert sha(path) == expected
    dump(BASE / 'summary.json', summary); csv_write(BASE / 'paired_results.csv', rows)
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>master_answer_query α 消融</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#20333d}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>',
        '<h1>master_answer_query：固定 λ=0.1 的 α 消融</h1>',
        '<p>8 个 domain、371 题、438 条记忆、88 个 master。各 master 视觉匹配完整候选陈述，使用扩展查询匹配对话＋完整 caption，在簇内在线赋权后合并取 Top-5。final = raw + α × std_pop(raw in master) × tanh(z_white)。</p>',
        '<p>α=0.01、0.1、10 在空闲 GPU 0 分别重新评测，均含新执行的未赋权对照；α=1 为前次在 GPU 1 完成并通过来源验证的 202/371 结果。不同 GPU 的型号相同，未赋权逐题答案及相同完整输入的回答均检查一致性。仅同一次运行中的完全相同输入共享新推理，无历史回答复用。</p>',
        '<table><tr><th>设置</th><th>答对</th><th>准确率</th><th>目标记忆命中</th><th>至少一条目标</th><th>相对未赋权</th></tr>']
    for name, alpha in ALPHAS.items():
        c = summary['overall'][name]; label = '未赋权' if alpha is None else f'α={alpha:g}'
        delta = c['correct'] - summary['overall']['raw']['correct']
        parts.append(f'<tr><td>{label}</td><td>{c["correct"]}/371</td><td>{c["accuracy"]:.4%}</td><td>{c["gold_hits_top5"]}/1041</td><td>{c["questions_with_any_gold"]}/371</td><td>{delta:+d}</td></tr>')
    parts.append('</table><h2>配对比较</h2>')
    for reference, comparisons in [('未赋权', summary['comparison_vs_raw']), ('α=1', summary['comparison_vs_alpha1'])]:
        for name, c in comparisons.items():
            parts.append(f'<p>α={ALPHAS[name]:g} 相对 {reference}：错→对 {c["wrong_to_right"]}，对→错 {c["right_to_wrong"]}，净变化 {c["net_correct"]:+d}；McNemar 双侧 p={c["mcnemar_exact_two_sided_p_unadjusted"]:.4f}。Top-5 顺序变化 {c["changed_top5_order"]} 题，集合变化 {c["changed_top5_set"]} 题。</p>')
    parts.append('<p>所有 p 值均未校正多次比较；这是同一测试集上的参数实验，非独立验证集结论。</p><h2>各 domain</h2><table><tr><th>Domain</th><th>题数</th><th>未赋权</th><th>α=0.01</th><th>α=0.1</th><th>α=1</th><th>α=10</th></tr>')
    for domain, value in summary['by_domain'].items():
        cells = ''.join(f'<td>{value["conditions"][name]["correct"]}</td>' for name in ALPHAS)
        parts.append(f'<tr><td>{html.escape(domain)}</td><td>{value["questions"]}</td>{cells}</tr>')
    parts.append('</table><h2>五项方法标注与来源</h2>')
    for name in ALPHAS:
        parts.append('<details><summary>' + html.escape(name) + '</summary><pre>' + html.escape(json.dumps(annotations[name], ensure_ascii=False, indent=2)) + '</pre></details>')
    parts.append('<p><a href="summary.json">完整统计及来源哈希</a> · <a href="paired_results.csv">全部设置逐题对照</a> · <a href="alpha_0p01/reader/independent_audit.json">α=0.01 核验</a> · <a href="alpha_0p1/reader/independent_audit.json">α=0.1 核验</a> · <a href="alpha_10/reader/independent_audit.json">α=10 核验</a></p></html>')
    (BASE / 'report.html').write_text('\n'.join(parts))
    print(json.dumps({'overall': summary['overall'], 'validation': summary['validation']}, ensure_ascii=False))


if __name__ == '__main__': main()
