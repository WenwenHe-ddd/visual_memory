"""Independent controls and raw-answer comparisons for lambda=.001 vs .01/.1.

No encoder, ranking implementation, API transport or experiment runner imports.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'results/memeye_gpt54_caption_zscore_lambda0p01_alpha1_20261024'
PRIORS = {'0.01': PRIOR, '0.1': ROOT / 'results/memeye_gpt54_caption_zscore_lambda0p1_alpha1_20261024'}
CURRENT = ROOT / 'results/memeye_gpt54_caption_zscore_lambda0p001_alpha1_20261024'
NAMES = ('baseline_question_only_zscore', 'master_answer_query_zscore')
EXPECTED = {'0.01': (242, 237), '0.1': (236, 237)}
REFERENCE_FILES = ('protocol.json', 'retrieval_protocol.json', 'retrieval_results.jsonl',
                   'inputs/gold.jsonl', 'inputs/requests.jsonl', 'api_request_manifest.jsonl',
                   'responses.jsonl', 'fresh_responses.jsonl', 'summary.json', 'independent_audit.json')


def read(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def indexed(rows, key):
    result = {r[key]: r for r in rows}
    assert len(result) == len(rows)
    return result
def choice(s):
    m = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', s.strip(), re.I)
    return m.group(1).upper() if m else None
def dump(p, value): Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def compare(output=CURRENT, inputs_only=False):
    output = Path(output).resolve()
    protocol = read(output / 'protocol.json'); c = protocol['contract']
    assert digest(c) == protocol['experiment_signature']
    assert (c['lambda'], c['alpha']) == (.001, 1.)
    assert c['lambda_reference_paths'] == {key: str(value) for key, value in PRIORS.items()}
    gold = indexed(lines(output / 'inputs/gold.jsonl'), 'id')
    retrieval = indexed(lines(output / 'retrieval_results.jsonl'), 'question_id')
    requests = indexed(lines(output / 'inputs/requests.jsonl'), 'record_id')
    assert set(retrieval) == set(gold) and len(gold) == 371 and len(requests) == 742
    controls, references = {}, {}
    for label, prior in PRIORS.items():
        p = read(prior / 'protocol.json'); old = p['contract']
        assert digest(old) == p['experiment_signature'] and old['lambda'] == float(label) and old['alpha'] == 1.
        for key in ('memory_path', 'caption_model', 'model', 'conditions', 'questions', 'domains',
                    'top_k', 'base_url', 'api_config', 'temperature', 'seed', 'reasoning_effort',
                    'max_completion_tokens', 'image_detail', 'reader_system_prompt'):
            assert old[key] == c[key], (label, key)
        audit = read(prior / 'independent_audit.json')
        assert audit['passed']
        assert tuple(audit['overall'][k] for k in ('baseline_correct', 'expanded_correct')) == EXPECTED[label]
        for filename, expected in audit['artifact_sha256'].items(): assert sha(prior / filename) == expected
        for name in REFERENCE_FILES:
            assert sha(prior / name) == c['source_sha256'][str(prior / name)]
        old_gold = indexed(lines(prior / 'inputs/gold.jsonl'), 'id')
        old_ret = indexed(lines(prior / 'retrieval_results.jsonl'), 'question_id')
        old_requests = indexed(lines(prior / 'inputs/requests.jsonl'), 'record_id')
        assert old_gold == gold and set(old_ret) == set(retrieval) and set(old_requests) == set(requests)
        group_count = leaf_count = 0
        changed = dict.fromkeys(NAMES, 0)
        for qid in gold:
            a, b = old_ret[qid], retrieval[qid]
            assert a['task'] == b['task'] == gold[qid]['task']
            aa, bb = indexed(a['conditions'], 'name'), indexed(b['conditions'], 'name')
            assert set(aa) == set(bb) == set(NAMES)
            for name in NAMES:
                ca, cb = aa[name], bb[name]
                assert len(ca['groups']) == len(cb['groups']) and len(ca['leaves']) == len(cb['leaves'])
                for ga, gb in zip(ca['groups'], cb['groups']):
                    for k in ('group_id', 'size', 'query', 'selected_choice', 'selected_statement', 'raw_std', 'raw_status'):
                        assert ga[k] == gb[k], (label, qid, name, k)
                    group_count += 1
                for la, lb in zip(ca['leaves'], cb['leaves']):
                    for k in ('memory_id', 'leaf_index', 'group_id', 'raw'):
                        assert la[k] == lb[k], (label, qid, name, k)
                    leaf_count += 1
                changed[name] += ca['top5_memory_ids'] != cb['top5_memory_ids']
        same = {name: sum(old_requests[rid]['payload_sha256'] == r['payload_sha256']
                         for rid, r in requests.items() if r['condition'] == name) for name in NAMES}
        controls[label] = {'passed': True, 'prior_lambda': float(label), 'current_lambda': .001, 'alpha': 1.,
                           'questions': 371, 'raw_scores_exactly_equal': leaf_count, 'query_and_group_checks': group_count,
                           'all_other_model_and_reader_settings_equal': True, 'ordered_top5_changed_questions': changed,
                           'identical_reader_payload_questions': same, 'prior_result_artifacts_unchanged': True,
                           'prior_reference': str(prior)}
        references[label] = {'requests': old_requests, 'audit': audit}
    proof = {'passed': True, 'current_lambda': .001, 'alpha': 1., 'caption_model': c['caption_model'],
             'reader_model': c['model'], 'memory_path': c['memory_path'], 'comparisons': controls}
    dump(output / 'lambda_control_validation.json', proof)
    if inputs_only:
        print(json.dumps({'stage': 'lambda_controls_verified', **proof}), flush=True)
        return proof
    audit = read(output / 'independent_audit.json')
    assert audit['passed']
    for filename, expected in audit['artifact_sha256'].items(): assert sha(output / filename) == expected
    answers = indexed(lines(output / 'responses.jsonl'), 'record_id')
    assert set(answers) == set(requests)
    for label, prior in PRIORS.items():
        references[label]['answers'] = indexed(lines(prior / 'responses.jsonl'), 'record_id')
        assert set(references[label]['answers']) == set(requests)
    pairs = []
    for qid, g in gold.items():
        r = {'question_id': qid, 'domain': g['task'], 'gold_choice': g['gold_choice']}
        for name in NAMES:
            rid = qid + '::' + name
            pred = choice(answers[rid]['raw_response'])
            r.update({name + '_lambda0p001_answer': pred, name + '_lambda0p001_correct': pred == g['gold_choice'],
                      name + '_lambda0p001_top5': requests[rid]['top_memory_ids']})
            for label, ref in references.items():
                tag = 'lambda' + label.replace('.', 'p')
                oldpred = choice(ref['answers'][rid]['raw_response'])
                r.update({name + '_' + tag + '_answer': oldpred, name + '_' + tag + '_correct': oldpred == g['gold_choice'],
                          name + '_' + tag + '_top5': ref['requests'][rid]['top_memory_ids'],
                          name + '_same_payload_as_' + tag: requests[rid]['payload_sha256'] == ref['requests'][rid]['payload_sha256']})
        pairs.append(r)
    metrics, results = {}, {label: {} for label in ('0.001', '0.01', '0.1')}
    for name, field in zip(NAMES, ('baseline_correct', 'expanded_correct')):
        current_count = sum(r[name + '_lambda0p001_correct'] for r in pairs)
        assert current_count == audit['overall'][field]
        results['0.001'][name] = {'correct': current_count, 'accuracy': current_count / 371}
        for label, ref in references.items():
            tag = 'lambda' + label.replace('.', 'p')
            old_count = sum(r[name + '_' + tag + '_correct'] for r in pairs)
            assert old_count == ref['audit']['overall'][field]
            results[label][name] = {'correct': old_count, 'accuracy': old_count / 371}
            wins = sum(not r[name + '_' + tag + '_correct'] and r[name + '_lambda0p001_correct'] for r in pairs)
            losses = sum(r[name + '_' + tag + '_correct'] and not r[name + '_lambda0p001_correct'] for r in pairs)
            d = wins + losses
            metrics.setdefault(label, {})[name] = {'reference_lambda': float(label), 'current_lambda': .001,
                'reference_correct': old_count, 'current_correct': current_count, 'net_correct': current_count - old_count,
                'delta_percentage_points': 100 * (current_count - old_count) / 371,
                'wrong_to_right': wins, 'right_to_wrong': losses,
                'mcnemar_exact_two_sided_p': min(1., 2 * sum(math.comb(d, i) for i in range(min(wins, losses) + 1)) / 2 ** d) if d else 1.,
                'identical_payload_questions': controls[label]['identical_reader_payload_questions'][name],
                'identical_payload_answer_disagreements': [r['question_id'] for r in pairs if r[name + '_same_payload_as_' + tag] and r[name + '_' + tag + '_answer'] != r[name + '_lambda0p001_answer']]}
    summary = {'status': 'complete', 'alpha': 1., 'current_lambda': .001, 'questions': 371, 'results': results,
               'comparisons': metrics, 'controls': proof,
               'caveat': 'Lambda=0.01 and 0.1 are historical runs. Identical model inputs may yield different API responses; differences are not wholly attributable to lambda.'}
    dump(output / 'lambda_comparison.json', summary)
    with (output / 'lambda_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>GPT caption λ 消融</title>',
            '<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap}</style>',
            '<h1>GPT-5.4 caption＋GPT-5.4 回答：α=1，λ=0.001 / 0.01 / 0.1</h1>',
            '<p>同一套 GPT caption、记忆簇、查询、原始匹配分数及回答设置；MemEye 全部 371 题。本次 λ=0.001 全部重新回答，其余两组是已验证历史结果。</p>',
            '<table><tr><th>方案</th><th>λ=0.001（本次）</th><th>λ=0.01</th><th>λ=0.1</th></tr>']
    for name in NAMES:
        body.append('<tr><td>' + name + '</td>' + ''.join(f'<td>{results[label][name]["correct"]}/371 ({results[label][name]["accuracy"]:.4%})</td>' for label in ('0.001', '0.01', '0.1')) + '</tr>')
    body += ['</table><p>相同输入也有跨轮 API 波动，不能把全部差异归因于 λ；配对检验未作多重比较校正。</p>',
             '<pre>' + html.escape(json.dumps(metrics, ensure_ascii=False, indent=2)) + '</pre>',
             '<p><a href="report.html">本轮详情与五项方法标注</a> · <a href="lambda_comparison.csv">三个 λ 逐题对比</a> · <a href="lambda_comparison.json">统计</a> · <a href="independent_audit.json">本轮核验</a> · <a href="lambda_control_validation.json">变量控制核验</a></p></html>']
    (output / 'lambda_comparison.html').write_text('\n'.join(body))
    dump(output / 'lambda_comparison_audit.json', {'passed': True, 'questions': 371,
         'independent_raw_answer_recount': True, 'controls_passed': True,
         'artifact_sha256': {n: sha(output / n) for n in ('lambda_comparison.json', 'lambda_comparison.csv', 'lambda_comparison.html', 'lambda_control_validation.json')},
         'results': results, 'comparisons': metrics})
    print(json.dumps({'stage': 'lambda_comparison_complete', 'results': results, 'comparisons': metrics}), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=CURRENT)
    p.add_argument('--inputs-only', action='store_true')
    args = p.parse_args(); compare(args.output, args.inputs_only)
