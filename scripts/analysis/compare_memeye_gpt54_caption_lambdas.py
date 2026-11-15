"""Read-only source/control checks and paired statistics for GPT-caption lambdas.

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
CURRENT = ROOT / 'results/memeye_gpt54_caption_zscore_lambda0p1_alpha1_20261024'
NAMES = ('baseline_question_only_zscore', 'master_answer_query_zscore')
REFERENCE_FILES = ('protocol.json', 'retrieval_protocol.json', 'retrieval_results.jsonl',
                   'inputs/gold.jsonl', 'inputs/requests.jsonl', 'api_request_manifest.jsonl',
                   'responses.jsonl', 'fresh_responses.jsonl', 'summary.json', 'independent_audit.json')


def read(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def indexed(rows, key):
    d = {r[key]: r for r in rows}
    assert len(d) == len(rows)
    return d
def choice(s):
    m = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', s.strip(), re.I)
    return m.group(1).upper() if m else None
def dump(p, value): Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def compare(output=CURRENT, prior=PRIOR, inputs_only=False):
    output, prior = Path(output).resolve(), Path(prior).resolve()
    oldp, newp = read(prior / 'protocol.json'), read(output / 'protocol.json')
    old, new = oldp['contract'], newp['contract']
    assert digest(old) == oldp['experiment_signature'] and digest(new) == newp['experiment_signature']
    assert (old['lambda'], new['lambda'], old['alpha'], new['alpha']) == (.01, .1, 1., 1.)
    for key in ('memory_path', 'caption_model', 'model', 'conditions', 'questions', 'domains',
                'top_k', 'base_url', 'api_config', 'temperature', 'seed', 'reasoning_effort',
                'max_completion_tokens', 'image_detail', 'reader_system_prompt'):
        assert old[key] == new[key], key
    assert new['prior_lambda_reference'] == str(prior)
    oa = read(prior / 'independent_audit.json')
    assert oa['passed'] and oa['overall']['baseline_correct'] == 242 and oa['overall']['expanded_correct'] == 237
    for filename, expected in oa['artifact_sha256'].items(): assert sha(prior / filename) == expected
    for name in REFERENCE_FILES:
        assert sha(prior / name) == new['source_sha256'][str(prior / name)]
    old_gold, gold = indexed(lines(prior / 'inputs/gold.jsonl'), 'id'), indexed(lines(output / 'inputs/gold.jsonl'), 'id')
    assert old_gold == gold and len(gold) == 371
    old_ret = indexed(lines(prior / 'retrieval_results.jsonl'), 'question_id')
    new_ret = indexed(lines(output / 'retrieval_results.jsonl'), 'question_id')
    assert set(old_ret) == set(new_ret) == set(gold)
    group_count = leaf_count = 0
    retrieval_changes = dict.fromkeys(NAMES, 0)
    for qid in gold:
        a, b = old_ret[qid], new_ret[qid]
        assert a['task'] == b['task'] == gold[qid]['task']
        aa, bb = indexed(a['conditions'], 'name'), indexed(b['conditions'], 'name')
        assert set(aa) == set(bb) == set(NAMES)
        for name in NAMES:
            ca, cb = aa[name], bb[name]
            assert len(ca['groups']) == len(cb['groups']) and len(ca['leaves']) == len(cb['leaves'])
            for ga, gb in zip(ca['groups'], cb['groups']):
                for k in ('group_id', 'size', 'query', 'selected_choice', 'selected_statement', 'raw_std', 'raw_status'):
                    assert ga[k] == gb[k], (qid, name, k)
                group_count += 1
            for la, lb in zip(ca['leaves'], cb['leaves']):
                for k in ('memory_id', 'leaf_index', 'group_id', 'raw'):
                    assert la[k] == lb[k], (qid, name, k)
                leaf_count += 1
            retrieval_changes[name] += ca['top5_memory_ids'] != cb['top5_memory_ids']
    old_requests = indexed(lines(prior / 'inputs/requests.jsonl'), 'record_id')
    requests = indexed(lines(output / 'inputs/requests.jsonl'), 'record_id')
    assert set(old_requests) == set(requests) and len(requests) == 742
    same = {name: sum(old_requests[rid]['payload_sha256'] == r['payload_sha256']
                     for rid, r in requests.items() if r['condition'] == name) for name in NAMES}
    checks = {'passed': True, 'prior_lambda': .01, 'current_lambda': .1, 'alpha': 1.,
              'caption_model': new['caption_model'], 'reader_model': new['model'], 'memory_path': new['memory_path'],
              'questions': 371, 'raw_scores_exactly_equal': leaf_count, 'query_and_group_checks': group_count,
              'all_other_model_and_reader_settings_equal': True, 'ordered_top5_changed_questions': retrieval_changes,
              'identical_reader_payload_questions': same, 'prior_result_artifacts_unchanged': True,
              'prior_reference': str(prior)}
    dump(output / 'lambda_control_validation.json', checks)
    if inputs_only:
        print(json.dumps({'stage': 'lambda_controls_verified', **checks}), flush=True)
        return checks
    audit = read(output / 'independent_audit.json')
    assert audit['passed']
    for filename, expected in audit['artifact_sha256'].items(): assert sha(output / filename) == expected
    old_answers = indexed(lines(prior / 'responses.jsonl'), 'record_id')
    answers = indexed(lines(output / 'responses.jsonl'), 'record_id')
    assert set(answers) == set(old_answers) == set(requests)
    pairs = []
    for qid, g in gold.items():
        r = {'question_id': qid, 'domain': g['task'], 'gold_choice': g['gold_choice']}
        for name in NAMES:
            rid = qid + '::' + name
            a, b = choice(old_answers[rid]['raw_response']), choice(answers[rid]['raw_response'])
            r.update({name + '_lambda0p01_answer': a, name + '_lambda0p1_answer': b,
                      name + '_lambda0p01_correct': a == g['gold_choice'], name + '_lambda0p1_correct': b == g['gold_choice'],
                      name + '_identical_payload': old_requests[rid]['payload_sha256'] == requests[rid]['payload_sha256'],
                      name + '_old_top5': old_requests[rid]['top_memory_ids'], name + '_new_top5': requests[rid]['top_memory_ids']})
        pairs.append(r)
    metrics = {}
    for name, field in zip(NAMES, ('baseline_correct', 'expanded_correct')):
        old_count = sum(r[name + '_lambda0p01_correct'] for r in pairs)
        new_count = sum(r[name + '_lambda0p1_correct'] for r in pairs)
        assert old_count == oa['overall'][field] and new_count == audit['overall'][field]
        wins = sum(not r[name + '_lambda0p01_correct'] and r[name + '_lambda0p1_correct'] for r in pairs)
        losses = sum(r[name + '_lambda0p01_correct'] and not r[name + '_lambda0p1_correct'] for r in pairs)
        d = wins + losses
        metrics[name] = {'lambda0p01_correct': old_count, 'lambda0p1_correct': new_count,
                         'lambda0p01_accuracy': old_count / 371, 'lambda0p1_accuracy': new_count / 371,
                         'net_correct': new_count - old_count, 'delta_percentage_points': 100 * (new_count - old_count) / 371,
                         'wrong_to_right': wins, 'right_to_wrong': losses,
                         'mcnemar_exact_two_sided_p': min(1., 2 * sum(math.comb(d, i) for i in range(min(wins, losses) + 1)) / 2 ** d) if d else 1.,
                         'identical_payload_questions': same[name],
                         'identical_payload_answer_disagreements': [r['question_id'] for r in pairs if r[name + '_identical_payload'] and r[name + '_lambda0p01_answer'] != r[name + '_lambda0p1_answer']]}
    summary = {'status': 'complete', 'controls': checks, 'conditions': metrics,
               'caveat': 'Prior lambda=0.01 is a historical run. Identical model inputs may produce different API responses; differences are not wholly attributable to lambda.'}
    dump(output / 'lambda_comparison.json', summary)
    with (output / 'lambda_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>GPT caption λ 消融</title>',
            '<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap}</style>',
            '<h1>GPT-5.4 caption＋GPT-5.4 回答：α=1，λ=0.01 / 0.1</h1>',
            '<p>同一套 GPT caption、记忆簇、查询、原始匹配分数及回答设置；MemEye 全部 371 题。λ=0.01 为此前已验证结果，λ=0.1 全部重新回答。</p>',
            '<table><tr><th>方案</th><th>λ=0.01</th><th>λ=0.1</th><th>λ=0.1 准确率</th><th>变化</th></tr>']
    for name, v in metrics.items():
        body.append(f'<tr><td>{name}</td><td>{v["lambda0p01_correct"]}/371</td><td>{v["lambda0p1_correct"]}/371</td><td>{v["lambda0p1_accuracy"]:.4%}</td><td>{v["net_correct"]:+d}</td></tr>')
    body += ['</table><p>相同输入也可能出现跨轮 API 回答波动，不能将全部差异归因于 λ；逐题变化与配对统计如下。</p>',
             '<pre>' + html.escape(json.dumps(metrics, ensure_ascii=False, indent=2)) + '</pre>',
             '<p><a href="report.html">本轮详情与五项方法标注</a> · <a href="lambda_comparison.csv">λ 逐题对比</a> · <a href="lambda_comparison.json">对比统计</a> · <a href="independent_audit.json">本轮核验</a> · <a href="lambda_control_validation.json">变量控制核验</a></p></html>']
    (output / 'lambda_comparison.html').write_text('\n'.join(body))
    dump(output / 'lambda_comparison_audit.json', {'passed': True, 'questions': 371,
         'independent_raw_answer_recount': True, 'controls_passed': True,
         'artifact_sha256': {n: sha(output / n) for n in ('lambda_comparison.json', 'lambda_comparison.csv', 'lambda_comparison.html', 'lambda_control_validation.json')},
         'conditions': metrics})
    print(json.dumps({'stage': 'lambda_comparison_complete', 'conditions': metrics}), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=CURRENT)
    p.add_argument('--prior', type=Path, default=PRIOR)
    p.add_argument('--inputs-only', action='store_true')
    args = p.parse_args(); compare(args.output, args.prior, args.inputs_only)
