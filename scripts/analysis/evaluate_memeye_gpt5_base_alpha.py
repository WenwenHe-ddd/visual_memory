"""GPT-5 alpha ablation at lambda=.01; frozen memory, six jointly answered conditions."""
from __future__ import annotations
import argparse
import copy
import csv
import html
import itertools
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memeye_gpt5_reader as api
from scripts.analysis import evaluate_memeye_gpt5_base as previous
from scripts.analysis.retrieve_memeye_gpt5_base_alpha import DEFAULT_OUTPUT, ALPHAS, BASE_NAMES, SPECS, NAMES, verify_source

read, lines, sha, digest, dump = api.read, api.lines, api.sha, api.digest, api.dump
indexed, write_lines, verify = api.indexed, api.write_lines, api.verify
SOURCE = previous.BASE
run = previous.run


def prepare(output=DEFAULT_OUTPUT, source=SOURCE, workers=3):
    output, source = Path(output).resolve(), Path(source).resolve()
    out = output / 'reader'
    if (out / 'protocol.json').exists():
        p = verify(out)
        assert p['contract']['memory_path'] == str(source)
        assert p['contract']['condition_specs'] == SPECS and p['contract']['lambda'] == .01
        return p
    proof = verify_source(source)
    pc = verify(source / 'reader')['contract']
    retrieval = read(output / 'retrieval_protocol.json')
    assert (retrieval['lambda'], retrieval['alphas'], retrieval['memory_path']) == (.01, list(ALPHAS), str(source))
    for path, value in {**retrieval['source_sha256'], **retrieval['output_sha256']}.items():
        assert sha(path) == value, path
    qs = indexed(lines(source / 'inputs/source_questions.jsonl'), 'id')
    gold = lines(source / 'reader/inputs/gold.jsonl')
    assert {r['id'] for r in gold} == set(qs)
    memories = {m['id']: m['retrieval_text'] for m in lines(source / 'offline/memories.jsonl')}
    records = []
    for i, row in enumerate(lines(output / 'retrieval_results.jsonl')):
        q = qs[row['question_id']]
        conditions = indexed(row['conditions'], 'name')
        assert set(conditions) == set(NAMES)
        batch = []
        for name in NAMES:
            ids = conditions[name]['top5_memory_ids']
            assert len(ids) == len(set(ids)) == 5
            text = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
            text += '\n\n'.join(f'Memory {j}: {memories[mid]}' for j, mid in enumerate(ids, 1))
            text += '\n\nQuestion:\n' + q['question'] + '\n\nOptions:\n' + '\n'.join(k + '. ' + q['options'][k] for k in 'ABCD') + '\n\nAnswer with one letter only.'
            payload = {'system': pc['reader_system_prompt'], 'user_text': text, 'query_images': q['query_images']}
            batch.append({'record_id': q['id'] + '::' + name, 'id': q['id'], 'task': q['task'], 'condition': name,
                          'reader_text': text, 'query_images': q['query_images'], 'query_image_paths': [im['path'] for im in q['query_images']],
                          'top_memory_ids': ids, 'payload_sha256': digest(payload)})
        offset = i % len(NAMES)
        records.extend(batch[offset:] + batch[:offset])
    assert len(records) == 2226 and len(gold) == 371
    (out / 'inputs').mkdir(parents=True, exist_ok=True)
    write_lines(out / 'inputs/requests.jsonl', records)
    write_lines(out / 'inputs/gold.jsonl', gold)
    sources = {**proof['source_sha256'], **retrieval['source_sha256'], **retrieval['output_sha256']}
    paths = [Path(__file__), ROOT / 'scripts/analysis/audit_memeye_gpt5_base_alpha.py',
             ROOT / 'scripts/tests/test_memeye_gpt5_base_alpha.py', ROOT / 'scripts/analysis/retrieve_memeye_gpt5_base_alpha.py',
             ROOT / 'scripts/analysis/evaluate_memeye_gpt5_base.py', ROOT / 'scripts/analysis/audit_memeye_gpt5_base.py',
             output / 'retrieval_protocol.json', out / 'inputs/requests.jsonl', out / 'inputs/gold.jsonl']
    paths.extend(Path(im['path']) for q in qs.values() for im in q['query_images'])
    sources.update({str(p): sha(p) for p in paths})
    keys = ('model', 'requested_model', 'expected_returned_model', 'caption_model', 'seed', 'caption_seed', 'temperature',
            'temperature_wire_policy', 'reasoning_effort', 'max_completion_tokens', 'image_detail', 'base_url', 'api_config',
            'api_env_file', 'reader_system_prompt', 'timeout_seconds', 'max_attempts')
    contract = {k: copy.deepcopy(pc[k]) for k in keys}
    annotations = {}
    for name, spec in SPECS.items():
        m = copy.deepcopy(pc['method_annotations'][spec['base']])
        m['1. 候选记忆结构'].update(caption_source='沿用已核验的同一批 GPT-5 caption，未重新生成', text='原始对话＋同一批 GPT-5 完整 caption')
        m['1. 候选记忆结构']['weighting_group'] = m['1. 候选记忆结构']['weighting_group'].replace('新 master', '已有 master')
        m['2. 查询内容'] = m['2. 查询内容'].replace('本轮 GPT-5', '原有 GPT-5')
        m['3. 检索方式'].update(alpha=spec['alpha'])
        m['4. 编码方式']['CLIP'] = '沿用已有视觉编码器、外部基底、master 和视觉聚合量；仅重新编码检索文本'
        m['5. 回答方式']['same_run_identical_requests'] = '六种条件中完整请求相同则共享本轮新回答；不复用历史回答'
        annotations[name] = m
    contract.update(questions=371, logical_records=2226, unique_payloads=len({r['payload_sha256'] for r in records}),
                    conditions=list(NAMES), condition_specs=SPECS, memory_path=str(source), alpha_reference_path=str(source),
                    source_run_id=read(source / 'protocol.json')['run_id'], alphas=list(ALPHAS), reference_alpha=1., top_k=5,
                    workers=workers, source_sha256=sources, method_annotations=annotations, historical_api_answer_reuse=False,
                    gold_in_api_payload=False, caption_regenerated=False, memory_rebuilt=False,
                    within_run_sharing='One fresh answer per complete request across six conditions; no historical answer reuse.',
                    interpretation='Only alpha changes at fixed lambda=.01. Frozen GPT-5 captions, memory, queries, raw/white/z scores and reader settings. Alpha=1 and unweighted controls are historical; identical-request cross-run differences are reported separately.')
    contract['lambda'] = .01
    manifest = [{'record_id': r['record_id'], 'payload_sha256': r['payload_sha256'],
                 'request_sha256': digest(previous.request_body({'contract': contract}, r)),
                 'query_image_count': len(r['query_images']), 'requested_model': contract['model']} for r in records]
    write_lines(out / 'api_request_manifest.jsonl', manifest)
    contract['request_manifest_sha256'] = digest(manifest)
    p = {'run_id': str(uuid.uuid4()), 'experiment_signature': digest(contract), 'contract': contract,
         'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    dump(out / 'protocol.json', p)
    verify(out)
    print(json.dumps({'stage': 'prepared', 'records': len(records), 'unique_requests': contract['unique_payloads']}), flush=True)
    return p


def comparison_with_requests(pairs, before, after, flag):
    metric = previous.compare(pairs, before, after)
    same = [r for r in pairs if r[flag]]
    changed = [r for r in pairs if not r[flag]]
    metric.update(identical_request_questions=len(same), changed_request_questions=len(changed),
                  identical_request_answer_disagreements=[r['question_id'] for r in same if r[before + '_answer'] != r[after + '_answer']],
                  identical_request_net_correct=sum(r[after + '_correct'] - r[before + '_correct'] for r in same),
                  changed_request_net_correct=sum(r[after + '_correct'] - r[before + '_correct'] for r in changed))
    return metric


def analyze(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    out = output / 'reader'
    c = verify(out)['contract']
    source = Path(c['memory_path'])
    validation = read(out / 'validation_output.json')
    assert validation['passed']
    for n in ('responses', 'fresh_responses'):
        assert sha(out / (n + '.jsonl')) == validation[n + '_sha256']
    requests = indexed(lines(out / 'inputs/requests.jsonl'), 'record_id')
    answers = indexed(lines(out / 'responses.jsonl'), 'record_id')
    old_requests = indexed(lines(source / 'reader/inputs/requests.jsonl'), 'record_id')
    old_answers = indexed(lines(source / 'reader/responses.jsonl'), 'record_id')
    manifests = indexed(lines(out / 'api_request_manifest.jsonl'), 'record_id')
    old_manifests = indexed(lines(source / 'reader/api_request_manifest.jsonl'), 'record_id')
    gold = lines(out / 'inputs/gold.jsonl')
    assert len(gold) == 371 and set(requests) == set(answers) and len(answers) == 2226
    pairs = []
    for g in gold:
        qid = g['id']
        row = {'question_id': qid, 'domain': g['task'], 'gold_choice': g['gold_choice']}
        for name, spec in SPECS.items():
            rid = qid + '::' + name
            old_id = qid + '::' + spec['base']
            raw_id = qid + '::' + spec['base'].removesuffix('_zscore')
            current = api.parse_choice(answers[rid]['raw_response'])
            old = api.parse_choice(old_answers[old_id]['raw_response'])
            raw = api.parse_choice(old_answers[raw_id]['raw_response'])
            ids = requests[rid]['top_memory_ids']
            row.update({name + '_answer': current, name + '_correct': current == g['gold_choice'],
                        name + '_reference_answer': old, name + '_reference_correct': old == g['gold_choice'],
                        name + '_unweighted_answer': raw, name + '_unweighted_correct': raw == g['gold_choice'],
                        name + '_top5': ids, name + '_gold_hits': len(set(ids) & set(g['gold_ids'])),
                        name + '_same_request': manifests[rid]['request_sha256'] == old_manifests[old_id]['request_sha256'],
                        name + '_same_unweighted_request': manifests[rid]['request_sha256'] == old_manifests[raw_id]['request_sha256']})
            assert row[name + '_same_request'] == (requests[rid]['payload_sha256'] == old_requests[old_id]['payload_sha256'])
        for base in BASE_NAMES:
            for a, b in itertools.combinations([n for n in NAMES if SPECS[n]['base'] == base], 2):
                same = manifests[qid + '::' + a]['request_sha256'] == manifests[qid + '::' + b]['request_sha256']
                row[a + '__' + b + '_same_request'] = same
                if same:
                    assert row[a + '_answer'] == row[b + '_answer']
        pairs.append(row)
    overall = {n: {'correct': sum(r[n + '_correct'] for r in pairs), 'questions': 371,
                   'accuracy': sum(r[n + '_correct'] for r in pairs) / 371,
                   'gold_hits_top5': sum(r[n + '_gold_hits'] for r in pairs), 'gold_references': 1041,
                   'questions_with_any_gold': sum(r[n + '_gold_hits'] > 0 for r in pairs)} for n in NAMES}
    comparisons = {n: comparison_with_requests(pairs, n + '_reference', n, n + '_same_request') for n in NAMES}
    unweighted = {n: comparison_with_requests(pairs, n + '_unweighted', n, n + '_same_unweighted_request') for n in NAMES}
    within = {a + '__' + b: comparison_with_requests(pairs, a, b, a + '__' + b + '_same_request')
              for base in BASE_NAMES for a, b in itertools.combinations([n for n in NAMES if SPECS[n]['base'] == base], 2)}
    by_domain = {d: {n: {'correct': sum(r[n + '_correct'] for r in pairs if r['domain'] == d),
                        'questions': sum(r['domain'] == d for r in pairs)} for n in NAMES} for d in sorted({r['domain'] for r in pairs})}
    methods = copy.deepcopy(c['method_annotations'])
    for n in NAMES:
        methods[n]['5. 回答方式'].update(correct=overall[n]['correct'], returned_models=validation['returned_models'])
    fresh = lines(out / 'fresh_responses.jsonl')
    historical = read(source / 'summary.json')['overall']
    summary = {'status': 'complete', 'requested_model': c['model'], 'returned_model': c['expected_returned_model'],
               'caption_model': c['caption_model'], 'reader_seed': 42, 'lambda': .01, 'alphas': list(ALPHAS), 'reference_alpha': 1.,
               'memory_path': str(source), 'condition_specs': SPECS, 'overall': overall, 'alpha1_comparisons': comparisons,
               'unweighted_comparisons': unweighted, 'within_run_alpha_comparisons': within, 'historical_references': historical,
               'by_domain': by_domain, 'method_annotations': methods, 'validation': validation, 'interpretation': c['interpretation'],
               'usage': {k: sum((r.get('usage') or {}).get(k, 0) or 0 for r in fresh) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}}
    dump(output / 'summary.json', summary)
    dump(output / 'paired_results.json', pairs)
    with (output / 'paired_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    table = []
    for base in BASE_NAMES:
        for label, metrics, origin in [('未赋权', historical[base.removesuffix('_zscore')], '历史参照')] + [
                (str(alpha), historical[base] if alpha == 1. else overall[next(n for n in NAMES if SPECS[n] == {'base': base, 'alpha': alpha})],
                 '历史参照' if alpha == 1. else '本轮新回答') for alpha in (.01, .1, 1., 10.)]:
            table.append({'method': base, 'alpha': label, 'source': origin, **metrics})
    dump(output / 'alpha_comparison.json', table)
    with (output / 'alpha_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>GPT-5 α 消融</title>',
            '<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap}</style>',
            '<h1>GPT-5：固定 λ=0.01 的 α 消融</h1><p>同一批 GPT-5 caption、438 条记忆、72 个 master，371 题。α=0.01、0.1、10 新评测；α=1 与未赋权为历史参照。回答 seed=42、minimal、不传 temperature、2048 token 总预算。六种新条件中的相同请求共享本轮新回答。</p>',
            '<table><tr><th>方案</th><th>α</th><th>来源</th><th>答对</th><th>准确率</th><th>目标记忆命中</th></tr>']
    for row in table:
        body.append(f'<tr><td>{row["method"]}</td><td>{row["alpha"]}</td><td>{row["source"]}</td><td>{row["correct"]}/371</td><td>{row["accuracy"]:.4%}</td><td>{row["gold_hits_top5"]}/1041</td></tr>')
    body += ['</table><h2>与历史 α=1 的逐题变化</h2><p>相同请求跨轮也可能改变答案，单独记录；以下是同一测试集上的单轮参数比较，不能作为独立验证的最优参数结论。</p>',
             '<pre>' + html.escape(json.dumps(comparisons, ensure_ascii=False, indent=2)) + '</pre>',
             '<h2>本轮三个 α 的配对比较</h2><pre>' + html.escape(json.dumps(within, ensure_ascii=False, indent=2)) + '</pre>',
             '<h2>分 domain</h2><pre>' + html.escape(json.dumps(by_domain, ensure_ascii=False, indent=2)) + '</pre>',
             '<h2>核验</h2><pre>' + html.escape(json.dumps(validation, ensure_ascii=False, indent=2)) + '</pre>']
    for n in NAMES:
        body.append('<details><summary>' + n + '</summary><pre>' + html.escape(json.dumps(methods[n], ensure_ascii=False, indent=2)) + '</pre></details>')
    body.append('<p><a href="summary.json">完整统计</a> · <a href="alpha_comparison.csv">α 对照表</a> · <a href="paired_results.csv">逐题 CSV</a> · <a href="independent_audit.json">独立核验</a></p></html>')
    (output / 'report.html').write_text('\n'.join(body))
    print(json.dumps({'stage': 'analyzed', 'overall': overall}), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--memory', type=Path, default=SOURCE)
    p.add_argument('--stage', choices=['prepare', 'smoke', 'run', 'analyze', 'all'], default='all')
    p.add_argument('--api-workers', type=int, default=3)
    a = p.parse_args()
    assert a.api_workers > 0
    from scripts.analysis.audit_memeye_gpt5_base_alpha import audit
    if a.stage in ('prepare', 'all'):
        prepare(a.output, a.memory, a.api_workers)
    if a.stage in ('prepare', 'smoke', 'run', 'all'):
        audit(a.output, inputs_only=True)
    if a.stage in ('smoke', 'all'):
        run(a.output / 'reader', smoke=True)
    if a.stage in ('run', 'all'):
        run(a.output / 'reader')
    if a.stage in ('analyze', 'all'):
        analyze(a.output)
        verification = audit(a.output)
        dump(a.output / 'execution_status.json', {'status': 'complete', 'independent_audit_passed': verification['passed'], 'overall': verification['overall']})


if __name__ == '__main__':
    main()
