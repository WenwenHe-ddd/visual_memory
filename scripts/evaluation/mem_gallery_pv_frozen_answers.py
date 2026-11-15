"""PV lambda ablation anchored to the original 778/798 saved reader answers.

Only previously unseen complete reader payloads invoke Qwen. Captions, graphs,
queries and audited PV rankings are frozen. Existing signed runs are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import mem_gallery_positive_variance_ablation as pv
from scripts.evaluation import mem_gallery_master_reader as reader

ORIGINAL = ROOT / 'results/mem_gallery_master_answer_20261021'
RETRIEVAL = ROOT / 'results/memgallery_qwen_positive_variance_lambda_ablation_20261107'
DEFAULT_OUTPUT = ROOT / 'results/memgallery_qwen_pv_original_answers_lambda_ablation_20261107'
AUDITOR = ROOT / 'scripts/analysis/audit_memgallery_pv_frozen_answers.py'
read, lines, dump, sha = pv.read, pv.lines, pv.dump, pv.sha
METHODS, SPECS, LAMBDAS = pv.METHODS, pv.SPECS, pv.LAMBDAS
ANSWER_FIELDS = ('raw_response', 'input_tokens', 'generated_tokens', 'hit_token_limit',
                 'batch_elapsed_seconds', 'batch_size', 'execution_id')


def write_lines(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    temporary.replace(path)


def copy_identical(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(source) == sha(target), target
    else:
        shutil.copyfile(source, target)


def original_cache():
    protocol = read(ORIGINAL / 'reader/protocol.json')
    assert reader.digest(protocol['contract']) == protocol['experiment_signature']
    pv.verify_hashes(protocol['contract']['source_sha256'])
    manifest = lines(ORIGINAL / 'reader/qa_manifest.jsonl')
    assert reader.digest(manifest) == protocol['contract']['manifest_sha256']
    responses = reader.by_id(lines(ORIGINAL / 'reader/responses.jsonl'), 'record_id')
    mindex = reader.by_id(manifest, 'record_id')
    assert set(responses) == set(mindex) and len(manifest) == 3422
    cache = {}
    for m in manifest:
        r = responses[m['record_id']]
        assert r['payload_sha256'] == m['payload_sha256']
        assert r['top5_memory_ids'] == m['top5_memory_ids']
        assert r['run_id'] == protocol['run_id']
        assert r['experiment_signature'] == protocol['experiment_signature']
        digest = m['payload_sha256']
        if digest in cache:
            assert cache[digest]['raw_response'] == r['raw_response']
        if digest not in cache or r['fresh_inference_executed_for_this_record']:
            cache[digest] = r
    assert len(cache) == 3072
    return protocol, mindex, responses, cache


def annotations():
    result = pv.annotations()
    for meta in result.values():
        meta['5. 回答方式'] += (' 完整请求与原778/798实验一致时复用原答案；仅新请求推理。'
            '跨参数相同请求共享答案；新请求按首次出现条件分别固定分批。')
    return result


def prepare(base, device):
    original_protocol, old_manifest, _, cache = original_cache()
    source = pv.verify(RETRIEVAL)
    base.mkdir(parents=True, exist_ok=True)
    (base / 'reader').mkdir(exist_ok=True)
    for name in ['questions.jsonl', 'memories.jsonl', 'gold.jsonl', 'protocol.json']:
        copy_identical(ORIGINAL / 'inputs' / name, base / 'inputs' / name)
    copy_identical(RETRIEVAL / 'retrieval_results.jsonl', base / 'retrieval_results.jsonl')
    # Read-only reuse; the independent auditor checks every diagnostic hash.
    diagnostics = base / 'diagnostics'
    if not diagnostics.exists():
        diagnostics.symlink_to(RETRIEVAL / 'diagnostics', target_is_directory=True)
    assert diagnostics.resolve() == (RETRIEVAL / 'diagnostics').resolve()
    p = copy.deepcopy(source)
    p.update(reader_answer_policy='Reuse original exact payload answers; infer only unseen payloads.',
             frozen_retrieval_source=str(RETRIEVAL), method_annotations=annotations())
    tracked = [Path(__file__), AUDITOR, RETRIEVAL / 'retrieval_protocol.json',
               ORIGINAL / 'reader/protocol.json', ORIGINAL / 'reader/qa_manifest.jsonl',
               ORIGINAL / 'reader/responses.jsonl']
    p['source_sha256'].update({str(f): sha(f) for f in tracked})
    p['source_sha256'].update({str(base / 'inputs' / n): sha(base / 'inputs' / n)
                              for n in ['questions.jsonl', 'memories.jsonl', 'gold.jsonl', 'protocol.json']})
    p['artifact_sha256'][str(base / 'retrieval_results.jsonl')] = sha(base / 'retrieval_results.jsonl')
    if (base / 'retrieval_protocol.json').exists():
        assert read(base / 'retrieval_protocol.json') == p
    else:
        dump(base / 'retrieval_protocol.json', p)

    reader.CONDITIONS = tuple(SPECS)
    manifest = reader.build_reader_manifest(lines(base / 'inputs/questions.jsonl'),
        lines(base / 'inputs/memories.jsonl'), lines(base / 'retrieval_results.jsonl'))
    mindex = reader.by_id(manifest, 'record_id')
    for rid, old in old_manifest.items():
        for key in ['payload_sha256', 'reader_text', 'query_images', 'top5_memory_ids']:
            assert old[key] == mindex[rid][key], (rid, key)
    plans, owners, assignment = [], {}, []
    # Freeze the entire plan before inference. Conditions do not share batches;
    # an identical new payload has one declared owner and one generated answer.
    for condition in SPECS:
        selected = []
        for m in manifest:
            digest = m['payload_sha256']
            if m['condition'] != condition or digest in cache or digest in owners:
                continue
            assert condition not in METHODS
            owners[digest] = m
            selected.append(m)
        for has_image in [False, True]:
            items = [m for m in selected if bool(m['query_image_paths']) == has_image]
            size = 4 if has_image else 16
            for start in range(0, len(items), size):
                batch = items[start:start + size]
                plans.append({'batch_index': len(plans), 'owner_condition': condition,
                              'has_image': has_image, 'record_ids': [m['record_id'] for m in batch],
                              'payloads': [m['payload_sha256'] for m in batch]})
    for m in manifest:
        digest = m['payload_sha256']
        source_answer = cache.get(digest)
        assignment.append({'record_id': m['record_id'], 'payload_sha256': digest,
            'origin': 'original' if source_answer else 'new',
            'source_record_id': source_answer['record_id'] if source_answer else owners[digest]['record_id'],
            'same_as_own_unweighted_input': digest == old_manifest[m['id'] + '::' + SPECS[m['condition']]['method']]['payload_sha256']})
    for filename, rows in [('qa_manifest.jsonl', manifest), ('reuse_manifest.jsonl', assignment)]:
        f = base / 'reader' / filename
        if f.exists(): assert lines(f) == rows
        else: write_lines(f, rows)
    plan = {'batches': plans, 'new_unique_payloads': len(owners), 'seed': 42,
            'batch_policy': 'First owning condition, source question order, no-image then image, batch16/4; persisted before inference.'}
    if (base / 'reader/inference_plan.json').exists():
        assert read(base / 'reader/inference_plan.json') == plan
    else:
        dump(base / 'reader/inference_plan.json', plan)
    old = original_protocol['contract']
    # Hash the actual local weights, rather than only copying their old metadata.
    assert reader.model_provenance() == old['model']
    contract = copy.deepcopy(old)
    contract.update(stage='original_answer_anchored_positive_variance_ablation',
        logical_conditions=len(manifest), unique_payloads=len({m['payload_sha256'] for m in manifest}),
        manifest_sha256=reader.digest(manifest), source_sha256={**p['source_sha256'], **p['artifact_sha256']},
        historical_response_reuse=True, historical_response_source=str(ORIGINAL / 'reader/responses.jsonl'),
        original_run_id=original_protocol['run_id'], condition_specs=SPECS, method_annotations=annotations(),
        device=device, inference_order=plan['batch_policy'],
        deduplication='Original complete payload cache takes precedence; every new payload generated exactly once.',
        parameter_policy='alpha=1; lambda=.001/.003/.01/.03/.1; original controls never regenerated.',
        new_unique_payloads=len(owners), batch_plan_sha256=sha(base / 'reader/inference_plan.json'),
        reuse_manifest_sha256=sha(base / 'reader/reuse_manifest.jsonl'))
    contract['source_sha256'][str(base / 'retrieval_protocol.json')] = sha(base / 'retrieval_protocol.json')
    protocol = reader.ensure_protocol(base / 'reader/protocol.json', contract)
    report = {'status': 'prepared', 'logical_records': len(manifest),
        'original_reused_records': sum(a['origin'] == 'original' for a in assignment),
        'new_logical_records': sum(a['origin'] == 'new' for a in assignment),
        'new_unique_payloads': len(owners), 'new_batches': len(plans),
        'control_records_reused': len(old_manifest), 'control_model_calls': 0, 'gold_loaded': False}
    dump(base / 'preparation.json', report)
    print(json.dumps(report), flush=True)
    return protocol, manifest, cache, plans, assignment


def run(base, device):
    protocol, manifest, cache, plans, assignment = prepare(base, device)
    mindex = reader.by_id(manifest, 'record_id')
    saved, fresh = {}, {}
    batch_root = base / 'reader/batches'
    batch_root.mkdir(exist_ok=True)
    for plan in plans:
        f = batch_root / f'{plan["batch_index"]:04d}.json'
        if f.exists():
            record = read(f)
            assert record['plan'] == plan and record['experiment_signature'] == protocol['experiment_signature']
            assert len(record['responses']) == len(plan['payloads'])
            for digest, r in zip(plan['payloads'], record['responses']):
                assert digest == r['payload_sha256'] and digest not in cache and digest not in fresh
                fresh[digest] = r
            saved[plan['batch_index']] = record
    pending = [p for p in plans if p['batch_index'] not in saved]
    if pending:
        runtime = reader.load_runtime(device)
        execution_id = str(uuid.uuid4())
        dump(base / 'reader' / f'generation_environment_{execution_id}.json', runtime[2])
        for plan in pending:
            batch = [mindex[rid] for rid in plan['record_ids']]
            assert all(m['condition'] not in METHODS and m['payload_sha256'] not in cache for m in batch)
            outputs = reader.generate_batch(runtime, [{'system': reader.READER_SYSTEM,
                'user_text': m['reader_text'], 'query_image_paths': m['query_image_paths']} for m in batch], 256)
            assert len(outputs) == len(batch)
            responses = [{**r, 'record_id': m['record_id'], 'payload_sha256': m['payload_sha256'],
                          'execution_id': execution_id, 'batch_index': plan['batch_index']}
                         for m, r in zip(batch, outputs)]
            record = {'plan': plan, 'experiment_signature': protocol['experiment_signature'], 'responses': responses}
            dump(batch_root / f'{plan["batch_index"]:04d}.json', record)
            saved[plan['batch_index']] = record
            for r in responses:
                assert r['payload_sha256'] not in fresh
                fresh[r['payload_sha256']] = r
            progress = {'stage': 'new_inputs_only', 'batches_completed': len(saved), 'batches_total': len(plans),
                'fresh_inferences': len(fresh), 'fresh_total': protocol['contract']['new_unique_payloads'],
                'control_model_calls': 0,
                'empty_responses': sum(not r['raw_response'] for r in fresh.values()),
                'token_limit_records': sum(r['hit_token_limit'] for r in fresh.values())}
            dump(base / 'reader/progress.json', progress)
            print(json.dumps(progress), flush=True)
        del runtime
    assert len(fresh) == protocol['contract']['new_unique_payloads']
    responses = []
    for m, origin in zip(manifest, assignment):
        digest = m['payload_sha256']
        historic = digest in cache
        source = cache[digest] if historic else fresh[digest]
        is_fresh = not historic and m['record_id'] == source['record_id']
        r = {key: source[key] for key in ANSWER_FIELDS}
        r.update({key: m[key] for key in ['id', 'task', 'point', 'record_id', 'condition',
                                        'payload_sha256', 'top5_memory_ids']})
        r.update(run_id=protocol['run_id'], experiment_signature=protocol['experiment_signature'],
            status='ok' if r['raw_response'] else 'empty_response',
            fresh_inference_executed_for_this_record=is_fresh,
            historical_response_reused=historic,
            historical_run_id=protocol['contract']['original_run_id'] if historic else None,
            historical_record_id=source['record_id'] if historic else None,
            same_run_shared_inference_record_id=None if historic or is_fresh else source['record_id'],
            inference_batch_index=None if historic else source['batch_index'],
            same_as_own_unweighted_input=origin['same_as_own_unweighted_input'])
        responses.append(r)
    write_lines(base / 'reader/responses.jsonl', responses)
    report = {'passed': True, 'questions': 1711, 'logical_records': len(responses),
        'fresh_inferences': len(fresh),
        'historical_reused_records': sum(r['historical_response_reused'] for r in responses),
        'same_run_shared_records': sum(r['same_run_shared_inference_record_id'] is not None for r in responses),
        'control_model_calls': 0, 'gold_loaded': False, 'candidate_expansion_in_reader': False,
        'empty_responses': sum(not r['raw_response'] for r in responses),
        'token_limit_records': sum(r['hit_token_limit'] for r in responses),
        'responses_sha256': sha(base / 'reader/responses.jsonl')}
    pv.verify_hashes(protocol['contract']['source_sha256'])
    dump(base / 'reader/validation_output.json', report)
    return report


def analyze(base):
    pv.verify(base)
    reader.CONDITIONS = tuple(SPECS)
    scored = reader.analyze(base)
    pairs = read(base / 'reader/paired_results.json')
    originals = reader.by_id(read(ORIGINAL / 'reader/paired_results.json'))
    responses = reader.by_id(lines(base / 'reader/responses.jsonl'), 'record_id')
    overall = {}
    for name in SPECS:
        values = [p[name] for p in pairs]
        records = [responses[p['id'] + '::' + name] for p in pairs]
        overall[name] = {**scored['overall'][name], 'correct': sum(int(v['em']) for v in values),
            'questions': len(values), 'gold_hits': sum(v['gold_hits'] for v in values),
            'gold_links': sum(v['gold_clues'] for v in values),
            'original_answer_reused': sum(r['historical_response_reused'] for r in records),
            'new_answer_records': sum(not r['historical_response_reused'] for r in records),
            'own_unweighted_input_changed': sum(not r['same_as_own_unweighted_input'] for r in records)}
    for method, expected in zip(METHODS, [778, 798]):
        assert overall[method]['correct'] == expected
        assert all(p[method] == originals[p['id']][method] for p in pairs)
        assert overall[method]['new_answer_records'] == 0
    table, changes, detail = [], {}, []
    for ridge in LAMBDAS:
        row = {'lambda': ridge, 'alpha': 1.}
        for prefix, method in zip(['baseline', 'master'], METHODS):
            name = pv.condition(method, ridge)
            metrics = overall[name]
            wins, losses = [], []
            for p in pairs:
                a, b = p[method], p[name]
                if a['payload_sha256'] == b['payload_sha256']:
                    assert a['raw_response'] == b['raw_response'] and a['em'] == b['em']
                if b['em'] > a['em']: wins.append(p['id'])
                if b['em'] < a['em']: losses.append(p['id'])
                r = responses[p['id'] + '::' + name]
                detail.append({'id': p['id'], 'condition': name, 'lambda': ridge, 'alpha': 1.,
                    'original_answer': a['raw_response'], 'weighted_answer': b['raw_response'],
                    'original_em': int(a['em']), 'weighted_em': int(b['em']),
                    'original_answer_reused': r['historical_response_reused'],
                    'same_as_own_unweighted_input': r['same_as_own_unweighted_input'],
                    'payload_sha256': r['payload_sha256'],
                    'source_record_id': r['historical_record_id'] or r['same_run_shared_inference_record_id'] or r['record_id']})
            row.update({prefix + '_correct': metrics['correct'], prefix + '_em': metrics['em'],
                prefix + '_f1': metrics['f1'], prefix + '_recall_at_5': metrics['recall_at_5_micro'],
                prefix + '_gold_hits': metrics['gold_hits'], prefix + '_gain': metrics['correct'] - overall[method]['correct'],
                prefix + '_wrong_to_right': len(wins), prefix + '_right_to_wrong': len(losses),
                prefix + '_original_answer_reused': metrics['original_answer_reused'],
                prefix + '_new_answer_records': metrics['new_answer_records']})
            changes[name] = {'wrong_to_right': wins, 'right_to_wrong': losses}
        table.append(row)
    summary = {'status': 'complete', 'dataset': 'Mem-Gallery', 'model': 'Qwen3-VL-8B-Instruct',
        'scheme': 'positive_variance', 'alpha': 1., 'lambdas': list(LAMBDAS), 'seed': 42,
        'questions': 1711, 'conditions': 12, 'original_answer_source': str(ORIGINAL),
        'frozen_retrieval_source': str(RETRIEVAL), 'overall': overall, 'ablation': table,
        'paired_changes': changes, 'reader_validation': read(base / 'reader/validation_output.json'),
        'method_annotations': annotations(), 'metric_policy': scored['metric_policy'],
        'recall_definition': 'Micro Recall@5: target hits / 4109 target links, 1527 nonempty-clue questions; all 1711 enter EM.',
        'interpretation': 'Original-answer-cache anchored descriptive ablation, not an all-fresh rerun or held-out validation.'}
    dump(base / 'summary.json', summary)
    for filename, records in [('ablation.csv', table), ('paired_results.csv', detail),
                              ('metrics.csv', [{'condition': k, **v} for k, v in overall.items()])]:
        with (base / filename).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader(); w.writerows(records)
    pv.plot(base, table, overall)
    html = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>MemGallery PV：原始回答固定</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%}</style>',
        '<h1>MemGallery Positive Variance λ 消融：固定原始回答</h1>',
        '<p>Qwen3-VL-8B-Instruct，seed=42，α=1。未赋权对照直接复用原实验逐题回答：Caption baseline 778/1711；Caption visual master 798/1711。对照没有重新调用模型。</p>',
        '<p>原始caption、对话、图、master分组、候选陈述和CLIP选择均固定。复用已审计的PV检索分数与Top-5。仅完整回答输入变化且原答案缓存中不存在时进行新推理；跨λ相同输入共享答案。新请求按条件分别固定分批，保留原文本batch16/图像batch4与其他生成设置。</p>',
        '<p>这里报告的是原始回答缓存固定的消融，不是所有条件重新生成答案。EM为归一化严格匹配；Recall@5分母4109条目标关联，184道无clue题仍进入EM。</p>',
        '<img src="lambda_ablation.png"><table><tr><th>λ</th><th>Baseline + PV</th><th>Master + PV</th><th>Baseline Recall@5</th><th>Master Recall@5</th><th>Baseline复用/1711</th><th>Master复用/1711</th></tr>']
    for r in table:
        html.append(f'<tr><td>{r["lambda"]}</td><td>{r["baseline_correct"]}/1711</td><td>{r["master_correct"]}/1711</td><td>{r["baseline_recall_at_5"]:.2%}</td><td>{r["master_recall_at_5"]:.2%}</td><td>{r["baseline_original_answer_reused"]}</td><td>{r["master_original_answer_reused"]}</td></tr>')
    html += ['</table><p><a href="ablation.csv">消融CSV</a> · <a href="paired_results.csv">逐题来源和得失</a> · <a href="summary.json">完整统计</a></p></html>']
    (base / 'report.html').write_text('\n'.join(html))
    print(json.dumps({'stage': 'scored', 'ablation': table}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stage', choices=['prepare', 'reader', 'analyze', 'all'], default='all')
    args = parser.parse_args()
    base = args.output.resolve()
    if args.stage == 'prepare': prepare(base, args.device)
    if args.stage in ['reader', 'all']: run(base, args.device)
    if args.stage in ['analyze', 'all']: analyze(base)
    if args.stage == 'all':
        from scripts.analysis.audit_memgallery_pv_frozen_answers import audit
        audit(base)


if __name__ == '__main__':
    main()
