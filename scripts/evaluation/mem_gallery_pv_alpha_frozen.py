"""MemGallery PV alpha ablation with original and audited lambda-answer caches.

Freeze lambda=.03 and all source data. Recompute only alpha-scaled ranking;
invoke the unchanged Qwen reader only for complete payloads absent from caches.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import mem_gallery_pv_frozen_answers as frozen
from scripts.evaluation import mem_gallery_positive_variance_ablation as previous
from scripts.evaluation import mem_gallery_master_reader as reader
import numpy as np

ORIGINAL = frozen.ORIGINAL
DIAGNOSTICS = frozen.RETRIEVAL
LAMBDA_RUN = frozen.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / 'results/memgallery_qwen_pv_original_answers_alpha_ablation_l0p03_20261107'
AUDITOR = ROOT / 'scripts/analysis/audit_memgallery_pv_alpha_frozen.py'
RIDGE = .03
ALPHAS = (.1, .3, .6, 1., 3., 6., 10.)
METHODS = frozen.METHODS
read, lines, dump, sha = frozen.read, frozen.lines, frozen.dump, frozen.sha
write_lines, copy_identical = frozen.write_lines, frozen.copy_identical
check = previous.verify_hashes


def condition(method, alpha):
    return method if alpha is None else method + '+positive_variance__l0p03_a' + format(alpha, 'g').replace('.', 'p')


SPECS = {m: {'method': m, 'lambda': None, 'alpha': 0.} for m in METHODS}
SPECS.update({condition(m, a): {'method': m, 'lambda': RIDGE, 'alpha': a}
              for a in ALPHAS for m in METHODS})
N_RECORDS = 1711 * len(SPECS)


def annotations():
    old = previous.annotations()
    result = {}
    for name, spec in SPECS.items():
        result[name] = copy.deepcopy(old[condition(spec['method'], 1.) if spec['lambda'] else spec['method']])
        result[name]['3. 检索方式'].update(alpha=spec['alpha'] if spec['lambda'] else None)
        result[name]['5. 回答方式'] += (' 优先复用原778/798回答，其次复用已审计的原回答固定λ消融缓存；'
            '完整请求相同必须使用同一答案，只有缓存中不存在的新请求才推理。α=1逐题结果保持不变。')
    return result


def verify(base):
    p = read(base / 'retrieval_protocol.json')
    assert p['condition_specs'] == SPECS
    check(p['source_sha256']); check(p['artifact_sha256'])
    return p


def load_caches():
    original_p, original_m, original_r, initial = frozen.original_cache()
    completion = read(LAMBDA_RUN / 'completion_manifest.json')
    assert completion['status'] == 'complete' and read(LAMBDA_RUN / 'independent_audit.json')['passed']
    check(completion['artifact_sha256'])
    p = read(LAMBDA_RUN / 'reader/protocol.json')
    assert reader.digest(p['contract']) == p['experiment_signature']
    check(p['contract']['source_sha256'])
    old_contract = original_p['contract']
    for k in ['model', 'system_prompt', 'task_instructions', 'seed', 'do_sample', 'max_new_tokens',
              'batch_size_text', 'batch_size_images', 'dtype', 'query_image_max_pixels', 'input_truncation']:
        assert old_contract[k] == p['contract'][k], k
    lm = lines(LAMBDA_RUN / 'reader/qa_manifest.jsonl')
    assert reader.digest(lm) == p['contract']['manifest_sha256']
    lr = reader.by_id(lines(LAMBDA_RUN / 'reader/responses.jsonl'), 'record_id')
    assert len(lr) == len(lm) == 20532
    cache = {h: {**r, 'cache_source': 'original', 'cache_root': str(ORIGINAL),
                 'cache_run_id': original_p['run_id']} for h, r in initial.items()}
    for m in lm:
        r = lr[m['record_id']]
        h = m['payload_sha256']
        assert h == r['payload_sha256'] and m['top5_memory_ids'] == r['top5_memory_ids']
        assert r['run_id'] == p['run_id'] and r['experiment_signature'] == p['experiment_signature']
        if h in cache:
            assert cache[h]['raw_response'] == r['raw_response']
        else:
            assert not r['historical_response_reused']
            cache[h] = {**r, 'cache_source': 'lambda_ablation', 'cache_root': str(LAMBDA_RUN), 'cache_run_id': p['run_id']}
    assert len(cache) == 4588
    return original_p, original_m, original_r, reader.by_id(lm, 'record_id'), lr, cache


def retrieve(base):
    if (base / 'retrieval_protocol.json').exists(): return verify(base)
    source = previous.verify(DIAGNOSTICS)
    check(read(LAMBDA_RUN / 'completion_manifest.json')['artifact_sha256'])
    assert read(LAMBDA_RUN / 'independent_audit.json')['passed']
    base.mkdir(parents=True, exist_ok=True)
    for name in ['questions.jsonl', 'memories.jsonl', 'gold.jsonl', 'protocol.json']:
        copy_identical(ORIGINAL / 'inputs' / name, base / 'inputs' / name)
    prior = reader.by_id(lines(DIAGNOSTICS / 'retrieval_results.jsonl'))
    by_question = {}
    checks = Counter()
    max_unit_error = 0.
    for path in sorted((DIAGNOSTICS / 'diagnostics').glob('*.jsonl')):
        for d in lines(path):
            before = prior[d['id']]
            old_conditions = {c['name']: c for c in before['conditions']}
            row = {k: copy.deepcopy(before[k]) for k in ['id', 'task', 'candidate_count', 'master_choices']}
            row['conditions'] = []
            ids = d['memory_ids']
            for method, data in d['methods'].items():
                raw = np.asarray(data['raw'])
                stats = data['weighting']['0.03']
                unit = np.asarray(stats['correction'])
                z = np.asarray(stats['z'])
                assert np.array_equal(raw + unit, np.asarray(stats['weighted']))
                assert np.isfinite(raw).all() and np.isfinite(unit).all()
                for group in data['groups']:
                    ix = group['indices']
                    expected = np.zeros(len(ix))
                    if len(ix) > 1 and group['raw_status'] != 'zero_variance':
                        expected = np.std(raw[ix]) * np.tanh(z[ix])
                    error = float(np.max(np.abs(unit[ix] - expected)))
                    assert error < 1e-12
                    max_unit_error = max(max_unit_error, error)
                    checks['unit_group_checks'] += 1
                if method == METHODS[1]:
                    assert not unit[d['text_only_indices']].any()
                for alpha in [None, *ALPHAS]:
                    values = raw if alpha is None else raw + alpha * unit
                    ix = np.argsort(-values, kind='stable')[:5]
                    item = {'name': condition(method, alpha), 'top5_memory_ids': [ids[i] for i in ix],
                            'top5_scores': values[ix].tolist()}
                    if alpha in [None, 1.]:
                        anchor = old_conditions[item['name']]
                        assert item == anchor, (d['id'], item['name'])
                        checks['raw_matches' if alpha is None else 'alpha1_matches'] += 1
                    row['conditions'].append(item)
            by_question[d['id']] = row
    questions = lines(base / 'inputs/questions.jsonl')
    assert len(questions) == len(by_question) == 1711
    assert checks['raw_matches'] == checks['alpha1_matches'] == 3422
    rows = [by_question[q['id']] for q in questions]
    write_lines(base / 'retrieval_results.jsonl', rows)
    sources = {**source['source_sha256'], **source['artifact_sha256']}
    tracked = [Path(__file__), AUDITOR, Path(frozen.__file__), DIAGNOSTICS / 'retrieval_protocol.json',
        LAMBDA_RUN / 'completion_manifest.json', LAMBDA_RUN / 'independent_audit.json',
        LAMBDA_RUN / 'reader/protocol.json', LAMBDA_RUN / 'reader/qa_manifest.jsonl',
        LAMBDA_RUN / 'reader/responses.jsonl', LAMBDA_RUN / 'reader/paired_results.json',
        ORIGINAL / 'reader/protocol.json', ORIGINAL / 'reader/qa_manifest.jsonl', ORIGINAL / 'reader/responses.jsonl']
    sources.update({str(f): sha(f) for f in tracked})
    sources.update({str(f): sha(f) for f in (base / 'inputs').iterdir()})
    p = {'scheme': 'positive_variance', 'dataset': 'Mem-Gallery', 'lambda': RIDGE, 'alphas': list(ALPHAS),
         'condition_specs': SPECS, 'method_annotations': annotations(), 'source_run': str(ORIGINAL),
         'fixed_lambda_diagnostics': str(DIAGNOSTICS / 'diagnostics'), 'answer_cache_run': str(LAMBDA_RUN),
         'questions': 1711, 'conditions': len(SPECS), 'only_alpha_scaled': True,
         'scoring': 'raw + alpha * unit_correction; unit_correction from signed lambda=.03, alpha=1',
         'gold_loaded': False, **dict(checks), 'max_unit_correction_error': max_unit_error,
         'source_sha256': sources, 'artifact_sha256': {str(base / 'retrieval_results.jsonl'): sha(base / 'retrieval_results.jsonl')}}
    dump(base / 'retrieval_protocol.json', p)
    print(json.dumps({'stage': 'alpha_retrieval_complete', **dict(checks)}), flush=True)
    return p


def prepare(base, device):
    source = retrieve(base)
    old_p, old_m, _, lambda_m, _, cache = load_caches()
    (base / 'reader').mkdir(exist_ok=True)
    reader.CONDITIONS = tuple(SPECS)
    manifest = reader.build_reader_manifest(lines(base / 'inputs/questions.jsonl'),
        lines(base / 'inputs/memories.jsonl'), lines(base / 'retrieval_results.jsonl'))
    mindex = reader.by_id(manifest, 'record_id')
    for m in manifest:
        if m['condition'] in METHODS or SPECS[m['condition']]['alpha'] == 1.:
            old = old_m[m['record_id']] if m['condition'] in METHODS else lambda_m[m['record_id']]
            for field in ['payload_sha256', 'reader_text', 'query_images', 'top5_memory_ids']:
                assert m[field] == old[field]
            assert m['payload_sha256'] in cache
    owners, plans = {}, []
    for name in SPECS:
        selected = []
        for m in manifest:
            h = m['payload_sha256']
            if m['condition'] != name or h in cache or h in owners: continue
            assert name not in METHODS and SPECS[name]['alpha'] != 1.
            owners[h] = m
            selected.append(m)
        for has_image in [False, True]:
            items = [m for m in selected if bool(m['query_image_paths']) == has_image]
            size = 4 if has_image else 16
            for start in range(0, len(items), size):
                batch = items[start:start + size]
                plans.append({'batch_index': len(plans), 'owner_condition': name, 'has_image': has_image,
                    'record_ids': [m['record_id'] for m in batch], 'payloads': [m['payload_sha256'] for m in batch]})
    assignments = []
    for m in manifest:
        h = m['payload_sha256']; cached = cache.get(h)
        assignments.append({'record_id': m['record_id'], 'payload_sha256': h,
            'origin': cached['cache_source'] if cached else 'new',
            'source_record_id': cached['record_id'] if cached else owners[h]['record_id'],
            'source_root': cached['cache_root'] if cached else str(base),
            'same_as_own_unweighted_input': h == old_m[m['id'] + '::' + SPECS[m['condition']]['method']]['payload_sha256']})
    plan = {'batches': plans, 'new_unique_payloads': len(owners), 'seed': 42,
        'batch_policy': 'First owning condition, source question order, no-image then image, batch16/4; frozen before inference.'}
    for filename, data, is_lines in [('qa_manifest.jsonl', manifest, True), ('reuse_manifest.jsonl', assignments, True),
                                    ('inference_plan.json', plan, False)]:
        f = base / 'reader' / filename
        if f.exists(): assert (lines(f) if is_lines else read(f)) == data
        elif is_lines: write_lines(f, data)
        else: dump(f, data)
    assert reader.model_provenance() == old_p['contract']['model']
    contract = copy.deepcopy(old_p['contract'])
    contract.update(stage='original_answer_anchored_positive_variance_alpha_ablation',
        logical_conditions=len(manifest), unique_payloads=len({m['payload_sha256'] for m in manifest}),
        manifest_sha256=reader.digest(manifest), source_sha256={**source['source_sha256'], **source['artifact_sha256']},
        historical_response_reuse=True, answer_cache_priority=[str(ORIGINAL), str(LAMBDA_RUN)],
        original_run_id=old_p['run_id'], condition_specs=SPECS, method_annotations=annotations(), device=device,
        inference_order=plan['batch_policy'], new_unique_payloads=len(owners),
        deduplication='Original cache first, audited lambda-run cache second; new complete payloads once only.',
        parameter_policy='lambda=.03; alpha=.1/.3/.6/1/3/6/10; controls and alpha1 never regenerated.',
        batch_plan_sha256=sha(base / 'reader/inference_plan.json'),
        reuse_manifest_sha256=sha(base / 'reader/reuse_manifest.jsonl'))
    contract['source_sha256'][str(base / 'retrieval_protocol.json')] = sha(base / 'retrieval_protocol.json')
    protocol = reader.ensure_protocol(base / 'reader/protocol.json', contract)
    report = {'status': 'prepared', 'logical_records': len(manifest),
        'origins': dict(Counter(a['origin'] for a in assignments)), 'new_unique_payloads': len(owners),
        'new_batches': len(plans), 'control_model_calls': 0, 'alpha1_model_calls': 0, 'gold_loaded': False}
    dump(base / 'preparation.json', report)
    print(json.dumps(report), flush=True)
    return protocol, manifest, cache, plans, assignments


def run(base, device):
    protocol, manifest, cache, plans, assignments = prepare(base, device)
    mindex = reader.by_id(manifest, 'record_id')
    batch_dir = base / 'reader/batches'; batch_dir.mkdir(exist_ok=True)
    fresh, done = {}, set()
    for plan in plans:
        f = batch_dir / f'{plan["batch_index"]:04d}.json'
        if not f.exists(): continue
        saved = read(f)
        assert saved['plan'] == plan and saved['experiment_signature'] == protocol['experiment_signature']
        assert len(saved['responses']) == len(plan['payloads'])
        for h, r in zip(plan['payloads'], saved['responses']):
            assert h == r['payload_sha256'] and h not in cache and h not in fresh
            fresh[h] = r
        done.add(plan['batch_index'])
    pending = [p for p in plans if p['batch_index'] not in done]
    if pending:
        runtime = reader.load_runtime(device)
        execution_id = str(uuid.uuid4())
        dump(base / 'reader' / f'generation_environment_{execution_id}.json', runtime[2])
        old_envs = list((ORIGINAL / 'reader').glob('generation_environment_*.json'))
        assert len(old_envs) == 1
        old_env = read(old_envs[0])
        keys = ['torch', 'transformers', 'device', 'gpu', 'dtype', 'context_limit', 'generation_config', 'query_image_max_pixels']
        for key in keys: assert runtime[2][key] == old_env[key], key
        dump(base / 'environment_audit.json', {'passed': True, 'equal_fields': keys})
        for plan in pending:
            batch = [mindex[rid] for rid in plan['record_ids']]
            assert all(m['payload_sha256'] not in cache for m in batch)
            answers = reader.generate_batch(runtime, [{'system': reader.READER_SYSTEM, 'user_text': m['reader_text'],
                'query_image_paths': m['query_image_paths']} for m in batch], 256)
            assert len(answers) == len(batch)
            records = [{**r, 'record_id': m['record_id'], 'payload_sha256': m['payload_sha256'],
                'execution_id': execution_id, 'batch_index': plan['batch_index']} for m, r in zip(batch, answers)]
            dump(batch_dir / f'{plan["batch_index"]:04d}.json', {'plan': plan,
                'experiment_signature': protocol['experiment_signature'], 'responses': records})
            done.add(plan['batch_index'])
            for r in records:
                assert r['payload_sha256'] not in fresh
                fresh[r['payload_sha256']] = r
            progress = {'stage': 'new_alpha_inputs_only', 'batches_completed': len(done), 'batches_total': len(plans),
                'fresh_inferences': len(fresh), 'fresh_total': protocol['contract']['new_unique_payloads'],
                'control_model_calls': 0, 'alpha1_model_calls': 0,
                'empty_responses': sum(not r['raw_response'] for r in fresh.values()),
                'token_limit_records': sum(r['hit_token_limit'] for r in fresh.values())}
            dump(base / 'reader/progress.json', progress); print(json.dumps(progress), flush=True)
        del runtime
    assert len(fresh) == protocol['contract']['new_unique_payloads']
    responses = []
    for m, a in zip(manifest, assignments):
        h = m['payload_sha256']; historic = h in cache
        source = cache[h] if historic else fresh[h]
        is_fresh = not historic and source['record_id'] == m['record_id']
        row = {k: source[k] for k in frozen.ANSWER_FIELDS}
        row.update({k: m[k] for k in ['id', 'task', 'point', 'record_id', 'condition', 'payload_sha256', 'top5_memory_ids']})
        row.update(run_id=protocol['run_id'], experiment_signature=protocol['experiment_signature'],
            status='ok' if row['raw_response'] else 'empty_response', historical_response_reused=historic,
            cache_source=source['cache_source'] if historic else None,
            historical_root=source['cache_root'] if historic else None,
            historical_run_id=source['cache_run_id'] if historic else None,
            historical_record_id=source['record_id'] if historic else None,
            fresh_inference_executed_for_this_record=is_fresh,
            same_run_shared_inference_record_id=None if historic or is_fresh else source['record_id'],
            inference_batch_index=None if historic else source['batch_index'],
            same_as_own_unweighted_input=a['same_as_own_unweighted_input'])
        responses.append(row)
    write_lines(base / 'reader/responses.jsonl', responses)
    check(protocol['contract']['source_sha256'])
    report = {'passed': True, 'logical_records': len(responses), 'fresh_inferences': len(fresh),
        'cached_records': dict(Counter(r['cache_source'] for r in responses if r['historical_response_reused'])),
        'same_run_shared_records': sum(r['same_run_shared_inference_record_id'] is not None for r in responses),
        'control_model_calls': 0, 'alpha1_model_calls': 0, 'gold_loaded': False,
        'empty_responses': sum(not r['raw_response'] for r in responses),
        'token_limit_records': sum(r['hit_token_limit'] for r in responses),
        'responses_sha256': sha(base / 'reader/responses.jsonl')}
    dump(base / 'reader/validation_output.json', report)


def analyze(base):
    verify(base); reader.CONDITIONS = tuple(SPECS)
    scored = reader.analyze(base)
    pairs = read(base / 'reader/paired_results.json')
    old = reader.by_id(read(ORIGINAL / 'reader/paired_results.json'))
    previous_pairs = reader.by_id(read(LAMBDA_RUN / 'reader/paired_results.json'))
    responses = reader.by_id(lines(base / 'reader/responses.jsonl'), 'record_id')
    overall = {}
    for name in SPECS:
        values = [p[name] for p in pairs]
        records = [responses[p['id'] + '::' + name] for p in pairs]
        origins = Counter(r['cache_source'] if r['historical_response_reused'] else 'new' for r in records)
        overall[name] = {**scored['overall'][name], 'questions': len(values),
            'correct': sum(int(v['em']) for v in values), 'gold_hits': sum(v['gold_hits'] for v in values),
            'gold_links': sum(v['gold_clues'] for v in values), 'original_answer_reused': origins['original'],
            'lambda_answer_reused': origins['lambda_ablation'], 'new_answer_records': origins['new']}
    for method, expected, alpha1_expected in zip(METHODS, [778, 798], [778, 768]):
        assert overall[method]['correct'] == expected
        name = condition(method, 1.)
        assert overall[name]['correct'] == alpha1_expected
        assert overall[method]['new_answer_records'] == overall[name]['new_answer_records'] == 0
        for p in pairs:
            assert p[method] == old[p['id']][method]
            assert p[name] == previous_pairs[p['id']][name]
    table, changes, detail = [], {}, []
    for alpha in ALPHAS:
        row = {'lambda': RIDGE, 'alpha': alpha}
        for prefix, method in zip(['baseline', 'master'], METHODS):
            name = condition(method, alpha); metrics = overall[name]
            wins, losses = [], []
            for p in pairs:
                a, b = p[method], p[name]
                if a['payload_sha256'] == b['payload_sha256']: assert a['raw_response'] == b['raw_response']
                if b['em'] > a['em']: wins.append(p['id'])
                if b['em'] < a['em']: losses.append(p['id'])
                r = responses[p['id'] + '::' + name]
                detail.append({'id': p['id'], 'condition': name, 'lambda': RIDGE, 'alpha': alpha,
                    'original_answer': a['raw_response'], 'weighted_answer': b['raw_response'],
                    'original_em': int(a['em']), 'weighted_em': int(b['em']),
                    'origin': r['cache_source'] or 'new', 'same_as_own_unweighted_input': r['same_as_own_unweighted_input'],
                    'payload_sha256': r['payload_sha256'], 'source_root': r['historical_root'] or str(base),
                    'source_record_id': r['historical_record_id'] or r['same_run_shared_inference_record_id'] or r['record_id']})
            for field in ['correct', 'em', 'f1', 'gold_hits', 'original_answer_reused', 'lambda_answer_reused', 'new_answer_records']:
                row[prefix + '_' + field] = metrics[field]
            row.update({prefix + '_recall_at_5': metrics['recall_at_5_micro'],
                prefix + '_gain': metrics['correct'] - overall[method]['correct'],
                prefix + '_wrong_to_right': len(wins), prefix + '_right_to_wrong': len(losses)})
            changes[name] = {'wrong_to_right': wins, 'right_to_wrong': losses}
        table.append(row)
    summary = {'status': 'complete', 'dataset': 'Mem-Gallery', 'model': 'Qwen3-VL-8B-Instruct',
        'scheme': 'positive_variance', 'lambda': RIDGE, 'alphas': list(ALPHAS), 'seed': 42,
        'questions': 1711, 'conditions': len(SPECS), 'overall': overall, 'ablation': table,
        'paired_changes': changes, 'reader_validation': read(base / 'reader/validation_output.json'),
        'method_annotations': annotations(), 'original_answer_source': str(ORIGINAL),
        'lambda_answer_source': str(LAMBDA_RUN), 'metric_policy': scored['metric_policy'],
        'recall_definition': 'Micro Recall@5 = gold hits / 4109 target links; 1527 nonempty-clue questions. All 1711 enter EM.',
        'interpretation': 'Original-answer-cache anchored descriptive ablation; not all-fresh or held-out validation.'}
    dump(base / 'summary.json', summary)
    for filename, data in [('ablation.csv', table), ('paired_results.csv', detail),
                           ('metrics.csv', [{'condition': k, **v} for k, v in overall.items()])]:
        with (base / filename).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    plot(base, table, overall)
    html = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>MemGallery PV α 消融</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1300px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%}</style>',
        '<h1>MemGallery Positive Variance α 消融</h1>',
        '<p>Qwen3-VL-8B-Instruct，λ=0.03，seed=42。原始未赋权对照778/1711、798/1711及α=1的778/1711、768/1711均直接复用，逐题保持一致。</p>',
        '<p>固定原caption、对话、master图、候选陈述、CLIP选择及λ=0.03的SVD修正项，只重新计算raw+α×unit_correction的排序。原始完整请求缓存优先，其次使用已审计λ消融的请求缓存，仅新请求在GPU上生成答案。所有相同完整输入共享答案。</p>',
        '<p>EM为归一化严格匹配；Recall@5分母4109条目标关联。184道无clue题仍计入EM。本实验为复用已验证回答的消融，不是全量重新生成答案。</p>',
        '<img src="alpha_ablation.png"><table><tr><th>α</th><th>Baseline + PV</th><th>Master + PV</th><th>Baseline Recall@5</th><th>Master Recall@5</th></tr>']
    for r in table:
        html.append(f'<tr><td>{r["alpha"]}</td><td>{r["baseline_correct"]}/1711</td><td>{r["master_correct"]}/1711</td><td>{r["baseline_recall_at_5"]:.2%}</td><td>{r["master_recall_at_5"]:.2%}</td></tr>')
    html += ['</table><p><a href="ablation.csv">消融CSV</a> · <a href="paired_results.csv">逐题结果与来源</a> · <a href="summary.json">完整统计</a></p></html>']
    (base / 'report.html').write_text('\n'.join(html))
    print(json.dumps({'stage': 'scored', 'ablation': table}), flush=True)


def plot(base, table, overall):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 14, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.labelweight': 'bold', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, metric, label in zip(axes, ['em', 'recall_at_5'], ['Exact match (×100)', 'Recall@5, micro (×100)']):
        for prefix, method, color, name in zip(['baseline', 'master'], METHODS,
                ['#E4B27C', '#A85B13'], ['Caption baseline', 'Caption visual master']):
            values = [r[prefix + '_' + metric] for r in table]
            ax.plot(ALPHAS, values, 'o-', color=color, label=name + ' + PV', linewidth=2)
            control = overall[method]['em' if metric == 'em' else 'recall_at_5_micro']
            ax.axhline(control, color=color, linestyle='--', alpha=.65, label=name + ' (unweighted)')
            for x, y in zip(ALPHAS, values):
                ax.annotate(f'{y*100:.2f}', (x, y), xytext=(0, 9), textcoords='offset points',
                            ha='center', color=color, fontsize=12, weight='bold')
        ax.set_xscale('log'); ax.set_xticks(ALPHAS, [format(a, 'g') for a in ALPHAS]); ax.minorticks_off()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v*100:.2f}'))
        ax.set_xlabel('Alpha (log scale)', fontsize=17); ax.set_ylabel(label, fontsize=17)
        for tick in ax.get_xticklabels() + ax.get_yticklabels(): tick.set_fontweight('bold')
        ax.margins(y=.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, .005),
               frameon=False, prop={'size': 13, 'weight': 'bold'})
    fig.subplots_adjust(left=.08, right=.98, top=.95, bottom=.28, wspace=.32)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(base / f'alpha_ablation.{ext}', dpi=220, facecolor='white', bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stage', choices=['prepare', 'reader', 'analyze', 'all'], default='all')
    args = parser.parse_args(); base = args.output.resolve()
    if args.stage == 'prepare': prepare(base, args.device)
    if args.stage in ['reader', 'all']: run(base, args.device)
    if args.stage in ['analyze', 'all']: analyze(base)
    if args.stage == 'all':
        from scripts.analysis.audit_memgallery_pv_alpha_frozen import audit
        audit(base)


if __name__ == '__main__': main()
