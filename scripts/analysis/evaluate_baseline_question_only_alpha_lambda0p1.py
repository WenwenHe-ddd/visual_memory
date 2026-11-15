"""Full-scan alpha ablation at lambda=.1: three fresh settings plus raw/alpha=1 references."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import html
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
from scripts import visual_memory_pipeline
from scripts.analysis.evaluate_adaptive_changed_qa import digest, dump, lines, read, reader_text, sha
from scripts.analysis.evaluate_baseline_question_only_zscore import jsonl, csv_write, check_sources
from scripts.analysis.evaluate_all_master_answer_reader import verify
from scripts.evaluation.paired_single_gpu_reader import run as run_reader

BASE = ROOT / 'reports/baseline_question_only_lambda0p1_alpha_ablation_20261024'
REFERENCE = ROOT / 'reports/baseline_question_only_zscore_20261023'
BASELINE = ROOT / 'reports/baseline_question_only_zscore_20261023'
HISTORICAL = ROOT / 'reports/all_master_answer_query_20261021'
INPUTS = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
STORES = ROOT / 'results/memory_stores_20261023'
ORIGINAL = ROOT / 'data/independent_visual_inputs'
SNAPSHOT = ROOT / 'results/experiment_source_snapshots/baseline_question_only_lambda0p1_alpha_20261024'
CONTROL = 'baseline_question_only'
ALPHAS = {CONTROL: None, 'alpha_0p01': .01, 'alpha_0p1': .1, 'alpha_1p0': 1., 'alpha_10p0': 10.}
RIDGE = .1
NEW = ('alpha_0p1', 'alpha_0p01', 'alpha_10p0')
LOGICAL_RECORDS = 371 * len(NEW)
OLD = {CONTROL: CONTROL, 'baseline_question_only_zscore': 'alpha_1p0'}
LABELS = {CONTROL: '原始基线（未赋权）', 'alpha_0p01': 'α=0.01', 'alpha_0p1': 'α=0.1', 'alpha_1p0': 'α=1', 'alpha_10p0': 'α=10'}
REFERENCE_ENTRYPOINT_CHANGE = {
    'path': str(ROOT / 'scripts/visual_memory_pipeline.py'),
    'historical_sha256': '7ec9368919a94fdf57f8ee2dfc787f19d1f5198970c52bee2d6ca8a2c4b586ad',
    'current_sha256': 'd234d70f6f41dc0b518ce61c542cbb5720934ede299debb6286624acdbbf6cb8',
    'validation': 'The current online entrypoint was inspected. Require other pinned sources unchanged except the separately verified comment-only edit; reproduce all raw/white/z scores and raw rankings before QA. Freeze current sources in the new protocol.'}
REFERENCE_COMMENT_CHANGE = {
    'path': str(ROOT / 'scripts/memory/online_retrieval.py'),
    'historical_sha256': '8475dea9aa059ee2909b66ea1ef0d23050d38ac5c9ce35e35282497d391376a9',
    'current_sha256': 'b1d6efb9cbdbd9fc4a76f4be285e7e6dfbf349d5ed9995fff03d3ef610103ddf',
    'added_comments': [' # baseline question only加上zscore的方案在这里', ' # full_scan or multi_path'],
    'validation': 'Removing exactly these two inline comments restores the historical SHA256.'}


def frozen_source(path):
    path = Path(path)
    candidate = SNAPSHOT / path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return candidate if candidate.is_file() else path


def activate_source_snapshot():
    """Load byte-identical saved modules, isolating ongoing workspace comments."""
    global visual_memory_pipeline
    manifest = read(SNAPSHOT / 'snapshot_manifest.json')
    for record in manifest['files']:
        assert sha(record['snapshot']) == record['sha256']
    names = ['scripts.memory.' + name for name in (
        'caption_visual_master', 'memory_store', 'ridge_whitening', 'online_zscore',
        'result_method_annotations', 'online_retrieval')]
    names.append('scripts.visual_memory_pipeline')
    for name in names:
        parent_name, attribute = name.rsplit('.', 1)
        parent = importlib.import_module(parent_name)
        path = SNAPSHOT.joinpath(*name.split('.')).with_suffix('.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(parent, attribute, module)
        spec.loader.exec_module(module)
    visual_memory_pipeline = sys.modules['scripts.visual_memory_pipeline']


def check_reference_sources(protocol):
    """Accept pinned entrypoint/comment revisions; never permit data drift."""
    for path, expected in protocol['source_sha256'].items():
        actual = sha(frozen_source(path))
        if actual == expected: continue
        if path == REFERENCE_COMMENT_CHANGE['path']:
            change = REFERENCE_COMMENT_CHANGE
            assert expected == change['historical_sha256'] and actual == change['current_sha256']
            source = frozen_source(path).read_text()
            for comment in change['added_comments']:
                assert source.count(comment) == 1
                source = source.replace(comment, '')
            assert hashlib.sha256(source.encode()).hexdigest() == expected
            continue
        assert path == REFERENCE_ENTRYPOINT_CHANGE['path'], path
        assert expected == REFERENCE_ENTRYPOINT_CHANGE['historical_sha256'], path
        assert actual == REFERENCE_ENTRYPOINT_CHANGE['current_sha256'], path


def reference_readers():
    """Raw and alpha=1 are attributed to their respective completed runs."""
    manifests = {}; responses = {}; protocols = {}
    for name, directory in ((CONTROL, BASELINE), ('baseline_question_only_zscore', REFERENCE)):
        check_reference_sources(read(directory / 'protocol.json'))
        protocol = read(directory / 'reader/protocol.json'); verify(protocol)
        signed = dict(protocol); signature = signed.pop('experiment_signature')
        assert digest(signed) == signature
        protocols[name] = protocol
        for target, filename in ((manifests, 'qa_manifest.jsonl'), (responses, 'responses.jsonl')):
            selected = [r for r in lines(directory / 'reader' / filename) if r['condition'] == name]
            assert len(selected) == len({r['record_id'] for r in selected}) == 371
            target.update({r['record_id']: r for r in selected})
    assert protocols[CONTROL]['reader_settings'] == protocols['baseline_question_only_zscore']['reader_settings']
    assert set(manifests) == set(responses) and len(manifests) == 742
    return manifests, responses, protocols


def reference_retrieval():
    result = {}
    for row in lines(REFERENCE / 'retrieval_results.jsonl'):
        conditions = [c for c in row['conditions'] if c['name'] in OLD]
        assert len(conditions) == 2
        result[row['question_id']] = {**row, 'conditions': conditions}
    assert len(result) == 371
    return result


def aggregate(retrieved, answers=None):
    n = len(retrieved); total = sum(len(r['evaluation']['gold_ids']) for r in retrieved)
    result = {'questions': n, 'gold_references': total, 'conditions': {}, 'comparisons': {}}
    for name, alpha in ALPHAS.items():
        selected = [next(c for c in r['conditions'] if c['name'] == name) for r in retrieved]
        hits = sum(c['gold_hits_top5'] for c in selected)
        stats = {'lambda': RIDGE if alpha is not None else None, 'alpha': alpha, 'gold_hits_top5': hits, 'gold_recall': hits / total,
                 'questions_with_any_gold': sum(c['gold_hits_top5'] > 0 for c in selected),
                 'top1_gold': sum(c['top5'][0]['is_gold'] for c in selected),
                 'answer_source': 'fresh_current_run' if name in NEW else 'verified_previous_run'}
        if answers is not None:
            stats.update(correct=sum(answers[r['question_id']][name]['correct'] for r in retrieved))
            stats['accuracy'] = stats['correct'] / n
        result['conditions'][name] = stats
    for anchor in (CONTROL, 'alpha_1p0'):
        comparison = {}
        for name in ALPHAS:
            if name == anchor: continue
            same_order = same_set = 0; wins = []; losses = []
            for row in retrieved:
                conditions = {c['name']: c for c in row['conditions']}
                a, b = [conditions[k]['top5_memory_ids'] for k in (anchor, name)]
                same_order += a == b; same_set += set(a) == set(b)
                if answers is not None:
                    a, b = [answers[row['question_id']][k]['correct'] for k in (anchor, name)]
                    if b and not a: wins.append(row['question_id'])
                    if a and not b: losses.append(row['question_id'])
            c = {'changed_top5_order': n - same_order, 'changed_top5_set': n - same_set}
            if answers is not None:
                m = len(wins) + len(losses)
                p = min(1., 2 * sum(math.comb(m, i) for i in range(min(len(wins), len(losses)) + 1)) / 2**m) if m else 1.
                c.update(wrong_to_right=len(wins), right_to_wrong=len(losses), net_correct=len(wins)-len(losses),
                    delta_percentage_points=100 * (len(wins)-len(losses)) / n,
                    mcnemar_exact_two_sided_p=p, win_question_ids=wins, loss_question_ids=losses)
            comparison[name] = c
        result['comparisons']['versus_' + anchor] = comparison
    return result


def retrieve(base, device):
    if base.exists(): raise FileExistsError('Choose a new output directory.')
    prior = read(REFERENCE / 'protocol.json'); check_reference_sources(prior)
    reference_readers()
    original = lines(ORIGINAL / 'qa_manifest.jsonl')
    previous = reference_retrieval()
    assert len(original) == len(previous) == 371 and set(previous) == {q['id'] for q in original}
    base.mkdir(parents=True)
    paths = [Path(__file__), ROOT / 'scripts/analysis/evaluate_baseline_question_only_zscore.py',
             ORIGINAL / 'qa_manifest.jsonl', ORIGINAL / 'retrieval_memories.jsonl']
    paths += [REFERENCE / p for p in ('protocol.json', 'retrieval_results.jsonl', 'reader/protocol.json',
                                     'reader/qa_manifest.jsonl', 'reader/responses.jsonl', 'reader/summary.json')]
    paths += [BASELINE / p for p in ('protocol.json', 'reader/protocol.json', 'reader/qa_manifest.jsonl', 'reader/responses.jsonl')]
    paths += [SNAPSHOT / 'snapshot_manifest.json', *SNAPSHOT.rglob('*.py')]
    paths += [frozen_source(path) for path in prior['source_sha256']]
    hashes = {str(p): sha(p) for p in paths}
    new_results = {name: {} for name in NEW}; max_raw_error = max_white_error = max_z_error = 0.
    for name in NEW:
        for domain in sorted({q['task'] for q in original}):
            source = INPUTS / domain / 'inputs.json'
            assert sha(source) == prior['source_sha256'][str(source)]
            destination = base / 'retrieval' / name / domain
            visual_memory_pipeline.main(['online', '--memory-path', str(STORES / domain / 'natural'),
                '--retrieval', 'full_scan', '--online-weighting', 'z_score',
                '--whitening-lambda', str(RIDGE), '--reweight-alpha', str(ALPHAS[name]),
                '--device', device, '--question-source', str(source), '--top-k', '5', '--output', str(destination)])
            path = destination / 'retrieval_results.json'; report = read(path)
            assert report['memory_unchanged'] and not report['vectors_persisted'] and not report['reader_called']
            assert report['execution_devices'] == {'text_encoder': device}
            assert (report['offline_method'], report['retrieval'], report['ridge_lambda'], report['alpha']) == ('natural', 'full_scan', RIDGE, ALPHAS[name])
            hashes.update(report['source_sha256']); hashes[str(path)] = sha(path)
            for row in report['questions']:
                qid = row['question_id']; old = previous[qid]
                assert qid not in new_results[name]
                for key in ('question', 'options', 'query_image_paths', 'retrieval_query', 'evaluation'):
                    assert row[key] == old[key]
                assert len(row['groups']) == 1 and row['groups'][0]['size'] == report['images']
                assert row['groups'][0]['query'] == row['retrieval_query']
                assert row['conditions'][0]['top5_memory_ids'] == old['conditions'][0]['top5_memory_ids']
                assert [l['memory_id'] for l in row['leaf_scores']] == [l['memory_id'] for l in old['leaf_scores']]
                error = max(abs(a['raw_score']-b['raw_score']) for a, b in zip(row['leaf_scores'], old['leaf_scores']))
                max_raw_error = max(max_raw_error, error); assert error < 1e-12
                white_error = max(abs(a['white_score']-b['white_score']) for a,b in zip(row['leaf_scores'],old['leaf_scores']))
                z_error = max(abs(a['white_z']-b['white_z']) for a,b in zip(row['leaf_scores'],old['leaf_scores']))
                max_white_error = max(max_white_error, white_error); max_z_error = max(max_z_error, z_error)
                assert white_error < 1e-12 and z_error < 1e-12
                for a, b in zip(row['leaf_scores'], old['leaf_scores']):
                    assert abs(a['correction'] - ALPHAS[name] * b['correction']) < 1e-14
                new_results[name][qid] = row
            print(json.dumps({'stage': 'retrieval_domain_done', 'lambda': RIDGE, 'alpha': ALPHAS[name],
                              'domain': domain, 'completed': len(new_results[name]), 'total': 371}), flush=True)
        assert set(new_results[name]) == set(previous)
    rows = []
    for q in original:
        qid = q['id']; old = previous[qid]
        conditions = {}; weighting = {'alpha_1p0': {'groups': old['groups'], 'leaf_scores': old['leaf_scores']}}
        for c in old['conditions']:
            name = OLD[c['name']]; conditions[name] = {**c, 'name': name,
                'lambda': RIDGE if ALPHAS[name] is not None else None, 'alpha': ALPHAS[name]}
        for name in NEW:
            r = new_results[name][qid]
            conditions[name] = {**r['conditions'][1], 'name': name, 'lambda': RIDGE, 'alpha': ALPHAS[name]}
            weighting[name] = {'groups': r['groups'], 'leaf_scores': r['leaf_scores']}
        for key in ('question', 'options', 'query_image_paths'): assert old[key] == q[key]
        assert old['evaluation'] == {key: q[key] for key in ('gold_ids', 'gold_choice')}
        rows.append({**{k: old[k] for k in ('question_id', 'domain', 'question', 'options', 'query_image_paths', 'retrieval_query', 'evaluation')},
                     'conditions': [conditions[name] for name in ALPHAS], 'weighting': weighting})
    annotations = {}
    old_annotations = read(REFERENCE / 'reader/summary.json')['method_annotations']
    for name, alpha in ALPHAS.items():
        annotation = copy.deepcopy(old_annotations[CONTROL if alpha is None else 'baseline_question_only_zscore'])
        annotation['scope'] = name + '；同 domain 全量记忆组成一个簇。'
        if alpha is not None: annotation['3. 检索方式'].update(alpha=alpha, **{'lambda': RIDGE})
        if name in NEW:
            annotation['5. 回答方式'].update(reader_called=False, measured_qa_accuracy=None, correct=None,
                note='检索已执行；新条件需完成本次 reader 后才报告准确率。')
        else:
            annotation['5. 回答方式']['note'] = '来自已完成实验的核验结果；本轮作为参照列。'
        annotation['result_origin'] = 'current_run' if name in NEW else str(BASELINE if name == CONTROL else REFERENCE)
        annotations[name] = annotation
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'variant': 'baseline_question_only: dialogue + full caption',
        'questions': 371, 'domains': 8, 'memory_leaves': 438, 'new_alphas': [.1, .01, 10.], 'reference_alpha': 1.,
        'lambda': RIDGE, 'top_k': 5, 'device': device, 'only_intended_change': 'Alpha magnitude of online z-score correction; ridge lambda fixed at .1.',
        'score_formula': 'raw + alpha * std_pop(raw over all same-domain leaves) * tanh(z_white(lambda=.1))',
        'query_policy': 'Original retrieval_query, including cached query-image caption; no option expansion.',
        'memory_policy': 'Original dialogue plus full caption; one whole-domain group per question.',
        'encoding_policy': 'Actual online pipeline: fresh per-question BGE encoding, discard vectors, read-only memory.',
        'reference_answers_source': {CONTROL: str(BASELINE / 'reader/responses.jsonl'), 'alpha_1p0': str(REFERENCE / 'reader/responses.jsonl')},
        'reader_policy': 'New alpha=.1/.01/10 conditions each run all 371 questions; only identical current-run payloads share inference.',
        'reference_entrypoint_change': REFERENCE_ENTRYPOINT_CHANGE,
        'reference_comment_change': REFERENCE_COMMENT_CHANGE,
        'execution_source_snapshot': read(SNAPSHOT / 'snapshot_manifest.json'),
        'reference_gpu_indices': {name: p['placement']['gpu_index'] for name, p in reference_readers()[2].items()},
        'method_annotations': annotations, 'source_sha256': hashes}
    check_sources(protocol); dump(base / 'protocol.json', protocol); jsonl(base / 'retrieval_results.jsonl', rows)
    dump(base / 'retrieval_summary.json', {'method_annotations': annotations, 'overall': aggregate(rows),
        'by_domain': {d: aggregate([r for r in rows if r['domain'] == d]) for d in sorted({r['domain'] for r in rows})}})
    dump(base / 'retrieval_validation.json', {'passed': True, 'raw_top5_matches_reference': LOGICAL_RECORDS,
        'max_raw_score_error_vs_reference': max_raw_error, 'unchanged_queries_pool_and_texts': True,
        'max_white_score_error_vs_alpha1': max_white_error, 'max_z_error_vs_alpha1': max_z_error,
        'correction_changes_only_by_alpha_factor': True,
        'single_whole_domain_group': True, 'vectors_persisted': False})


def prepare(base, device):
    method = read(base / 'protocol.json'); check_sources(method)
    old_manifest, _, old_protocols = reference_readers(); previous = old_protocols['baseline_question_only_zscore']
    original = {q['id']: q for q in lines(ORIGINAL / 'qa_manifest.jsonl')}
    memories = {r['id']: r['reader_text'] for r in lines(ORIGINAL / 'retrieval_memories.jsonl')}
    settings = previous['reader_settings']; manifest = []; image_records = {}; matches = 0
    reverse_old = {v: k for k, v in OLD.items()}
    for qi, row in enumerate(lines(base / 'retrieval_results.jsonl')):
        q = original[row['question_id']]; prepared = []
        for path in q['query_image_paths']: image_records.setdefault(path, {'path': path, 'sha256': sha(path)})
        images = [image_records[p] for p in q['query_image_paths']]
        for c in row['conditions']:
            name = c['name']; ids = c['top5_memory_ids']
            assert len(ids) == len(set(ids)) == 5 and all(mid.split('/', 1)[0] == q['task'] for mid in ids)
            for leaf in c['top5']: assert leaf['reader_text'] == memories[leaf['memory_id']]
            text = reader_text(q, ids, memories)
            payload = {'system': settings['reader_system_prompt'], 'user_text': text, 'query_images': images}
            if name not in NEW:
                old = old_manifest[q['id'] + '::' + reverse_old[name]]
                assert old['payload_sha256'] == digest(payload) and old['top_memory_ids'] == ids
                matches += 1; continue
            prepared.append({**{k: q[k] for k in ('id', 'task', 'question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths')},
                'record_id': q['id'] + '::' + name, 'condition': name, 'top_memory_ids': ids,
                'top_query_scores': [leaf['score'] for leaf in c['top5']], 'reader_text': text,
                'query_images': images, 'payload_sha256': digest(payload)})
        offset = qi % len(NEW)
        manifest.extend(prepared[offset:] + prepared[:offset])
    assert len(manifest) == LOGICAL_RECORDS and matches == 742
    output = base / 'reader'; output.mkdir(); jsonl(output / 'qa_manifest.jsonl', manifest)
    for path, expected in previous['model_source_sha256'].items(): assert sha(path) == expected
    for weight in previous['weight_files']: assert sha(weight['path']) == weight['sha256']
    paths = [base / 'protocol.json', base / 'retrieval_results.jsonl', output / 'qa_manifest.jsonl', Path(__file__),
        ROOT / 'scripts/evaluation/paired_single_gpu_reader.py', ROOT / 'scripts/analysis/evaluate_adaptive_changed_qa.py',
        ROOT / 'scripts/analysis/evaluate_all_master_answer_reader.py', REFERENCE / 'reader/protocol.json',
        REFERENCE / 'reader/qa_manifest.jsonl', REFERENCE / 'reader/responses.jsonl', ORIGINAL / 'qa_manifest.jsonl',
        ORIGINAL / 'retrieval_memories.jsonl']
    paths += [BASELINE / p for p in ('reader/protocol.json', 'reader/qa_manifest.jsonl', 'reader/responses.jsonl')]
    assert device.startswith('cuda:')
    protocol = {'design': 'Full-scan alpha ablation: three fresh conditions plus attributed raw/alpha=1 references.',
        'experiment_run_id': str(uuid.uuid4()), 'condition_names': list(NEW), 'questions': 371, 'logical_conditions': LOGICAL_RECORDS,
        'unique_reader_payloads': len({r['payload_sha256'] for r in manifest}), 'reader_settings': settings,
        'placement': {'gpu_index': int(device.split(':')[1]), 'minimum_free_gib': 24}, 'attention_implementation': 'sdpa',
        'query_images': list(image_records.values()), 'historical_response_reuse': False,
        'deduplication': 'Identical full payloads share one fresh inference only within this new run.',
        'reference_answer_policy': 'Raw and alpha=1 columns attributed to verified previous runs; no old answers reused for new conditions.',
        'source_sha256': {str(p): sha(p) for p in paths}, 'model_source_sha256': previous['model_source_sha256'],
        'weight_files': previous['weight_files'], 'method_annotations': method['method_annotations']}
    protocol['experiment_signature'] = digest(protocol); dump(output / 'protocol.json', protocol)
    dump(output / 'validation_input.json', {'passed': True, 'new_records': LOGICAL_RECORDS, 'reference_payload_matches': matches,
        'weights_sha256_verified': True, 'native_image_records': sum(bool(r['query_image_paths']) for r in manifest)})
    print(json.dumps({'stage': 'reader_prepared', 'records': LOGICAL_RECORDS, 'unique_payloads': protocol['unique_reader_payloads']}), flush=True)


def audit(output):
    """Reconstruct scalar ranks, covariance inverse, prompts, and token decoding."""
    protocol = read(output / 'protocol.json'); verify(protocol)
    signed = dict(protocol); signature = signed.pop('experiment_signature'); assert digest(signed) == signature
    method = read(output.parent / 'protocol.json'); check_sources(method)
    original = {q['id']: q for q in lines(ORIGINAL / 'qa_manifest.jsonl')}
    memories = {r['id']: r['reader_text'] for r in lines(ORIGINAL / 'retrieval_memories.jsonl')}
    retrieved = lines(output.parent / 'retrieval_results.jsonl')
    assert len(retrieved) == len(original) == 371 and {r['question_id'] for r in retrieved} == set(original)
    manifest = {r['record_id']: r for r in lines(output / 'qa_manifest.jsonl')}
    responses = lines(output / 'responses.jsonl'); indexed = {r['record_id']: r for r in responses}
    assert len(manifest) == len(responses) == len(indexed) == LOGICAL_RECORDS and set(manifest) == set(indexed)
    assert Counter(r['condition'] for r in responses) == Counter(dict.fromkeys(NEW, 371))
    previous_manifest, previous_responses, previous_protocols = reference_readers()
    assert all(protocol['reader_settings'] == p['reader_settings'] for p in previous_protocols.values())
    answers = {qid: {} for qid in original}; source_manifest = {}
    for record_id, r in previous_responses.items():
        source = previous_manifest[record_id]; q = original[r['question_id']]
        assert r['experiment_signature'] == previous_protocols[r['condition']]['experiment_signature']
        assert r['payload_sha256'] == source['payload_sha256'] and r['gold_choice'] == q['gold_choice']
        assert r['correct'] == (r['predicted_choice'] == q['gold_choice'])
        name = OLD[r['condition']]; answers[q['id']][name] = r; source_manifest[(q['id'], name)] = source
    for source in manifest.values(): source_manifest[(source['id'], source['condition'])] = source

    # Independent matrix inverse on pinned historical features; never used for
    # the online run. CPU/GPU encoding differences allow a 2e-6 score tolerance.
    inverse = {}; feature_hashes = {}; source_leaves = {}; source_queries = {}
    pinned = read(HISTORICAL / 'protocol.json')['source_sha256']
    for domain in sorted({q['task'] for q in original.values()}):
        directory = INPUTS / domain; data = read(directory / 'inputs.json'); path = directory / 'features.npz'
        feature_hashes[str(path)] = sha(path); assert feature_hashes[str(path)] == pinned[str(path)]
        with np.load(path) as features:
            x, q = [features[k].astype(np.float64) for k in ('bge_leaf', 'bge_query')]
        x /= np.linalg.norm(x, axis=1, keepdims=True); q /= np.linalg.norm(q, axis=1, keepdims=True)
        centered = x - x.mean(0); covariance = centered.T @ centered / (len(x)-1)
        raw = q @ x.T
        # Alpha does not affect whitening; independently solve once per domain.
        white = (x @ np.linalg.solve(covariance + RIDGE*np.eye(x.shape[1]), RIDGE*q.T)).T
        whites = {name: white for name, alpha in ALPHAS.items() if alpha is not None}
        for i, question in enumerate(data['questions']):
            qid = question['id']; inverse[qid] = (raw[i], {name: w[i] for name, w in whites.items()})
            source_queries[qid] = question['retrieval_query']
        source_leaves[domain] = data['leaves']
    hits = Counter(); max_errors = Counter(); score_count = 0
    for row in retrieved:
        qid = row['question_id']; q = original[qid]
        assert row['retrieval_query'] == source_queries[qid]
        assert row['evaluation'] == {k: q[k] for k in ('gold_ids', 'gold_choice')}
        rankings = {}
        for name, record in row['weighting'].items():
            leaves = record['leaf_scores']; groups = record['groups']; assert len(groups) == 1
            group = groups[0]
            assert group['group_id'] == 'all_memory' and group['size'] == len(leaves)
            assert group['query'] == row['retrieval_query']
            assert [l['memory_id'] for l in leaves] == [l['memory_id'] for l in source_leaves[q['task']]]
            raw, white, z, correction, weighted = [np.array([l[k] for l in leaves]) for k in
                ('raw_score', 'white_score', 'white_z', 'correction', 'weighted_score')]
            for label, actual, expected in (('raw', raw, inverse[qid][0]), (name, white, inverse[qid][1][name])):
                error = float(np.max(abs(actual-expected))); max_errors[label] = max(max_errors[label], error)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
            np.testing.assert_allclose(z, (white-white.mean())/white.std(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(correction, ALPHAS[name]*raw.std()*np.tanh(z), rtol=0, atol=1e-14)
            np.testing.assert_allclose(weighted, raw+correction, rtol=0, atol=1e-15)
            rankings[name] = [leaves[i]['memory_id'] for i in np.argsort(-weighted, kind='stable')[:5]]
            rankings[CONTROL] = [leaves[i]['memory_id'] for i in np.argsort(-raw, kind='stable')[:5]]
            score_count += len(leaves)
        for c in row['conditions']:
            name = c['name']; ids = rankings[name]; source = source_manifest[(qid, name)]
            assert ids == c['top5_memory_ids'] == source['top_memory_ids'] and len(set(ids)) == 5
            assert c['gold_hits_top5'] == len(set(ids) & set(q['gold_ids'])); hits[name] += c['gold_hits_top5']
            text = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
            text += '\n\n'.join(f'Memory {i}: {memories[mid]}' for i, mid in enumerate(ids, 1))
            text += '\n\nQuestion:\n' + q['question'] + '\n\nOptions:\n'
            text += '\n'.join(k+'. '+q['options'][k] for k in 'ABCD') + '\n\nAnswer with one letter only.'
            images = [{'path': p, 'sha256': sha(p)} for p in q['query_image_paths']]
            payload = {'system': protocol['reader_settings']['reader_system_prompt'], 'user_text': text, 'query_images': images}
            assert text == source['reader_text'] and images == source['query_images']
            assert digest(payload) == source['payload_sha256']
            for key in ('question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths'): assert source[key] == q[key]
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(protocol['reader_settings']['reader_model_path'], local_files_only=True)
    fresh = {}; shared = 0; comparisons = 0; disagreements = []
    for r in responses:
        source = manifest[r['record_id']]; qid = r['question_id']; q = original[qid]
        assert r['run_id'] == protocol['experiment_run_id'] and r['experiment_signature'] == signature
        assert r['payload_sha256'] == source['payload_sha256'] and not r['historical_response_reuse']
        raw = processor.batch_decode([r['generated_token_ids']], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw, re.I)
        assert raw == r['raw_response'] and match and r['status'] == 'ok' and not r['hit_token_limit']
        choice = match.group(1).upper()
        assert choice == r['predicted_choice'] and r['gold_choice'] == q['gold_choice']
        assert r['correct'] == (choice == q['gold_choice']) and r['batch_size'] == 1
        assert r['query_image_count'] == len(source['query_image_paths'])
        content = [{'type': 'image'} for _ in source['query_image_paths']] + [{'type': 'text', 'text': source['reader_text']}]
        rendered = processor.apply_chat_template([{'role': 'system', 'content': protocol['reader_settings']['reader_system_prompt']},
            {'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        assert hashlib.sha256(rendered.encode()).hexdigest() == r['rendered_prompt_sha256']
        if r['fresh_inference_executed_for_this_record']:
            assert r['payload_sha256'] not in fresh; fresh[r['payload_sha256']] = r
        else:
            origin = indexed[r['same_run_shared_inference_record_id']]; assert origin['fresh_inference_executed_for_this_record']
            for key in ('payload_sha256', 'generated_token_ids', 'raw_response', 'execution_id', 'rendered_prompt_sha256'):
                assert r[key] == origin[key]
            shared += 1
        for name in OLD.values():
            old = answers[qid][name]
            if r['payload_sha256'] == old['payload_sha256']:
                comparisons += 1
                if r['predicted_choice'] != old['predicted_choice']: disagreements.append({'record_id': r['record_id'], 'reference': name})
        answers[qid][r['condition']] = r
    assert all(set(a) == set(ALPHAS) for a in answers.values())
    assert sum(a[CONTROL]['correct'] for a in answers.values()) == 184
    assert sum(a['alpha_1p0']['correct'] for a in answers.values()) == 183
    assert len(fresh) == protocol['unique_reader_payloads'] and len(fresh)+shared == LOGICAL_RECORDS
    environment = read(output / 'generation_environment.json')
    assert environment['parameter_devices'] == [method['device']] and environment['dtype'] == 'torch.bfloat16'
    assert environment['processor_size'] == {'shortest_edge': 65536, 'longest_edge': protocol['reader_settings']['query_image_max_pixels']}
    assert not list(output.parent.rglob('*.npy')) and not list(output.parent.rglob('*.npz'))
    checks = {'passed': True, 'questions': 371, 'new_records': LOGICAL_RECORDS, 'candidate_scores_checked': score_count,
        'covariance_inverse_whitening_verified': True, 'max_frozen_cpu_vs_online_gpu_score_errors': dict(max_errors),
        'audit_feature_sha256': feature_hashes, 'z_correction_rank_payload_and_image_hashes_verified': True,
        'new_response_token_decodings_and_rendered_prompts_verified': LOGICAL_RECORDS,
        'correct': {name: sum(a[name]['correct'] for a in answers.values()) for name in ALPHAS},
        'gold_hits_top5': dict(hits), 'fresh_inferences': len(fresh), 'same_run_shared_records': shared,
        'native_image_records': sum(bool(r['query_image_paths']) for r in manifest.values()),
        'identical_payload_comparisons_vs_reference': comparisons, 'identical_payload_answer_disagreements': disagreements,
        'historical_response_reuse_for_new_conditions': False, 'format_errors': 0, 'token_limit_hits': 0}
    dump(output / 'independent_audit.json', checks)
    return retrieved, answers, checks


def analyze(output):
    retrieved, answers, checks = audit(output)
    method = read(output.parent / 'protocol.json'); annotations = copy.deepcopy(method['method_annotations'])
    for name in NEW:
        count = checks['correct'][name]
        annotations[name]['5. 回答方式'].update(reader_called=True, measured_qa_accuracy=count/371, correct=count,
            note='本轮已实际执行；仅同轮相同完整输入共享新推理。')
    summary = {'method_annotations': annotations, 'overall': aggregate(retrieved, answers),
        'by_domain': {d: aggregate([r for r in retrieved if r['domain'] == d], answers) for d in sorted({r['domain'] for r in retrieved})},
        'device': method['device'], 'lambda': RIDGE, 'alphas': [.01, .1, 1., 10.], 'reference_answers_source': method['reference_answers_source'],
        'new_logical_records': LOGICAL_RECORDS, 'fresh_inferences': checks['fresh_inferences'], 'same_run_shared_records': checks['same_run_shared_records'],
        'identical_payload_answer_disagreements': checks['identical_payload_answer_disagreements'],
        'format_errors': checks['format_errors'], 'token_limit_hits': checks['token_limit_hits'],
        'interpretation': 'Fixed-option local Qwen, descriptive test-set alpha ablation; no held-out best-alpha claim.'}
    pairs = []
    for row in retrieved:
        qid = row['question_id']; pair = {'question_id': qid, 'domain': row['domain'], 'gold_choice': row['evaluation']['gold_choice']}
        for c in row['conditions']:
            name = c['name']; a = answers[qid][name]
            pair.update({name+'_answer': a['predicted_choice'], name+'_correct': a['correct'],
                name+'_top5_ids': c['top5_memory_ids'], name+'_gold_hits': c['gold_hits_top5']})
        pairs.append(pair)
    dump(output / 'summary.json', summary); dump(output / 'paired_results.json', pairs); csv_write(output / 'paired_results.csv', pairs)
    dump(output / 'validation_output.json', checks); render(output.parent, summary)
    print(json.dumps({'stage': 'complete', 'overall': summary['overall'],
                      'reference_disagreements': checks['identical_payload_answer_disagreements']}, ensure_ascii=False), flush=True)


def render(base, summary):
    overall = summary['overall']
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>全量检索 α 消融</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#20333d}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>',
        '<h1>baseline_question_only：α=0.01 / 0.1 / 1 / 10，固定 λ=0.1</h1>',
        '<p>8 个 domain、371 题；每个 domain 的全量记忆作为一个簇。原始查询、对话＋完整 caption、BGE 编码及 Qwen 回答配置保持一致，固定 λ=0.1，仅改变在线辅助修正的 α。</p>',
        '<p>final = raw + α × std_pop(raw in domain) × tanh(z_white(λ=0.1))。白化分数和 z-score 保持一致，仅缩放加回原始匹配分数的修正量。</p>',
        '<table><tr><th>方案</th><th>答对</th><th>准确率</th><th>Top-5 目标召回</th><th>至少命中一条目标</th><th>回答来源</th></tr>']
    for name in ALPHAS:
        c = overall['conditions'][name]; source = '本轮新评测' if name in NEW else '已验证的上一轮结果'
        parts.append(f'<tr><td>{LABELS[name]}</td><td>{c["correct"]}/371</td><td>{c["accuracy"]:.4%}</td><td>{c["gold_hits_top5"]}/1041 ({c["gold_recall"]:.2%})</td><td>{c["questions_with_any_gold"]}/371</td><td>{source}</td></tr>')
    parts.append('</table>')
    for anchor in (CONTROL, 'alpha_1p0'):
        parts.append(f'<h2>相对{LABELS[anchor]}</h2><table><tr><th>方案</th><th>错→对</th><th>对→错</th><th>净变化</th><th>Top-5 顺序变化</th><th>Top-5 集合变化</th><th>McNemar 双侧 p</th></tr>')
        for name, c in overall['comparisons']['versus_'+anchor].items():
            parts.append(f'<tr><td>{LABELS[name]}</td><td>{c["wrong_to_right"]}</td><td>{c["right_to_wrong"]}</td><td>{c["net_correct"]:+d}</td><td>{c["changed_top5_order"]}</td><td>{c["changed_top5_set"]}</td><td>{c["mcnemar_exact_two_sided_p"]:.4f}</td></tr>')
        parts.append('</table>')
    parts.append('<h2>分 domain 答对数量</h2><table><tr><th>Domain</th><th>题数</th>'+''.join('<th>'+LABELS[n]+'</th>' for n in ALPHAS)+'</tr>')
    for domain, stats in summary['by_domain'].items():
        parts.append(f'<tr><td>{html.escape(domain)}</td><td>{stats["questions"]}</td>'+''.join(f'<td>{stats["conditions"][n]["correct"]}</td>' for n in ALPHAS)+'</tr>')
    parts.append('</table><h2>五项方法标注</h2>')
    for name, annotation in summary['method_annotations'].items():
        parts.append('<details><summary>'+LABELS[name]+'</summary><pre>'+html.escape(json.dumps(annotation, ensure_ascii=False, indent=2))+'</pre></details>')
    parts.append(f'<p>三个新 α 完整评测 1113 条记录，{summary["fresh_inferences"]} 次新推理、{summary["same_run_shared_records"]} 条同轮输入共享；{summary["device"]}、BF16、SDPA、seed=42、batch=1。新设置不复用历史答案。与上一轮完整输入相同时，答案不同 {len(summary["identical_payload_answer_disagreements"])} 条。</p>')
    parts.append('<p>配对检验 p 值未作多重比较校正。这是同一测试集上的描述性消融，不代表选出的参数具有独立验证的泛化优势。采用固定选项排列的本地 Qwen 评测。</p>')
    parts.append('<p><a href="reader/summary.json">完整统计</a> · <a href="reader/paired_results.csv">逐题对照</a> · <a href="reader/independent_audit.json">独立核验</a> · <a href="protocol.json">协议</a></p></html>')
    (base / 'report.html').write_text('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['retrieve', 'prepare', 'run', 'analyze', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=BASE); parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args(); base = args.output.resolve()
    activate_source_snapshot()
    if args.stage in ('retrieve', 'all'): retrieve(base, args.device)
    if args.stage in ('prepare', 'all'): prepare(base, args.device)
    if args.stage in ('run', 'all'): run_reader(base / 'reader', analyze, resume=args.resume)
    if args.stage == 'analyze': analyze(base / 'reader')


if __name__ == '__main__': main()
