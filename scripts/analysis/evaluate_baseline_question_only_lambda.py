"""Full-scan MemEye lambda ablation: fresh .01/1 QA, frozen raw/.1 references."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import html
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

BASE = ROOT / 'reports/baseline_question_only_lambda_ablation_20261023'
REFERENCE = ROOT / 'reports/baseline_question_only_zscore_20261023'
HISTORICAL = ROOT / 'reports/all_master_answer_query_20261021'
INPUTS = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
STORES = ROOT / 'results/memory_stores_20261023'
ORIGINAL = ROOT / 'data/independent_visual_inputs'
CONTROL = 'baseline_question_only'
LAMBDAS = {CONTROL: None, 'lambda_0p01': .01, 'lambda_0p1': .1, 'lambda_1p0': 1.}
NEW = ('lambda_0p01', 'lambda_1p0')
OLD = {CONTROL: CONTROL, 'baseline_question_only_zscore': 'lambda_0p1'}
LABELS = {CONTROL: '原始基线（未赋权）', 'lambda_0p01': 'λ=0.01', 'lambda_0p1': 'λ=0.1', 'lambda_1p0': 'λ=1.0'}


def aggregate(retrieved, answers=None):
    n = len(retrieved); total = sum(len(r['evaluation']['gold_ids']) for r in retrieved)
    result = {'questions': n, 'gold_references': total, 'conditions': {}, 'comparisons': {}}
    for name, ridge in LAMBDAS.items():
        selected = [next(c for c in r['conditions'] if c['name'] == name) for r in retrieved]
        hits = sum(c['gold_hits_top5'] for c in selected)
        stats = {'lambda': ridge, 'gold_hits_top5': hits, 'gold_recall': hits / total,
                 'questions_with_any_gold': sum(c['gold_hits_top5'] > 0 for c in selected),
                 'top1_gold': sum(c['top5'][0]['is_gold'] for c in selected),
                 'answer_source': 'fresh_current_run' if name in NEW else 'verified_previous_run'}
        if answers is not None:
            stats.update(correct=sum(answers[r['question_id']][name]['correct'] for r in retrieved))
            stats['accuracy'] = stats['correct'] / n
        result['conditions'][name] = stats
    for anchor in (CONTROL, 'lambda_0p1'):
        comparison = {}
        for name in LAMBDAS:
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
    prior = read(REFERENCE / 'protocol.json'); check_sources(prior)
    prior_reader = read(REFERENCE / 'reader/protocol.json'); verify(prior_reader)
    signed = dict(prior_reader); signature = signed.pop('experiment_signature'); assert digest(signed) == signature
    original = lines(ORIGINAL / 'qa_manifest.jsonl')
    previous = {r['question_id']: r for r in lines(REFERENCE / 'retrieval_results.jsonl')}
    assert len(original) == len(previous) == 371 and set(previous) == {q['id'] for q in original}
    base.mkdir(parents=True)
    paths = [Path(__file__), ROOT / 'scripts/analysis/evaluate_baseline_question_only_zscore.py',
             ORIGINAL / 'qa_manifest.jsonl', ORIGINAL / 'retrieval_memories.jsonl']
    paths += [REFERENCE / p for p in ('protocol.json', 'retrieval_results.jsonl', 'reader/protocol.json',
                                     'reader/qa_manifest.jsonl', 'reader/responses.jsonl', 'reader/summary.json')]
    hashes = {str(p): sha(p) for p in paths}
    new_results = {name: {} for name in NEW}; max_raw_error = 0.
    for name in NEW:
        for domain in sorted({q['task'] for q in original}):
            source = INPUTS / domain / 'inputs.json'
            assert sha(source) == prior['source_sha256'][str(source)]
            destination = base / 'retrieval' / name / domain
            visual_memory_pipeline.main(['online', '--memory-path', str(STORES / domain / 'natural'),
                '--retrieval', 'full_scan', '--online-weighting', 'z_score',
                '--whitening-lambda', str(LAMBDAS[name]), '--reweight-alpha', '1',
                '--device', device, '--question-source', str(source), '--top-k', '5', '--output', str(destination)])
            path = destination / 'retrieval_results.json'; report = read(path)
            assert report['memory_unchanged'] and not report['vectors_persisted'] and not report['reader_called']
            assert report['execution_devices'] == {'text_encoder': device}
            assert (report['offline_method'], report['retrieval'], report['ridge_lambda'], report['alpha']) == ('natural', 'full_scan', LAMBDAS[name], 1.)
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
                new_results[name][qid] = row
            print(json.dumps({'stage': 'retrieval_domain_done', 'lambda': LAMBDAS[name],
                              'domain': domain, 'completed': len(new_results[name]), 'total': 371}), flush=True)
        assert set(new_results[name]) == set(previous)
    rows = []
    for q in original:
        qid = q['id']; old = previous[qid]
        conditions = {}; weighting = {'lambda_0p1': {'groups': old['groups'], 'leaf_scores': old['leaf_scores']}}
        for c in old['conditions']:
            name = OLD[c['name']]; conditions[name] = {**c, 'name': name, 'lambda': LAMBDAS[name]}
        for name in NEW:
            r = new_results[name][qid]
            conditions[name] = {**r['conditions'][1], 'name': name, 'lambda': LAMBDAS[name]}
            weighting[name] = {'groups': r['groups'], 'leaf_scores': r['leaf_scores']}
        for key in ('question', 'options', 'query_image_paths'): assert old[key] == q[key]
        assert old['evaluation'] == {key: q[key] for key in ('gold_ids', 'gold_choice')}
        rows.append({**{k: old[k] for k in ('question_id', 'domain', 'question', 'options', 'query_image_paths', 'retrieval_query', 'evaluation')},
                     'conditions': [conditions[name] for name in LAMBDAS], 'weighting': weighting})
    annotations = {}
    old_annotations = read(REFERENCE / 'reader/summary.json')['method_annotations']
    for name, ridge in LAMBDAS.items():
        annotation = copy.deepcopy(old_annotations[CONTROL if ridge is None else 'baseline_question_only_zscore'])
        annotation['scope'] = name + '；同 domain 全量记忆组成一个簇。'
        if ridge is not None: annotation['3. 检索方式']['lambda'] = ridge
        if name in NEW:
            annotation['5. 回答方式'].update(reader_called=False, measured_qa_accuracy=None, correct=None,
                note='检索已执行；新条件需完成本次 reader 后才报告准确率。')
        annotation['result_origin'] = 'current_run' if name in NEW else str(REFERENCE)
        annotations[name] = annotation
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'variant': 'baseline_question_only: dialogue + full caption',
        'questions': 371, 'domains': 8, 'memory_leaves': 438, 'new_lambdas': [.01, 1.], 'reference_lambda': .1,
        'alpha': 1., 'top_k': 5, 'device': device, 'only_intended_change': 'Ridge lambda in online z-score weighting.',
        'score_formula': 'raw + std_pop(raw over all same-domain leaves) * tanh(z_white(lambda))',
        'query_policy': 'Original retrieval_query, including cached query-image caption; no option expansion.',
        'memory_policy': 'Original dialogue plus full caption; one whole-domain group per question.',
        'encoding_policy': 'Actual online pipeline: fresh per-question BGE encoding, discard vectors, read-only memory.',
        'reference_answers_source': str(REFERENCE / 'reader/responses.jsonl'),
        'reader_policy': 'New .01 and 1.0 conditions run all 371 questions; only identical current-run payloads share inference.',
        'method_annotations': annotations, 'source_sha256': hashes}
    check_sources(protocol); dump(base / 'protocol.json', protocol); jsonl(base / 'retrieval_results.jsonl', rows)
    dump(base / 'retrieval_summary.json', {'method_annotations': annotations, 'overall': aggregate(rows),
        'by_domain': {d: aggregate([r for r in rows if r['domain'] == d]) for d in sorted({r['domain'] for r in rows})}})
    dump(base / 'retrieval_validation.json', {'passed': True, 'raw_top5_matches_reference': 742,
        'max_raw_score_error_vs_reference': max_raw_error, 'unchanged_queries_pool_and_texts': True,
        'single_whole_domain_group': True, 'vectors_persisted': False})


def prepare(base, device):
    method = read(base / 'protocol.json'); check_sources(method)
    previous = read(REFERENCE / 'reader/protocol.json'); verify(previous)
    original = {q['id']: q for q in lines(ORIGINAL / 'qa_manifest.jsonl')}
    memories = {r['id']: r['reader_text'] for r in lines(ORIGINAL / 'retrieval_memories.jsonl')}
    old_manifest = {r['record_id']: r for r in lines(REFERENCE / 'reader/qa_manifest.jsonl')}
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
        manifest.extend(prepared if qi % 2 == 0 else list(reversed(prepared)))
    assert len(manifest) == matches == 742
    output = base / 'reader'; output.mkdir(); jsonl(output / 'qa_manifest.jsonl', manifest)
    for path, expected in previous['model_source_sha256'].items(): assert sha(path) == expected
    for weight in previous['weight_files']: assert sha(weight['path']) == weight['sha256']
    paths = [base / 'protocol.json', base / 'retrieval_results.jsonl', output / 'qa_manifest.jsonl', Path(__file__),
        ROOT / 'scripts/evaluation/paired_single_gpu_reader.py', ROOT / 'scripts/analysis/evaluate_adaptive_changed_qa.py',
        ROOT / 'scripts/analysis/evaluate_all_master_answer_reader.py', REFERENCE / 'reader/protocol.json',
        REFERENCE / 'reader/qa_manifest.jsonl', REFERENCE / 'reader/responses.jsonl', ORIGINAL / 'qa_manifest.jsonl',
        ORIGINAL / 'retrieval_memories.jsonl']
    assert device.startswith('cuda:')
    protocol = {'design': 'Full-scan lambda ablation: two fresh conditions plus attributed raw/.1 references.',
        'experiment_run_id': str(uuid.uuid4()), 'condition_names': list(NEW), 'questions': 371, 'logical_conditions': 742,
        'unique_reader_payloads': len({r['payload_sha256'] for r in manifest}), 'reader_settings': settings,
        'placement': {'gpu_index': int(device.split(':')[1]), 'minimum_free_gib': 24}, 'attention_implementation': 'sdpa',
        'query_images': list(image_records.values()), 'historical_response_reuse': False,
        'deduplication': 'Identical full payloads share one fresh inference only within this new run.',
        'reference_answer_policy': 'Raw and .1 columns attributed to verified previous run; no old answers reused for new conditions.',
        'source_sha256': {str(p): sha(p) for p in paths}, 'model_source_sha256': previous['model_source_sha256'],
        'weight_files': previous['weight_files'], 'method_annotations': method['method_annotations']}
    protocol['experiment_signature'] = digest(protocol); dump(output / 'protocol.json', protocol)
    dump(output / 'validation_input.json', {'passed': True, 'new_records': 742, 'reference_payload_matches': matches,
        'weights_sha256_verified': True, 'native_image_records': sum(bool(r['query_image_paths']) for r in manifest)})
    print(json.dumps({'stage': 'reader_prepared', 'records': 742, 'unique_payloads': protocol['unique_reader_payloads']}), flush=True)


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
    assert len(manifest) == len(responses) == len(indexed) == 742 and set(manifest) == set(indexed)
    assert Counter(r['condition'] for r in responses) == Counter(dict.fromkeys(NEW, 371))
    previous_manifest = {r['record_id']: r for r in lines(REFERENCE / 'reader/qa_manifest.jsonl')}
    previous_responses = {r['record_id']: r for r in lines(REFERENCE / 'reader/responses.jsonl')}
    previous_protocol = read(REFERENCE / 'reader/protocol.json'); verify(previous_protocol)
    assert protocol['reader_settings'] == previous_protocol['reader_settings']
    answers = {qid: {} for qid in original}; source_manifest = {}
    for record_id, r in previous_responses.items():
        source = previous_manifest[record_id]; q = original[r['question_id']]
        assert r['experiment_signature'] == previous_protocol['experiment_signature']
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
        whites = {name: (x @ np.linalg.solve(covariance + ridge*np.eye(x.shape[1]), ridge*q.T)).T
                  for name, ridge in LAMBDAS.items() if ridge is not None}
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
            np.testing.assert_allclose(correction, raw.std()*np.tanh(z), rtol=0, atol=1e-14)
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
    assert all(set(a) == set(LAMBDAS) for a in answers.values())
    assert sum(a[CONTROL]['correct'] for a in answers.values()) == 184
    assert sum(a['lambda_0p1']['correct'] for a in answers.values()) == 183
    assert len(fresh) == protocol['unique_reader_payloads'] and len(fresh)+shared == 742
    environment = read(output / 'generation_environment.json')
    assert environment['parameter_devices'] == [method['device']] and environment['dtype'] == 'torch.bfloat16'
    assert environment['processor_size'] == {'shortest_edge': 65536, 'longest_edge': protocol['reader_settings']['query_image_max_pixels']}
    assert not list(output.parent.rglob('*.npy')) and not list(output.parent.rglob('*.npz'))
    checks = {'passed': True, 'questions': 371, 'new_records': 742, 'candidate_scores_checked': score_count,
        'covariance_inverse_whitening_verified': True, 'max_frozen_cpu_vs_online_gpu_score_errors': dict(max_errors),
        'audit_feature_sha256': feature_hashes, 'z_correction_rank_payload_and_image_hashes_verified': True,
        'new_response_token_decodings_and_rendered_prompts_verified': 742,
        'correct': {name: sum(a[name]['correct'] for a in answers.values()) for name in LAMBDAS},
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
        'alpha': 1., 'lambdas': [.01, .1, 1.], 'reference_answers_source': str(REFERENCE / 'reader/responses.jsonl'),
        'new_logical_records': 742, 'fresh_inferences': checks['fresh_inferences'], 'same_run_shared_records': checks['same_run_shared_records'],
        'identical_payload_answer_disagreements': checks['identical_payload_answer_disagreements'],
        'format_errors': checks['format_errors'], 'token_limit_hits': checks['token_limit_hits'],
        'interpretation': 'Fixed-option local Qwen, descriptive test-set lambda ablation; no held-out best-lambda claim.'}
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
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>全量检索 λ 消融</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#20333d}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>',
        '<h1>baseline_question_only：λ=0.01 / 0.1 / 1.0，固定 α=1</h1>',
        '<p>8 个 domain、371 题；每个 domain 的全量记忆作为一个簇。原始查询、对话＋完整 caption、BGE 编码及 Qwen 回答配置保持一致，仅改变在线白化的 λ。</p>',
        '<p>final = raw + std_pop(raw in domain) × tanh(z_white(λ))。增大 λ 减弱白化；α=1 仍有辅助修正，λ=1.0 不等于关闭赋权。</p>',
        '<table><tr><th>方案</th><th>答对</th><th>准确率</th><th>Top-5 目标召回</th><th>至少命中一条目标</th><th>回答来源</th></tr>']
    for name in LAMBDAS:
        c = overall['conditions'][name]; source = '本轮新评测' if name in NEW else '已验证的上一轮结果'
        parts.append(f'<tr><td>{LABELS[name]}</td><td>{c["correct"]}/371</td><td>{c["accuracy"]:.4%}</td><td>{c["gold_hits_top5"]}/1041 ({c["gold_recall"]:.2%})</td><td>{c["questions_with_any_gold"]}/371</td><td>{source}</td></tr>')
    parts.append('</table>')
    for anchor in (CONTROL, 'lambda_0p1'):
        parts.append(f'<h2>相对{LABELS[anchor]}</h2><table><tr><th>方案</th><th>错→对</th><th>对→错</th><th>净变化</th><th>Top-5 顺序变化</th><th>Top-5 集合变化</th><th>McNemar 双侧 p</th></tr>')
        for name, c in overall['comparisons']['versus_'+anchor].items():
            parts.append(f'<tr><td>{LABELS[name]}</td><td>{c["wrong_to_right"]}</td><td>{c["right_to_wrong"]}</td><td>{c["net_correct"]:+d}</td><td>{c["changed_top5_order"]}</td><td>{c["changed_top5_set"]}</td><td>{c["mcnemar_exact_two_sided_p"]:.4f}</td></tr>')
        parts.append('</table>')
    parts.append('<h2>分 domain 答对数量</h2><table><tr><th>Domain</th><th>题数</th>'+''.join('<th>'+LABELS[n]+'</th>' for n in LAMBDAS)+'</tr>')
    for domain, stats in summary['by_domain'].items():
        parts.append(f'<tr><td>{html.escape(domain)}</td><td>{stats["questions"]}</td>'+''.join(f'<td>{stats["conditions"][n]["correct"]}</td>' for n in LAMBDAS)+'</tr>')
    parts.append('</table><h2>五项方法标注</h2>')
    for name, annotation in summary['method_annotations'].items():
        parts.append('<details><summary>'+LABELS[name]+'</summary><pre>'+html.escape(json.dumps(annotation, ensure_ascii=False, indent=2))+'</pre></details>')
    parts.append(f'<p>两个新 λ 完整评测 742 条记录，{summary["fresh_inferences"]} 次新推理、{summary["same_run_shared_records"]} 条同轮输入共享；GPU 1、BF16、SDPA、seed=42、batch=1。新设置不复用历史答案。与上一轮完整输入相同时，答案不同 {len(summary["identical_payload_answer_disagreements"])} 条。</p>')
    parts.append('<p>这是同一测试集上的描述性消融，不代表选出的参数具有独立验证的泛化优势。采用固定选项排列的本地 Qwen 评测。</p>')
    parts.append('<p><a href="reader/summary.json">完整统计</a> · <a href="reader/paired_results.csv">逐题对照</a> · <a href="reader/independent_audit.json">独立核验</a> · <a href="protocol.json">协议</a></p></html>')
    (base / 'report.html').write_text('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['retrieve', 'prepare', 'run', 'analyze', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=BASE); parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args(); base = args.output.resolve()
    if args.stage in ('retrieve', 'all'): retrieve(base, args.device)
    if args.stage in ('prepare', 'all'): prepare(base, args.device)
    if args.stage in ('run', 'all'): run_reader(base / 'reader', analyze, resume=args.resume)
    if args.stage == 'analyze': analyze(base / 'reader')


if __name__ == '__main__': main()
