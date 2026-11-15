"""Full MemEye paired QA: original master-answer query versus online ridge-z correction.

Only alpha changes: one group per saved master, lambda=.1, alpha=.01, .1, 1 or 10.
Both conditions receive fresh Qwen inference, sharing identical payloads only
within this run. Historical 195/371 is verified separately, never reused.
"""
from __future__ import annotations
import argparse
import copy
import csv
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
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
from scripts.analysis.evaluate_all_master_answer_reader import verify
from scripts.evaluation.paired_single_gpu_reader import run as run_reader

BASE = ROOT / 'reports/master_answer_query_lambda0p1_alpha_ablation_20261024'
ALPHA = .1
INPUTS = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
STORES = ROOT / 'results/memory_stores_20261023'
ORIGINAL = ROOT / 'data/independent_visual_inputs'
REFERENCE = ROOT / 'reports/all_master_answer_query_20261021'
CANDIDATES = REFERENCE / 'candidates.jsonl'
ALPHA1_REFERENCE = ROOT / 'reports/master_answer_query_lambda_ablation_20261023/lambda_0p1'
REFERENCE_TREATMENT = 'master_answer_query_zscore_lambda0p1'
CONTROL, TREATMENT = 'master_answer_query', 'master_answer_query_zscore_lambda0p1_alpha0p1'
CONDITIONS = (CONTROL, TREATMENT)


def jsonl(path, rows):
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def csv_write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def check_sources(protocol):
    for path, expected in protocol['source_sha256'].items():
        assert sha(path) == expected, path


def aggregate(retrieved, answers=None):
    n = len(retrieved); total = sum(len(r['evaluation']['gold_ids']) for r in retrieved)
    result = {'questions': n, 'gold_references': total, 'conditions': {}}
    for name in CONDITIONS:
        selected = [next(c for c in r['conditions'] if c['name'] == name) for r in retrieved]
        hits = sum(c['gold_hits_top5'] for c in selected)
        stats = {'gold_hits_top5': hits, 'gold_recall': hits / total,
                 'questions_with_any_gold': sum(c['gold_hits_top5'] > 0 for c in selected),
                 'top1_gold': sum(c['top5'][0]['is_gold'] for c in selected)}
        if answers is not None:
            stats['correct'] = sum(answers[r['question_id']][name]['correct'] for r in retrieved)
            stats['accuracy'] = stats['correct'] / n
        result['conditions'][name] = stats
    same_order = same_set = 0; wins = []; losses = []
    for row in retrieved:
        a, b = row['conditions']; assert (a['name'], b['name']) == CONDITIONS
        same_order += a['top5_memory_ids'] == b['top5_memory_ids']
        same_set += set(a['top5_memory_ids']) == set(b['top5_memory_ids'])
        if answers is not None:
            a, b = [answers[row['question_id']][name] for name in CONDITIONS]
            if b['correct'] and not a['correct']: wins.append(row['question_id'])
            if a['correct'] and not b['correct']: losses.append(row['question_id'])
    result['comparison'] = {'same_top5_order': same_order, 'same_top5_set': same_set,
                            'changed_top5_order': n - same_order, 'changed_top5_set': n - same_set}
    if answers is not None:
        m = len(wins) + len(losses)
        p = min(1., 2 * sum(math.comb(m, i) for i in range(min(len(wins), len(losses)) + 1)) / 2**m) if m else 1.
        result['comparison'].update(wrong_to_right=len(wins), right_to_wrong=len(losses),
            net_correct=len(wins) - len(losses), delta_percentage_points=100 * (len(wins) - len(losses)) / n,
            win_question_ids=wins, loss_question_ids=losses, mcnemar_exact_two_sided_p=p)
    return result


def retrieve(base, device):
    if base.exists(): raise FileExistsError('Use a fresh output directory.')
    check_sources(read(ALPHA1_REFERENCE / 'protocol.json'))
    verify(read(ALPHA1_REFERENCE / 'reader/protocol.json'))
    alpha1 = {r['question_id']: r for r in lines(ALPHA1_REFERENCE / 'retrieval_results.jsonl')}
    base.mkdir(parents=True)
    original = lines(ORIGINAL / 'qa_manifest.jsonl')
    reference = {r['question_id']: r for r in lines(REFERENCE / 'retrieval_results.jsonl')}
    assert len(original) == len(reference) == 371
    frozen_sources = read(REFERENCE / 'protocol.json')['source_sha256']
    sources = [Path(__file__), ROOT / 'scripts/analysis/evaluate_master_answer_query_alpha.py', CANDIDATES, ORIGINAL / 'qa_manifest.jsonl', ORIGINAL / 'retrieval_memories.jsonl',
               REFERENCE / 'retrieval_results.jsonl', REFERENCE / 'reader/qa_manifest.jsonl',
               REFERENCE / 'reader/responses.jsonl', REFERENCE / 'reader/protocol.json']
    sources += [ALPHA1_REFERENCE / p for p in ('protocol.json', 'retrieval_results.jsonl', 'reader/protocol.json',
                                              'reader/qa_manifest.jsonl', 'reader/responses.jsonl', 'reader/summary.json')]
    hashes = {str(p): sha(p) for p in sources}
    assert sha(CANDIDATES) == frozen_sources[str(CANDIDATES)]
    by_id = {}; counts = {}; master_counts = {}; descriptions = None; max_error = max_clip_error = 0.; choice_checks = 0
    max_alpha1_error = max_scale_error = 0.
    for domain in sorted({q['task'] for q in original}):
        source = INPUTS / domain / 'inputs.json'
        assert sha(source) == frozen_sources[str(source)]
        destination = base / 'retrieval' / domain
        visual_memory_pipeline.main(['online', '--memory-path', str(STORES / domain / 'caption_visual_master'),
            '--retrieval', 'multi_path', '--online-weighting', 'z_score', '--whitening-lambda', '.1',
            '--reweight-alpha', str(ALPHA), '--candidates', str(CANDIDATES), '--device', device, '--question-source', str(source),
            '--top-k', '5', '--output', str(destination)])
        path = destination / 'retrieval_results.json'; report = read(path)
        assert report['execution_devices'] == {'clip_text_encoder': device, 'text_encoder': device}
        assert report['memory_unchanged'] and not report['reader_called'] and not report['vectors_persisted']
        assert (report['offline_method'], report['retrieval'], report['ridge_lambda'], report['alpha']) == ('caption_visual_master', 'multi_path', .1, ALPHA)
        hashes.update(report['source_sha256']); hashes[str(path)] = sha(path)
        counts[domain] = report['images']; descriptions = report['method_annotations']
        master_counts[domain] = len(report['questions'][0]['groups'])
        for row in report['questions']:
            qid = row['question_id']; ref = reference[qid]
            assert qid not in by_id
            prior = alpha1[qid]
            for key in ('question', 'options', 'query_image_paths', 'retrieval_query', 'evaluation', 'groups'):
                assert row[key] == prior[key]
            assert len(row['leaf_scores']) == len(prior['leaf_scores'])
            for a, b in zip(row['leaf_scores'], prior['leaf_scores']):
                for key in ('memory_id', 'image_id', 'group_id'): assert a[key] == b[key]
                for key in ('raw_score', 'white_score', 'white_z'):
                    error = abs(a[key]-b[key]); max_alpha1_error = max(max_alpha1_error, error)
                    assert error < 1e-12
                error = abs(a['correction'] - ALPHA*b['correction']); max_scale_error = max(max_scale_error, error)
                assert error < 1e-14
            assert row['retrieval_query'] == ref['baseline_retrieval_query']
            ref_masters = {m['master_id']: m for m in ref['masters']}
            assert {g['group_id'] for g in row['groups']} == set(ref_masters)
            for group in row['groups']:
                old = ref_masters[group['group_id']]
                assert group['size'] == old['n']
                assert group['choice'] == old['selected_choice']
                assert group['selected_statement'] == old['selected_statement']
                assert group['query'] == old['augmented_query']
                assert {l['image_id'] for l in row['leaf_scores'] if l['group_id'] == group['group_id']} == set(old['member_ids'])
                max_clip_error = max(max_clip_error, max(abs(group['clip_statement_scores'][k]-old['clip_candidate_scores'][k]) for k in 'ABCD'))
                choice_checks += 1
            assert sum(g['size'] for g in row['groups']) == report['images']
            ref_condition = next(c for c in ref['conditions'] if c['name'] == CONTROL)
            assert row['conditions'][0]['top5_memory_ids'] == ref_condition['top5_memory_ids']
            ref_leaves = {leaf['image_id']: leaf for leaf in ref['leaves']}
            assert set(ref_leaves) == {leaf['image_id'] for leaf in row['leaf_scores']}
            max_error = max(max_error, max(abs(leaf['raw_score'] - ref_leaves[leaf['image_id']]['expanded_score']) for leaf in row['leaf_scores']))
            for c, name in zip(row['conditions'], CONDITIONS): c['name'] = name
            by_id[qid] = {**row, 'domain': domain}
        print(json.dumps({'stage': 'retrieval_domain_done', 'domain': domain, 'total': len(by_id)}), flush=True)
    assert set(by_id) == {q['id'] for q in original} and max_error < 2e-6 and max_clip_error < 2e-6
    assert sum(master_counts.values()) == 88 and choice_checks == 3862
    rows = [by_id[q['id']] for q in original]
    for q, row in zip(original, rows):
        for key in ('question', 'options', 'query_image_paths'): assert q[key] == row[key]
        assert row['evaluation'] == {key: q[key] for key in ('gold_ids', 'gold_choice')}
    annotations = {name: copy.deepcopy(descriptions) for name in CONDITIONS}
    for name, description in annotations.items():
        description['scope'] = name + '；每个已有 master 及其全部叶子为一个簇。'
        description['1. 候选记忆结构'].update(memory_path=str(STORES), candidate_count=sum(counts.values()), per_domain_counts=counts,
            master_count=sum(master_counts.values()), per_domain_master_counts=master_counts)
        description['4. 编码方式']['allowed_leaf_truncations'] = '按各 domain memory.json 固定策略；Card 一条 577-token 记忆截到 512，保持历史基线一致。'
        if name == CONTROL:
            description['3. 检索方式'].update(online_weighting='none', final_score='raw', weighting_note='不做白化或 z-score。', **{'lambda': None, 'alpha': None})
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'conditions': list(CONDITIONS),
        'questions': 371, 'domains': 8, 'memory_leaves': 438, 'masters': 88, 'lambda': .1, 'alpha': ALPHA, 'top_k': 5,
        'only_intended_change': 'Relative to lambda=.1 alpha=1, change only alpha; preserve raw/white/z scores, master queries, groups and reader.',
        'formula': 'raw + alpha * std_pop(raw within each master) * tanh(z_white)',
        'query_policy': 'Unchanged complete candidate statements selected by each saved master; original retrieval_query plus selected statement.',
        'memory_policy': 'Original dialogue + full caption; each saved master forms one group.',
        'reader_policy': 'Fresh paired inference for all 371 questions; identical payloads may share only within this run.',
        'alpha1_reference': str(ALPHA1_REFERENCE),
        'reference_policy': 'Strict current hashes for the completed lambda=.1 alpha=1 reference; preserve all queries, scores and groups.',
        'method_annotations': annotations, 'source_sha256': hashes, 'device': device,
        'historical_source_policy': 'Verify current pinned inputs and exact historical baseline payloads; no dependency on retired historical source paths.'}
    check_sources(protocol)
    jsonl(base / 'retrieval_results.jsonl', rows); dump(base / 'protocol.json', protocol)
    dump(base / 'retrieval_summary.json', {'method_annotations': annotations, 'overall': aggregate(rows),
        'by_domain': {d: aggregate([r for r in rows if r['domain'] == d]) for d in counts}})
    dump(base / 'retrieval_validation.json', {'passed': True, 'baseline_top5_matches_historical': 371,
        'max_baseline_score_error': max_error, 'master_choices_and_expanded_queries_match': choice_checks,
        'max_clip_score_error': max_clip_error, 'master_memberships_unchanged': True,
        'alpha1_raw_white_z_max_difference': max_alpha1_error, 'correction_alpha_times_reference_max_difference': max_scale_error, 'alpha': ALPHA,
        'query_and_leaf_text_unchanged': True, 'all_masters_participate': True, 'no_vectors_persisted': True})


def prepare(base, device):
    method = read(base / 'protocol.json'); check_sources(method)
    assert (method['lambda'], method['alpha']) == (.1, ALPHA)
    previous = read(REFERENCE / 'reader/protocol.json')
    signature = previous.pop('experiment_signature'); assert digest(previous) == signature
    old_manifest = {r['record_id']: r for r in lines(REFERENCE / 'reader/qa_manifest.jsonl')}
    old_answers = {r['record_id']: r for r in lines(REFERENCE / 'reader/responses.jsonl')}
    original = {r['id']: r for r in lines(ORIGINAL / 'qa_manifest.jsonl')}
    memories = {r['id']: r['reader_text'] for r in lines(ORIGINAL / 'retrieval_memories.jsonl')}
    output = base / 'reader'; output.mkdir()
    settings = previous['reader_settings']; manifest = []; image_records = {}; matches = 0; baseline_correct = 0
    assert settings['do_sample'] is False and settings['seed'] == 42 and settings['max_new_tokens'] == 16
    for qi, row in enumerate(lines(base / 'retrieval_results.jsonl')):
        q = original[row['question_id']]; prepared = []
        for p in q['query_image_paths']: image_records.setdefault(p, {'path': p, 'sha256': sha(p)})
        images = [image_records[p] for p in q['query_image_paths']]
        for c in row['conditions']:
            ids = c['top5_memory_ids']; assert len(ids) == len(set(ids)) == 5
            assert all(mid.split('/', 1)[0] == q['task'] for mid in ids)
            for leaf in c['top5']: assert leaf['reader_text'] == memories[leaf['memory_id']]
            text = reader_text(q, ids, memories)
            payload = {'system': settings['reader_system_prompt'], 'user_text': text, 'query_images': images}
            if c['name'] == CONTROL:
                ref = old_manifest[q['id'] + '::' + CONTROL]; answer = old_answers[ref['record_id']]
                assert ref['top_memory_ids'] == ids and ref['reader_text'] == text and ref['payload_sha256'] == digest(payload)
                assert answer['payload_sha256'] == ref['payload_sha256'] and answer['experiment_signature'] == signature
                assert answer['gold_choice'] == q['gold_choice'] and answer['correct'] == (answer['predicted_choice'] == q['gold_choice'])
                matches += 1; baseline_correct += answer['correct']
            prepared.append({**{k: q[k] for k in ('id', 'task', 'question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths')},
                'record_id': q['id'] + '::' + c['name'], 'condition': c['name'], 'top_memory_ids': ids,
                'top_query_scores': [leaf['score'] for leaf in c['top5']], 'reader_text': text,
                'query_images': images, 'payload_sha256': digest(payload)})
        manifest.extend(prepared if qi % 2 == 0 else list(reversed(prepared)))
    assert len(manifest) == 742 and matches == 371 and baseline_correct == 195
    jsonl(output / 'qa_manifest.jsonl', manifest)
    for path, expected in previous['model_source_sha256'].items(): assert sha(path) == expected
    for weight in previous['weight_files']:
        assert sha(weight['path']) == weight['sha256']
    paths = [base / 'protocol.json', base / 'retrieval_results.jsonl', output / 'qa_manifest.jsonl', Path(__file__),
             ROOT / 'scripts/evaluation/paired_single_gpu_reader.py', ROOT / 'scripts/analysis/evaluate_all_master_answer_reader.py',
             ROOT / 'scripts/analysis/evaluate_adaptive_changed_qa.py', ORIGINAL / 'qa_manifest.jsonl', ORIGINAL / 'retrieval_memories.jsonl',
             REFERENCE / 'reader/protocol.json', REFERENCE / 'reader/qa_manifest.jsonl', REFERENCE / 'reader/responses.jsonl']
    assert device.startswith('cuda:')
    protocol = {'design': f'Fresh paired master-answer-query baseline versus lambda=.1 alpha={ALPHA:g} z-score; no historical answers reused.',
        'experiment_run_id': str(uuid.uuid4()), 'condition_names': list(CONDITIONS), 'questions': 371, 'logical_conditions': 742,
        'unique_reader_payloads': len({r['payload_sha256'] for r in manifest}), 'reader_settings': settings,
        'placement': {'gpu_index': int(device.split(':')[1]), 'minimum_free_gib': 24}, 'attention_implementation': 'sdpa',
        'placement_difference_from_alpha1_reference': 'Alpha=1 reference used GPU 1; each fresh run records its selected GPU and verifies all raw baseline answers.',
        'placement_difference_from_historical': 'Both new conditions on one GPU; historical reference used GPU/CPU offload. Verify all baseline answers after fresh inference.',
        'query_images': list(image_records.values()), 'historical_response_reuse': False,
        'deduplication': 'Identical payloads share one fresh inference within this paired run only.',
        'source_sha256': {str(p): sha(p) for p in paths}, 'model_source_sha256': previous['model_source_sha256'],
        'weight_files': previous['weight_files'], 'method_annotations': method['method_annotations']}
    protocol['experiment_signature'] = digest(protocol); dump(output / 'protocol.json', protocol)
    dump(output / 'validation_input.json', {'passed': True, 'records': 742, 'historical_baseline_payload_matches': matches,
        'historical_baseline_correct': baseline_correct, 'model_weights_sha256_verified': True,
        'image_condition_records': sum(bool(r['query_image_paths']) for r in manifest),
        'unique_reader_payloads': protocol['unique_reader_payloads']})
    print(json.dumps({'stage': 'reader_prepared', 'records': 742, 'unique_payloads': protocol['unique_reader_payloads']}), flush=True)


def analyze(output):
    protocol = read(output / 'protocol.json'); verify(protocol)
    signed = dict(protocol); signature = signed.pop('experiment_signature'); assert digest(signed) == signature
    method = read(output.parent / 'protocol.json'); check_sources(method)
    assert (method['lambda'], method['alpha']) == (.1, ALPHA)
    manifest = {r['record_id']: r for r in lines(output / 'qa_manifest.jsonl')}
    responses = lines(output / 'responses.jsonl'); indexed = {r['record_id']: r for r in responses}
    assert len(indexed) == len(responses) == len(manifest) == 742 and set(indexed) == set(manifest)
    answers = {}; fresh = {}; shared = 0
    for r in responses:
        source = manifest[r['record_id']]
        assert r['payload_sha256'] == source['payload_sha256'] and r['experiment_signature'] == signature
        assert r['run_id'] == protocol['experiment_run_id'] and r['gold_choice'] == source['gold_choice']
        assert r['correct'] == (r['predicted_choice'] == source['gold_choice'])
        assert r['historical_response_reuse'] is False and r['batch_size'] == 1
        assert r['query_image_count'] == len(source['query_image_paths'])
        if r['fresh_inference_executed_for_this_record']:
            assert r['payload_sha256'] not in fresh; fresh[r['payload_sha256']] = r
        else:
            prior = indexed[r['same_run_shared_inference_record_id']]; assert prior['fresh_inference_executed_for_this_record']
            for key in ('payload_sha256', 'rendered_prompt_sha256', 'raw_response', 'generated_token_ids', 'execution_id'): assert r[key] == prior[key]
            shared += 1
        answers.setdefault(r['question_id'], {})[r['condition']] = r
    assert len(fresh) == protocol['unique_reader_payloads'] and len(fresh) + shared == 742
    assert len(answers) == 371 and all(set(v) == set(CONDITIONS) for v in answers.values())
    old = {r['question_id']: r for r in lines(REFERENCE / 'reader/responses.jsonl') if r['condition'] == CONTROL}
    disagreements = [qid for qid in answers if answers[qid][CONTROL]['predicted_choice'] != old[qid]['predicted_choice']]
    check_sources(read(ALPHA1_REFERENCE / 'protocol.json'))
    alpha1_protocol = read(ALPHA1_REFERENCE / 'reader/protocol.json'); verify(alpha1_protocol)
    signed_alpha1 = dict(alpha1_protocol); alpha1_signature = signed_alpha1.pop('experiment_signature')
    assert digest(signed_alpha1) == alpha1_signature
    assert alpha1_protocol['reader_settings'] == protocol['reader_settings']
    alpha1_manifest = {r['record_id']: r for r in lines(ALPHA1_REFERENCE / 'reader/qa_manifest.jsonl')}
    alpha1_answers = {r['question_id']: r for r in lines(ALPHA1_REFERENCE / 'reader/responses.jsonl')
                      if r['condition'] == REFERENCE_TREATMENT}
    alpha1_retrieved = {r['question_id']: r for r in lines(ALPHA1_REFERENCE / 'retrieval_results.jsonl')}
    assert len(alpha1_answers) == len(alpha1_retrieved) == 371
    for qid, answer in alpha1_answers.items():
        prior = alpha1_manifest[answer['record_id']]
        assert answer['experiment_signature'] == alpha1_signature
        assert answer['payload_sha256'] == prior['payload_sha256']
        assert answer['gold_choice'] == answers[qid][CONTROL]['gold_choice']
        assert answer['correct'] == (answer['predicted_choice'] == answer['gold_choice'])
    assert sum(a['correct'] for a in alpha1_answers.values()) == 202
    retrieved = lines(output.parent / 'retrieval_results.jsonl'); pairs = []
    alpha1_comparison_rows = []; alpha1_comparison_answers = {}; alpha1_input_matches = 0; alpha1_disagreements = []
    for row in retrieved:
        qid = row['question_id']; a, b = [answers[qid][name] for name in CONDITIONS]
        first, second = row['conditions']
        pairs.append({'question_id': qid, 'domain': row['domain'], 'gold_choice': row['evaluation']['gold_choice'],
            'baseline_answer': a['predicted_choice'], 'zscore_answer': b['predicted_choice'],
            'baseline_correct': a['correct'], 'zscore_correct': b['correct'],
            'baseline_top5': first['top5_memory_ids'], 'zscore_top5': second['top5_memory_ids'],
            'baseline_gold_hits': first['gold_hits_top5'], 'zscore_gold_hits': second['gold_hits_top5'],
            'same_payload': a['payload_sha256'] == b['payload_sha256']})
        prior_condition = alpha1_retrieved[qid]['conditions'][1]; prior_answer = alpha1_answers[qid]
        assert prior_condition['name'] == REFERENCE_TREATMENT
        pairs[-1].update(alpha1_answer=prior_answer['predicted_choice'], alpha1_correct=prior_answer['correct'],
                         alpha1_top5=prior_condition['top5_memory_ids'], alpha1_gold_hits=prior_condition['gold_hits_top5'])
        alpha1_comparison_rows.append({**row, 'conditions': [{**prior_condition, 'name': CONTROL}, second]})
        alpha1_comparison_answers[qid] = {CONTROL: prior_answer, TREATMENT: b}
        if prior_answer['payload_sha256'] == b['payload_sha256']:
            alpha1_input_matches += 1
            if prior_answer['predicted_choice'] != b['predicted_choice']: alpha1_disagreements.append(qid)
    alpha1_comparison = aggregate(alpha1_comparison_rows, alpha1_comparison_answers)
    annotations = copy.deepcopy(method['method_annotations'])
    for name, annotation in annotations.items():
        reference = annotation['5. 回答方式']['reference_reader_only']
        count = sum(answer[name]['correct'] for answer in answers.values())
        annotation['5. 回答方式'] = {**reference, 'note': '本次已实际执行，两个条件均为本轮新推理；仅同轮相同输入共享。',
            'reader_called': True, 'measured_qa_accuracy': count / 371, 'correct': count, 'questions': 371}
    alpha1_annotation = copy.deepcopy(read(ALPHA1_REFERENCE / 'reader/summary.json')['method_annotations'][REFERENCE_TREATMENT])
    alpha1_annotation['scope'] = 'master_answer_query_zscore_alpha1；已完成的 α=1 参照，非本轮新回答。'
    alpha1_annotation['5. 回答方式']['note'] = '来自已完成且输入/输出哈希验证的 α=1 实验。'
    alpha1_annotation['result_origin'] = str(ALPHA1_REFERENCE)
    annotations['master_answer_query_zscore_alpha1'] = alpha1_annotation
    summary = {'method_annotations': annotations, 'overall': aggregate(retrieved, answers),
        'by_domain': {d: aggregate([r for r in retrieved if r['domain'] == d], answers) for d in sorted({r['domain'] for r in retrieved})},
        'lambda': .1, 'alpha': ALPHA, 'device': 'cuda:' + str(protocol['placement']['gpu_index']), 'logical_records': 742, 'fresh_inferences': len(fresh), 'same_run_shared_records': shared,
        'historical_baseline_correct': 195, 'baseline_answer_disagreements_vs_historical': disagreements,
        'alpha1_reference': {**alpha1_comparison['conditions'][CONTROL], 'source': str(ALPHA1_REFERENCE / 'reader/responses.jsonl'),
            'identical_payload_matches': alpha1_input_matches, 'identical_payload_answer_disagreements': alpha1_disagreements},
        'comparison_vs_alpha1': alpha1_comparison['comparison'],
        'alpha1_by_domain': {d: aggregate([r for r in alpha1_comparison_rows if r['domain'] == d], alpha1_comparison_answers)['conditions'][CONTROL]
                            for d in sorted({r['domain'] for r in retrieved})},
        'format_errors': sum(r['status'] != 'ok' for r in responses), 'token_limit_hits': sum(r['hit_token_limit'] for r in responses),
        'interpretation': 'Fixed-option-order local-Qwen accuracy; paired test with only online retrieval correction changed.'}
    dump(output / 'summary.json', summary); dump(output / 'paired_results.json', pairs); csv_write(output / 'paired_results.csv', pairs)
    dump(output / 'validation_output.json', {'passed': True, 'records': 742, 'questions': 371,
        'payloads_signatures_ids_shared_inferences_verified': True, 'fresh_inferences': len(fresh),
        'historical_response_reuse': False, 'baseline_answer_disagreements_vs_historical': disagreements,
        'format_errors': summary['format_errors'], 'token_limit_hits': summary['token_limit_hits']})
    render(output.parent, summary)
    print(json.dumps({'stage': 'complete', 'overall': summary['overall'], 'baseline_disagreements': disagreements}, ensure_ascii=False), flush=True)


def render(base, summary):
    overall = summary['overall']; comparison = overall['comparison']
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>多路径检索加 z-score</title><style>body{font:16px/1.7 system-ui;max-width:1180px;margin:32px auto;padding:0 20px;color:#20333d}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}details{margin:12px 0}</style>',
        f'<h1>master_answer_query ＋在线 z-score：λ=0.1，α={ALPHA:g}</h1><p>8 个 domain、371 题、438 条记忆；每题在各 master 内分别计算白化和 z-score，再合并所有叶子取 Top-5。查询、对话＋完整 caption、回答模型及参数保持一致。</p>',
        f'<p>final = raw + {ALPHA:g} × std_pop(raw in master) × tanh(z_white)。两个条件本轮重新回答；相同完整输入仅在本轮共享一次新推理。</p>',
        '<table><tr><th>方案</th><th>答对</th><th>准确率</th><th>Top-5 目标召回</th><th>至少命中一条目标</th></tr>']
    for label, c in [('原始多路径检索（本轮复现）', overall['conditions'][CONTROL]),
                     ('多路径检索＋z-score（α=1，已有）', summary['alpha1_reference']),
                     (f'多路径检索＋z-score（α={ALPHA:g}，本轮）', overall['conditions'][TREATMENT])]:
        parts.append(f'<tr><td>{label}</td><td>{c["correct"]}/371</td><td>{c["accuracy"]:.2%}</td><td>{c["gold_hits_top5"]}/1041 ({c["gold_recall"]:.2%})</td><td>{c["questions_with_any_gold"]}/371</td></tr>')
    parts.append(f'</table><p>相对未赋权：错→对 {comparison["wrong_to_right"]}，对→错 {comparison["right_to_wrong"]}，净变化 {comparison["net_correct"]:+d} 题（{comparison["delta_percentage_points"]:+.2f} 个百分点）；McNemar 双侧 p={comparison["mcnemar_exact_two_sided_p"]:.4f}。</p>')
    c = summary['comparison_vs_alpha1']
    parts.append(f'<p>相对 α=1：错→对 {c["wrong_to_right"]}，对→错 {c["right_to_wrong"]}，净变化 {c["net_correct"]:+d} 题（{c["delta_percentage_points"]:+.2f} 个百分点）；McNemar 双侧 p={c["mcnemar_exact_two_sided_p"]:.4f}。</p>')
    parts.append(f'<p>Top-5 顺序变化 {comparison["changed_top5_order"]}/371，集合变化 {comparison["changed_top5_set"]}/371。原始基线与历史答案不同 {len(summary["baseline_answer_disagreements_vs_historical"])} 题；格式错误 {summary["format_errors"]}，输出截断 {summary["token_limit_hits"]}。</p>')
    parts.append(f'<h2>各 domain</h2><table><tr><th>Domain</th><th>题数</th><th>未赋权</th><th>α=1（已有）</th><th>α={ALPHA:g}</th><th>相对未赋权</th></tr>')
    for domain, stats in summary['by_domain'].items():
        a, b = [stats['conditions'][name]['correct'] for name in CONDITIONS]
        parts.append(f'<tr><td>{html.escape(domain)}</td><td>{stats["questions"]}</td><td>{a}</td><td>{summary["alpha1_by_domain"][domain]["correct"]}</td><td>{b}</td><td>{b-a:+d}</td></tr>')
    parts.append('</table><h2>五项方法标注</h2>')
    for name, annotation in summary['method_annotations'].items():
        parts.append('<details><summary>'+html.escape(name)+'</summary><pre>'+html.escape(json.dumps(annotation, ensure_ascii=False, indent=2))+'</pre></details>')
    parts.append(f'<p>本轮 {summary["logical_records"]} 条回答记录，{summary["fresh_inferences"]} 次新推理，{summary["same_run_shared_records"]} 条同轮共享。{summary["device"]}、BF16、SDPA、seed=42、batch=1；保留题目原图。历史基线为 GPU/CPU offload，已单独检查基线答案的一致性。</p>')
    parts.append(f'<p>α=1 列来自前次完成的 202/371 实验。本次固定 λ=0.1；原始/白化/z 分数及 master 选择须与 α=1 完全复现，仅将修正量乘以 {ALPHA:g}。参照与当前来源均通过严格哈希核验；实际 GPU 及相对参照的设备变化记录在协议中。本次参数比较是同一测试集上的描述性结果。</p>')
    parts.append('<p>固定选项顺序的本地 Qwen 结果；不是论文四轮旋转 EM。<a href="reader/summary.json">完整统计</a> · <a href="reader/paired_results.csv">逐题结果</a> · <a href="reader/independent_audit.json">独立核验</a> · <a href="protocol.json">协议</a></p></html>')
    (base / 'report.html').write_text('\n'.join(parts))


def main():
    global ALPHA, TREATMENT, CONDITIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha', type=float, choices=[.01, .1, 1., 10.], default=.1)
    parser.add_argument('--stage', choices=['retrieve', 'prepare', 'run', 'analyze', 'all'], default='all')
    parser.add_argument('--output', type=Path); parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    ALPHA = args.alpha
    suffix = format(ALPHA, 'g').replace('.', 'p')
    TREATMENT = 'master_answer_query_zscore_lambda0p1_alpha' + suffix
    CONDITIONS = (CONTROL, TREATMENT)
    base = (args.output or BASE / ('alpha_' + suffix)).resolve()
    if args.stage in ('retrieve', 'all'): retrieve(base, args.device)
    if args.stage in ('prepare', 'all'): prepare(base, args.device)
    if args.stage in ('run', 'all'): run_reader(base / 'reader', analyze, resume=args.resume)
    if args.stage == 'analyze': analyze(base / 'reader')


if __name__ == '__main__': main()
