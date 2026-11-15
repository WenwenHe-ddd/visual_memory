"""Audit a completed paired Mem-Gallery run and write CPU-only result reports.

This utility never generates candidates, retrieves memories, or runs a reader.
It refuses partial runs and verifies stored provenance before computing results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]

import numpy as np

from scripts.evaluation.mem_gallery_master_reader import (
    CONDITIONS, READER_SYSTEM, build_reader_manifest, candidate_user_text,
    digest, parse_candidates, score_answer, summarize_pairs,
)

DEFAULT_OUTPUT = ROOT / 'results/mem_gallery_master_answer_20261021'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def index(rows, key='id'):
    result = {row[key]: row for row in rows}
    require(len(rows) == len(result), f'Duplicate {key}')
    return result


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class HashAudit:
    """Cache repeated file reads while checking every declared digest."""

    def __init__(self):
        self.checked = {}

    def verify(self, path, expected):
        path = Path(path).resolve()
        if path not in self.checked:
            self.checked[path] = sha(path)
        require(self.checked[path] == expected, f'Frozen hash changed: {path}')

    def mapping(self, mapping, base=ROOT):
        for name, expected in mapping.items():
            path = Path(name)
            self.verify(path if path.is_absolute() else base / path, expected)


def audit_shared_responses(responses, manifest, protocol):
    """Reject provenance-only, payload-only, and copied-answer-only reuse errors."""
    rmap, mmap = index(responses, 'record_id'), index(manifest, 'record_id')
    require(set(rmap) == set(mmap), 'Reader coverage is incomplete or contains extra rows')
    fresh = {}
    for rid, row in rmap.items():
        item = mmap[rid]
        require(row['run_id'] == protocol['run_id'] and
                row['experiment_signature'] == protocol['experiment_signature'], 'Reader run mismatch')
        require(row['payload_sha256'] == item['payload_sha256'] and
                row['top5_memory_ids'] == item['top5_memory_ids'] and
                row['condition'] == item['condition'] and row['id'] == item['id'], 'Reader payload mismatch')
        require(row['status'] == ('ok' if row['raw_response'] else 'empty_response'), 'Wrong reader status')
        if row['fresh_inference_executed_for_this_record']:
            require(row['payload_sha256'] not in fresh, 'Duplicate fresh inference for one payload')
            require(row['same_run_shared_inference_record_id'] is None, 'Fresh row marked shared')
            fresh[row['payload_sha256']] = row
    for row in responses:
        if row['fresh_inference_executed_for_this_record']:
            continue
        source = fresh.get(row['payload_sha256'])
        require(source is not None, 'Shared row lacks a fresh source')
        require(row['same_run_shared_inference_record_id'] == source['record_id'], 'Wrong fresh source')
        for key in ('raw_response', 'input_tokens', 'generated_tokens', 'hit_token_limit',
                    'execution_id', 'batch_size', 'batch_elapsed_seconds'):
            require(row[key] == source[key], f'Shared inference differs in {key}')
    require(len(fresh) == protocol['contract']['unique_payloads'], 'Fresh payload count mismatch')
    return {'logical_records': len(responses), 'fresh_inferences': len(fresh),
            'shared_records': len(responses) - len(fresh),
            'execution_ids': sorted({r['execution_id'] for r in responses}),
            'empty_responses': sum(not r['raw_response'] for r in responses),
            'token_limit_records': sum(bool(r['hit_token_limit']) for r in responses),
            'max_input_tokens': max(r['input_tokens'] for r in responses),
            'max_generated_tokens': max(r['generated_tokens'] for r in responses),
            'historical_responses_reused': False}


def paired_bootstrap(rows, repetitions=10000, seed=42):
    require(repetitions >= 100, 'At least 100 bootstrap replicates required')
    delta = np.array([r[CONDITIONS[1]]['f1'] - r[CONDITIONS[0]]['f1'] for r in rows])
    require(len(delta) > 0, 'No paired questions')
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, repetitions, 100):
        count = min(100, repetitions - start)
        samples.extend(delta[rng.integers(len(delta), size=(count, len(delta)))].mean(axis=1))
    domains = sorted({r['task'] for r in rows})
    sums = np.array([sum(d for r, d in zip(rows, delta) if r['task'] == task) for task in domains])
    sizes = np.array([sum(r['task'] == task for r in rows) for task in domains])
    draws = rng.integers(len(domains), size=(repetitions, len(domains)))
    cluster_samples = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {'metric': 'method minus baseline mean token F1', 'delta': float(delta.mean()),
            'replicates': repetitions, 'seed': seed,
            'paired_question_percentile_95': np.quantile(samples, [.025, .975]).tolist(),
            'paired_domain_cluster_percentile_95': np.quantile(cluster_samples, [.025, .975]).tolist(),
            'domain_cluster_policy': 'Sample 20 whole domains with replacement; retain all questions in each sampled domain and compute a question-weighted mean.',
            'interpretation': 'Descriptive uncertainty only. Question bootstrap assumes independent questions; shared domain history violates that assumption. Domain bootstrap retains within-domain dependence but has only 20 domains.'}


def compact_summary(rows):
    result = summarize_pairs(rows)
    for condition in CONDITIONS:
        result[condition]['exact_matches'] = int(sum(r[condition]['em'] for r in rows))
        result[condition]['gold_hits'] = sum(r[condition]['gold_hits'] for r in rows)
    result['gold_references'] = sum(r[CONDITIONS[0]]['gold_clues'] for r in rows)
    result['f1_tied'] = len(rows) - result['f1_improved'] - result['f1_worsened']
    result['em_tied'] = len(rows) - result['em_gained'] - result['em_lost']
    result['delta'] = {key: result[CONDITIONS[1]][key] - result[CONDITIONS[0]][key]
                       if result[CONDITIONS[0]][key] is not None else None
                       for key in ('em', 'f1', 'bleu', 'recall_at_5_micro', 'hit_rate_at_5_nonempty')}
    return result


def batch_seconds(rows, stage):
    """Use repeated per-batch timing only when full batch membership is observed."""
    groups = defaultdict(list)
    selected = [r for r in rows if r.get('fresh_inference_executed_for_this_record', True)]
    for row in selected:
        key = (row.get('execution_id', row.get('run_id')), row.get('attempt'),
               row['batch_elapsed_seconds'], row['batch_size'])
        groups[key].append(row)
    complete = all(len(group) % key[-1] == 0 for key, group in groups.items())
    return {'stage': stage, 'complete_batch_membership': complete,
            'summed_generation_batch_seconds': sum(key[-2] * (len(group) // key[-1])
                                                   for key, group in groups.items()) if complete else None,
            'note': 'Sum of generate() batch timings, inferred from execution/attempt/elapsed/size fields; excludes model loading, tokenization, disk I/O and validation. It is not end-to-end latency or an independent per-condition timing.'}


def audit(root):
    root = Path(root).resolve()
    required = ('reader/summary.json', 'reader/paired_results.json', 'reader/validation_output.json',
                'retrieval_summary.json', 'candidate_validation.json')
    require(all((root / name).is_file() for name in required), 'Report requires a fully completed and analyzed run')
    hashes = HashAudit()
    hashes.mapping(read(root / 'inputs/artifact_manifest.json'), root / 'inputs')
    inputs = read(root / 'inputs/protocol.json')
    hashes.mapping(inputs['source_sha256'])
    memories, questions, gold = [lines(root / 'inputs' / name)
                                 for name in ('memories.jsonl', 'questions.jsonl', 'gold.jsonl')]
    mmap, qmap, gmap = map(index, (memories, questions, gold))
    require(len(memories) == 3962 and sum(r['is_visual'] for r in memories) == 1003, 'Memory coverage differs')
    require(len(questions) == 1711 and set(qmap) == set(gmap), 'QA/gold coverage differs')
    require(sum(bool(q['query_image_paths']) for q in questions) == 487, 'Query image coverage differs')
    require(len({q['task'] for q in questions}) == 20, 'Domain coverage differs')
    require(sum(not r['gold_ids'] for r in gold) == 184 and
            sum(len(set(r['gold_ids'])) for r in gold) == 4109, 'Clue denominator differs')
    require(all(not {'answer', 'clue', 'gold_ids', 'session_id'} & set(q) for q in questions), 'Questions contain gold/cutoff fields')
    offline = read(root / 'offline/protocol.json')
    hashes.mapping(offline['source_sha256'])
    hashes.mapping(offline['artifact_sha256'])
    require(offline['leaf_visual_dimension'] == 2 and not offline['per_image_visual_embeddings_saved'] and
            offline['image_copies'] == offline['old_image_reencodes'] == 0, 'Visual storage policy differs')
    persistent = lines(root / 'offline/memories.jsonl')
    require(persistent == [{k: v for k, v in r.items() if k not in ('image_path', 'image_sha256')}
                           for r in memories], 'Persistent store differs from input text')
    masters, approximate = 0, 0
    for domain in offline['domain_summaries']:
        folder = root / 'offline/memory' / domain['task']
        graph = read(folder / 'graph.json')
        nodes = index(graph['image_nodes'])
        visual = {r['node_id']: r for r in memories if r['task'] == domain['task'] and r['is_visual']}
        require(set(nodes) == set(visual), 'Graph leaf coverage differs')
        require(all(set(n) == {'id', 'caption', 'dialogue', 'timestamp', 'visual_coords', 'master_id'}
                    and len(n['visual_coords']) == 2 for n in nodes.values()), 'Leaf stores more than two coordinates')
        for nid, n in nodes.items():
            require(all(n[k] == visual[nid][k] for k in ('caption', 'dialogue', 'timestamp')), 'Leaf text changed')
        memberships = [nid for m in graph['masters'] for nid in m['member_ids']]
        require(len(memberships) == len(nodes) and set(memberships) == set(nodes), 'Masters do not partition leaves')
        sums = np.load(folder / 'master_sums.npy', allow_pickle=False)
        require(sums.shape == (len(graph['masters']), 512) and np.isfinite(sums).all(), 'Invalid master vectors')
        masters += len(graph['masters'])
        approximate += sum(m['approximate'] for m in graph['masters'])
    require(masters == 240, 'Unexpected master count')

    cp = read(root / 'candidate_generation_protocol.json')
    rp = read(root / 'reader/protocol.json')
    for protocol in (cp, rp):
        require(digest(protocol['contract']) == protocol['experiment_signature'], 'Protocol signature differs')
        hashes.mapping(protocol['contract']['source_sha256'])
        model = protocol['contract']['model']
        for filename, metadata in model['files'].items():
            path = Path(model['path']) / filename
            require(path.stat().st_size == metadata['bytes'], 'Model file size changed')
            hashes.verify(path, metadata['sha256'])
    require(cp['contract']['model'] == rp['contract']['model'], 'Candidate and reader models differ')
    for protocol in (cp, rp):
        require(protocol['contract']['device'] == 'cuda:0' and
                protocol['contract']['seed'] == 42 and protocol['contract']['do_sample'] is False,
                'GPU or generation protocol differs')
    require(cp['contract']['gold_or_clues_or_history_used'] is False and
            cp['contract']['input_fields'] == ['id', 'question', 'query_image_caption'], 'Candidate input policy differs')
    for key in ('candidate_expansion_in_reader', 'gold_loaded_before_analysis', 'historical_response_reuse', 'input_truncation'):
        require(rp['contract'][key] is False, f'Reader policy differs: {key}')
    require(rp['contract']['system_prompt'] == READER_SYSTEM, 'Reader prompt changed')
    candidate_inputs = index(lines(root / 'inputs/sanitized_candidate_inputs.jsonl'))
    candidates = index(lines(root / 'candidates.jsonl'))
    attempts = lines(root / 'candidate_attempts.jsonl')
    validation = read(root / 'candidate_validation.json')
    hashes.verify(root / 'candidates.jsonl', validation['candidates_sha256'])
    require(validation['passed'] and validation['questions'] == 1711 and
            set(candidate_inputs) == set(candidates) == set(qmap), 'Candidate coverage mismatch')
    from transformers import AutoTokenizer
    from scripts.evaluation.mem_gallery_master_reader import CLIP
    tokenizer = AutoTokenizer.from_pretrained(str(CLIP), local_files_only=True)
    hashes.mapping(cp['contract']['clip_tokenizer_sha256'], CLIP)
    repair_path = root / 'candidate_format_repair.json'
    repaired = {}
    if repair_path.exists():
        repair = read(repair_path)
        require(repair['no_new_generation'] and repair['decoded_strings_unchanged'], 'Format repair changed generation/content')
        require(repair['status'] == 'resolved' and repair['unresolved_ids'] == [] and
                repair['gold_or_history_used'] is False, 'Format repair is unresolved or used labels')
        hashes.verify(root / 'candidate_attempts.jsonl', repair['input_attempts_sha256'])
        hashes.verify(root / 'candidate_failures.json', repair['initial_failure_file_sha256'])
        hashes.verify(root / 'candidate_generation_protocol.json', repair['original_protocol_sha256'])
        hashes.verify(root / 'inputs/sanitized_candidate_inputs.jsonl', repair['sanitized_input_sha256'])
        hashes.verify(ROOT / 'scripts/data/normalize_mem_gallery_candidate_format.py', repair['repair_script_sha256'])
        hashes.verify(root / 'candidates.jsonl', repair['candidates_after_sha256'])
        repaired = index(repair['repaired_records'])
        require(len(repaired) == 3, 'Only the three documented format repairs are allowed')
        require(set(repaired) == set(repair['initial_failure_ids']) == set(read(root / 'candidate_failures.json')['ids']),
                'Format repair did not cover the exact failure set')
    accepted = defaultdict(list)
    for row in attempts:
        require(row['run_id'] == cp['run_id'] and row['experiment_signature'] == cp['experiment_signature'], 'Candidate attempt provenance differs')
        if row['validation_error'] is None:
            accepted[row['id']].append(row)
    for qid, row in candidates.items():
        source = candidate_inputs[qid]
        require(set(source) == {'id', 'question', 'query_image_caption'}, 'Candidate source contains extra fields')
        require(row['source_input_sha256'] == digest(source) and row['gold_or_history_used'] is False,
                'Candidate source isolation mismatch')
        require(row['run_id'] == cp['run_id'] and row['experiment_signature'] == cp['experiment_signature'], 'Candidate signature differs')
        if qid in repaired:
            from scripts.data.normalize_mem_gallery_candidate_format import parse_four_singleton_lists
            evidence = repaired[qid]
            require(row['response_format'] == 'four_json_singleton_lists' and row['attempt'] == 3,
                    'Unexpected repaired candidate format/attempt')
            matches = [a for a in attempts if a['id'] == qid and digest(a) == evidence['attempt_record_sha256']]
            require(len(matches) == 1, 'Repaired candidate lacks one original generation')
            raw = matches[0]
            require(hashlib.sha256(raw['raw_response'].encode()).hexdigest() == evidence['raw_response_sha256'],
                    'Repaired raw response hash differs')
            decoded = parse_four_singleton_lists(raw['raw_response'])
            values, lengths = parse_candidates(json.dumps(decoded), tokenizer)
            require(values == decoded == evidence['candidate_statements'] and lengths == evidence['clip_token_lengths'],
                    'Format repair changed decoded strings or lengths')
            require(all(evidence[k] == row[k] == raw[k] for k in ('id', 'attempt', 'run_id', 'experiment_signature', 'prompt_sha256')),
                    'Repaired candidate provenance differs')
            provenance = row.get('format_repair_provenance', {})
            require(all(provenance.get(k) == repair[k] for k in ('input_attempts_sha256', 'repair_script_sha256')) and
                    all(provenance.get(k) == evidence[k] for k in ('attempt_record_sha256', 'raw_response_sha256')) and
                    provenance.get('no_new_generation') is True and provenance.get('decoded_strings_unchanged') is True and
                    provenance.get('gold_or_history_used') is False, 'Repaired candidate lacks matching repair provenance')
            form = 'four_json_singleton_lists'
        else:
            matches = [a for a in accepted[qid] if a['attempt'] == row['attempt'] and a['prompt_sha256'] == row['prompt_sha256']]
            require(len(matches) == 1, 'Candidate lacks one accepted raw generation')
            raw = matches[0]
            values, lengths, form = parse_candidates(raw['raw_response'], tokenizer, return_format=True)
        require(values == row['candidate_statements'] and lengths == row['clip_token_lengths'] and
                form == row['response_format'] and not raw['hit_token_limit'], 'Candidate differs from generated response')
        require(row['prompt_sha256'] == digest({'system': cp['contract']['system_prompt'],
                    'user_text': candidate_user_text(source, row['attempt'])}), 'Candidate prompt hash differs')

    retrieval = lines(root / 'retrieval_results.jsonl')
    retrieval_map = index(retrieval)
    retrieval_protocol = read(root / 'retrieval_protocol.json')
    hashes.mapping(retrieval_protocol['source_sha256'])
    hashes.verify(root / 'retrieval_results.jsonl', retrieval_protocol['results_sha256'])
    require(set(retrieval_map) == set(qmap) and retrieval_protocol['top_k'] == 5 and
            retrieval_protocol['gold_used_for_ranking'] is False, 'Retrieval coverage/policy differs')
    retrieval_counts = Counter()
    for row in retrieval:
        pool = {r['id'] for r in memories if r['task'] == row['task']}
        require(row['candidate_count'] == len(pool), 'Retrieval pool size differs')
        conditions = index(row['conditions'], 'name')
        require(set(conditions) == set(CONDITIONS), 'Wrong retrieval conditions')
        for condition in conditions.values():
            ranking = condition['ranking_memory_ids']
            require(len(ranking) == len(pool) and set(ranking) == pool and
                    condition['top5_memory_ids'] == ranking[:5], 'Full domain ranking or Top5 differs')
        pure_rankings = [[mid for mid in conditions[c]['ranking_memory_ids'] if not mmap[mid]['is_visual']]
                        for c in CONDITIONS]
        require(pure_rankings[0] == pure_rankings[1], 'Pure text relative ranking changed')
        retrieval_counts['pure_text_relative_order_unchanged_questions'] += 1
        targets = set(gmap[row['id']]['gold_ids'])
        for prefix, condition in zip(('baseline', 'method'), CONDITIONS):
            top = set(conditions[condition]['top5_memory_ids'])
            retrieval_counts[f'{prefix}_visual_hits'] += sum(mmap[mid]['is_visual'] for mid in top & targets)
            retrieval_counts[f'{prefix}_text_hits'] += sum(not mmap[mid]['is_visual'] for mid in top & targets)
            retrieval_counts[f'{prefix}_image_slots'] += sum(mmap[mid]['is_visual'] for mid in top)
    diagnostic_path = root / 'diagnostics/retrieval_audit.json'
    diagnostic_sha = sha(diagnostic_path)
    diagnostic = read(diagnostic_path)
    hashes.verify(diagnostic_path, diagnostic_sha)
    hashes.mapping(diagnostic['checked_sha256'])
    require(diagnostic['status'] == 'passed' and all(diagnostic['checks'].values()), 'Independent retrieval audit failed')
    for key, value in retrieval_counts.items():
        actual = (diagnostic['text_rank_displacement'][key] if key.startswith('pure_text')
                  else diagnostic['overall'][key])
        require(actual == value, f'Independent retrieval audit disagrees: {key}')
    budget = diagnostic['bge_token_budget']
    hashes.mapping(budget['tokenizer_file_sha256'])
    require(budget['status'] == 'passed' and budget['max_seq_length'] == 512 and
            budget['total_inputs_checked'] == 13520 and budget['total_over_limit'] == 0 and
            budget['tokenization_truncation'] is False, 'BGE token audit differs')
    require(sum(g['count'] for g in budget['groups'].values()) == budget['total_inputs_checked'] and
            all(g['max_tokens'] <= 512 and g['over_limit'] == 0 for g in budget['groups'].values()),
            'BGE token counts are inconsistent')
    manifest = lines(root / 'reader/qa_manifest.jsonl')
    require(manifest == build_reader_manifest(questions, memories, retrieval), 'Reader payloads cannot be reproduced')
    require(digest(manifest) == rp['contract']['manifest_sha256'], 'Manifest digest differs')
    responses = lines(root / 'reader/responses.jsonl')
    response_audit = audit_shared_responses(responses, manifest, rp)
    require(response_audit['logical_records'] == 3422, 'Expected 1711 x 2 reader records')
    rv = read(root / 'reader/validation_output.json')
    hashes.verify(root / 'reader/responses.jsonl', rv['responses_sha256'])
    for key in ('logical_records', 'fresh_inferences', 'shared_records', 'empty_responses', 'token_limit_records'):
        require(rv[key] == response_audit[key], f'Reader validation mismatch: {key}')
    summary = read(root / 'reader/summary.json')
    require(summary['status'] == 'complete', 'Reader summary is incomplete')
    hashes.mapping(summary['artifacts_sha256'])
    pairs = read(root / 'reader/paired_results.json')
    require(set(index(pairs)) == set(qmap), 'Paired result coverage differs')
    response_map = index(responses, 'record_id')
    for row in pairs:
        qid = row['id']
        require(row['answer'] == gmap[qid]['answer'] and row['gold_ids'] == gmap[qid]['gold_ids'], 'Gold changed in paired results')
        require(row['task'] == qmap[qid]['task'] and row['point'] == qmap[qid]['point'], 'Paired category differs')
        targets = set(row['gold_ids'])
        visual = {mid for mid in targets if mmap[mid]['is_visual']}
        row['cohort'] = ('no_clue' if not targets else 'mixed' if visual and targets - visual
                         else 'visual_only' if visual else 'text_only')
        row['query_has_image'] = bool(qmap[qid]['query_image_paths'])
        for condition in CONDITIONS:
            stored = row[condition]
            response = response_map[qid + '::' + condition]
            require(stored['raw_response'] == response['raw_response'], 'Scored answer changed')
            require(all(stored[k] == response[k] for k in ('payload_sha256', 'top5_memory_ids', 'hit_token_limit')),
                    'Paired response provenance differs')
            for key, value in score_answer(response['raw_response'], row['answer']).items():
                require(abs(stored[key] - value) < 1e-12, f'Scoring differs: {qid}/{key}')
            require(stored['gold_hits'] == len(targets & set(response['top5_memory_ids'])) and
                    stored['gold_clues'] == len(targets), 'Paired recall numerator differs')
        require(row['same_reader_payload'] == (row[CONDITIONS[0]]['payload_sha256'] ==
                                               row[CONDITIONS[1]]['payload_sha256']), 'Paired payload equality differs')
    require(summarize_pairs(pairs) == summary['overall'], 'Reader summary not reproducible')
    audit_result = {'passed': True, 'frozen_files_verified': len(hashes.checked), **response_audit,
        'questions': 1711, 'domains': 20, 'memory_rounds': 3962, 'visual_rounds': 1003,
        'text_only_rounds': 2959, 'query_image_questions': 487, 'clue_questions': 1527,
        'no_clue_questions': 184, 'gold_references': 4109, 'masters': masters,
        'approximate_masters': approximate, 'leaf_visual_dimensions': 2,
        'candidate_attempt_records': len(attempts),
        'candidate_invalid_attempts': sum(r['validation_error'] is not None for r in attempts),
        'candidate_token_limit_attempts': sum(r['hit_token_limit'] for r in attempts),
        'candidate_format_repairs_without_new_generation': len(repaired),
        'final_candidate_formats': dict(Counter(r['response_format'] for r in candidates.values())),
        'final_candidate_attempts': dict(Counter(r['attempt'] for r in candidates.values())),
        'pilot_diagnostics_excluded_from_final_counts': str(root / 'diagnostics/candidate_format_pilot'),
        'input_policy': inputs['policy'],
        'retrieval_diagnostics': {'path': str(diagnostic_path), 'sha256': diagnostic_sha,
            'overall': diagnostic['overall'],
            'text_rank_displacement': {k: v for k, v in diagnostic['text_rank_displacement'].items()
                                       if k != 'lost_text_target_examples'},
            'bge_token_budget': budget}}
    timings = {'offline_seconds': offline['elapsed_seconds'],
               'retrieval_seconds': retrieval_protocol['elapsed_seconds'],
               'candidate_generation': batch_seconds(attempts, 'candidate_generation'),
               'reader_generation': batch_seconds(responses, 'reader_generation'),
               'comparison_limit': 'Shared vector encoding and paired-payload inference are amortized; these timings cannot establish a baseline-versus-method latency improvement.'}
    return pairs, audit_result, timings


def percentage(value, signed=False):
    return '—' if value is None else f'{value * 100:+.2f}' if signed else f'{value * 100:.2f}'


def write_report(root, repetitions=10000):
    root = Path(root).resolve()
    pairs, validation, timings = audit(root)
    overall = compact_summary(pairs)
    report = {'status': 'complete', 'audit': validation, 'overall': overall,
              'f1_uncertainty': paired_bootstrap(pairs, repetitions), 'timings': timings,
              'metric_policy': 'Official deterministic normalized EM, stemmed token F1, smoothed sentence BLEU-4. Open-ended answers; no LLM judge or MCQ accuracy.',
              'method_policy': 'Four question-only generated hypotheses; each visual master selects one by CLIP alignment; append it to that master\'s leaf retrieval queries. Text-only leaves retain original-query scores. Global Top5 over the full same-domain pool. Hypotheses never enter reader evidence.'}
    for field, key in (('task', 'by_domain'), ('point', 'by_question_type'), ('cohort', 'by_cohort'),
                       ('query_has_image', 'by_query_image')):
        report[key] = {str(value): compact_summary([r for r in pairs if r[field] == value])
                       for value in sorted({r[field] for r in pairs})}
    report['source_sha256'] = {str(path): sha(path) for path in [
        root / 'reader/summary.json', root / 'reader/paired_results.json', root / 'retrieval_summary.json',
        root / 'reader/protocol.json', root / 'retrieval_protocol.json', root / 'candidate_validation.json',
        root / 'diagnostics/retrieval_audit.json', Path(__file__)]}
    destination = root / 'report'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    fields = ['domain', 'questions', 'baseline_em', 'method_em', 'baseline_f1', 'method_f1', 'delta_f1',
              'baseline_bleu', 'method_bleu', 'baseline_recall5', 'method_recall5', 'baseline_any_hit', 'method_any_hit']
    with (destination / 'by_domain.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for domain, group in report['by_domain'].items():
            base, method = [group[c] for c in CONDITIONS]
            writer.writerow({'domain': domain, 'questions': group['questions'], 'delta_f1': group['delta']['f1'],
                **{f'{prefix}_{name}': row[key] for prefix, row in [('baseline', base), ('method', method)]
                   for name, key in [('em', 'em'), ('f1', 'f1'), ('bleu', 'bleu'),
                                     ('recall5', 'recall_at_5_micro'), ('any_hit', 'hit_rate_at_5_nonempty')]}})
    base, method = [overall[c] for c in CONDITIONS]
    text = ['# Mem-Gallery：全量检索基线与视觉 master 候选扩展', '',
        '完整覆盖 20 个领域、1,711 道开放式 QA。两种方法均在该领域的全部历史记忆中检索 Top5；'
        '候选池共 3,962 条记忆，其中 1,003 条含图、2,959 条纯文本。', '',
        '| 指标 | 全量检索基线 | 视觉 master 方法 | 差值（百分点） |', '|---|---:|---:|---:|']
    for label, key in [('Normalized EM', 'em'), ('Token F1', 'f1'), ('BLEU-4', 'bleu'),
                       ('Gold micro Recall@5', 'recall_at_5_micro'), ('Any-hit@5', 'hit_rate_at_5_nonempty')]:
        text.append(f'| {label} | {percentage(base[key])} | {percentage(method[key])} | {percentage(overall["delta"][key], True)} |')
    text.extend(['', 'EM/F1/BLEU 均使用全部 1,711 道题。检索指标只在 1,527 道有 clue 的题上计算，'
                 'micro Recall 分母为 4,109 条目标记忆引用；184 道无 clue 题仍计入回答指标。'
                 'BLEU 为逐题平滑 BLEU-4 的均值。这些是开放式答案的确定性词面指标，未使用 LLM 裁判，也不是选择题准确率。', '',
        f'逐题配对：F1 提升 {overall["f1_improved"]} 题，下降 {overall["f1_worsened"]} 题，持平 {overall["f1_tied"]} 题；'
        f'EM 新增命中 {overall["em_gained"]} 题、失去命中 {overall["em_lost"]} 题。'
        f'两种方法有 {overall["same_reader_payload"]} 题的输入完全相同。', ''])
    diagnostics = validation['retrieval_diagnostics']
    counts = diagnostics['overall']
    token_groups = diagnostics['bge_token_budget']['groups']
    text.extend([
        f'检索变化存在明显的记忆类型差异：图片目标命中 {counts["baseline_visual_hits"]} → {counts["method_visual_hits"]} '
        f'（净增 {counts["method_visual_hits"] - counts["baseline_visual_hits"]}），纯文本目标命中 '
        f'{counts["baseline_text_hits"]} → {counts["method_text_hits"]} '
        f'（净减 {counts["baseline_text_hits"] - counts["method_text_hits"]}）。'
        f'Top5 中图片记忆的平均条数从 {counts["baseline_mean_image_slots"]:.3f} 增至 {counts["method_mean_image_slots"]:.3f}。'
        '全部 1,711 道题的纯文本记忆内部次序不变；纯文本目标在总榜中的位置变化均对应其前方图片条数的变化。'
        '这些是冻结检索结果中的排序事实，不能单独证明回答效果的原因。', '',
        '对实际 BGE 输入的长度审计覆盖 13,520 条文本，超过 512 token 的数量为 0；'
        f'对话＋caption、聚类 caption、原 query、扩展 query 的最长长度分别为 '
        f'{token_groups["memory_retrieval_text"]["max_tokens"]}、{token_groups["clustering_captions"]["max_tokens"]}、'
        f'{token_groups["baseline_queries"]["max_tokens"]}、{token_groups["expanded_queries"]["max_tokens"]} token。', ''])
    uncertainty = report['f1_uncertainty']
    for label, key in [('逐题配对 bootstrap', 'paired_question_percentile_95'),
                       ('按领域整体重采样 bootstrap', 'paired_domain_cluster_percentile_95')]:
        low, high = uncertainty[key]
        text.append(f'{label} 的 F1 差值 95% 区间：[{low * 100:+.2f}, {high * 100:+.2f}] 个百分点。')
    text.extend(['两个区间均为描述性结果：同领域问题共享历史，逐题独立性不成立；领域重采样保留这种依赖，但只有 20 个领域。', '',
        '本次开放式适配先由 Qwen3-VL-8B 根据问题及官方问题图片 caption 生成四个假设；'
        '每个 master 通过 CLIP 选一个假设并追加到 query，其下图片叶子使用这个 query 计算 BGE 分数，纯文本记忆保留原 query 分数，最后全域排序取 Top5。'
        '基线使用原 query 对同一个完整候选池排序。候选假设只参与检索；reader 只读原始对话、完整历史 caption、日期与图片 ID，带图问题还使用原始问题图片。', '',
        '这四条假设没有标准答案监督，也未按下游成绩筛选；格式合格不保证它们是互斥答案或都适合视觉匹配。'
        '本次成绩衡量的是这个开放式候选生成与 master 扩展的整体实现。', '',
        f'离线生成 {validation["masters"]} 个 master，每个图片叶子仅保存二维坐标；其中 {validation["approximate_masters"]} 个 master 标记为近似视觉聚合。'
        '旧成员迁移或拆分时无法恢复全部残差，因此不能把这些 master 当作精确重算的图片均值。二维基底固定来自与本数据集无重叠的 MemEye 校准集。'
        '未保存每图完整视觉 embedding 或额外图片副本，未重编码旧图。', '',
        f'审计通过：{validation["frozen_files_verified"]} 个冻结文件的哈希已核对；3,422 条逻辑回答对应 '
        f'{validation["fresh_inferences"]} 次本次新推理及 {validation["shared_records"]} 条同次相同输入复用。'
        f'空回答 {validation["empty_responses"]} 条，达到输出 token 上限 {validation["token_limit_records"]} 条。'
        f'正式候选生成共 {validation["candidate_attempt_records"]} 次记录，其中 {validation["candidate_invalid_attempts"]} 条格式验证失败记录；'
        f'最终 {validation["candidate_format_repairs_without_new_generation"]} 条仅将四行单元素 JSON 数组合并，逐字保留原生成字符串，未重新生成或参考答案。'
        'diagnostics 下的先行格式试验未混入这些计数或最终成绩。', '',
        f'离线构建耗时 {timings["offline_seconds"]:.2f} 秒，联合检索阶段耗时 {timings["retrieval_seconds"]:.2f} 秒。'
        '两组共享编码与部分完全相同的 reader 推理，因此这些时间不代表两组独立延迟，也不能用于声称方法更快。', ''])
    for title, key in [('按问题类型', 'by_question_type'), ('按目标记忆类型', 'by_cohort'), ('按领域', 'by_domain')]:
        text.extend([f'## {title}', '', '| 分组 | 题数 | 基线 F1 | 方法 F1 | 差值（百分点） | 基线 Recall@5 | 方法 Recall@5 |',
                     '|---|---:|---:|---:|---:|---:|---:|'])
        for label, group in report[key].items():
            b, m = [group[c] for c in CONDITIONS]
            text.append(f'| {label} | {group["questions"]} | {percentage(b["f1"])} | {percentage(m["f1"])} | '
                        f'{percentage(group["delta"]["f1"], True)} | {percentage(b["recall_at_5_micro"])} | {percentage(m["recall_at_5_micro"])} |')
        text.append('')
    text.extend(['完整可核对数据：', '', '- [汇总 JSON](comparison.json)', '- [逐领域 CSV](by_domain.csv)',
                 '- [逐题答案与得分](../reader/paired_results.csv)', '- [冻结协议](../reader/protocol.json)',
                 '- [完整检索结果](../retrieval_results.jsonl)', ''])
    (destination / 'comparison.md').write_text('\n'.join(text))
    print(json.dumps({'report': str(destination / 'comparison.md'), 'overall': overall,
                      'audit': {key: validation[key] for key in ('passed', 'fresh_inferences', 'shared_records',
                                                                 'empty_responses', 'token_limit_records')}}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    args = parser.parse_args()
    write_report(args.output, args.bootstrap_replicates)
