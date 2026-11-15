#!/usr/bin/env python3
"""Fresh paired API readers for the three-seed MemEye visual-basis ablation.

Historical captions are frozen separately for each backbone. No historical
answer is read: only identical complete requests within this run share a new
answer. Evaluation labels are opened by analyze(), after generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation.memeye_gpt5_reader import parse_choice

CONDITIONS = ('pca_2d', 'random_2d_seed_0', 'random_2d_seed_1', 'random_2d_seed_2')
ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
ALLOWED_OLD_CODE_DRIFT = {str(ROOT / 'scripts/memory/caption_visual_master.py'),
                          str(ROOT / 'scripts/visual_memory_pipeline.py')}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def indexed(rows, key):
    result = {row[key]: row for row in rows}
    assert len(result) == len(rows), ('duplicate', key)
    return result


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
    tmp.replace(path)


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()


def transport_for(backend):
    if backend == 'gpt5':
        from scripts.evaluation import openai_gpt5_base_transport as transport
    elif backend == 'deepseek':
        from scripts.evaluation import deepseek_flash_seed42_audited_transport as transport
    else:
        raise ValueError('Unsupported backend')
    return transport


def build_wire(backend, payload):
    # Deliberately reject, rather than silently strip, unexpected metadata.
    assert set(payload) == {'system', 'user_text', 'query_images'}
    transport = transport_for(backend)
    if backend == 'gpt5':
        return transport.build_request(payload, seed=42, max_output_tokens=2048, image_detail='high')
    return transport.build_request(payload, model=transport.DEEPSEEK_MODEL,
                                   max_output_tokens=16, image_detail='high')


def build_record(backend, system, question, condition, memories):
    name, ids = condition['name'], condition['top5_memory_ids']
    assert name in CONDITIONS and len(ids) == len(set(ids)) == 5
    assert all(memories[mid]['task'] == question['task'] for mid in ids)
    context = '\n\n'.join(f'Memory {i}: {memories[mid]["retrieval_text"]}'
                            for i, mid in enumerate(ids, 1))
    assert context == condition['reader_context'], (question['id'], name, 'context differs')
    text = ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + context
            + '\n\nQuestion:\n' + question['question'] + '\n\nOptions:\n'
            + '\n'.join(f'{letter}. {question["options"][letter]}' for letter in 'ABCD')
            + '\n\nAnswer with one letter only.')
    payload = {'system': system, 'user_text': text, 'query_images': question['query_images']}
    wire = build_wire(backend, payload)
    assert wire['seed'] == 42
    return {'record_id': question['id'] + '::' + name, 'question_id': question['id'],
            'task': question['task'], 'condition': name, 'top_memory_ids': ids,
            **payload, 'payload_sha256': digest(payload), 'request_sha256': digest(wire)}


def unique_requests(records):
    unique = {}
    for row in records:
        unique.setdefault(row['request_sha256'], row)
    return unique


def materialize(records, fresh):
    result = []
    for row in records:
        answer = fresh.get(row['request_sha256'])
        if answer is None:
            continue
        result.append({**answer, 'record_id': row['record_id'], 'question_id': row['question_id'],
                       'task': row['task'], 'condition': row['condition'],
                       'top_memory_ids': row['top_memory_ids'], 'payload_sha256': row['payload_sha256'],
                       'shared_from_record_id': answer['record_id'],
                       'same_run_shared': answer['record_id'] != row['record_id']})
    return result


def verify_hashes(hashes, base=None):
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute():
            assert base is not None
            path = Path(base) / path
        assert path.is_file() and sha(path) == expected, str(path)


def old_source_audit(source):
    """Freeze historical data; explicitly disclose only known entrypoint drift."""
    root = read(source / 'protocol.json')
    old_digest = hashlib.sha256(json.dumps(root['contract'], sort_keys=True, ensure_ascii=False,
                                           separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert old_digest == root['experiment_signature'], 'Historical protocol signature changed'
    checks, drift = {}, []
    for name in ('protocol.json', 'inputs/protocol.json', 'retrieval_protocol.json'):
        value = read(source / name)
        for path, expected in value.get('source_sha256', value.get('contract', {}).get('source_sha256', {})).items():
            actual = sha(path)
            if actual != expected:
                assert path in ALLOWED_OLD_CODE_DRIFT, ('Unexpected historical source change', path)
                drift.append({'protocol': name, 'path': path, 'historical_sha256': expected,
                              'current_sha256': actual, 'reason': 'Existing Random-2D entrypoint extension'})
            checks[path] = actual
        for field in ('artifact_sha256', 'output_sha256'):
            verify_hashes(value.get(field, {}), source / 'inputs' if name.startswith('inputs/') else source)
    return root, checks, drift


def prepare(backend, source, output, workers=3):
    source, output = Path(source).resolve(), Path(output).resolve()
    folder = output / 'reader'
    if (folder / 'protocol.json').exists():
        p = verify(output)
        assert p['contract']['backend'] == backend and p['contract']['memory_path'] == str(source)
        return p
    assert 1 <= workers <= 8
    ret = read(output / 'retrieval_protocol.json')
    assert ret.get('status') == 'complete' or ret.get('completed') is True
    for field in ('source_sha256', 'output_sha256', 'artifact_sha256'):
        verify_hashes(ret.get(field, {}), output)
    old, sources, drift = old_source_audit(source)
    questions = indexed(lines(source / 'inputs/source_questions.jsonl'), 'id')
    memories = indexed(lines(source / 'offline/memories.jsonl'), 'id')
    results = indexed(lines(output / 'retrieval_results.jsonl'), 'question_id')
    assert set(results) == set(questions) and len(questions) == 371 and len(memories) == 438
    system = read(source / 'inputs/protocol.json')['reader_system_prompt']
    model = 'gpt-5' if backend == 'gpt5' else 'deepseek-v4.1-flash'
    assert old['contract'].get('requested_model', old['contract'].get('model')) == model
    records = []
    for qi, (qid, question) in enumerate(questions.items()):
        conditions = indexed(results[qid]['conditions'], 'name')
        assert set(conditions) == set(CONDITIONS)
        batch = [build_record(backend, system, question, conditions[name], memories) for name in CONDITIONS]
        offset = qi % 4
        records.extend(batch[offset:] + batch[:offset])
    assert len(records) == 1484 and sum(bool(q['query_images']) for q in questions.values()) == 8
    transport = transport_for(backend)
    env_file = Path(old['contract'].get('api_env_file', ENV))
    config = transport.load_config(env_file)
    folder.mkdir(parents=True, exist_ok=True)
    write_lines(folder / 'requests.jsonl', records)
    # Gold bytes are hashed without parsing; run() never opens evaluation labels.
    paths = [Path(__file__), source / 'protocol.json', source / 'captions.jsonl',
             source / 'inputs/protocol.json', source / 'inputs/source_memories.jsonl',
             source / 'inputs/source_questions.jsonl', source / 'inputs/evaluation.jsonl',
             source / 'offline/memories.jsonl', output / 'retrieval_protocol.json',
             output / 'retrieval_results.jsonl', folder / 'requests.jsonl', Path(transport.__file__),
             ROOT / 'scripts/evaluation/memeye_gpt5_reader.py',
             ROOT / 'scripts/evaluation/openai_chat_transport.py',
             ROOT / 'scripts/evaluation/openai_backbone_transport.py',
             ROOT / 'scripts/evaluation/openai_gpt54_seeded_transport.py',
             ROOT / 'scripts/tests/test_memeye_api_random2d.py']
    paths.extend(Path(im['path']) for q in questions.values() for im in q['query_images'])
    for field in ('source_sha256', 'output_sha256', 'artifact_sha256'):
        for name in ret.get(field, {}):
            p = Path(name)
            paths.append(p if p.is_absolute() else output / p)
    sources.update({str(path.resolve()): sha(path) for path in paths})
    c = {'backend': backend, 'model': model, 'caption_model': old['contract']['model'],
         'historical_reader_model': old['contract']['model'], 'memory_path': str(source),
         'conditions': list(CONDITIONS), 'random_seeds': [0, 1, 2], 'questions': 371,
         'logical_records': 1484, 'unique_requests': len(unique_requests(records)),
         'seed': 42, 'temperature': None if backend == 'gpt5' else 0,
         'reasoning_effort': 'minimal' if backend == 'gpt5' else None,
         'enable_thinking': None if backend == 'gpt5' else False,
         'max_output_tokens': 2048 if backend == 'gpt5' else 16, 'image_detail': 'high',
         'api_env_file': str(env_file), 'api_config': config.public_dict(),
         'workers': workers, 'max_attempts': 3, 'timeout_seconds': 90,
         'historical_answer_reuse': False, 'same_run_complete_wire_deduplication': True,
         'gold_sent_to_api': False, 'gold_loaded_during_generation': False,
         'content_rejection_policy': 'data_inspection_failed/content_filter are terminal incorrect outcomes, no retry',
         'invalid_output_policy': 'Keep first original response and score incorrect; never resample',
         'transient_error_policy': 'Retry unchanged at most three attempts; exhausted errors leave run incomplete',
         'known_historical_code_drift': drift, 'source_sha256': sources}
    p = {'run_id': str(uuid.uuid4()), 'contract': c, 'experiment_signature': digest(c),
         'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    dump(folder / 'protocol.json', p)
    validation = {'passed': True, 'records': len(records), 'questions': 371,
                  'unique_requests': c['unique_requests'], 'query_image_questions': 8,
                  'complete_contexts_independently_rebuilt': len(records),
                  'historical_code_drift': drift, 'labels_loaded': False,
                  'requests_sha256': sha(folder / 'requests.jsonl')}
    dump(folder / 'validation_input.json', validation)
    verify(output)
    print(json.dumps({'stage': 'prepared', 'backend': backend, **validation}), flush=True)
    return p


def verify(output):
    folder = Path(output) / 'reader'
    p = read(folder / 'protocol.json')
    assert digest(p['contract']) == p['experiment_signature']
    verify_hashes(p['contract']['source_sha256'])
    return p


def safe_error(error):
    return {key: getattr(error, key, None) for key in
            ('kind', 'http_status', 'code', 'retryable', 'provider_error_code')}


def run(output):
    output = Path(output).resolve()
    folder = output / 'reader'
    with (folder / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(output)


def _run_locked(output):
    folder = output / 'reader'
    p = verify(output)
    c = p['contract']
    transport = transport_for(c['backend'])
    config = transport.load_config(c['api_env_file'])
    assert config.public_dict() == c['api_config']
    records = lines(folder / 'requests.jsonl')
    unique = unique_requests(records)
    assert len(records) == c['logical_records'] and len(unique) == c['unique_requests']
    saved = lines(folder / 'fresh_responses.jsonl') if (folder / 'fresh_responses.jsonl').exists() else []
    fresh = indexed(saved, 'request_sha256')
    for key, answer in fresh.items():
        assert key in unique and answer['run_id'] == p['run_id']
        assert answer['experiment_signature'] == p['experiment_signature']
        assert answer['historical_response_reuse'] is False
        assert answer['record_id'] == unique[key]['record_id']
        assert answer['parsed_choice'] == parse_choice(answer['raw_response'])
        assert answer['predicted_choice'] == (answer['parsed_choice'] if answer['status'] == 'ok' else None)
    pending = [row for key, row in unique.items() if key not in fresh]
    execution_id = str(uuid.uuid4())

    def execute(row):
        payload = {key: row[key] for key in ('system', 'user_text', 'query_images')}
        assert digest(payload) == row['payload_sha256']
        wire = build_wire(c['backend'], payload)
        assert digest(wire) == row['request_sha256']
        attempts = []
        for attempt in range(1, c['max_attempts'] + 1):
            meta = {'record_id': row['record_id'], 'request_sha256': row['request_sha256'],
                    'run_id': p['run_id'], 'execution_id': execution_id, 'attempt': attempt,
                    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            try:
                answer = transport.chat_request(config, wire, timeout=c['timeout_seconds'])
            except transport.APIError as error:
                info = safe_error(error)
                attempts.append({**meta, 'status': 'api_error', **info})
                rejected = (info['provider_error_code'] == 'data_inspection_failed'
                            or info['code'] == 'content_filter')
                if rejected:
                    answer = {'raw_response': '', 'requested_model': c['model'], 'returned_model': None,
                              'response_id': None, 'request_id': None, 'finish_reason': None,
                              'usage': None, 'system_fingerprint': None, 'refusal': None,
                              'content_rejection': info}
                    break
                if not error.retryable or attempt == c['max_attempts']:
                    return None, attempts
                time.sleep(min(2 ** attempt, 30))
                continue
            attempts.append({**meta, 'status': 'success', 'response_id': answer['response_id'],
                             'returned_model': answer['returned_model'],
                             'finish_reason': answer['finish_reason']})
            break
        predicted = parse_choice(answer['raw_response'])
        rejected = bool(answer.get('content_rejection'))
        valid = predicted is not None and answer['finish_reason'] == 'stop' and not answer.get('refusal') and not rejected
        result = {**answer, 'record_id': row['record_id'], 'question_id': row['question_id'],
                  'task': row['task'], 'condition': row['condition'],
                  'payload_sha256': row['payload_sha256'], 'request_sha256': row['request_sha256'],
                  'run_id': p['run_id'], 'experiment_signature': p['experiment_signature'],
                  'execution_id': execution_id, 'attempts': attempt,
                  'query_image_count': len(row['query_images']), 'parsed_choice': predicted,
                  'predicted_choice': predicted if valid else None,
                  'status': 'content_rejected' if rejected else 'ok' if valid else 'format_error',
                  'hit_token_limit': answer['finish_reason'] == 'length', 'historical_response_reuse': False}
        return result, attempts

    failures = []
    with ThreadPoolExecutor(max_workers=c['workers']) as pool:
        for start in range(0, len(pending), c['workers']):
            futures = {pool.submit(execute, row): row for row in pending[start:start + c['workers']]}
            for future in as_completed(futures):
                row = futures[future]
                answer, attempts = future.result()
                for attempt in attempts:
                    append(folder / 'api_attempts.jsonl', attempt)
                if answer is None:
                    failures.append({'record_id': row['record_id'], 'last_attempt': attempts[-1]})
                else:
                    fresh[row['request_sha256']] = answer
                    append(folder / 'fresh_responses.jsonl', answer)
            logical = materialize(records, fresh)
            write_lines(folder / 'responses.jsonl', logical)
            progress = {'stage': 'api_reader', 'backend': c['backend'], 'logical_records': len(logical),
                        'total': len(records), 'fresh_requests': len(fresh), 'expected_fresh_requests': len(unique),
                        'format_errors': sum(r['status'] == 'format_error' for r in logical),
                        'content_rejections': sum(r['status'] == 'content_rejected' for r in logical),
                        'token_limit_hits': sum(r['hit_token_limit'] for r in logical),
                        'failures': len(failures)}
            dump(folder / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            if failures:
                dump(folder / 'failed_requests.json', failures)
                dump(folder / 'execution_status.json', {'status': 'incomplete', **progress})
                raise RuntimeError('API failure; successful responses retained for verified resume')
    assert set(fresh) == set(unique)
    logical = materialize(records, fresh)
    write_lines(folder / 'responses.jsonl', logical)
    models = sorted({r['returned_model'] for r in fresh.values() if r['returned_model'] is not None})
    assert len(models) == 1, ('Mixed API revisions; do not publish complete comparison', models)
    verify(output)
    attempts = lines(folder / 'api_attempts.jsonl')
    report = {'passed': True, 'logical_records': len(logical), 'fresh_unique_requests': len(fresh),
              'fresh_model_inferences': sum(r['status'] != 'content_rejected' for r in fresh.values()),
              'api_calls': len(attempts), 'same_run_reused_records': len(logical) - len(fresh),
              'historical_answer_reuse': False, 'returned_models': models,
              'historical_reader_model': c['historical_reader_model'],
              'model_revision_changed_from_history': models != [c['historical_reader_model']],
              'format_errors': sum(r['status'] == 'format_error' for r in logical),
              'content_rejections': sum(r['status'] == 'content_rejected' for r in logical),
              'token_limit_hits': sum(r['hit_token_limit'] for r in logical),
              'requests_sha256': sha(folder / 'requests.jsonl'),
              'fresh_responses_sha256': sha(folder / 'fresh_responses.jsonl'),
              'responses_sha256': sha(folder / 'responses.jsonl'),
              'api_attempts_sha256': sha(folder / 'api_attempts.jsonl')}
    dump(folder / 'validation_output.json', report)
    return report


def aggregate(rows):
    n = len(rows)
    correct = sum(row['correct'] for row in rows)
    return {'questions': n, 'correct': correct, 'accuracy': correct / n,
            'gold_hits_top5': sum(row['gold_hits_top5'] for row in rows),
            'gold_references': sum(row['gold_references'] for row in rows),
            'format_errors': sum(row['status'] == 'format_error' for row in rows),
            'content_rejections': sum(row['status'] == 'content_rejected' for row in rows),
            'token_limit_hits': sum(row['hit_token_limit'] for row in rows)}


def stats(values):
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
            'min': min(values), 'max': max(values), 'values': values}


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(output):
    output = Path(output).resolve()
    folder = output / 'reader'
    p = verify(output)
    c = p['contract']
    validation = read(folder / 'validation_output.json')
    assert validation['passed'] and validation['logical_records'] == 1484
    for name in ('requests', 'fresh_responses', 'responses', 'api_attempts'):
        assert sha(folder / (name + '.jsonl')) == validation[name + '_sha256']
    requests = indexed(lines(folder / 'requests.jsonl'), 'record_id')
    responses = indexed(lines(folder / 'responses.jsonl'), 'record_id')
    assert set(requests) == set(responses) and len(responses) == 1484
    gold = indexed(lines(Path(c['memory_path']) / 'inputs/evaluation.jsonl'), 'id')
    assert len(gold) == 371
    rows = []
    for rid, r in responses.items():
        q = requests[rid]
        g = gold[r['question_id']]
        parsed = parse_choice(r['raw_response'])
        pred = parsed if r['status'] == 'ok' else None
        assert parsed == r['parsed_choice'] and pred == r['predicted_choice'] and r['request_sha256'] == q['request_sha256']
        rows.append({'record_id': rid, 'question_id': r['question_id'], 'domain': r['task'],
                     'condition': r['condition'], 'gold_choice': g['gold_choice'],
                     'predicted_choice': pred, 'correct': r['status'] == 'ok' and pred == g['gold_choice'],
                     'gold_hits_top5': len(set(q['top_memory_ids']) & set(g['gold_ids'])),
                     'gold_references': len(g['gold_ids']), 'status': r['status'],
                     'hit_token_limit': r['hit_token_limit'], 'returned_model': r['returned_model'],
                     'top_memory_ids': q['top_memory_ids'], 'request_sha256': r['request_sha256']})
    overall = {name: aggregate([row for row in rows if row['condition'] == name]) for name in CONDITIONS}
    assert all(m['questions'] == 371 and m['gold_references'] == 1041 for m in overall.values())
    domains = sorted({r['domain'] for r in rows})
    by_domain = {name: {domain: aggregate([r for r in rows if r['condition'] == name and r['domain'] == domain])
                        for domain in domains} for name in CONDITIONS}
    by_condition = {name: indexed([r for r in rows if r['condition'] == name], 'question_id') for name in CONDITIONS}
    paired = {}
    for name in CONDITIONS[1:]:
        base, other = by_condition['pca_2d'], by_condition[name]
        wins = sum(not base[q]['correct'] and other[q]['correct'] for q in base)
        losses = sum(base[q]['correct'] and not other[q]['correct'] for q in base)
        n = wins + losses
        paired[name] = {'wrong_to_right': wins, 'right_to_wrong': losses, 'net_correct': wins - losses,
                        'answer_changes': sum(base[q]['predicted_choice'] != other[q]['predicted_choice'] for q in base),
                        'ordered_top5_changes': sum(base[q]['top_memory_ids'] != other[q]['top_memory_ids'] for q in base),
                        'mcnemar_exact_p': min(1., 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.}
    random = {'seeds': [0, 1, 2], 'std_definition': 'sample standard deviation, ddof=1',
              'accuracy': stats([overall[name]['accuracy'] for name in CONDITIONS[1:]]),
              'correct': stats([overall[name]['correct'] for name in CONDITIONS[1:]]),
              'recall_at_5': stats([overall[name]['gold_hits_top5'] / 1041 for name in CONDITIONS[1:]])}
    summary = {'status': 'complete', 'backend': c['backend'], 'requested_model': c['model'],
               'caption_model': c['caption_model'], 'conditions': list(CONDITIONS),
               'overall': overall, 'by_domain': by_domain, 'random_2d_across_seeds': random,
               'paired_vs_pca': paired, **validation}
    write_lines(folder / 'per_question.jsonl', rows)
    write_csv(folder / 'per_question.csv', rows)
    write_csv(folder / 'metrics.csv', [{'condition': name, **values} for name, values in overall.items()])
    dump(folder / 'summary.json', summary)
    dump(folder / 'execution_status.json', {'status': 'complete', 'summary_sha256': sha(folder / 'summary.json')})
    print(json.dumps({'stage': 'reader_complete', 'backend': c['backend'], 'overall': overall,
                      'random_2d_across_seeds': random, 'validation': validation}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=('gpt5', 'deepseek'))
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', choices=('prepare', 'run', 'analyze', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):
        prepare(args.backend, args.source, args.output, args.workers)
    if args.stage in ('run', 'all'):
        p = verify(args.output)
        assert p['contract']['backend'] == args.backend and p['contract']['memory_path'] == str(args.source.resolve())
        run(args.output)
    if args.stage in ('analyze', 'all'):
        analyze(args.output)


if __name__ == '__main__':
    main()
