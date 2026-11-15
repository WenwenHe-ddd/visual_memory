"""Continue frozen generic readers while retaining terminal HTTP-400 outcomes.

Successful provider records remain in the original append-only responses file.
HTTP-400 rejections are separate, never represented as successful generations.
The sealed evaluation projection covers the full cohort before gold is opened.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memeye_deepseek_open_reader as original

sha, digest, rows, read, dump = original.sha, original.digest, original.rows, original.read, original.dump
MODEL = original.MODEL
POLICY = 'Only durable nonretryable HTTP400 with kind http/data_inspection_failed is a permanent_http_error; authentication, quota, model/protocol mismatch, exhausted transient failures remain fatal. Never retry a saved terminal failure.'


def permanent_400(event):
    code = str(event.get('code') or '').lower()
    excluded = ('quota', 'billing', 'authentication', 'api_key', 'permission', 'model', 'not_found')
    return (event.get('status') == 'error' and event.get('retryable') is False
            and event.get('http_status') == 400
            and event.get('error_kind') in {'http', 'data_inspection_failed'}
            and not any(word in code for word in excluded))


def failure_evidence(cache, request):
    key = digest(request)
    directory = cache / 'calls' / key
    path = directory / 'attempts.jsonl'
    if not path.exists():
        return None
    attempts = rows(path)
    if not attempts or not permanent_400(attempts[-1]):
        return None
    if (directory / 'response.json').exists():
        raise ValueError('Terminal error conflicts with a saved successful provider response')
    wire_path = directory / 'wire.json.gz'
    wire = json.loads(gzip.decompress(wire_path.read_bytes()))
    if wire != request or digest(wire) != key:
        raise ValueError('Rejected wire differs from frozen reader input')
    if [a.get('attempt') for a in attempts] != list(range(1, len(attempts) + 1)):
        raise ValueError('Rejected attempt sequence differs')
    if any(a.get('request_sha256') != key or a.get('status') != 'error' for a in attempts):
        raise ValueError('Rejected wire has conflicting attempt provenance')
    if any(not a.get('retryable') for a in attempts[:-1]):
        raise ValueError('A saved terminal failure was retried')
    return {'wire_path': str(wire_path.resolve()), 'wire_file_sha256': sha(wire_path),
            'attempts_path': str(path.resolve()), 'attempts_sha256': sha(path),
            'attempt_count': len(attempts), 'terminal_attempt': attempts[-1],
            'provider_response_received': False}


def original_successes(args, protocol, manifest):
    saved = original.validate(args.root, protocol, manifest)
    path = args.root / 'reader/responses.jsonl'
    return saved, path.read_bytes() if path.exists() else b''


def prepare(args):
    protocol, manifest, config = original.prepare(args)
    saved, success_bytes = original_successes(args, protocol, manifest)
    path = args.root / 'reader/evaluation_protocol.json'
    previous = read(path) if path.exists() else None
    initial = previous['contract']['initial_success_prefix'] if previous else {
        'bytes': len(success_bytes), 'sha256': hashlib.sha256(success_bytes).hexdigest(),
        'records': len(saved), 'ids': [r['id'] for r in saved]}
    if hashlib.sha256(success_bytes[:initial['bytes']]).hexdigest() != initial['sha256']:
        raise ValueError('An original successful response was changed')
    prefix_rows = [json.loads(line) for line in success_bytes[:initial['bytes']].decode().splitlines() if line.strip()]
    if len(prefix_rows) != initial['records'] or [r['id'] for r in prefix_rows] != initial['ids']:
        raise ValueError('Initial successful response prefix differs')
    sources = [Path(__file__), Path(original.__file__), args.root / 'reader/protocol.json',
               args.root / 'reader_manifest.jsonl', args.root / 'protocol.json', args.cache / 'protocol.json']
    contract = {'schema': 'memeye_deepseek_open_http400_evaluation_v1', 'dataset': 'MemEye Open',
        'method_id': args.method_id, 'questions': 371, 'answer_model': MODEL,
        'answer_model_revision': None, 'model_identity_kind': 'provider_alias',
        'original_reader_signature': protocol['experiment_signature'],
        'original_settings': protocol['contract']['settings'], 'cache_root': str(args.cache.resolve()),
        'source_sha256': {str(p.resolve()): sha(p) for p in sources},
        'initial_success_prefix': initial, 'failure_policy': POLICY,
        'projection_policy': 'Manifest order; every original successful provider record copied with source_status=answered; permanent HTTP400 gets an explicit empty-output failure record without response_id, usage or returned-model metadata.',
        'gold_sha256': protocol['contract']['gold_sha256'], 'gold_parsed_during_generation': False,
        'success_file_policy': 'Append real successes only; preserve every initial success byte; never edit old reader or shared-cache protocols.',
        'judge_policy': 'Judge the full 371-row projection with the real fixed judge; no fabricated grade for missing provider output.'}
    recovery = original.common.ensure_protocol(path, contract)
    return protocol, recovery, manifest, config


def rejected_record(row, protocol, recovery, request, evidence):
    return {**{k: row[k] for k in ('id', 'task', 'point')},
            'method_id': recovery['contract']['method_id'], 'answer_model': MODEL,
            'raw_response': '', 'status': 'permanent_http_error', 'source_status': 'permanent_http_error',
            'returned_model': None, 'hit_token_limit': None,
            'request_sha256': digest(request), 'api_failure': evidence,
            'experiment_signature': protocol['experiment_signature'], 'run_id': protocol['run_id'],
            'recovery_signature': recovery['experiment_signature']}


def validate(args, protocol, recovery, manifest):
    contract = recovery['contract']
    if digest(contract) != recovery['experiment_signature']:
        raise ValueError('Recovery signature differs')
    for path, expected in contract['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen recovery source changed: ' + path)
    successes, raw = original_successes(args, protocol, manifest)
    prefix = contract['initial_success_prefix']
    if hashlib.sha256(raw[:prefix['bytes']]).hexdigest() != prefix['sha256']:
        raise ValueError('Original success prefix changed')
    source = original.common.by_id(manifest)
    success_map = original.common.by_id(successes)
    path = args.root / 'reader/rejected_responses.jsonl'
    rejected = rows(path) if path.exists() else []
    rejected_map = original.common.by_id(rejected)
    if set(success_map) & set(rejected_map) or not set(rejected_map) <= set(source):
        raise ValueError('Overlapping or foreign answer/rejection IDs')
    for qid, record in rejected_map.items():
        request = original.build_request(source[qid])
        evidence = failure_evidence(args.cache, request)
        if evidence is None or record != rejected_record(source[qid], protocol, recovery, request, evidence):
            raise ValueError('Rejection differs from durable original API failure')
    return success_map, rejected_map


def run(args):
    protocol, recovery, manifest, config = prepare(args)
    successes, rejected = validate(args, protocol, recovery, manifest)
    finished = set(successes) | set(rejected)
    pending = [row for row in manifest if row['id'] not in finished]
    if args.limit:
        pending = pending[:args.limit]
    lock, stop = threading.Lock(), threading.Event()

    def one(row):
        if stop.is_set():
            return
        try:
            request = original.build_request(row)
            evidence = failure_evidence(args.cache, request)
            output = None
            if evidence is None:
                try:
                    output = original.call(config, args.cache, request)
                except Exception:
                    evidence = failure_evidence(args.cache, request)
                    if evidence is None:
                        raise
            if evidence is not None:
                record = rejected_record(row, protocol, recovery, request, evidence)
                name = 'rejected_responses.jsonl'
            else:
                original.validate_output(output, digest(request))
                record = {**output, **{k: row[k] for k in ('id', 'task', 'point')},
                    'method_id': args.method_id, 'answer_model': MODEL,
                    'top_memory_ids': row.get('top_memory_ids', []),
                    'experiment_signature': protocol['experiment_signature'], 'run_id': protocol['run_id']}
                name = 'responses.jsonl'
            with lock:
                original.append(args.root / 'reader' / name, record)
                finished.add(row['id'])
                if evidence is not None or len(finished) % 10 == 0 or len(finished) == 371:
                    print(json.dumps({'method': args.method_id, 'completed': len(finished),
                                      'source_status': 'permanent_http_error' if evidence else 'answered',
                                      'id': row['id']}), flush=True)
        except Exception:
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one, row) for row in pending]):
            future.result()
    successes, rejected = validate(args, protocol, recovery, manifest)
    if len(successes) + len(rejected) != 371:
        raise ValueError('Require all 371 outcomes before writing evaluation projection')
    projection = [dict(successes[q['id']], source_status='answered') if q['id'] in successes
                  else rejected[q['id']] for q in manifest]
    output = args.root / 'reader/evaluation_responses.jsonl'
    original.write_frozen(output, projection)
    artifact_names = ['responses.jsonl', 'evaluation_protocol.json', 'evaluation_responses.jsonl']
    if rejected:
        artifact_names.append('rejected_responses.jsonl')
    result = {'status': 'complete', 'questions': 371, 'questions_answered': len(successes),
        'questions_rejected': len(rejected), 'question_status_counts': {'answered': len(successes), 'permanent_http_error': len(rejected)},
        'responses_sha256': sha(output), 'evaluation_responses_sha256': sha(output),
        'successful_responses_sha256': sha(args.root / 'reader/responses.jsonl'),
        'protocol_sha256': sha(args.root / 'reader/evaluation_protocol.json'),
        'source_sha256': {str((args.root / 'reader' / n).resolve()): sha(args.root / 'reader' / n) for n in artifact_names},
        'rejected_ids': [q['id'] for q in manifest if q['id'] in rejected],
        'empty_successful_answers': sum(not r['raw_response'] for r in successes.values()),
        'token_limit_hits': sum(r['hit_token_limit'] for r in successes.values()),
        'gold_parsed_during_generation': False, 'judge_scores_imputed': False}
    dump(args.root / 'reader/evaluation_completion.json', result)
    print(json.dumps(result), flush=True)


def verified_evaluation(root, *, with_provenance=False):
    """Offline validation; return 371 rows, optionally (rows, provenance_by_id).

    This entrypoint never loads credentials or performs network requests.
    Successful provenance points to the original provider response.json;
    rejection provenance points to the actual terminal attempts.jsonl.
    """
    root = Path(root).resolve()
    protocol = read(root / 'reader/protocol.json')
    recovery = read(root / 'reader/evaluation_protocol.json')
    args = SimpleNamespace(root=root, cache=Path(recovery['contract']['cache_root']))
    manifest = rows(root / 'reader_manifest.jsonl')
    successes, rejected = validate(args, protocol, recovery, manifest)
    seal = read(root / 'reader/evaluation_completion.json')
    path = root / 'reader/evaluation_responses.jsonl'
    if seal['status'] != 'complete' or len(successes) + len(rejected) != 371 or sha(path) != seal['responses_sha256']:
        raise ValueError('Require the sealed full evaluation projection')
    for name, expected in seal['source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Sealed evaluation source changed: ' + name)
    expected = [dict(successes[q['id']], source_status='answered') if q['id'] in successes else rejected[q['id']] for q in manifest]
    if rows(path) != expected:
        raise ValueError('Evaluation projection differs from original successful/rejected outcomes')
    if (seal['questions_answered'] != len(successes) or seal['questions_rejected'] != len(rejected)
            or seal['question_status_counts'] != {'answered': len(successes), 'permanent_http_error': len(rejected)}
            or seal['rejected_ids'] != [q['id'] for q in manifest if q['id'] in rejected]
            or seal['protocol_sha256'] != sha(root / 'reader/evaluation_protocol.json')):
        raise ValueError('Evaluation seal counts or protocol hash differ')
    provenance = {}
    for q, outcome in zip(manifest, expected):
        key = outcome['request_sha256']
        answered = q['id'] in successes
        record = args.cache / 'calls' / key / ('response.json' if answered else 'attempts.jsonl')
        provenance[q['id']] = {'source_answer_api_request_sha256': key,
            'source_answer_record': str(record), 'source_answer_record_sha256': sha(record),
            'answer_model_revision': None,
            'answer_returned_model': outcome['returned_model'] if answered else None,
            'source_status': 'answered' if answered else 'permanent_http_error'}
    return (expected, provenance) if with_provenance else expected


def collect(args):
    expected = verified_evaluation(args.root)
    recovery = read(args.root / 'reader/evaluation_protocol.json')
    manifest = rows(args.root / 'reader_manifest.jsonl')
    seal = read(args.root / 'reader/evaluation_completion.json')
    gold_path = original.CANONICAL / 'inputs/gold.jsonl'
    if sha(gold_path) != recovery['contract']['gold_sha256']:
        raise ValueError('Canonical Open gold changed')
    gold = original.common.by_id(rows(gold_path))
    records = []
    for q, outcome in zip(manifest, expected):
        answered = outcome['source_status'] == 'answered'
        records.append({**{k: q[k] for k in ('id', 'task', 'point', 'question')},
            'ground_truth': gold[q['id']]['answer'], 'model_output': outcome['raw_response'],
            'answer_model': MODEL, 'answer_model_revision': None,
            'answer_returned_model': outcome['returned_model'] if answered else None,
            'source_record_id': q['id'], 'source_status': outcome['source_status'],
            'source_hit_token_limit': outcome['hit_token_limit'] if answered else None,
            'source_responses_sha256': seal['responses_sha256'],
            'source_api_request_sha256': outcome['request_sha256']})
    original.write_frozen(args.root / 'judge_inputs.jsonl', records)
    dump(args.root / 'reader/evaluation_collection_audit.json', {
        'status': 'complete', 'questions': 371, 'questions_answered': seal['questions_answered'],
        'questions_rejected': seal['questions_rejected'], 'judge_inputs_sha256': sha(args.root / 'judge_inputs.jsonl'),
        'evaluation_completion_sha256': sha(args.root / 'reader/evaluation_completion.json'),
        'evaluation_protocol_sha256': sha(args.root / 'reader/evaluation_protocol.json'),
        'all_successful_records_match_original_provider_cache': True,
        'all_rejections_match_durable_nonretryable_http400_attempts': True,
        'all_original_success_bytes_preserved': True, 'gold_loaded_only_after_full_coverage': True,
        'judge_scores_imputed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run', 'collect'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method-id', required=True)
    parser.add_argument('--reference-correct', type=int, required=True)
    parser.add_argument('--cache', type=Path, default=original.RUN / 'answer_api_cache')
    parser.add_argument('--env-file', type=Path, default=Path('/storage/hewenwen/NTU Decentralized Agent/.env'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0 or args.method_id == 'hippomm':
        parser.error('Positive workers, nonnegative limit, and a generic reader method are required')
    args.root, args.cache = args.root.resolve(), args.cache.resolve()
    with (args.root / 'reader.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run if args.phase == 'run' else collect)(args)


if __name__ == '__main__':
    main()
