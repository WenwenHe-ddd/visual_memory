"""Judge frozen MemGallery answers with the same five-grade DeepSeek protocol.

The manifest lists methods and their frozen 1711-row JSONL inputs. Identical
question/gold/prediction requests share one score across methods. Compatible
scores from the completed Semantic RAG evaluation are reused with provenance.
Retrieval, original reader answers, and source metrics are never regenerated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import deepseek_flash_seed42_audited_transport as transport
from scripts.evaluation import memgallery_five_grade_judge as original

PROMPT = original.PROMPT
GRADES = original.GRADES
sha = original.sha
read_rows = original.read_rows
dump = original.dump
write_rows = original.write_rows
parse_grade = original.parse_grade
normalize_answer = original.normalize_answer_universal
DEFAULT_CACHE = ROOT / 'results/memgallery_qwen_semrag_deepseek_judge_20261104'
REQUIRED_FIELDS = ('id', 'task', 'point', 'question', 'ground_truth', 'model_output',
                   'em', 'source_record_id')


def build_request(row, template):
    fields = {key: row[key] for key in ('question', 'ground_truth', 'model_output')}
    prompt = re.sub(r'\{\{(question|ground_truth|model_output)\}\}',
                    lambda match: fields[match.group(1)], template)
    return transport.build_request({'user_text': prompt}, model=transport.DEEPSEEK_MODEL,
                                   max_output_tokens=1024)


def judge_settings():
    return {'judge_model': transport.DEEPSEEK_MODEL,
            'judge_base_url': transport.DASHSCOPE_BASE_URL,
            'temperature': 0, 'seed': 42, 'enable_thinking': False, 'max_tokens': 1024,
            'grades': list(GRADES), 'official_prompt_sha256': sha(PROMPT),
            'judge_input_fields': ['question', 'ground_truth', 'model_output'],
            'images_context_captions_sent_to_judge': False}


def validate_score(row, expected_hash):
    if row.get('request_sha256') != expected_hash:
        raise ValueError('Saved response request hash mismatch')
    if (row.get('returned_model') != transport.DEEPSEEK_MODEL or
            row.get('requested_model') != transport.DEEPSEEK_MODEL or
            row.get('finish_reason') != 'stop' or row.get('refusal')):
        raise ValueError('Saved response model/finish reason/refusal mismatch')
    if row.get('seed') != 42 or row.get('enable_thinking') is not False:
        raise ValueError('Saved response seed/thinking mismatch')
    if row.get('reasoning_content_present') or row.get('reasoning_tokens') not in (None, 0):
        raise ValueError('Saved response contains disabled model reasoning')
    grade = parse_grade(row['raw_response'])
    if any(row.get(key) != value for key, value in grade.items()):
        raise ValueError('Saved parsed score differs from the raw judge output')


def load_baseline_cache(root, template):
    """Accept only the completed baseline under exactly the same wire protocol."""
    names = ('protocol.json', 'requests.jsonl', 'scores.jsonl', 'summary.json',
             'judge_prompt.txt')
    paths = {name: root / name for name in names}
    protocol = json.loads(paths['protocol.json'].read_text())
    for key, expected in judge_settings().items():
        if protocol.get(key) != expected:
            raise ValueError(f'Baseline cache protocol differs: {key}')
    if paths['judge_prompt.txt'].read_bytes() != PROMPT.read_bytes():
        raise ValueError('Baseline cache prompt bytes differ')
    requests = read_rows(paths['requests.jsonl'])
    expected = original.unique_map(requests)
    scores = original.unique_map(read_rows(paths['scores.jsonl']))
    summary = json.loads(paths['summary.json'].read_text())
    if (len(expected) != 1711 or set(expected) != set(scores) or
            summary.get('status') != 'complete' or summary.get('questions_scored') != 1711 or
            summary.get('source_em_correct') != 778):
        raise ValueError('Baseline cache is incomplete or has wrong source coverage')
    if (summary.get('requests_sha256') != sha(paths['requests.jsonl']) or
            summary.get('scores_sha256') != sha(paths['scores.jsonl'])):
        raise ValueError('Baseline cache differs from its completed summary hashes')
    cached = {}
    for source in requests:
        request = build_request(source, template)
        digest = transport.request_sha256(request)
        if source.get('request') != request or source.get('request_sha256') != digest:
            raise ValueError('Baseline request does not reproduce from frozen judge inputs')
        score = scores[source['id']]
        validate_score(score, digest)
        provenance = {'root': str(root), 'id': source['id'],
                      'scores_path': str(paths['scores.jsonl']),
                      'scores_sha256': summary['scores_sha256'],
                      'response_id': score.get('response_id')}
        if digest in cached:
            if cached[digest]['score'] != score['score']:
                raise ValueError('Baseline contains conflicting grades for identical requests')
            cached[digest]['reuse_sources'].append(provenance)
        else:
            cached[digest] = {**{k: v for k, v in score.items() if k != 'id'},
                              'score_origin': 'baseline_cache', 'reuse_sources': [provenance]}
    return expected, cached, {str(path): sha(path) for path in paths.values()}


def prepare(args):
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    methods = manifest.get('methods')
    if not isinstance(methods, list) or not methods:
        raise ValueError('Manifest needs a nonempty methods list')
    if manifest.get('questions_per_method', 1711) != 1711:
        raise ValueError('This evaluation requires the complete 1711-question cohort')
    template = PROMPT.read_text()
    canonical, baseline, cache_hashes = load_baseline_cache(args.baseline_cache.resolve(), template)
    unique_requests, question_map, method_protocols, verified_sources = {}, [], [], {}
    seen_methods = set()
    for method in methods:
        mid = method.get('id', '')
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', mid) or mid in seen_methods):
            raise ValueError('Method IDs must be unique and safe directory names')
        seen_methods.add(mid)
        if not isinstance(method.get('label'), str) or not method['label'].strip():
            raise ValueError(f'Method {mid} needs a label')
        path = Path(method['records_path'])
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        source_hashes = method.get('source_sha256')
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise ValueError(f'Method {mid} requires source_sha256 provenance')
        for source, expected_digest in source_hashes.items():
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = manifest_path.parent / source_path
            source_path = source_path.resolve()
            digest = verified_sources.setdefault(str(source_path), sha(source_path))
            if digest != expected_digest:
                raise ValueError(f'Source hash mismatch for {source_path}')
        records_digest = sha(path)
        if method.get('records_sha256', records_digest) != records_digest:
            raise ValueError(f'Frozen records hash mismatch for {mid}')
        records = read_rows(path)
        source_map = original.unique_map(records)
        if len(records) != 1711 or set(source_map) != set(canonical):
            raise ValueError(f'Method {mid} does not cover the canonical 1711 questions')
        correct = 0
        for row in records:
            if any(key not in row for key in REQUIRED_FIELDS):
                raise ValueError(f'Method {mid} record lacks required fields')
            if any(not isinstance(row[key], str) for key in REQUIRED_FIELDS if key != 'em'):
                raise ValueError(f'Method {mid} judge fields must contain original strings')
            canon = canonical[row['id']]
            if any(row[key] != canon[key] for key in ('task', 'point', 'question', 'ground_truth')):
                raise ValueError(f'Method {mid} changed the canonical question or gold answer')
            em = int(normalize_answer(row['model_output']) == normalize_answer(row['ground_truth']))
            if type(row['em']) not in (int, bool) or row['em'] != em:
                raise ValueError(f'Method {mid} per-question EM does not reproduce')
            correct += em
            request = build_request(row, template)
            digest = transport.request_sha256(request)
            origin = {'method_id': mid, 'id': row['id']}
            if digest in unique_requests:
                if unique_requests[digest]['request'] != request:
                    raise ValueError('Request hash collision')
                unique_requests[digest]['uses'].append(origin)
            else:
                unique_requests[digest] = {
                    'request_sha256': digest, 'request': request,
                    **{key: row[key] for key in ('question', 'ground_truth', 'model_output')},
                    'point': row['point'], 'uses': [origin]}
            # Retain original metadata, including pre-existing truncated reader outputs.
            question_map.append({**row, 'method_id': mid, 'request_sha256': digest})
        if correct != method.get('expected_em_correct'):
            raise ValueError(f'Method {mid}: recomputed EM={correct}/1711 differs from expected')
        if 'em' in method and abs(method['em'] - correct / 1711) > 1e-12:
            raise ValueError(f'Method {mid}: reported source EM differs')
        method_protocols.append({**method, 'records_path': str(path),
                                 'records_sha256': records_digest,
                                 'questions': 1711, 'em': correct / 1711})
    requests = list(unique_requests.values())
    reused = [baseline[digest] for digest in unique_requests if digest in baseline]
    code_paths = [Path(__file__), Path(original.__file__), Path(transport.__file__),
                  ROOT / 'scripts/evaluation/openai_chat_transport.py',
                  ROOT / 'scripts/evaluation/mem_gallery_master_reader.py']
    protocol = {'dataset': 'MemGallery', 'answer_model': 'Qwen/Qwen3-VL-8B-Instruct',
        **judge_settings(), 'judge_model_identity_kind': 'provider_alias',
        'official_prompt_url': original.PROMPT_URL,
        'parser_difference_from_public_code': 'Keep all five grades exactly; no three-level remapping.',
        'aggregation': 'Per-method arithmetic mean over all 1711 questions; no missing-score imputation.',
        'deduplication': 'One score per exact serialized request hash; reused across identical inputs.',
        'original_answer_policy': 'Keep every frozen raw answer, including any source token-limit outputs.',
        'methods': method_protocols, 'method_count': len(method_protocols),
        'total_method_questions': len(question_map), 'unique_requests': len(requests),
        'baseline_reused_unique_requests': len(reused),
        'new_unique_requests_expected': len(requests) - len(reused),
        'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'source_sha256': verified_sources, 'baseline_cache_sha256': cache_hashes,
        'implementation_sha256': {str(path.resolve()): sha(path) for path in code_paths}}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    frozen = {'requests.jsonl': requests, 'question_map.jsonl': question_map,
              'reused_scores.jsonl': reused}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing protocol differs; use a new output root')
        for filename, rows in frozen.items():
            if read_rows(args.output / filename) != rows:
                raise ValueError(f'Frozen artifact differs: {filename}')
        if (args.output / 'judge_prompt.txt').read_bytes() != PROMPT.read_bytes():
            raise ValueError('Frozen prompt differs')
        if (args.output / 'methods_manifest.json').read_bytes() != manifest_path.read_bytes():
            raise ValueError('Frozen methods manifest differs')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Output directory is nonempty without protocol')
        for filename, rows in frozen.items():
            write_rows(args.output / filename, rows)
        (args.output / 'judge_prompt.txt').write_bytes(PROMPT.read_bytes())
        (args.output / 'methods_manifest.json').write_bytes(manifest_path.read_bytes())
        dump(protocol_path, protocol)
    return protocol, requests, question_map


def validated_results(output, requests):
    expected = {row['request_sha256']: row for row in requests}
    saved = {}
    for name, origin in (('reused_scores.jsonl', 'baseline_cache'), ('scores.jsonl', 'new_api')):
        path = output / name
        rows = read_rows(path) if path.exists() else []
        for row in rows:
            digest = row.get('request_sha256')
            if digest not in expected or digest in saved:
                raise ValueError('Unknown or duplicate request in persisted scores')
            if row.get('score_origin') != origin:
                raise ValueError('Score origin does not match its storage file')
            validate_score(row, digest)
            saved[digest] = row
    return saved


def write_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def analyze(args, protocol, requests, question_map):
    saved = validated_results(args.output, requests)
    reports = []
    method_rows = {}
    for row in question_map:
        method_rows.setdefault(row['method_id'], []).append(row)
    for method in protocol['methods']:
        sources = method_rows[method['id']]
        rows = []
        for source in sources:
            score = saved.get(source['request_sha256'])
            rows.append({**source, **(score or {}), 'judge_status': 'ok' if score else 'missing'})
        scored = [row for row in rows if row['judge_status'] == 'ok']
        complete = len(scored) == 1711
        counts = Counter(row['score'] for row in scored)
        score_sum = sum(row['score'] for row in scored)
        def group(field):
            grouped = {}
            for key in sorted({row[field] for row in rows}):
                expected = [row for row in rows if row[field] == key]
                subset = [row for row in expected if row['judge_status'] == 'ok']
                grouped[str(key)] = {'questions_expected': len(expected),
                    'questions_scored': len(subset),
                    'mean_score': sum(row['score'] for row in subset) / len(expected)
                                  if len(subset) == len(expected) else None}
            return grouped
        report = {**method, 'method_id': method['id'],
            'status': 'complete' if complete else 'partial',
            'questions_expected': 1711, 'questions_scored': len(scored),
            'questions_missing': 1711 - len(scored), 'judge_model': transport.DEEPSEEK_MODEL,
            'source_em_correct': method['expected_em_correct'], 'source_em': method['em'],
            'score_sum': score_sum, 'llm_judge': score_sum / 1711 if complete else None,
            'llm_judge_percent': 100 * score_sum / 1711 if complete else None,
            'score_distribution': {str(grade): counts[grade] for grade in GRADES},
            'score_origin_question_counts': dict(Counter(row['score_origin'] for row in scored)),
            'by_question_type': group('point'), 'by_scene': group('task'), 'by_source_em': group('em'),
            'missing_ids': [row['id'] for row in rows if row['judge_status'] == 'missing']}
        method_dir = args.output / 'methods' / method['id']
        method_dir.mkdir(parents=True, exist_ok=True)
        dump(method_dir / 'summary.json', report)
        write_rows(method_dir / 'per_question.jsonl', rows)
        fields = ['method_id', 'id', 'task', 'point', 'em', 'question', 'ground_truth',
                  'model_output', 'source_record_id', 'source_hit_token_limit', 'judge_status',
                  'score', 'reasoning', 'score_origin', 'request_sha256', 'returned_model',
                  'response_id', 'reuse_sources']
        write_csv(method_dir / 'per_question.csv', fields, rows)
        reports.append(report)
    new_scores = [row for row in saved.values() if row['score_origin'] == 'new_api']
    attempts_path = args.output / 'attempts.jsonl'
    attempts = read_rows(attempts_path) if attempts_path.exists() else []
    summary = {'status': 'complete' if len(saved) == len(requests) else 'partial',
        'judge_model': transport.DEEPSEEK_MODEL, 'grades': list(GRADES),
        'method_count': len(reports), 'total_method_questions': len(question_map),
        'total_scored_method_questions': sum(row['questions_scored'] for row in reports),
        'unique_requests': len(requests), 'unique_requests_scored': len(saved),
        'unique_requests_missing': len(requests) - len(saved),
        'baseline_reused_unique_requests': len(saved) - len(new_scores),
        'new_successful_unique_requests': len(new_scores), 'api_attempt_count': len(attempts),
        'failed_api_attempt_count': sum(row.get('status') != 'ok' for row in attempts),
        'new_successful_request_usage': {
            key: sum((row.get('usage') or {}).get(key, 0) or 0 for row in new_scores)
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        'methods': reports,
        'artifact_sha256': {name: sha(args.output / name) for name in
            ('protocol.json', 'requests.jsonl', 'question_map.jsonl', 'reused_scores.jsonl',
             'scores.jsonl', 'attempts.jsonl') if (args.output / name).exists()}}
    dump(args.output / 'summary.json', summary)
    write_csv(args.output / 'summary.csv',
              ['method_id', 'label', 'display_label', 'code_method', 'source_em_correct', 'source_em',
               'f1', 'bleu_1', 'llm_judge', 'llm_judge_percent', 'questions_expected',
               'questions_scored', 'questions_missing', 'status', 'context'], reports)
    print(json.dumps({key: summary[key] for key in ('status', 'method_count',
        'total_scored_method_questions', 'unique_requests', 'unique_requests_scored',
        'unique_requests_missing', 'baseline_reused_unique_requests')}, ensure_ascii=False), flush=True)
    return summary


def run(args, protocol, requests, question_map):
    saved = validated_results(args.output, requests)
    attempts_path = args.output / 'attempts.jsonl'
    previous = {}
    expected = {row['request_sha256'] for row in requests}
    for attempt in read_rows(attempts_path) if attempts_path.exists() else []:
        digest = attempt.get('request_sha256')
        if digest not in expected:
            raise ValueError('Unknown request in attempts log')
        if attempt.get('attempt') != previous.get(digest, {}).get('attempt', 0) + 1:
            raise ValueError('Attempts log is not sequential per request')
        previous[digest] = attempt
    # The success audit is fsynced before its score. Recover that narrow crash
    # window from the durable raw response instead of issuing another API call.
    recovered = []
    for digest, attempt in previous.items():
        if digest not in saved and attempt.get('status') == 'ok':
            score = {key: value for key, value in attempt.items()
                     if key not in ('attempt', 'started_at_unix', 'status', 'retryable')}
            score.update(parse_grade(score['raw_response']), score_origin='new_api',
                         recovered_from_successful_attempt_log=True)
            validate_score(score, digest)
            recovered.append(score)
            saved[digest] = score
    if recovered:
        with (args.output / 'scores.jsonl').open('a') as recovery:
            for score in recovered:
                recovery.write(json.dumps(score, ensure_ascii=False, allow_nan=False) + '\n')
            recovery.flush()
            os.fsync(recovery.fileno())
        print(f'Recovered {len(recovered)} scores from durable successful attempt logs', flush=True)
    pending = [row for row in requests if row['request_sha256'] not in saved and
               previous.get(row['request_sha256'], {}).get('attempt', 0) < args.max_attempts and
               previous.get(row['request_sha256'], {}).get('retryable', True)]
    if args.limit:
        seen, first, rest = set(), [], []
        for row in pending:
            if row['point'] in seen:
                rest.append(row)
            else:
                first.append(row)
                seen.add(row['point'])
        pending = (first + rest)[:args.limit]
    if not pending:
        return analyze(args, protocol, requests, question_map)
    config = transport.load_config(args.env_file)
    print(f'Judging {len(pending)} unique pending requests; {len(saved)} validated cached scores', flush=True)
    io_lock = threading.Lock()
    def persist(handle, row):
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    with attempts_path.open('a') as log, (args.output / 'scores.jsonl').open('a') as out:
        def judge(row):
            digest = row['request_sha256']
            start = previous.get(digest, {}).get('attempt', 0) + 1
            for number in range(start, args.max_attempts + 1):
                audit = {'request_sha256': digest, 'attempt': number,
                         'started_at_unix': time.time()}
                result = None
                retryable = False
                try:
                    response = transport.chat_request(config, row['request'], timeout=60)
                    audit.update(response)
                    grade = parse_grade(response['raw_response'])
                    result = {**response, **grade, 'score_origin': 'new_api'}
                    validate_score(result, digest)
                    audit.update(status='ok', retryable=False)
                except transport.APIError as error:
                    retryable = error.retryable
                    audit.update(status='error', error_kind=error.kind,
                                 provider_error_code=getattr(error, 'provider_error_code', None),
                                 retryable=retryable)
                except (ValueError, TypeError, KeyError) as error:
                    result = None
                    retryable = True
                    audit.update(status='error', error_kind='invalid_judge_output',
                                 error_class=type(error).__name__, retryable=True)
                with io_lock:
                    persist(log, audit)
                    if result is not None:
                        persist(out, result)
                if result is not None:
                    return True
                if not retryable or number == args.max_attempts:
                    return False
                time.sleep(min(2 ** number, 16))
            return False
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(judge, row) for row in pending]
            succeeded = 0
            for count, future in enumerate(as_completed(futures), 1):
                success = future.result()
                succeeded += int(success)
                if count % 50 == 0 or count == len(pending) or not success:
                    print(f'Finished {count}/{len(pending)}; successes={succeeded}; '
                          f'last_success={success}', flush=True)
    return analyze(args, protocol, requests, question_map)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'run', 'analyze'], required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--env-file', type=Path,
                        default=Path('/storage/hewenwen/NTU Decentralized Agent/.env'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-attempts', type=int, default=5,
                        help='Maximum total attempts per unique request, including earlier runs')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit new unique requests for a smoke run; 0 means all pending')
    args = parser.parse_args()
    if args.workers < 1 or args.max_attempts < 1 or args.limit < 0:
        parser.error('workers/max-attempts must be positive; limit must be nonnegative')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / (args.output.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, requests, question_map = prepare(args)
        if args.phase == 'run':
            report = run(args, protocol, requests, question_map)
            if report['status'] != 'complete' and not args.limit:
                raise SystemExit(2)
        else:
            analyze(args, protocol, requests, question_map)


if __name__ == '__main__':
    main()
