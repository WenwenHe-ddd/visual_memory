"""Canonical 371-question Judge execution, with backbone metadata from the protocol.

The durable retry and scoring implementation is preserved from the signed Qwen
runner. This separate module leaves all earlier experiment bytes unchanged.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import threading
import time
from scripts.evaluation import memeye_open_five_grade_judge as original
shared, transport = original.shared, original.transport
sha, read_rows, dump, write_rows = original.sha, original.read_rows, original.dump, original.write_rows
COHORT_SIZE = 371
ANSWER_MODEL = 'gpt-5'
FIELDS = original.FIELDS


def analyze(args, protocol, requests, question_map):
    saved = shared.validated_results(args.output, requests)
    rows = []
    for source in question_map:
        score = saved.get(source['request_sha256'])
        rows.append({**source, **(score or {}), 'judge_status': 'ok' if score else 'missing'})
    scored = [row for row in rows if row['judge_status'] == 'ok']
    complete = len(scored) == COHORT_SIZE
    counts = Counter(row['score'] for row in scored)
    score_sum = sum(row['score'] for row in scored)

    def group(field):
        result = {}
        for value in sorted({row[field] for row in rows}):
            expected = [row for row in rows if row[field] == value]
            selected = [row for row in expected if row['judge_status'] == 'ok']
            result[str(value)] = {'questions_expected': len(expected),
                'questions_scored': len(selected),
                'mean_score': sum(row['score'] for row in selected) / len(expected)
                              if len(selected) == len(expected) else None}
        return result

    attempts_path = args.output / 'attempts.jsonl'
    attempts = read_rows(attempts_path) if attempts_path.exists() else []
    report = {'status': 'complete' if complete else 'partial', 'dataset': 'MemEye',
        'task': 'open_ended_answering', 'method': protocol['method'],
        'answer_model': protocol['answer_model'], 'judge_model': transport.DEEPSEEK_MODEL,
        'grades': list(shared.GRADES), 'questions_expected': COHORT_SIZE,
        'questions_scored': len(scored), 'questions_missing': COHORT_SIZE - len(scored),
        'source_open_em_correct': protocol['source_open_em_correct'],
        'source_open_em': protocol['source_open_em_correct'] / COHORT_SIZE,
        'score_sum': score_sum, 'llm_judge': score_sum / COHORT_SIZE if complete else None,
        'llm_judge_percent': 100 * score_sum / COHORT_SIZE if complete else None,
        'score_distribution': {str(grade): counts[grade] for grade in shared.GRADES},
        'unique_requests': len(requests), 'unique_requests_scored': len(saved),
        'api_attempt_count': len(attempts),
        'failed_api_attempt_count': sum(row.get('status') != 'ok' for row in attempts),
        'successful_unique_request_usage': {
            key: sum((row.get('usage') or {}).get(key, 0) or 0 for row in saved.values())
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        'by_question_type': group('point'), 'by_scene': group('task'), 'by_source_open_em': group('em'),
        'missing_ids': [row['id'] for row in rows if row['judge_status'] == 'missing'],
        'artifact_sha256': {name: sha(args.output / name) for name in
            ('protocol.json', 'requests.jsonl', 'question_map.jsonl', 'reused_scores.jsonl',
             'scores.jsonl', 'attempts.jsonl') if (args.output / name).exists()}}
    dump(args.output / 'summary.json', report)
    write_rows(args.output / 'per_question.jsonl', rows)
    fields = ['id', 'task', 'point', 'em', 'question', 'ground_truth', 'model_output',
              'source_record_id', 'judge_status', 'score', 'reasoning', 'request_sha256',
              'returned_model', 'response_id']
    shared.write_csv(args.output / 'per_question.csv', fields, rows)
    print(json.dumps({key: report[key] for key in ('status', 'questions_scored', 'questions_missing',
        'llm_judge', 'score_distribution', 'failed_api_attempt_count')}, ensure_ascii=False), flush=True)
    return report


def run(args, protocol, requests, question_map):
    """Adapt the audited shared runner's fsync, crash recovery, and bounded retries."""
    saved = shared.validated_results(args.output, requests)
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
    recovered = []
    for digest, attempt in previous.items():
        if digest not in saved and attempt.get('status') == 'ok':
            score = {key: value for key, value in attempt.items()
                     if key not in ('attempt', 'started_at_unix', 'status', 'retryable')}
            score.update(shared.parse_grade(score['raw_response']), score_origin='new_api',
                         recovered_from_successful_attempt_log=True)
            shared.validate_score(score, digest)
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
                audit = {'request_sha256': digest, 'attempt': number, 'started_at_unix': time.time()}
                result = None
                retryable = False
                try:
                    response = transport.chat_request(config, row['request'], timeout=60)
                    audit.update(response)
                    grade = shared.parse_grade(response['raw_response'])
                    result = {**response, **grade, 'score_origin': 'new_api'}
                    shared.validate_score(result, digest)
                    audit.update(status='ok', retryable=False)
                except transport.APIError as error:
                    retryable = error.retryable
                    audit.update(status='error', error_kind=error.kind,
                        provider_error_code=getattr(error, 'provider_error_code', None), retryable=retryable)
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
                if count % 25 == 0 or count == len(pending) or not success:
                    print(f'Finished {count}/{len(pending)}; successes={succeeded}; '
                          f'last_success={success}', flush=True)
    return analyze(args, protocol, requests, question_map)
