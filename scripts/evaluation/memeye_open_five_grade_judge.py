"""Judge 371 frozen MemEye open-ended answers with the five-grade rubric.

Only the original question, textual reference answer, and generated answer are
sent to the judge. Retrieval, answer generation, images, and MCQ options are not
part of this stage. The rubric and audited DeepSeek transport are shared with
the MemGallery evaluation; aggregation here uses the MemEye cohort of 371.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memgallery_multi_method_judge as shared

transport = shared.transport
sha, read_rows, dump, write_rows = shared.sha, shared.read_rows, shared.dump, shared.write_rows
COHORT_SIZE = 371
ANSWER_MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
FIELDS = ('id', 'task', 'point', 'question', 'ground_truth', 'model_output', 'source_record_id')


def prepare(args):
    source = args.input.resolve()
    records = read_rows(source)
    if len(records) != COHORT_SIZE or len({row.get('id') for row in records}) != COHORT_SIZE:
        raise ValueError('Expected exactly 371 unique MemEye question IDs')
    template = shared.PROMPT.read_text()
    requests, question_map = {}, []
    reserved = {'request', 'request_sha256', 'score', 'reasoning', 'score_origin',
                'judge_status', 'returned_model', 'requested_model'}
    for source_row in records:
        if any(not isinstance(source_row.get(key), str) for key in FIELDS):
            raise ValueError('Judge input fields must contain original strings')
        if any(not source_row[key].strip() for key in FIELDS if key != 'model_output'):
            raise ValueError('Only the original model answer may be empty')
        if reserved.intersection(source_row):
            raise ValueError('Input contains reserved judge output fields')
        if source_row.get('answer_model', ANSWER_MODEL) != ANSWER_MODEL:
            raise ValueError('This evaluation requires the Qwen3-VL-8B answer backbone')
        em = int(shared.normalize_answer(source_row['model_output']) ==
                 shared.normalize_answer(source_row['ground_truth']))
        if 'em' in source_row and (type(source_row['em']) not in (int, bool) or source_row['em'] != em):
            raise ValueError('Input open-ended EM does not reproduce')
        request = shared.build_request(source_row, template)
        digest = transport.request_sha256(request)
        if digest in requests:
            if requests[digest]['request'] != request:
                raise ValueError('Request hash collision')
            requests[digest]['question_ids'].append(source_row['id'])
        else:
            requests[digest] = {'request_sha256': digest, 'request': request,
                **{key: source_row[key] for key in ('question', 'ground_truth', 'model_output')},
                'point': source_row['point'], 'question_ids': [source_row['id']]}
        question_map.append({**source_row, 'em': em, 'request_sha256': digest})
    source_files = [source]
    if args.source_protocol is not None:
        source_files.append(args.source_protocol.resolve())
    code_paths = [Path(__file__), Path(shared.__file__), Path(shared.original.__file__),
                  Path(transport.__file__), ROOT / 'scripts/evaluation/openai_chat_transport.py',
                  ROOT / 'scripts/evaluation/mem_gallery_master_reader.py']
    protocol = {'dataset': 'MemEye', 'task': 'open_ended_answering',
        'method': 'Semantic RAG (caption baseline)', 'answer_model': ANSWER_MODEL,
        'questions': COHORT_SIZE, 'source_open_em_correct': sum(row['em'] for row in question_map),
        **shared.judge_settings(), 'judge_model_identity_kind': 'provider_alias',
        'official_prompt_url': shared.original.PROMPT_URL,
        'parser_difference_from_public_code': 'Keep all five grades exactly; no three-level remapping.',
        'aggregation': 'Arithmetic mean over all 371 questions; no missing-score imputation.',
        'original_answer_policy': 'Keep every frozen answer, including empty or truncated outputs.',
        'deduplication': 'One grade per exact serialized request hash; all 371 questions remain in the mean.',
        'cross_dataset_score_cache_reuse': False, 'unique_requests': len(requests),
        'max_attempts_per_unique_request': args.max_attempts,
        'input_path': str(source), 'source_sha256': {str(path): sha(path) for path in source_files},
        'implementation_sha256': {str(path.resolve()): sha(path) for path in code_paths}}
    frozen = {'requests.jsonl': list(requests.values()), 'question_map.jsonl': question_map,
              'reused_scores.jsonl': []}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing protocol differs; use a new output root')
        for filename, rows in frozen.items():
            if read_rows(args.output / filename) != rows:
                raise ValueError(f'Frozen artifact differs: {filename}')
        if (args.output / 'judge_prompt.txt').read_bytes() != shared.PROMPT.read_bytes():
            raise ValueError('Frozen judge prompt differs')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Output directory is nonempty without protocol')
        for filename, rows in frozen.items():
            write_rows(args.output / filename, rows)
        (args.output / 'judge_prompt.txt').write_bytes(shared.PROMPT.read_bytes())
        dump(protocol_path, protocol)
    return protocol, list(requests.values()), question_map


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
        'answer_model': ANSWER_MODEL, 'judge_model': transport.DEEPSEEK_MODEL,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'run', 'analyze'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-protocol', type=Path,
                        help='Reader protocol to hash alongside the frozen answer inputs')
    parser.add_argument('--env-file', type=Path,
                        default=Path('/storage/hewenwen/NTU Decentralized Agent/.env'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-attempts', type=int, default=5)
    parser.add_argument('--limit', type=int, default=0,
                        help='New unique requests for a smoke test; zero means all pending')
    args = parser.parse_args()
    if args.workers < 1 or args.max_attempts < 1 or args.limit < 0:
        parser.error('workers/max-attempts must be positive; limit must be nonnegative')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / (args.output.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, requests, question_map = prepare(args)
        report = (run if args.phase == 'run' else analyze)(args, protocol, requests, question_map)
        if args.phase == 'run' and report['status'] != 'complete' and not args.limit:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
