"""Verify native successes and actual provider-rejected requests without inventing outputs."""
import gzip
import hashlib
import json
from pathlib import Path

MODEL = 'deepseek-v4.1-flash'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha(path, expected=None):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Provider-rejection provenance changed: ' + str(path))
    return digest


def hashes(values):
    for path, expected in values.items():
        sha(path, expected)


def unique(values):
    result = {r['id']: r for r in values}
    if len(result) != len(values):
        raise ValueError('Duplicate native evaluation ID')
    return result


def source_path(root):
    protocol = read(root / 'collection_protocol.json')
    name = protocol.get('source_responses_file', 'responses.jsonl')
    if name not in ('responses.jsonl', 'evaluation_responses.jsonl'):
        raise ValueError('Unexpected native evaluation source file')
    return root / name


def hippo_answers(root, method, questions):
    from scripts.evaluation import memeye_deepseek_open_visual_methods as native
    from scripts.memory.hippomm_core import parse_fast
    collection = read(root / 'collection_protocol.json')
    validation = read(root / 'evaluation_validation.json')
    independent = read(root / 'independent_output_audit.json')
    seal = read(root / 'completion.json')
    answers = unique(rows(source_path(root)))
    originals = unique(rows(root / 'responses.jsonl'))
    rejected = {qid: r for qid, r in answers.items() if r.get('status') == 'permanent_http_error'}
    if (set(answers) != set(questions) or len(answers) != 371 or not rejected
            or set(originals) != set(answers) - set(rejected)
            or any(answers[qid] != row for qid, row in originals.items())):
        raise ValueError('Native evaluation projection changes original successful answers')
    if (collection['status'] != 'complete' or collection['answer_model'] != MODEL
            or collection['reference_mcq_correct'] != method['source_mcq_correct_for_identification_only']
            or collection['provider_rejections'] != len(rejected)
            or validation['successful_records'] != len(originals)
            or validation['provider_rejections'] != len(rejected)
            or validation['successful_native_full_replay'] is not True
            or validation['failed_native_prefix_replay'] is not True
            or independent['status'] != 'complete' or independent['questions'] != 371
            or independent['successful_records'] != len(originals)
            or independent['provider_rejections'] != len(rejected)
            or independent['successful_native_full_replay'] is not True
            or independent['failed_native_prefix_replay'] is not True
            or seal['answered'] != len(originals) or seal['provider_rejections'] != len(rejected)):
        raise ValueError('Native successful and rejected cohort counts do not reproduce')
    for document in (collection, validation, read(root / 'protocol.json')):
        hashes(document['source_sha256'])
    for field in ('attempt_files_sha256', 'call_files_sha256', 'failed_wire_files_sha256'):
        hashes(independent[field])
    sha(root / 'responses.jsonl', collection['native_success_responses_sha256'])
    sha(source_path(root), collection['source_responses_sha256'])
    sha(source_path(root), seal['responses_sha256'])
    sha(source_path(root), independent['responses_sha256'])
    sha(root / 'completion.json', independent['completion_sha256'])
    sha(root / 'protocol.json', independent['protocol_sha256'])
    sha(root / 'evaluation_validation.json', independent['evaluation_validation_sha256'])
    checked, provenance = {}, {}

    def rebuilt(request):
        return native.gpt.build_request({'system': native.runtime.READER_SYSTEM,
            'user_text': request['prompt'], 'query_images': [
                {'path': p, 'sha256': sha(p)} for p in request['image_paths']]},
            model=MODEL, max_output_tokens=2048, image_detail='high')

    for qid, row in answers.items():
        if (any(row[k] != questions[qid][k] for k in ('id', 'task', 'point', 'question'))
                or row['answer_model'] != MODEL or row['model'] != MODEL):
            raise ValueError('Native evaluation identity changed')
        failure = row.get('failed_call') if qid in rejected else None
        for name in row['llm_calls']:
            if failure and name == failure['path']:
                continue
            if name in checked:
                continue
            path = root / name
            call = read(path)
            wire = json.loads(gzip.decompress(path.with_suffix('.wire.json.gz').read_bytes()))
            sha(path.with_suffix('.wire.json.gz'), call['wire_file_sha256'])
            hashes(call['image_sha256'])
            if wire != rebuilt(call['request']) or native.runtime.digest(wire) != call['request_sha256']:
                raise ValueError('Native success wire differs')
            raw = call['response']
            attempts = [a for a in rows(path.parent / 'api_attempts.jsonl') if a['call_id'] == call['call_id']]
            successes = [a for a in attempts if a['status'] == 'success']
            if (len(successes) != 1 or successes[0]['response'] != raw
                    or raw['returned_model'] != MODEL or raw['requested_model'] != MODEL
                    or raw['finish_reason'] != 'stop' or raw.get('refusal')):
                raise ValueError('Native answer lacks its original successful raw attempt')
            checked[name] = call
        if failure:
            if (row['llm_calls'][-1] != failure['path'] or (root / failure['path']).exists()
                    or row['raw_response'] != '' or row['answer'] != ''
                    or row.get('returned_model') is not None or row.get('response_id')
                    or row.get('usage') or row['hit_token_limit']):
                raise ValueError('Rejected request has a fabricated model answer')
            wire_path = root / failure['wire_file']
            sha(wire_path, failure['wire_file_sha256'])
            wire = json.loads(gzip.decompress(wire_path.read_bytes()))
            if wire != rebuilt(failure['request']) or native.runtime.digest(wire) != failure['request_sha256']:
                raise ValueError('Rejected final native request differs')
            attempts_path = root / failure['attempts_file']
            attempts = [a for a in rows(attempts_path) if a['call_id'] == failure['call_id']]
            if (not attempts or any(a['status'] == 'success' for a in attempts)
                    or attempts[-1]['status'] != 'error' or attempts[-1]['http_status'] != 400
                    or attempts[-1]['retryable'] is not False
                    or attempts[-1]['request_sha256'] != failure['request_sha256']):
                raise ValueError('Empty native evaluation output lacks a permanent HTTP400 attempt')
            path, key, returned = attempts_path, failure['request_sha256'], None
        else:
            name = row['llm_calls'][-1]
            final = checked[name]
            expected = (parse_fast(final['response']['raw_response'], row['fast_context_event_ids'])['answer'].strip()
                        if row['route'] == 'fast' else final['response']['raw_response'].strip())
            if expected != row['raw_response'] or expected != row['answer']:
                raise ValueError('Original native final answer changed')
            path, key, returned = root / name, final['request_sha256'], MODEL
        if row['payload_sha256'] != key:
            raise ValueError('Native final request identity differs')
        provenance[qid] = {'source_answer_api_request_sha256': key,
            'source_answer_record': str(path), 'source_answer_record_sha256': sha(path),
            'answer_model_revision': None, 'answer_returned_model': returned,
            'source_status': 'permanent_http_error' if failure else 'answered'}
    return answers, provenance
