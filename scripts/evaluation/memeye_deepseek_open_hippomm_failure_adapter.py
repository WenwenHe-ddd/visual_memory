"""Seal native HippoMM successes and explicit HTTP failures for Open judging.

This adapter makes no API calls. The signed generator, successful answers and
failed original wire are retained. A failed final answer is represented by an
empty evaluation output and a failure record, never a fabricated API response.
"""
from __future__ import annotations

import json
import gzip
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memeye_deepseek_open_visual_methods as native

MODEL = native.MODEL
OUTPUT = native.DEFAULT / 'hippomm'
read, rows, sha, dump = lambda p: json.loads(Path(p).read_text()), native.read, native.sha, native.dump


class FailedFinalReplay(native.AuditedChat):
    """Replay successful originals; represent the observed failed final call."""

    def __init__(self, directory, output):
        super().__init__(directory, allow_api=False)
        self.output = Path(output)
        self.failure = None

    def __call__(self, role, prompt, image_paths=()):
        request = {'role': role, 'prompt': prompt, 'image_paths': [str(p) for p in image_paths],
                   'backbone': 'deepseek', 'max_output_tokens': 2048}
        if native.runtime.digest(request) in self.index:
            return super().__call__(role, prompt, image_paths)
        if role != 'answer_with_representatives' or self.failure is not None:
            raise ValueError('Only the observed permanent final-answer failure can be adapted')
        call_id = f'call_{len(self.index) + 1:04d}'
        wire_path = self.directory / (call_id + '.wire.json.gz')
        if (self.directory / (call_id + '.json')).exists():
            raise ValueError('A successful original call cannot be replaced by a failure')
        images = [{'path': str(p), 'sha256': sha(p)} for p in image_paths]
        wire = native.gpt.build_request({'system': native.runtime.READER_SYSTEM,
            'user_text': prompt, 'query_images': images}, model=MODEL,
            max_output_tokens=2048, image_detail='high')
        assert json.loads(gzip.decompress(wire_path.read_bytes())) == wire
        key = native.runtime.digest(wire)
        attempts_path = self.directory / 'api_attempts.jsonl'
        attempts = [a for a in rows(attempts_path) if a['call_id'] == call_id]
        assert len(attempts) == 1 and attempts[0]['attempt'] == 1
        attempt = attempts[0]
        assert attempt['status'] == 'error' and attempt['kind'] == 'http'
        assert attempt['http_status'] == 400 and attempt['retryable'] is False
        assert attempt['request_sha256'] == key
        relative = Path('failed_call_records') / self.directory.name / (call_id + '.json')
        self.failure = {'status': 'permanent_http_error', 'call_id': call_id,
            'request': request, 'request_sha256': key,
            'requested_model': MODEL, 'returned_model': None, 'response': None,
            'http_status': 400, 'retryable': False,
            'provider_specific_cause': 'unknown; provider error body/code was not persisted',
            'wire_file': str(wire_path.relative_to(self.output)),
            'wire_file_sha256': sha(wire_path),
            'image_sha256': {r['path']: r['sha256'] for r in images},
            'source_attempts_file': str(attempts_path.relative_to(self.output)),
            'source_attempts_sha256': sha(attempts_path),
            'source_attempt': attempt, 'failure_record_file': str(relative)}
        dump(self.output / relative, self.failure)
        self.last_call_id = call_id
        # Explicit evaluation representation of the absence of a generated answer.
        return ''


def replay_failure(output, index, question, memory, events, vectors, dialogues, images, encode):
    emap = {e['event_id']: e for e in events}
    directory = output / 'llm_cache' / f'Q{index:04d}'
    chat = FailedFinalReplay(directory, output)
    calls, recovery, image_order = [], [], []
    rewrite_count = 0

    def llm(role, prompt):
        nonlocal rewrite_count, image_order
        paths = question['query_image_paths'] if role in ('fast', 'answer') else []
        actual_role = role
        if role == 'answer':
            evidence = json.loads(prompt.rsplit('\n', 1)[-1])
            selected = [e['event_id'] for e in evidence['memories']]
            if selected:
                prompt, paths, image_order = native.hip_original.reader_payload(question,
                    {'route': 'detailed', 'selected_event_ids': selected},
                    emap, images, dialogues, memory)
                actual_role = 'answer_with_representatives'
        value = chat(actual_role, prompt, paths)
        calls.append(chat.last_call_id)
        if role == 'rewrite':
            rewrite_count += 1
            if rewrite_count == 3:
                value = native.hip_api.repaired_rewrite(chat, value,
                    native.hip_core.query_prompt(question), calls, recovery)
        return value

    trace = native.hip_core.answer_question(question, events, vectors, dialogues, llm, encode)
    failure = chat.failure
    assert failure is not None and trace['answer'] == '' and trace['route'] == 'detailed'
    assert len(calls) == 68 and calls[-1] == failure['call_id']
    success_calls = [str((directory / (c + '.json')).relative_to(output)) for c in calls[:-1]]
    failed_original_path = str((directory / (calls[-1] + '.json')).relative_to(output))
    failed_call = {'path': failed_original_path, 'request': failure['request'],
        'request_sha256': failure['request_sha256'], 'wire_file': failure['wire_file'],
        'wire_file_sha256': failure['wire_file_sha256'],
        'attempts_file': failure['source_attempts_file'], 'call_id': failure['call_id'],
        'error': {'kind': 'http', 'http_status': 400, 'retryable': False}}
    trace.update(point=question['point'], question=question['question'],
        raw_response='', answer_model=MODEL, model=MODEL, backbone='deepseek',
        requested_model=MODEL, returned_model=None,
        status='permanent_http_error', http_status=400,
        model_identity_kind='provider_alias', resolved_model_revision=None,
        llm_calls=success_calls + [failed_original_path], failed_call=failed_call,
        failed_call_record_file=failure['failure_record_file'],
        query_format_recovery=recovery, reader_image_order=image_order,
        historical_image_reads=sum(r['kind'] == 'historical_representative' for r in image_order),
        hit_token_limit=False, source_mcq_correct=207,
        payload_sha256=failure['request_sha256'], request_sha256=failure['request_sha256'],
        failed_call_wire=failure['wire_file'], failed_call_id=failure['call_id'],
        source_error_attempt_file=failure['source_attempts_file'],
        source_error_attempt_sha256=failure['source_attempts_sha256'],
        evaluation_adaptation='Permanent final HTTP failure retained as empty model output; Judge supplies the score.')
    assert not set(trace) & {'response_id', 'usage', 'input_tokens', 'generated_tokens'}
    return trace


def validate_calls(output, names):
    """Independently verify every original success, wire and attempt."""
    calls, attempts_hashes, roles, transient_errors = {}, {}, Counter(), Counter()
    for name in sorted(set(names)):
        path = output / name
        call = read(path)
        assert call.get('status') != 'permanent_http_error'
        request, response = call['request'], call['response']
        assert request['backbone'] == 'deepseek' and request['max_output_tokens'] == 2048
        images = [{'path': p, 'sha256': sha(p)} for p in request['image_paths']]
        assert {r['path']: r['sha256'] for r in images} == call['image_sha256']
        body = native.gpt.build_request({'system': native.runtime.READER_SYSTEM,
            'user_text': request['prompt'], 'query_images': images}, model=MODEL,
            max_output_tokens=2048, image_detail='high')
        wire_path = path.with_suffix('.wire.json.gz')
        assert json.loads(gzip.decompress(wire_path.read_bytes())) == body
        assert sha(wire_path) == call['wire_file_sha256']
        assert native.runtime.digest(body) == call['request_sha256']
        assert response['returned_model'] == response['requested_model'] == MODEL
        assert response['seed'] == 42 and response['enable_thinking'] is False
        assert response['finish_reason'] == 'stop' and not response.get('refusal')
        assert response['raw_response'].strip()
        attempts_path = path.parent / 'api_attempts.jsonl'
        attempts = [a for a in rows(attempts_path) if a['call_id'] == call['call_id']]
        assert 1 <= len(attempts) <= 4
        assert [a['attempt'] for a in attempts] == list(range(1, len(attempts) + 1))
        successes = [a for a in attempts if a['status'] == 'success']
        assert len(successes) == 1 and attempts[-1]['status'] == 'success'
        assert successes[0]['response'] == response
        assert all(a['request_sha256'] == call['request_sha256'] for a in attempts)
        for a in attempts:
            if a['status'] == 'error':
                assert a['retryable']
                transient_errors[a['kind']] += 1
        roles[request['role']] += 1
        calls[name] = sha(path)
        attempts_hashes[str(attempts_path.relative_to(output))] = sha(attempts_path)
    return {'successful_native_calls': len(calls), 'roles': dict(roles),
        'transient_errors': dict(transient_errors), 'call_sha256': calls,
        'attempts_sha256': attempts_hashes}


def seal(output=OUTPUT):
    output = Path(output)
    status = read(output / 'generation_status.json')
    assert status['status'] == 'partial' and status['answered'] == 370 and status['questions'] == 371
    source_bytes = (output / 'responses.jsonl').read_bytes()
    assert source_bytes.endswith(b'\n')
    original = rows(output / 'responses.jsonl')
    assert len(original) == len({r['id'] for r in original}) == 370
    assert status['responses_sha256'] == sha(output / 'responses.jsonl')
    protocol, questions, memory, events, vectors, dialogues, images = native.load_hippo(output)
    by_id = {r['id']: r for r in original}
    failed = [q for q in questions if q['id'] not in by_id]
    assert [q['id'] for q in failed] == ['Cartoon_Entertainment_Companion/COMIC_Q48']
    assert {e['id'] for e in status['errors_this_pass']} == {q['id'] for q in failed}
    import torch
    torch.set_num_threads(8)
    encoder = native.hip_base.ClipText('cpu')
    names, failures = [], []
    for i, q in enumerate(questions, 1):
        if q['id'] in by_id:
            replay = native.native_answer(output, i, q, memory, events, vectors,
                dialogues, images, encoder, allow_api=False)
            assert replay == by_id[q['id']], 'Original successful native replay differs: ' + q['id']
            names.extend(replay['llm_calls'])
        else:
            replay = replay_failure(output, i, q, memory, events, vectors, dialogues, images, encoder)
            failures.append(replay)
            names.extend(replay['llm_calls'][:-1])
    verified = validate_calls(output, names)
    value = source_bytes + b''.join((json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n').encode() for r in failures)
    projection = output / 'evaluation_responses.jsonl'
    if projection.exists():
        assert projection.read_bytes() == value
    else:
        projection.write_bytes(value)
    assert len(rows(projection)) == 371
    sources = {str(p): sha(p) for p in [Path(__file__), Path(native.__file__),
        output / 'protocol.json', output / 'generation_status.json', output / 'responses.jsonl',
        output / 'question_errors.jsonl', output / 'questions.jsonl',
        output / 'http400_Q0149_diagnostic.json']}
    validation = {'status': 'complete', 'stage': 'evaluation_records', 'questions': 371,
        'answered': 370, 'successful_records': 370, 'provider_rejections': 1, 'source_mcq_correct': 207,
        'answer_model': MODEL, 'successful_full_native_replay': 370,
        'failed_full_native_replay_until_original_wire': 1,
        'successful_native_full_replay': True, 'failed_native_prefix_replay': True,
        'all_371_native_pipelines_accounted_for': True,
        'original_successes_byte_preserved': True, 'gold_accessed': False,
        'no_failed_api_retries_or_synthetic_successes': True,
        'failure_policy': 'Unknown-cause permanent HTTP400 final answer becomes explicitly empty evaluation output; no grade imputation.',
        'failed_ids': [r['id'] for r in failures],
        'responses_file': projection.name, 'responses_sha256': sha(projection),
        'native_success_responses_sha256': sha(output / 'responses.jsonl'),
        'source_sha256': sources, **verified}
    dump(output / 'evaluation_validation.json', validation)
    dump(output / 'completion.json', {'status': 'complete', 'stage': 'evaluation_records',
        'questions': 371, 'answered': 370, 'provider_rejections': 1,
        'responses_file': projection.name, 'responses_sha256': sha(projection),
        'native_success_responses_file': 'responses.jsonl',
        'native_success_responses_sha256': sha(output / 'responses.jsonl'),
        'validation_file': 'evaluation_validation.json',
        'validation_sha256': sha(output / 'evaluation_validation.json'),
        'token_limit_hits': 0, 'empty_answers': 1,
        'format_recovery_ids': [r['id'] for r in [*original, *failures] if r['query_format_recovery']],
        'failed_ids': [r['id'] for r in failures]})
    print(json.dumps({k: v for k, v in validation.items()
                      if k not in ('source_sha256', 'call_sha256', 'attempts_sha256')}), flush=True)
    collect(output)


def collect(output=OUTPUT):
    output = Path(output)
    seal = read(output / 'completion.json')
    assert seal['stage'] == 'evaluation_records' and seal['answered'] == 370
    source = output / seal['responses_file']
    assert sha(source) == seal['responses_sha256']
    assert sha(output / seal['validation_file']) == seal['validation_sha256']
    gold_path = native.OPEN.parent / 'gold.jsonl'
    gold = {r['id']: r for r in rows(gold_path)}
    values = {r['id']: r for r in rows(source)}
    assert len(gold) == len(values) == 371 and set(gold) == set(values)
    result = []
    for q in native.questions():
        r = values[q['id']]
        result.append({'id': q['id'], 'task': q['task'], 'point': q['point'],
            'question': q['question'], 'ground_truth': gold[q['id']]['answer'],
            'model_output': r['raw_response'], 'answer_model': MODEL,
            'source_record_id': q['id'], 'source_hit_token_limit': r['hit_token_limit'],
            'source_responses_sha256': seal['responses_sha256'],
            'source_protocol_sha256': sha(output / 'protocol.json'),
            'source_payload_sha256': r['payload_sha256'], 'source_native_route': r['route'],
            'source_format_recovery': bool(r['query_format_recovery']),
            'source_status': r.get('status', 'answered')})
    native.write(output / 'judge_inputs.jsonl', result)
    files = [output / n for n in ('protocol.json', 'responses.jsonl', 'generation_status.json',
        'question_errors.jsonl', 'completion.json', 'evaluation_validation.json',
        'evaluation_responses.jsonl', 'judge_inputs.jsonl', 'questions.jsonl')]
    files += [native.OPEN, gold_path, Path(__file__), Path(native.__file__)]
    files += list((output / 'failed_call_records').glob('*/call_*.json'))
    dump(output / 'collection_protocol.json', {'status': 'complete', 'method_id': 'hippomm',
        'display_label': 'HippoMM', 'dataset': 'MemEye Open', 'questions': 371,
        'answer_model': MODEL, 'reference_mcq_correct': 207, 'max_completion_tokens': 2048,
        'source_responses_file': seal['responses_file'],
        'source_responses_sha256': seal['responses_sha256'],
        'native_success_responses_file': 'responses.jsonl',
        'native_success_responses_sha256': seal['native_success_responses_sha256'],
        'native_success_answers': 370, 'provider_rejections': 1,
        'raw_successful_answers_unchanged': True,
        'empty_failure_output_requires_real_judge': True,
        'failure_cause': 'Unknown-cause permanent HTTP400; not asserted to be content filtering.',
        'gold_read_only_after_answers_sealed': True, 'generic_reader_rerun': False,
        'format_recovery_ids': seal['format_recovery_ids'],
        'source_sha256': {str(p): sha(p) for p in files}})


if __name__ == '__main__':
    seal()
