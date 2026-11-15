"""Audit sealed native HippoMM Open answers without retrieval, generation or judging."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.memory import hippomm_core as core, hippomm_original as original

RUN = ROOT / 'results/memeye_qwen_other_open_20261105'
HIP = RUN / 'visual_methods/hippomm'
CANONICAL = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs'
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def keyed(values):
    result = {v['id']: v for v in values}
    assert len(result) == len(values)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    collection = read(HIP / 'collection_protocol.json')
    seal, protocol = read(HIP / 'completion.json'), read(HIP / 'protocol.json')
    assert collection['status'] == seal['status'] == 'complete'
    assert collection['questions'] == seal['answered'] == 371
    assert collection['method_id'] == protocol['method_id'] == 'hippomm'
    assert collection['answer_model'] == protocol['answer_model'] == MODEL
    assert collection['max_new_tokens'] == protocol['max_new_tokens'] == 1024
    assert seal['responses_sha256'] == sha(HIP / 'responses.jsonl')
    verified_hashes = {}
    for contract in (collection, protocol):
        for path, expected in contract['source_sha256'].items():
            actual = sha(path)
            assert actual == expected, path
            if path in verified_hashes:
                assert verified_hashes[path] == actual
            verified_hashes[path] = actual
    answers = keyed(rows(HIP / 'responses.jsonl'))
    questions, gold = keyed(rows(CANONICAL / 'questions.jsonl')), keyed(rows(CANONICAL / 'gold.jsonl'))
    judge = keyed(rows(HIP / 'judge_inputs.jsonl'))
    assert len(answers) == len(questions) == len(gold) == len(judge) == 371
    assert set(answers) == set(questions) == set(gold) == set(judge)
    assert keyed(rows(HIP / 'questions.jsonl')) == questions
    service = read(HIP / 'service_provenance.json')
    assert service['service_signature'] == read(RUN / 'qwen_service/protocol.json')['experiment_signature']
    memory = Path(protocol['memory_dir'])
    events = {r['event_id']: r for r in rows(memory / 'events.jsonl')}
    images = {r['event_id']: r for r in rows(memory / 'representative_images.jsonl')}
    dialogues = keyed(rows(memory / 'dialogues.jsonl'))
    assert len(events) == len(images) == 250
    checked_calls, checked_images, routes, recovered = {}, {}, Counter(), []
    extra_service_sources = {}
    historical_errors = rows(HIP / 'question_errors.jsonl') if (HIP / 'question_errors.jsonl').exists() else []

    def check_call(key, snapshot=None):
        if key in checked_calls:
            if snapshot is not None:
                assert read(snapshot) == checked_calls[key][0]
            return checked_calls[key]
        cache_path = Path(snapshot) if snapshot is not None else HIP / 'llm_cache' / (key + '.json')
        cache = read(cache_path)
        request, output = cache['request'], cache['response']
        assert cache['request_sha256'] == digest(request) == key
        assert request['max_new_tokens'] == 1024
        image_parts = [part for msg in request['reader_messages'] if isinstance(msg['content'], list)
                       for part in msg['content'] if part['type'] == 'image']
        paths = [p['path'] for p in image_parts]
        hashes = {}
        for part in image_parts:
            path = part['path']
            if path not in checked_images:
                checked_images[path] = sha(path)
            assert part['sha256'] == checked_images[path]
            hashes[path] = checked_images[path]
        assert request['reader_image_paths'] == paths
        assert request['reader_image_hashes'] == hashes
        assert request['payload_sha256'] == digest({'messages': request['reader_messages'], 'image_hashes': hashes})
        assert output['input_tokens'] + 1024 <= protocol['context_budget']
        wire = {'conversations': [request['reader_messages']], 'cap': 1024,
                'sampling_temperature': 0.0, 'service_signature': service['service_signature']}
        wire_sha = digest(wire)
        service_path = RUN / 'qwen_service/calls' / (wire_sha + '.json')
        actual = read(service_path)
        assert actual['request'] == wire and actual['outputs'] == [output]
        assert str(cache_path) in collection['source_sha256']
        if str(service_path) not in collection['source_sha256']:
            # Original failed calls are frozen through recovery snapshots. Also
            # lock their actual service record in this independent audit.
            extra_service_sources[str(service_path)] = sha(service_path)
        checked_calls[key] = (cache, wire_sha)
        return cache, wire_sha

    for qid, answer in answers.items():
        question, scored = questions[qid], judge[qid]
        assert not {'answer', 'options', 'gold_ids', 'gold_choice'} & question.keys()
        for field in ('question', 'task', 'point'):
            assert answer[field] == question[field] == scored[field]
        assert scored['ground_truth'] == gold[qid]['answer']
        assert scored['model_output'] == answer['raw_response'] == answer['answer']
        assert answer['answer_model'] == scored['answer_model'] == MODEL
        assert scored['source_record_id'] == qid
        assert scored['source_responses_sha256'] == sha(HIP / 'responses.jsonl')
        assert scored['source_protocol_sha256'] == sha(HIP / 'protocol.json')
        assert scored['source_native_route'] == answer['route']
        calls = [check_call(key) for key in answer['llm_calls']]
        first, last = calls[0][0], calls[-1][0]
        assert first['request']['role'] == 'fast'
        local = [e for e in events.values() if e['task'] == question['task']]
        assert first['request']['reader_messages'][1]['content'][-1]['text'] == core.fast_prompt(question, local)
        assert first['request']['reader_image_paths'] == question['query_image_paths']
        try:
            fast = core.parse_fast(first['response']['raw_response'], [e['event_id'] for e in local])
            fast_error = None
        except (ValueError, TypeError) as exc:
            fast = {'answer': None, 'confidence': 0.0, 'evidence_event_ids': []}
            fast_error = str(exc)
        assert fast == answer['fast'] and fast_error == answer['fast_parse_error']
        if answer['route'] == 'fast':
            assert fast['answer'] is not None and fast['confidence'] > protocol['confidence_threshold']
            assert answer['raw_response'] == fast['answer']
            assert len(calls) == 1 and answer['historical_image_reads'] == 0
        else:
            assert answer['route'] == 'detailed'
            assert fast['answer'] is None or fast['confidence'] <= protocol['confidence_threshold']
            assert answer['raw_response'] == last['response']['raw_response'].strip()
            assert last['request']['role'] in ('answer', 'answer_with_representatives')
            ranked = sorted([r for r in answer['detail_candidates'] if r['selected']],
                            key=lambda r: (-r['score'], r['order']))[:5]
            assert ranked == answer['ranked_candidates']
            assert [r['event_id'] for r in ranked] == answer['selected_event_ids']
            if ranked:
                prompt, paths, order = original.reader_payload(question, answer, events, images, dialogues, memory)
                assert last['request']['reader_messages'][1]['content'][-1]['text'] == prompt
                assert last['request']['reader_image_paths'] == paths
                assert answer['reader_image_order'] == order
                assert answer['historical_image_reads'] == len(ranked)
            else:
                assert last['request']['role'] == 'answer'
                assert last['request']['reader_image_paths'] == question['query_image_paths']
        for call, _ in calls:
            if call['request']['role'] in ('rewrite', 'caption_fallback'):
                assert not call['request']['reader_image_paths']
        for field in ('input_tokens', 'generated_tokens', 'hit_token_limit'):
            assert answer[field] == last['response'][field]
        assert answer['payload_sha256'] == last['request']['payload_sha256'] == scored['source_payload_sha256']
        assert scored['source_service_request_sha256'] == calls[-1][1]
        assert scored['source_hit_token_limit'] == answer['hit_token_limit']
        assert scored['source_format_recovery_protocol_sha256'] == answer.get('format_recovery_protocol_sha256')
        routes[answer['route']] += 1
        if not answer.get('format_recovery_dir'):
            continue
        directory = Path(answer['format_recovery_dir'])
        assert directory.resolve().is_relative_to((HIP / 'format_recovery').resolve())
        recovery = read(directory / 'recovery_protocol.json')
        assert sha(directory / 'recovery_protocol.json') == answer['format_recovery_protocol_sha256']
        assert recovery['eligible_id'] == qid
        for path, expected in recovery['source_sha256'].items():
            assert sha(path) == expected
        failure = read(directory / 'original_failure.json')
        assert failure['id'] == qid and failure['error'].startswith('rewrite: invalid output after 3 attempts:')
        assert failure in historical_errors
        assert failure['llm_calls'][0] == answer['llm_calls'][0]
        assert recovery['original_call_hashes'] == failure['llm_calls']
        assert len(failure['llm_calls']) == 4
        previous_prompt = core.query_prompt(question)
        for failed_key in failure['llm_calls'][1:]:
            snapshot = directory / 'original_calls' / (failed_key + '.json')
            failed, _ = check_call(failed_key, snapshot)
            assert failed['request']['role'] == 'rewrite' and not failed['request']['reader_image_paths']
            assert failed['request']['reader_messages'][1]['content'][-1]['text'] == previous_prompt
            raw = failed['response']['raw_response']
            try:
                core.parse_query(raw)
            except (ValueError, TypeError) as exc:
                parse_error = str(exc)
            else:
                raise AssertionError('Recovered a rewrite that was already valid')
            previous_prompt = (core.query_prompt(question) + '\nYour previous output failed validation: '
                               + parse_error + '\nPrevious output: ' + raw
                               + '\nReturn corrected JSON only. Use only the original evidence; do not guess answer attributes.')
        assert failure['error'] == 'rewrite: invalid output after 3 attempts: ' + parse_error
        first_key = answer['llm_calls'][0] + '.json'
        assert sha(directory / 'llm_cache' / first_key) == sha(HIP / 'llm_cache' / first_key)
        child, = rows(directory / 'responses.jsonl')
        assert all(answer[k] == v for k, v in child.items())
        recovery_seal = read(directory / 'recovery_completion.json')
        assert recovery_seal['status'] == 'complete' and recovery_seal['eligible_id'] == qid
        assert recovery_seal['answered'] == 1 and recovery_seal['questions'] == 1
        assert recovery_seal['responses_sha256'] == sha(directory / 'responses.jsonl')
        assert recovery_seal['recovery_protocol_sha256'] == sha(directory / 'recovery_protocol.json')
        attempts = rows(directory / 'recovery_attempts.jsonl')
        is_array = recovery['kind'] == 'open_word_array_format_adapter'
        is_fields = recovery['kind'] == 'open_three_field_format_adapter'
        assert is_array or is_fields or recovery['kind'] == 'open_rewrite_format_recovery'
        if is_fields:
            assert qid == 'Personal_Health_Dashboard_Assistant/Q49'
            assert recovery['max_distinct_adapter_requests'] == 1
        budget = 1 if is_fields else 2 if is_array else 3
        assert recovery['max_new_rewrite_prompts'] == budget and 1 <= len(attempts) <= budget
        assert [a['attempt'] for a in attempts] == list(range(1, len(attempts) + 1))
        assert attempts[-1]['valid'] and not any(a['valid'] for a in attempts[:-1])
        keys = {a['request_sha256'] for a in attempts}
        assert keys <= set(answer['llm_calls'])
        for attempt in attempts:
            call, _ = check_call(attempt['request_sha256'])
            assert call['response']['raw_response'] == attempt['raw_response']
            text = call['request']['reader_messages'][1]['content'][-1]['text']
            if is_fields:
                assert 'Question: ' + question['question'] in text
                assert 'word_1, word_2, word_3' in text
                assert 'ONE FINAL FIELD-FORMAT REQUEST; no further retries.' in text
                assert 'Previous invalid response:' not in text
            elif is_array:
                assert 'Question: ' + question['question'] in text
                assert '2 to 5 English word strings' in text and 'Previous invalid response:' not in text
            else:
                assert text.startswith(core.query_prompt(question))
        if is_fields:
            parsed = core.parse_object(attempts[-1]['raw_response'])
            assert set(parsed) == {'word_1', 'word_2', 'word_3'}
            words = [parsed[key] for key in ('word_1', 'word_2', 'word_3')]
            assert all(isinstance(word, str) and word.strip() == word and len(word.split()) == 1 for word in words)
            query = ' '.join(words)
            assert core.parse_query(json.dumps({'visual_query': query})) == answer['visual_query']
            assert attempts[-1]['words'] == words and attempts[-1]['adapted_query'] == query
            previous_directory = Path(recovery['previous_array_recovery_dir'])
            assert previous_directory.resolve().is_relative_to((HIP / 'format_recovery').resolve())
            previous_attempts = rows(previous_directory / 'recovery_attempts.jsonl')
            previous_keys = {attempt['request_sha256'] for attempt in previous_attempts}
            assert len(previous_attempts) == len(previous_keys) == 2 and not previous_keys & keys
            assert previous_keys == set(recovery['previous_array_recovery_unique_requests'])
            assert not recovery['previous_phrase_recovery_unique_requests']
            previous_seal = read(previous_directory / 'recovery_completion.json')
            assert previous_seal['status'] == 'failed' and previous_seal['answered'] == 0
            for prior in previous_attempts:
                assert not prior['valid']
                failed, _ = check_call(prior['request_sha256'],
                    previous_directory / 'llm_cache' / (prior['request_sha256'] + '.json'))
                assert failed['response']['raw_response'] == prior['raw_response']
                old_text = failed['request']['reader_messages'][1]['content'][-1]['text']
                assert 'Question: ' + question['question'] in old_text and '2 to 5 English word strings' in old_text
                value = core.parse_object(prior['raw_response'])
                old_words = value.get('words')
                valid_array = (set(value) == {'words'} and isinstance(old_words, list)
                    and 2 <= len(old_words) <= 5
                    and all(isinstance(w, str) and w.strip() == w and len(w.split()) == 1 for w in old_words))
                assert not valid_array, 'Prior array request was already valid'
            keys |= previous_keys
            assert len(keys) == 3
            assert recovery_seal['previous_array_adapter_requests'] == 2
            assert recovery_seal['new_format_adapter_requests'] == 1
        elif is_array:
            words = core.parse_object(attempts[-1]['raw_response'])['words']
            assert 2 <= len(words) <= 5
            assert all(isinstance(word, str) and word.strip() == word and len(word.split()) == 1 for word in words)
            query = ' '.join(words)
            assert core.parse_query(json.dumps({'visual_query': query})) == answer['visual_query']
            assert attempts[-1]['words'] == words and attempts[-1]['adapted_query'] == query
            previous = set(recovery['previous_phrase_recovery_unique_requests'])
            assert len(previous) <= 3 and not keys & previous
            actual_previous = set()
            for previous_directory in recovery.get('previous_recovery_dirs', []):
                previous_directory = Path(previous_directory)
                for prior in rows(previous_directory / 'recovery_attempts.jsonl'):
                    assert not prior['valid']
                    actual_previous.add(prior['request_sha256'])
                    failed, _ = check_call(prior['request_sha256'],
                        previous_directory / 'llm_cache' / (prior['request_sha256'] + '.json'))
                    assert failed['response']['raw_response'] == prior['raw_response']
                    assert failed['request']['reader_messages'][1]['content'][-1]['text'].startswith(core.query_prompt(question))
                    try:
                        core.parse_query(prior['raw_response'])
                    except (ValueError, TypeError):
                        pass
                    else:
                        raise AssertionError('Earlier phrase recovery was already valid')
            assert actual_previous == previous
            keys |= previous
        else:
            assert core.parse_query(attempts[-1]['raw_response']) == answer['visual_query']
        if recovery.get('prior_recovery_dir'):
            previous = rows(Path(recovery['prior_recovery_dir']) / 'recovery_attempts.jsonl')
            assert not any(a['valid'] for a in previous)
            previous_keys = {a['request_sha256'] for a in previous}
            assert previous_keys == set(recovery['prior_unique_requests'])
            keys |= previous_keys
        assert len(keys) <= (5 if is_array else 3)
        assert recovery_seal['new_rewrite_attempts'] == len(keys)
        recovered.append({'id': qid, 'kind': recovery['kind'], 'extra_distinct_rewrite_requests': len(keys),
                          'adapter_attempts': len(attempts),
                          'explicit_additional_three_field_exception': is_fields,
                          'protocol_sha256': sha(directory / 'recovery_protocol.json')})
    assert set(seal.get('format_recovery_ids', [])) == {r['id'] for r in recovered}
    assert {r['id'] for r in historical_errors} <= {r['id'] for r in recovered}
    assert collection['format_recovery_questions'] == len(recovered)
    assert collection['format_recovery_new_rewrite_attempts'] == sum(r['extra_distinct_rewrite_requests'] for r in recovered)
    result = {'passed': True, 'method_id': 'hippomm', 'questions_checked': 371,
        'routes': dict(routes), 'unique_native_calls_checked_against_actual_service': len(checked_calls),
        'unique_original_image_hashes_checked': len(checked_images), 'collection_frozen_hashes_checked': len(verified_hashes),
        'all_canonical_open_questions_gold_and_raw_answer_mapping_verified': True,
        'fast_answer_extracted_from_native_json': True, 'detailed_answer_preserved_from_last_raw_response': True,
        'format_recovery': recovered, 'format_recovery_questions': len(recovered),
        'token_limit_answers_preserved': sum(r['hit_token_limit'] for r in answers.values()),
        'source_sha256': {str(p): sha(p) for p in (Path(__file__), Path(core.__file__), Path(original.__file__),
            HIP / 'collection_protocol.json', HIP / 'responses.jsonl', HIP / 'completion.json', HIP / 'judge_inputs.jsonl',
            *([HIP / 'question_errors.jsonl'] if (HIP / 'question_errors.jsonl').exists() else []))}
            | extra_service_sources}
    output = RUN / 'audits/hippomm_native_independent_20261105.json'
    assert not output.exists(), 'Do not overwrite a completed signed audit'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
