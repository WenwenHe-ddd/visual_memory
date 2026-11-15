"""Read-only independent answer/wire/grade/cache audit of a completed Open method."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memeye_open_method_judge as strict
from scripts.evaluation import memeye_open_method_reader as reader
from scripts.analysis import export_memeye_open_llm_judge as exporter

RUN = ROOT / 'results/memeye_qwen_other_open_20261105'
GRADES = (0, .25, .5, .75, 1)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(values, key='id'):
    result = {row[key]: row for row in values}
    assert len(result) == len(values), ('duplicate', key)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wire_hash(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def parse_raw(text):
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        assert lines[0] in ('```', '```json') and lines[-1] == '```'
        text = '\n'.join(lines[1:-1])
    value = json.loads(text)
    assert isinstance(value, dict)
    assert type(value['score']) in (int, float) and value['score'] in GRADES
    assert isinstance(value['reasoning'], str) and value['reasoning'].strip()
    return value['score'], value['reasoning']


def make_wire(row, template):
    text = re.sub(r'\{\{(question|ground_truth|model_output)\}\}',
                  lambda match: row[match.group(1)], template)
    return {'model': 'deepseek-v4.1-flash', 'temperature': 0, 'seed': 42,
            'max_tokens': 1024, 'n': 1, 'stream': False, 'enable_thinking': False,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]}


def inspect(method):
    root = Path(method['root']); target = root / 'judge'
    protocol, summary = read(target / 'protocol.json'), read(target / 'summary.json')
    assert summary['status'] == 'complete'
    canonical, _ = strict.load_canonical()
    # These existing validators cover all signed hashes, canonical source strings,
    # original reader answers and complete cache aggregates before the second pass.
    strict.load_completed_cache(target, strict.shared.PROMPT.read_text(), canonical)
    questions = unique(rows(exporter.CANONICAL / 'inputs/questions.jsonl'))
    gold = unique(rows(exporter.CANONICAL / 'inputs/gold.jsonl'))
    export_summary, _ = exporter.verify_method(method, questions, gold)
    native = method['id'] == 'hippomm'
    reader_root = root if native else root / 'reader'
    response_path = reader_root / 'responses.jsonl'
    response_sha256 = sha(response_path)
    raw_answers = unique(rows(response_path))
    input_rows = unique(rows(root / 'judge_inputs.jsonl'))
    mapped = unique(rows(target / 'question_map.jsonl'))
    result_rows = unique(rows(target / 'per_question.jsonl'))
    assert len(input_rows) == 371 and set(input_rows) == set(questions) == set(raw_answers) == set(mapped) == set(result_rows)
    reader_protocol = read(reader_root / 'protocol.json') if (reader_root / 'protocol.json').exists() else None
    service_raw_verified = False
    if not native and reader_protocol and 'service_signature' in reader_protocol.get('contract', {}):
        source_rows = rows(root / 'reader_manifest.jsonl')
        validated_answers = reader.validate(root, reader_protocol, source_rows)
        assert len(validated_answers) == 371
        assert all(r['run_id'] == reader_protocol['run_id'] and r['method_id'] == method['id']
                   and r['answer_model'] == 'Qwen/Qwen3-VL-8B-Instruct' for r in validated_answers)
        service_raw_verified = True
    requests = unique(rows(target / 'requests.jsonl'), 'request_sha256')
    saved, origins = {}, Counter()
    for name, expected_origin in (('scores.jsonl', 'new_api'), ('reused_scores.jsonl', 'baseline_cache')):
        path = target / name
        for score in rows(path) if path.exists() else []:
            key = score['request_sha256']
            assert key not in saved and key in requests and score['score_origin'] == expected_origin
            parsed, reasoning = parse_raw(score['raw_response'])
            assert parsed == score['score'] and reasoning == score['reasoning']
            assert score['returned_model'] == score['requested_model'] == 'deepseek-v4.1-flash'
            assert score['seed'] == 42 and score['enable_thinking'] is False
            assert score['finish_reason'] == 'stop' and not score.get('refusal')
            assert not score.get('reasoning_content_present') and score.get('reasoning_tokens') in (None, 0)
            saved[key] = score; origins[expected_origin] += 1
    assert set(saved) == set(requests)
    source_cache_maps, cache_provenance_count = {}, 0
    for cache_root in protocol.get('cache_roots', []):
        cache = Path(cache_root)
        strict.load_completed_cache(cache, strict.shared.PROMPT.read_text(), canonical)
        source_cache_maps[cache_root] = {
            'requests': unique(rows(cache / 'requests.jsonl'), 'request_sha256'),
            'scores': {record['request_sha256']: record for name in ('scores.jsonl', 'reused_scores.jsonl')
                       if (cache / name).exists() for record in rows(cache / name)},
            'summary_sha256': sha(cache / 'summary.json')}
    latest_attempt = {}
    attempts = rows(target / 'attempts.jsonl') if (target / 'attempts.jsonl').exists() else []
    for attempt in attempts:
        key = attempt['request_sha256']; assert key in requests
        assert attempt['attempt'] == latest_attempt.get(key, {}).get('attempt', 0) + 1
        latest_attempt[key] = attempt
    for key, score in saved.items():
        if score['score_origin'] == 'new_api':
            attempt = latest_attempt[key]
            assert attempt['status'] == 'ok'
            assert attempt['raw_response'] == score['raw_response'] and attempt['response_id'] == score['response_id']
        else:
            assert score['reuse_sources']
            for provenance in score['reuse_sources']:
                source = source_cache_maps[provenance['root']]
                cached = source['scores'][key]
                assert provenance['source_request_sha256'] == key
                assert provenance['summary_sha256'] == source['summary_sha256']
                assert provenance['response_id'] == cached['response_id']
                assert provenance['source_score_origin'] == cached['score_origin']
                assert provenance['question_ids'] == source['requests'][key]['question_ids']
                assert source['requests'][key]['request'] == requests[key]['request']
                assert parse_raw(cached['raw_response'])[0] == score['score']
                cache_provenance_count += 1
            first = source_cache_maps[score['reuse_sources'][0]['root']]['scores'][key]
            assert score['raw_response'] == first['raw_response'] and score['response_id'] == first['response_id']
    template = (target / 'judge_prompt.txt').read_text()
    units, counts, question_origins, used = 0, Counter(), Counter(), set()
    for qid, question in questions.items():
        source, answer = input_rows[qid], raw_answers[qid]
        assert source['question'] == question['question'] and source['ground_truth'] == gold[qid]['answer']
        assert source['task'] == question['task'] and source['point'] == question['point']
        assert source['model_output'] == answer['raw_response']
        assert source['source_record_id'] == qid and source['source_responses_sha256'] == response_sha256
        assert source.get('source_hit_token_limit', False) == answer['hit_token_limit']
        request = make_wire(source, template); key = wire_hash(request)
        assert request == requests[key]['request'] and mapped[qid]['request_sha256'] == key
        assert result_rows[qid]['request_sha256'] == key and result_rows[qid]['judge_status'] == 'ok'
        for field in ('question', 'ground_truth', 'model_output', 'task', 'point'):
            assert mapped[qid][field] == result_rows[qid][field] == source[field]
        grade, reasoning = parse_raw(saved[key]['raw_response'])
        assert result_rows[qid]['score'] == grade and result_rows[qid]['reasoning'] == reasoning
        units += int(4 * grade); counts[grade] += 1
        question_origins[saved[key]['score_origin']] += 1; used.add(key)
    assert used == set(requests)
    exact_mean = Fraction(units, 4 * 371)
    distribution = {str(grade): counts[grade] for grade in GRADES}
    assert summary['score_sum'] == units / 4 and summary['llm_judge'] == float(exact_mean)
    assert summary['score_distribution'] == distribution and export_summary['llm_judge'] == float(exact_mean)
    assert sum(counts.values()) == 371
    paths = [Path(__file__), Path(strict.__file__), Path(reader.__file__), Path(exporter.__file__),
             RUN / 'methods_manifest.json', root / 'judge_inputs.jsonl', response_path,
             reader_root / 'completion.json', *[target / name for name in ('protocol.json', 'summary.json',
                 'question_map.jsonl', 'requests.jsonl', 'reused_scores.jsonl', 'per_question.jsonl', 'judge_prompt.txt')]]
    paths += [target / name for name in ('scores.jsonl', 'attempts.jsonl') if (target / name).exists()]
    result = {'passed': True, 'method_id': method['id'], 'method': method['label'], 'questions_checked': 371,
        'llm_judge': float(exact_mean), 'score_sum': units / 4, 'exact_mean_fraction': str(exact_mean),
        'score_distribution': distribution, 'unique_wire_requests': len(requests),
        'unique_score_origins': dict(origins), 'question_score_origins': dict(question_origins),
        'source_cache_provenance_entries_checked': cache_provenance_count,
        'all_original_answers_and_canonical_questions_gold_verified': True,
        'original_answers_verified_against_qwen_service_calls': service_raw_verified,
        'independent_wire_reconstruction_and_sha256': True, 'independent_raw_json_grade_parser': True,
        'integer_quarter_units_aggregation': True, 'all_frozen_hashes_and_completed_caches_verified': True,
        'api_attempts': len(attempts), 'failed_api_attempts': sum(r.get('status') != 'ok' for r in attempts),
        'empty_answers_preserved': sum(not r['raw_response'] for r in raw_answers.values()),
        'token_limit_answers_preserved': sum(r['hit_token_limit'] for r in raw_answers.values()),
        'source_sha256': {str(path.resolve()): sha(path) for path in paths}}
    output = RUN / 'audits' / (method['id'] + '_judge_independent_20261105.json')
    strict.dump(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'source_sha256'}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method-id', required=True)
    args = parser.parse_args()
    manifest = read(RUN / 'methods_manifest.json')
    selected = [m for m in manifest['methods'] if m['id'] == args.method_id and m['id'] != 'semrag']
    assert len(selected) == 1
    inspect(selected[0])


if __name__ == '__main__':
    main()
