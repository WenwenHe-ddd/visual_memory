"""Recover an unambiguous explicit judge grade from preserved malformed JSON.

This writes a separate evaluation root. It never changes provider responses,
question/answer requests, grades, prompts, or the original evaluation files.
Only identical numeric grades explicitly emitted in every saved attempt qualify.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from scripts.evaluation import memeye_open_method_judge as judge


def objects(raw):
    decoder = json.JSONDecoder()
    result = []
    cursor = 0
    while cursor < len(raw):
        start = raw.find('{', cursor)
        if start < 0:
            break
        try:
            item, size = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = start + size
        if isinstance(item, dict) and set(item) == {'score', 'reasoning'}:
            grade = judge.shared.parse_grade(json.dumps(item))
            result.append(grade)
    return result


def recover(source, output):
    source, output = Path(source), Path(output)
    protocol = judge.read(source / 'protocol.json')
    requests = judge.read_rows(source / 'requests.jsonl')
    question_map = judge.read_rows(source / 'question_map.jsonl')
    saved = judge.shared.validated_results(source, requests)
    pending = {r['request_sha256'] for r in requests} - set(saved)
    if len(pending) != 1:
        raise ValueError('This narrow recovery requires exactly one missing unique request')
    digest = next(iter(pending))
    attempts = [r for r in judge.read_rows(source / 'attempts.jsonl') if r['request_sha256'] == digest]
    if len(attempts) != 5 or any(r.get('error_class') != 'JSONDecodeError' or r.get('finish_reason') != 'stop'
            or r.get('seed') != 42 or r.get('enable_thinking') is not False for r in attempts):
        raise ValueError('Recovery is restricted to the five recorded JSON-format failures')
    proposals = [objects(a['raw_response']) for a in attempts]
    if not all(proposals) or len({r['score'] for group in proposals for r in group}) != 1:
        raise ValueError('Provider outputs do not contain one unambiguous explicit grade')
    chosen, parsed = attempts[-1], proposals[-1][-1]
    fields = {'attempt', 'started_at_unix', 'status', 'retryable', 'error_kind', 'error_class'}
    recovered = {k: v for k, v in chosen.items() if k not in fields}
    recovered.update(parsed, score_origin='new_api', recovered_from_preserved_format_error=True,
                     original_attempt=chosen['attempt'], original_attempts_sha256=judge.sha(source / 'attempts.jsonl'))
    raw_to_grade = {chosen['raw_response']: parsed}
    strict_parser = judge.shared.parse_grade

    def parse(raw):
        if raw in raw_to_grade:
            return dict(raw_to_grade[raw])
        return strict_parser(raw)

    parent_hashes = {str((source / name).resolve()): judge.sha(source / name) for name in
        ('protocol.json', 'requests.jsonl', 'question_map.jsonl', 'reused_scores.jsonl',
         'scores.jsonl', 'attempts.jsonl', 'judge_prompt.txt', 'summary.json')}
    amended = copy.deepcopy(protocol)
    amended['format_recovery'] = {'source_root': str(source.resolve()), 'source_sha256': parent_hashes,
        'implementation_sha256': judge.sha(__file__), 'missing_request_sha256': digest,
        'provider_raw_response_preserved': True, 'new_api_calls': 0,
        'grades_in_all_five_attempts': [[r['score'] for r in group] for group in proposals],
        'rule': 'Extract the last explicit score/reasoning JSON object only when every object in all five original outputs states the same grade; preserve complete original raw strings.',
        'strict_cache_reuse_compatible': False}
    output.mkdir(parents=True, exist_ok=True)
    frozen = {'requests.jsonl': requests, 'question_map.jsonl': question_map,
              'reused_scores.jsonl': judge.read_rows(source / 'reused_scores.jsonl'),
              'scores.jsonl': [*judge.read_rows(source / 'scores.jsonl'), recovered],
              'attempts.jsonl': judge.read_rows(source / 'attempts.jsonl')}
    if (output / 'protocol.json').exists():
        if judge.read(output / 'protocol.json') != amended:
            raise ValueError('Recovery protocol differs')
        for name, values in frozen.items():
            if judge.read_rows(output / name) != values:
                raise ValueError('Recovered artifact differs: ' + name)
    else:
        if any(output.iterdir()):
            raise ValueError('Recovery output must be fresh')
        judge.dump(output / 'protocol.json', amended)
        for name, values in frozen.items():
            judge.write_rows(output / name, values)
        shutil.copyfile(source / 'judge_prompt.txt', output / 'judge_prompt.txt')
    args = SimpleNamespace(output=output)
    judge.shared.parse_grade = parse
    try:
        judge.shared.validate_score(recovered, digest)
        report = judge.base.analyze(args, amended, requests, question_map)
        judge.publish_summary(args, amended, report)
    finally:
        judge.shared.parse_grade = strict_parser
    if report['status'] != 'complete' or report['questions_scored'] != 371:
        raise ValueError('Recovery did not produce complete coverage')
    judge.dump(output / 'recovery_audit.json', {'status': 'complete', 'questions': 371,
        'preserved_strict_scores': len(saved), 'format_recovered_requests': 1,
        'source_sha256': parent_hashes, 'original_raw_responses_preserved': True,
        'all_five_attempt_grades_agree': True, 'new_api_calls': 0,
        'llm_judge': report['llm_judge'], 'summary_sha256': judge.sha(output / 'summary.json')})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = recover(args.source, args.output)
    print(json.dumps({'status': report['status'], 'questions': report['questions_scored'],
                      'llm_judge': report['llm_judge']}), flush=True)


if __name__ == '__main__':
    main()
