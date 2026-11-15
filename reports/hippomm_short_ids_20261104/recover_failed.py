"""Explicit query-format recovery; preserve completed answers and all raw calls.

Only rewrites that exhausted the original three attempts get an additional
three-word JSON-array formatting stage. Retrieval parameters and gold boundaries
remain unchanged. This script is recorded in recovery/experiment_manifest.json.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import visual_memory_pipeline as cli
from scripts.memory import hippomm as method, hippomm_core as core

REPORT = Path(__file__).resolve().parent / 'recovery'
OUTPUT = ROOT / 'results/hippomm_short_ids_20261104'
ORIGINAL_STRUCTURED_CALL = core.structured_call
ORIGINAL_APPEND = method.append
POLICY = 'three_word_array_after_original_rewrite_attempts'


def corrected_query(llm, role, prompt, parse, attempts=3):
    try:
        return ORIGINAL_STRUCTURED_CALL(llm, role, prompt, parse, attempts)
    except ValueError:
        if role != 'rewrite':
            raise
    correction = (
        'Generate a concise visual search cue from the original question below. '
        'Return exactly THREE separate English words as a JSON array in this object: '
        '{"words": ["word1", "word2", "word3"]}. Each element must be one word, '
        'with no whitespace. Focus on the visually identifiable objects or attributes '
        'explicitly mentioned in the question. Do not infer the answer or invent '
        'attributes. Do not return an explanation.\nOriginal request:\n' + prompt)
    for attempt in range(3):
        raw = llm('rewrite_format_recovery', correction)
        try:
            obj = core.parse_object(raw)
            words = obj.get('words')
            if (set(obj) != {'words'} or not isinstance(words, list) or len(words) != 3
                    or any(not isinstance(w, str) or not w.strip()
                           or any(c.isspace() for c in w) for w in words)):
                raise ValueError('Return exactly three nonempty single-word strings')
            return parse(json.dumps({'visual_query': ' '.join(words)}))
        except (TypeError, ValueError) as exc:
            if attempt == 2:
                raise ValueError('Additional query formatting failed: ' + str(exc)) from exc
            correction += '\nPrevious output: ' + raw + '\nValidation error: ' + str(exc)


def append_with_recovery_note(path, row):
    if Path(path).name == 'qa_responses.jsonl':
        row = {**row, 'recovery_policy': POLICY,
               'recovery_amendment': str(REPORT / 'amendment.json')}
    ORIGINAL_APPEND(path, row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['memeye', 'mem_gallery'], required=True)
    args = parser.parse_args()
    directory = OUTPUT / args.dataset
    original = method.inputs.read_lines(directory / 'online/qa_responses.jsonl')
    questions = method.inputs.read_lines(directory / 'online/questions.jsonl')
    missing = sorted({q['id'] for q in questions} - {r['id'] for r in original})
    method.dump(REPORT / (args.dataset + '_before.json'),
                {'completed': len(original), 'missing_ids': missing})
    core.structured_call = corrected_query
    method.append = append_with_recovery_note
    cli.main(['online', '--memory', 'hippomm', '--dataset', args.dataset,
              '--device', 'cuda:0', '--stage', 'all',
              '--memory-dir', str(directory / 'offline'), '--output', str(directory / 'online')])
    final = method.inputs.read_lines(directory / 'online/qa_responses.jsonl')
    assert final[:len(original)] == original, 'Existing answers were changed'
    assert len(final) == len(questions)
    assert len({r['id'] for r in final}) == len(questions)
    assert {r['id'] for r in final} == {q['id'] for q in questions}
    recovered = final[len(original):]
    for row in recovered:
        assert row['recovery_policy'] == POLICY
        if row['route'] == 'detailed':
            core.parse_query(json.dumps({'visual_query': row['visual_query']}))
    method.dump(REPORT / (args.dataset + '_verification.json'), {
        'status': 'complete', 'finished_at': datetime.now(timezone.utc).isoformat(),
        'questions': len(final), 'previous_answers_unchanged': len(original),
        'recovered_questions': len(recovered), 'recovered_ids': [r['id'] for r in recovered],
        'summary': json.loads((directory / 'online/summary.json').read_text())})


if __name__ == '__main__':
    main()
