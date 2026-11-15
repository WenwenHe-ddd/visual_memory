"""Recover four unambiguous singleton-JSON candidate lines without generation.

This separate, audited formatting adapter leaves the frozen generator, its
protocol and raw attempts untouched. It never reads answers or memory history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]

from scripts.evaluation.mem_gallery_master_reader import (
    CANDIDATE_SYSTEM, CLIP, DEFAULT_OUTPUT, by_id, candidate_user_text, digest,
    dump, parse_candidates, read_lines, sha,
)


def parse_four_singleton_lists(raw):
    """Flatten exactly four independent JSON [string] lines, preserving strings."""
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 4:
        raise ValueError('Expected exactly four nonempty singleton-array lines')
    result = []
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str):
            raise ValueError('Every line must be an independent JSON array of exactly one string')
        if not value[0] or value[0] != value[0].strip():
            raise ValueError('Candidate string is empty or would require whitespace alteration')
        result.append(value[0])
    return result


def recover(root):
    from transformers import AutoTokenizer
    root = Path(root).resolve()
    protocol_path = root / 'candidate_generation_protocol.json'
    attempt_path = root / 'candidate_attempts.jsonl'
    failure_path = root / 'candidate_failures.json'
    source_path = root / 'inputs/sanitized_candidate_inputs.jsonl'
    output_path = root / 'candidates.jsonl'
    report_path = root / 'candidate_format_repair.json'
    protocol = json.loads(protocol_path.read_text())
    if digest(protocol['contract']) != protocol['experiment_signature']:
        raise ValueError('Frozen candidate protocol signature mismatch')
    for path, expected in protocol['contract']['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Frozen generation source changed: {path}')
    for filename, expected in protocol['contract']['clip_tokenizer_sha256'].items():
        if sha(CLIP / filename) != expected:
            raise ValueError(f'CLIP tokenizer changed: {filename}')
    source = by_id(read_lines(source_path))
    if any(set(row) != {'id', 'question', 'query_image_caption'} for row in source.values()):
        raise ValueError('Only sanitized question inputs are permitted')
    existing = read_lines(output_path)
    existing_by_id = by_id(existing)
    for row in existing:
        if (row['id'] not in source or row['run_id'] != protocol['run_id'] or
                row['experiment_signature'] != protocol['experiment_signature']):
            raise ValueError('Existing candidate provenance mismatch')
    if report_path.exists():
        report = json.loads(report_path.read_text())
        for record in report['repaired_records']:
            row = existing_by_id.get(record['id'])
            if row is None or row['candidate_statements'] != record['candidate_statements']:
                raise ValueError('Previously repaired candidate is missing or changed')
        if (report['input_attempts_sha256'] != sha(attempt_path) or
                report['original_protocol_sha256'] != sha(protocol_path) or
                report['sanitized_input_sha256'] != sha(source_path) or
                report['repair_script_sha256'] != sha(__file__)):
            raise ValueError('Existing repair provenance changed')
        return report
    failed_ids = json.loads(failure_path.read_text())['ids']
    if len(set(failed_ids)) != len(failed_ids) or set(source) - set(existing_by_id) != set(failed_ids):
        raise ValueError('Only the exact outstanding failure set may be repaired')
    attempts = read_lines(attempt_path)
    latest = {}
    for attempt in attempts:
        if attempt['id'] in failed_ids:
            latest[attempt['id']] = attempt
    if set(latest) != set(failed_ids):
        raise ValueError('Missing failed generation attempt')
    attempts_hash = sha(attempt_path)
    tokenizer = AutoTokenizer.from_pretrained(str(CLIP), local_files_only=True)
    additions, records = [], []
    for qid in source:
        if qid not in latest:
            continue
        attempt = latest[qid]
        if (attempt['attempt'] != 3 or not attempt['validation_error'] or attempt['hit_token_limit'] or
                attempt['run_id'] != protocol['run_id'] or
                attempt['experiment_signature'] != protocol['experiment_signature']):
            raise ValueError('Recovery requires a complete failed third attempt in this run')
        expected_prompt = digest({'system': CANDIDATE_SYSTEM,
                                  'user_text': candidate_user_text(source[qid], attempt=3)})
        if expected_prompt != attempt['prompt_sha256']:
            raise ValueError('Frozen generation prompt hash mismatch')
        values = parse_four_singleton_lists(attempt['raw_response'])
        checked, lengths = parse_candidates(json.dumps(values, ensure_ascii=False), tokenizer)
        if checked != values:
            raise ValueError('Original validation would alter candidate text')
        provenance = {'operation': 'Flatten four valid singleton JSON arrays; decoded strings unchanged.',
            'input_attempts_sha256': attempts_hash,
            'attempt_record_sha256': digest(attempt),
            'raw_response_sha256': hashlib.sha256(attempt['raw_response'].encode()).hexdigest(),
            'repair_script_sha256': sha(__file__), 'no_new_generation': True,
            'decoded_strings_unchanged': True, 'gold_or_history_used': False}
        additions.append({'id': qid, 'candidate_statements': values,
            'clip_token_lengths': lengths, 'source_input_sha256': digest(source[qid]),
            'attempt': attempt['attempt'], 'prompt_sha256': attempt['prompt_sha256'],
            'response_format': 'four_json_singleton_lists', 'run_id': protocol['run_id'],
            'experiment_signature': protocol['experiment_signature'], 'gold_or_history_used': False,
            'format_repair_provenance': provenance})
        records.append({'id': qid, 'attempt': attempt['attempt'], 'run_id': protocol['run_id'],
            'experiment_signature': protocol['experiment_signature'],
            'prompt_sha256': attempt['prompt_sha256'], 'raw_response_sha256': provenance['raw_response_sha256'],
            'attempt_record_sha256': provenance['attempt_record_sha256'],
            'candidate_statements': values, 'clip_token_lengths': lengths})
    if len(additions) != len(failed_ids):
        raise ValueError('Incomplete recovery')
    original_bytes = output_path.read_bytes()
    if original_bytes and not original_bytes.endswith(b'\n'):
        raise ValueError('Existing candidate JSONL lacks its terminating newline')
    merged = {**existing_by_id, **by_id(additions)}
    # Match the frozen finalizer's exact source-order serialization so its later
    # coverage-only resume does not change this audited final artifact hash.
    new_bytes = ''.join(json.dumps(merged[qid], ensure_ascii=False) + '\n' for qid in source).encode()
    report = {'status': 'resolved', 'initial_failure_ids': failed_ids, 'unresolved_ids': [],
        'initial_failures_retained_at': str(failure_path), 'initial_failure_file_sha256': sha(failure_path),
        'input_attempts_path': str(attempt_path), 'input_attempts_sha256': attempts_hash,
        'original_protocol_sha256': sha(protocol_path), 'sanitized_input_sha256': sha(source_path),
        'repair_script_path': str(Path(__file__).resolve()), 'repair_script_sha256': sha(__file__),
        'candidates_before_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'candidates_after_sha256': hashlib.sha256(new_bytes).hexdigest(),
        'previous_valid_candidates': len(existing), 'repaired_candidates': len(additions),
        'total_candidates': len(existing) + len(additions), 'repaired_records': records,
        'no_new_generation': True, 'decoded_strings_unchanged': True, 'gold_or_history_used': False,
        'previous_valid_candidate_records_unchanged': True, 'serialization_order': 'sanitized source order',
        'frozen_generator_and_protocol_unchanged': True,
        'note': 'These were original format failures. Only unambiguous container syntax was normalized; original failure and attempt files are retained.'}
    temp = output_path.with_name(output_path.name + '.format-repair.tmp')
    temp.write_bytes(new_bytes)
    temp.replace(output_path)
    dump(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = recover(args.output)
    print(json.dumps({key: report[key] for key in ('status', 'previous_valid_candidates',
          'repaired_candidates', 'total_candidates', 'no_new_generation')}), flush=True)


if __name__ == '__main__':
    main()
