#!/usr/bin/env python3
"""Merge and validate fixed, assistant-authored hypotheses from sanitized QA text.

This compiler does not call an inference API, read memories/golds, or choose
candidate wording based on scores. Parts A/B are editable authored inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports/all_master_answer_query_20261021'
CHOICES = ('A', 'B', 'C', 'D')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    rows = json.loads((output / 'sanitized_candidate_inputs.json').read_text())
    assert len(rows) == 371 and len({q['id'] for q in rows}) == 371
    assert all(set(q) == {'id', 'question', 'options'} for q in rows)
    authored = {}
    for name in ('candidates_part_a.jsonl', 'candidates_part_b.jsonl'):
        for line in (output / name).read_text().splitlines():
            row = json.loads(line)
            assert row['id'] not in authored, row['id']
            assert set(row['candidate_statements']) == set(CHOICES)
            authored[row['id']] = row
    assert set(authored) == {q['id'] for q in rows}
    os.environ['HF_HUB_OFFLINE'] = '1'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/clip-vit-base-patch32'), local_files_only=True)
    results = []
    failures = []
    for question in rows:
        row = authored[question['id']]
        lengths = {c: len(tokenizer(row['candidate_statements'][c], truncation=False)['input_ids']) for c in CHOICES}
        for c, length in lengths.items():
            if length > 77:
                failures.append({'id': row['id'], 'choice': c, 'tokens': length,
                                 'text': row['candidate_statements'][c]})
        results.append({**question, **row, 'clip_token_lengths': lengths})
    if failures:
        raise ValueError(json.dumps(failures, indent=2))
    q10 = next(row for row in results if row['id'] == 'Home_Renovation_Interior_Design/Q10')
    old_q10_path = ROOT / 'reports/home_q10_master_answer_query_20261021/candidates.json'
    old_q10 = json.loads(old_q10_path.read_text())
    assert q10['candidate_statements'] == old_q10['candidate_statements']
    candidate_path = output / 'candidates.jsonl'
    candidate_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in results))
    protocol = {
        'status': 'complete', 'n_questions': 371, 'n_candidates': 1484,
        'authoring_method': 'Fixed assistant-authored hypotheses, compiled from manually authored templates and explicit statements; no local generation model was used.',
        'input_fields': ['id', 'question', 'options'],
        'gold_memory_caption_or_image_inputs_used': False,
        'selection_or_revision_using_gold_or_retrieval_scores': False,
        'wording_policy': 'One self-contained hypothesis for every option. Preserve candidate distinctions, negation, quantities and chronology. Shared question context may be shortened.',
        'q10_wording_frozen_from_previous_experiment': True,
        'q10_source_sha256': sha(old_q10_path),
        'clip_tokenizer': str(ROOT / 'models/clip-vit-base-patch32'),
        'clip_max_tokens_including_special_tokens': 77,
        'max_observed_clip_tokens': max(n for row in results for n in row['clip_token_lengths'].values()),
        'truncation_used': False,
        'preparation_endpoint_checks': [
            {'endpoint': f'http://127.0.0.1:{p}/v1/models', 'method': 'GET',
             'authenticated': False, 'status': 401, 'generation_requested': False}
            for p in (21080, 21081)],
        'endpoint_checks_followup': 'No credentials were used and no generation requests were made. Switched to transparent assistant-authored inputs.',
        'source_sha256': {name: sha(output / name) for name in (
            'sanitized_candidate_inputs.json', 'candidates_part_a.jsonl', 'candidates_part_b.jsonl')},
        'candidates_sha256': sha(candidate_path), 'compiler_sha256': sha(__file__),
        'frozen_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    (output / 'candidate_generation_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
