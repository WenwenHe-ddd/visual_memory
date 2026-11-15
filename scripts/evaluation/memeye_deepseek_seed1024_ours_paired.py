#!/usr/bin/env python3
"""Offline paired seed42/1024 scoring audit; never builds or sends requests."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memeye_deepseek_seed1024_ours as run
from scripts.evaluation import memeye_deepseek_seed1024_ours_open as open_run


def choice(raw):
    match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw.strip(), re.I) if isinstance(raw, str) else None
    return match.group(1).upper() if match else None


def audit(output=run.OUTPUT):
    output = Path(output).resolve()
    reference = open_run.validate_mcq_reference(output)
    old, source_paths = {}, []
    new_gold = run.native.unique(run.lines(output / 'inputs/gold.jsonl'))
    for folder in ('weighted_evaluation', 'complementary_evaluation'):
        old_root = run.SOURCE / folder / 'reader'
        old_gold = run.native.unique(run.lines(old_root / 'gold.jsonl'))
        assert set(old_gold) == set(new_gold) and len(new_gold) == 371
        assert all(old_gold[q]['gold_choice'] == new_gold[q]['gold_choice'] for q in new_gold)
        source_paths.extend(old_root / name for name in ('gold.jsonl', 'responses.jsonl'))
        for response in run.lines(old_root / 'responses.jsonl'):
            assert response['record_id'] not in old
            old[response['record_id']] = response
    new = run.native.unique(run.lines(output / 'reader/responses.jsonl'), 'record_id')
    assert set(old) == set(new) and len(new) == 1484
    detail, rejections, condition_summary = [], [], {}
    for method, condition in zip(open_run.METHODS, run.CONDITIONS):
        counter = Counter()
        transitions = Counter()
        old_status, new_status = Counter(), Counter()
        group = [n for n in new.values() if n['condition'] == condition]
        assert len(group) == 371 and {n['question_id'] for n in group} == set(new_gold)
        for current in group:
            previous = old[current['record_id']]
            assert previous['seed'] == 42 and current['seed'] == 1024
            assert previous['question_id'] == current['question_id']
            assert previous['condition'] == current['condition']
            assert previous['predicted_choice'] == choice(previous.get('raw_response'))
            expected = new_gold[current['question_id']]['gold_choice']
            old_correct = previous['predicted_choice'] == expected
            new_correct = current['predicted_choice'] == expected
            assert previous['correct'] == old_correct and current['correct'] == new_correct
            category = ('both_correct' if old_correct and new_correct else 'seed42_only_correct'
                        if old_correct else 'seed1024_only_correct' if new_correct else 'both_wrong')
            changed = previous['predicted_choice'] != current['predicted_choice']
            counter[category] += 1
            counter['choice_changed'] += changed
            counter['both_wrong_choice_changed'] += changed and not old_correct and not new_correct
            old_status[previous['status']] += 1
            new_status[current['status']] += 1
            transitions[(previous['status'], current['status'])] += 1
            row = {'record_id': current['record_id'], 'question_id': current['question_id'],
                'method': method, 'condition': condition, 'seed42_status': previous['status'],
                'seed1024_status': current['status'], 'seed42_choice': previous['predicted_choice'],
                'seed1024_choice': current['predicted_choice'], 'gold_choice_for_scoring_only': expected,
                'seed42_correct': old_correct, 'seed1024_correct': new_correct,
                'paired_category': category, 'choice_changed': changed,
                'seed42_request_sha256': previous['request_sha256'],
                'seed1024_request_sha256': current['request_sha256']}
            detail.append(row)
            if current['status'] == 'provider_input_rejected':
                rejections.append({**row, 'error_kind': current.get('error_kind'),
                    'measured_correct_difference_vs_seed42': int(new_correct) - int(old_correct),
                    'counted_wrong_in_denominator_371': True,
                    'unobserved_counterfactual_answer_not_imputed': True})
        old_correct = counter['both_correct'] + counter['seed42_only_correct']
        new_correct = counter['both_correct'] + counter['seed1024_only_correct']
        assert new_correct == reference['per_method'][method]['correct']
        assert sum(counter[k] for k in ('both_correct', 'seed42_only_correct', 'seed1024_only_correct', 'both_wrong')) == 371
        condition_summary[method] = {'condition': condition, 'questions': 371,
            **{k: counter[k] for k in ('both_correct', 'seed42_only_correct', 'seed1024_only_correct',
                'both_wrong', 'choice_changed', 'both_wrong_choice_changed')},
            'seed42_correct': old_correct, 'seed1024_correct': new_correct,
            'correct_difference': new_correct - old_correct,
            'seed42_accuracy': old_correct / 371, 'seed1024_accuracy': new_correct / 371,
            'seed42_statuses': dict(old_status), 'seed1024_statuses': dict(new_status),
            'status_transitions': [{'seed42': a, 'seed1024': b, 'records': n}
                                   for (a, b), n in sorted(transitions.items())]}
    source_paths.extend(output / name for name in ('reader/responses.jsonl', 'summary.json', 'inputs/gold.jsonl'))
    report = {'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'api_calls': 0, 'paired_records': 1484, 'questions_per_condition': 371,
        'methods': condition_summary, 'permanent_rejections_seed1024': rejections,
        'rejection_scoring_boundary': {
            'permanent_rejections': len(rejections),
            'previously_wrong_records_with_zero_score_difference': sum(not r['seed42_correct'] for r in rejections),
            'previously_correct_records_now_counted_wrong': sum(r['seed42_correct'] for r in rejections),
            'measured_correct_difference_on_rejected_records': sum(r['measured_correct_difference_vs_seed42'] for r in rejections),
            'interpretation': 'This is the observed paired scoring difference on rejected records. No hypothetical answer under a successful seed1024 response is known or imputed.'},
        'gold_use_boundary': 'Gold choices were read only for this posthoc scoring audit. The running inference manifest permits only system, user_text and query_images; it supplies all MCQ choices without identifying the gold choice. This audit starts no inference and modifies no answer.',
        'formulas': {'seed42_correct': 'both_correct + seed42_only_correct',
            'seed1024_correct': 'both_correct + seed1024_only_correct',
            'correct_difference': 'seed1024_only_correct - seed42_only_correct',
            'choice_changed': 'predicted_choice_seed42 != predicted_choice_seed1024, including None'},
        'interpretation_limit': 'Same provider alias and frozen inputs do not establish identical provider model weights across run dates; observed differences cannot be attributed exclusively to the seed.',
        'source_sha256': {str(p): run.sha(p) for p in [Path(__file__), *source_paths]}}
    run.write_lines(output / 'independent_mcq_paired_records.jsonl', detail)
    run.dump(output / 'independent_mcq_paired_summary.json', report)
    return report


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=run.OUTPUT)
    args = parser.parse_args()
    report = audit(args.output)
    print(json.dumps({'passed': report['passed'], 'paired_records': report['paired_records'],
                      'methods': report['methods'], 'rejection_scoring_boundary': report['rejection_scoring_boundary']},
                     ensure_ascii=False), flush=True)
