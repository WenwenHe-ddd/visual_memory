#!/usr/bin/env python3
"""Read-only canonical paired MCQ audit for all ten DeepSeek methods.

Missing completion evidence produces pending fields, never partial accuracy.
Only audit reports are written; no inference, API calls or metric exporters run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memeye_deepseek_seed1024_metrics as metrics
from scripts.evaluation import memeye_deepseek_seed1024_ours as ours

LABELS = {m: v[0] for m, v in metrics.OURS.items()} | metrics.TEXT | {
    'clip_image_dialogue': 'CLIP original images + dialogue', 'hippomm': 'HippoMM'}
PAIRED_FIELDS = ('both_correct', 'seed42_only_correct', 'seed1024_only_correct',
                 'both_wrong', 'choice_changed', 'both_wrong_choice_changed',
                 'native_prediction_changed', 'seed42_correct', 'seed1024_correct',
                 'correct_difference', 'seed42_accuracy', 'seed1024_accuracy')


def fail(message):
    raise ValueError(message)


def regex_choice(raw):
    match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw.strip(), re.I) if isinstance(raw, str) else None
    return match.group(1).upper() if match else None


def verify_cohort(values, canonical, key='id'):
    found = metrics.unique(values, key)
    if len(found) != 371 or set(found) != set(canonical):
        fail('Expected exactly 371 unique canonical MCQ identities')
    for identity, row in found.items():
        for field in ('task', 'question', 'options', 'gold_choice', 'gold_ids'):
            if field in row and row[field] != canonical[identity][field]:
                fail('Canonical field differs: ' + identity + '/' + field)
    return found


def source_layout(method, seed, output, reference):
    old = seed == 42
    root = Path(reference['source_root']) if old else output
    if method in metrics.OURS:
        condition = metrics.OURS[method][1]
        if old:
            folder = 'weighted_evaluation' if method in ('semrag', 'ours_1_ours_2') else 'complementary_evaluation'
            root = root / folder
            names = ['summary.json', 'reader/protocol.json', 'reader/responses.jsonl', 'reader/gold.jsonl']
        else:
            root = root / 'ours'
            names = ['summary.json', 'protocol.json', 'independent_audit.json', 'reader/responses.jsonl',
                     'inputs/questions.jsonl', 'inputs/gold.jsonl']
        return root, names, condition
    if method in metrics.TEXT:
        if not old:
            root = root / 'text_methods' / method
        names = ['summary.json', 'protocol.json', 'qa_responses.jsonl', 'questions.jsonl']
        if not old:
            names += ['completion_manifest.json', 'stability_audit.json']
        return root, names, None
    if method == 'clip_image_dialogue':
        if not old:
            root = root / 'visual_methods/clip_image_dialogue/online'
        names = ['summary.json', 'protocol.json', 'qa_responses.jsonl', 'qa_manifest.jsonl']
        if not old:
            names.append('completion.json')
        return root, names, None
    root = root / ('online' if old else 'visual_methods/hippomm/online')
    return root, ['summary.json', 'protocol.json', 'qa_responses.jsonl', 'questions.jsonl',
                  'completion_manifest.json'], None


def normalize(row, native_prediction, raw_output=None):
    out = raw_output if raw_output is not None else row
    state = row.get('status', out.get('status'))
    provider_rejected = state in ('provider_input_rejected', 'provider_rejection', 'content_filter')
    refusal = bool(out.get('refusal'))
    limited = bool(out.get('hit_token_limit')) or out.get('finish_reason') == 'length'
    prediction = native_prediction if native_prediction in ('A', 'B', 'C', 'D') else None
    outcome = ('provider_input_rejected' if provider_rejected else 'model_refusal' if refusal
               else 'token_limit' if limited else 'invalid_response' if prediction is None else 'ok')
    return {'choice': prediction, 'native_prediction': native_prediction,
            'native_status': state, 'outcome': outcome, 'model_refusal': refusal,
            'provider_rejected': provider_rejected, 'hit_token_limit': limited,
            'request_sha256': row.get('request_sha256', out.get('request_sha256'))}


def load_source(method, seed, output, reference, canonical):
    root, names, condition = source_layout(method, seed, output, reference)
    paths = [root / n for n in names]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        if seed == 42:
            fail('Missing historical source: ' + ', '.join(missing))
        return {'status': 'pending', 'root': str(root), 'missing_completion_evidence': missing}, None
    summary, protocol = metrics.read(root / 'summary.json'), metrics.read(root / ('reader/protocol.json' if method in metrics.OURS and seed == 42 else 'protocol.json'))
    if summary.get('status') not in (None, 'complete') or summary.get('questions', 371) != 371:
        fail('Source summary is not complete: ' + str(root))
    if method in metrics.OURS:
        expected_correct = summary['overall'][condition]['correct']
        raw = [r for r in metrics.rows(root / 'reader/responses.jsonl') if r['condition'] == condition]
        found = verify_cohort(raw, canonical, 'question_id')
        gold_path = root / ('reader/gold.jsonl' if seed == 42 else 'inputs/gold.jsonl')
        verify_cohort(metrics.rows(gold_path), canonical)
        if seed == 1024:
            if ours.digest(protocol['contract']) != protocol['experiment_signature']:
                fail('Ours protocol signature differs')
            audit = metrics.read(root / 'independent_audit.json')
            if not audit['passed']:
                fail('Ours independent completion audit failed')
            metrics.verify_hashes(audit['artifact_sha256'])
            verify_cohort(metrics.rows(root / 'inputs/questions.jsonl'), canonical)
        data = {}
        for identity, row in found.items():
            if row['seed'] != seed or row['predicted_choice'] != regex_choice(row.get('raw_response')):
                fail('Ours raw grading or seed differs: ' + identity)
            if row['record_id'] != identity + '::' + condition:
                fail('Ours condition identity differs')
            if seed == 1024 and (row['experiment_signature'] != protocol['experiment_signature'] or row['run_id'] != protocol['run_id'] or row.get('historical_answer_reuse') is not False):
                fail('Ours new response provenance differs')
            data[identity] = normalize(row, row['predicted_choice'])
        rule = 'Original case-insensitive full-match letter / Answer: letter parser, optional trailing period or exclamation.'
    elif method in metrics.TEXT:
        if protocol['seed'] != seed or protocol['model'] != metrics.MODEL:
            fail('Text seed/model differs')
        if metrics.digest({k: v for k, v in protocol.items() if k != 'experiment_signature'}) != protocol['experiment_signature']:
            fail('Text protocol signature differs')
        if seed == 1024:
            complete, audit = metrics.read(root / 'completion_manifest.json'), metrics.read(root / 'stability_audit.json')
            if complete.get('status') != 'complete' or not complete.get('passed') or not audit.get('passed') or audit['questions'] != 371:
                fail('Text completion evidence did not pass')
            metrics.verify_hashes({str(root / n): complete[k] for n, k in (
                ('protocol.json', 'protocol_sha256'), ('qa_responses.jsonl', 'responses_sha256'), ('stability_audit.json', 'audit_sha256'))})
            if audit['experiment_signature'] != protocol['experiment_signature']:
                fail('Text audit belongs to a different run')
        verify_cohort(metrics.rows(root / 'questions.jsonl'), canonical)
        found = verify_cohort(metrics.rows(root / 'qa_responses.jsonl'), canonical)
        data = {}
        for identity, row in found.items():
            text = row['raw_response'].strip()
            prediction = text if text in ('A', 'B', 'C', 'D') and not row['hit_token_limit'] and not row.get('refusal') else None
            if row['seed'] != seed or row['returned_model'] != metrics.MODEL or row['predicted_choice'] != prediction:
                fail('Text raw grading, model or seed differs: ' + identity)
            if row['experiment_signature'] != protocol['experiment_signature']:
                fail('Text response belongs to a different run')
            data[identity] = normalize(row, prediction)
        expected_correct = summary['correct']
        rule = 'Native exact uppercase A/B/C/D after strip, with token-limit and refusal outputs invalid.'
    elif method == 'clip_image_dialogue':
        pseed = protocol['settings']['seed'] if seed == 42 else protocol['seed']
        pmodel = protocol['model'] if seed == 42 else protocol['answer_model']
        if pseed != seed or pmodel != metrics.MODEL or metrics.compact_digest({k: v for k, v in protocol.items() if k != 'experiment_signature'}) != protocol['experiment_signature']:
            fail('CLIP protocol seed/model/signature differs')
        if seed == 1024:
            completion = metrics.read(root / 'completion.json')
            if completion['status'] != 'complete' or completion['questions'] != 371 or completion['answered'] != 371 or completion['responses_sha256'] != metrics.sha(root / 'qa_responses.jsonl'):
                fail('CLIP completion evidence differs')
        verify_cohort(metrics.rows(root / 'qa_manifest.jsonl'), canonical)
        found = verify_cohort(metrics.rows(root / 'qa_responses.jsonl'), canonical)
        data = {}
        for identity, row in found.items():
            if row['predicted_choice'] != regex_choice(row.get('raw_response')) or row['experiment_signature'] != protocol['experiment_signature']:
                fail('CLIP raw grading/provenance differs: ' + identity)
            if seed == 1024 and row['seed'] != seed:
                fail('CLIP new response seed differs')
            if row.get('status') == 'ok' and row['returned_model'] != metrics.MODEL:
                fail('CLIP returned model differs')
            data[identity] = normalize(row, row['predicted_choice'])
        expected_correct = summary['correct']
        rule = 'Original case-insensitive full-match letter parser; real provider rejection remains invalid.'
    else:
        contract = protocol['contract']
        if contract['seed'] != seed or contract['model'] != metrics.MODEL:
            fail('HippoMM source seed/model differs')
        completion = metrics.read(root / 'completion_manifest.json')
        if completion['status'] != 'complete' or completion['questions'] != 371:
            fail('HippoMM source is incomplete')
        verify_cohort(metrics.rows(root / 'questions.jsonl'), canonical)
        found = verify_cohort(metrics.rows(root / 'qa_responses.jsonl'), canonical)
        data = {}
        for identity, row in found.items():
            if row['model'] != metrics.MODEL:
                fail('HippoMM response model differs')
            calls = []
            for name in row['llm_calls']:
                path = (root / name).resolve()
                if not path.is_relative_to(root.resolve()):
                    fail('HippoMM foreign cache path')
                call = metrics.read(path)
                if call['response']['seed'] != seed or call['response']['returned_model'] != metrics.MODEL:
                    fail('HippoMM cached generation seed/model differs')
                paths.append(path)
                calls.append(call)
            final = [c for c in calls if c['request']['role'] in ('answer', 'answer_with_representatives', 'format_mcq')]
            if not final or final[-1]['response']['raw_response'].strip() != row['answer']:
                fail('HippoMM final answer differs from actual provider output')
            data[identity] = normalize(row, row['answer'], final[-1]['response'])
            data[identity].update(route=row['route'],
                any_stage_model_refusal=any(bool(c['response'].get('refusal')) for c in calls),
                any_stage_token_limit=any(c['response'].get('finish_reason') == 'length' for c in calls))
        expected_correct = summary['correct']
        rule = 'Native exact final answer after strip. Free-form invalid answers stay incorrect; no permissive reparsing.'
    correct = 0
    for identity, value in data.items():
        value['correct'] = value['choice'] == canonical[identity]['gold_choice']
        correct += value['correct']
        if 'correct' in found[identity] and found[identity]['correct'] != value['correct']:
            fail('Stored correctness differs: ' + identity)
    if correct != expected_correct or (seed == 42 and correct != reference['correct']):
        fail('Regraded accuracy differs from the source summary/reference for ' + method)
    info = {'status': 'complete', 'root': str(root), 'seed': seed, 'model': metrics.MODEL,
            'questions': 371, 'correct': correct, 'accuracy': correct / 371, 'native_grading_rule': rule,
            'normalized_outcomes': dict(Counter(r['outcome'] for r in data.values())),
            'source_sha256': {str(p): metrics.sha(p) for p in dict.fromkeys(paths)}}
    return info, data


def pair_method(method, previous, current, canonical):
    counters, transitions, native_states = Counter(), Counter(), Counter()
    flags = ('model_refusal', 'provider_rejected', 'hit_token_limit')
    flag_counts = {flag: Counter() for flag in flags}
    detail, exceptional = [], []
    for identity in canonical:
        old, new = previous[identity], current[identity]
        a, b = old['correct'], new['correct']
        category = 'both_correct' if a and b else 'seed42_only_correct' if a else 'seed1024_only_correct' if b else 'both_wrong'
        changed = old['choice'] != new['choice']
        counters[category] += 1
        counters['choice_changed'] += changed
        counters['both_wrong_choice_changed'] += changed and not a and not b
        counters['native_prediction_changed'] += old['native_prediction'] != new['native_prediction']
        transitions[(old['outcome'], new['outcome'])] += 1
        native_states[(str(old['native_status']), str(new['native_status']))] += 1
        for flag in flags:
            flag_counts[flag]['seed42'] += old[flag]
            flag_counts[flag]['seed1024'] += new[flag]
            flag_counts[flag]['both'] += old[flag] and new[flag]
            flag_counts[flag]['seed42_only'] += old[flag] and not new[flag]
            flag_counts[flag]['seed1024_only'] += new[flag] and not old[flag]
        row = {'method': method, 'id': identity, 'task': canonical[identity]['task'],
            'seed42_choice': old['choice'], 'seed1024_choice': new['choice'],
            'seed42_correct': a, 'seed1024_correct': b, 'paired_category': category,
            'choice_changed': changed, 'native_prediction_changed': old['native_prediction'] != new['native_prediction'],
            'seed42_outcome': old['outcome'], 'seed1024_outcome': new['outcome'],
            'seed42_native_status': old['native_status'], 'seed1024_native_status': new['native_status'],
            'seed42_request_sha256': old['request_sha256'], 'seed1024_request_sha256': new['request_sha256'],
            **{f'seed{seed}_{flag}': value[flag] for seed, value in ((42, old), (1024, new)) for flag in flags}}
        if 'route' in old:
            row.update(seed42_route=old['route'], seed1024_route=new['route'])
        detail.append(row)
        if any(old[f] or new[f] for f in flags):
            exceptional.append({**row, 'measured_correct_difference': int(b) - int(a),
                                'counterfactual_answer_not_imputed': True})
    n42 = counters['both_correct'] + counters['seed42_only_correct']
    n1024 = counters['both_correct'] + counters['seed1024_only_correct']
    return {'status': 'complete', 'questions': 371,
        **{key: counters[key] for key in PAIRED_FIELDS if key not in ('seed42_correct', 'seed1024_correct', 'correct_difference', 'seed42_accuracy', 'seed1024_accuracy')},
        'seed42_correct': n42, 'seed1024_correct': n1024, 'correct_difference': n1024 - n42,
        'seed42_accuracy': n42 / 371, 'seed1024_accuracy': n1024 / 371,
        'outcome_transitions': [{'seed42': a, 'seed1024': b, 'records': n} for (a, b), n in sorted(transitions.items())],
        'native_status_transitions': [{'seed42': a, 'seed1024': b, 'records': n} for (a, b), n in sorted(native_states.items())],
        'flag_changes': {key: {name: values[name] for name in ('seed42', 'seed1024', 'both', 'seed42_only', 'seed1024_only')} for key, values in flag_counts.items()},
        'exceptional_outcome_records': exceptional}, detail


def audit(output=metrics.OUT, report_dir=None):
    output = Path(output).resolve()
    report_dir = Path(report_dir or output / 'metrics').resolve()
    canonical = metrics.unique(metrics.rows(metrics.BENCHMARK))
    if len(canonical) != 371:
        fail('Canonical benchmark count differs')
    references, reports, details = metrics.reference_mapping(), {}, []
    for method in metrics.METHODS:
        entry = {'method': method, 'label': LABELS[method], 'expected_questions': 371}
        try:
            old_source, old = load_source(method, 42, output, references[method], canonical)
            new_source, new = load_source(method, 1024, output, references[method], canonical)
            entry.update(seed42_source=old_source, seed1024_source=new_source)
            if new is None:
                entry.update(status='pending', questions=None, **{k: None for k in PAIRED_FIELDS})
            else:
                paired, rows = pair_method(method, old, new, canonical)
                entry.update(paired)
                details.extend(rows)
        except (ValueError, AssertionError, KeyError, TypeError) as exc:
            entry.update(status='failed_audit', error_type=type(exc).__name__, error=str(exc))
        reports[method] = entry
    counts = Counter(r['status'] for r in reports.values())
    report = {'schema': 'memeye-deepseek-paired-mcq-v1',
        'status': 'failed_audit' if counts['failed_audit'] else 'complete' if counts['complete'] == 10 else 'partial',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'api_calls': 0,
        'seed42': 42, 'seed1024': 1024, 'model': metrics.MODEL,
        'canonical_questions': 371, 'methods_expected': 10, 'methods_complete': counts['complete'],
        'methods_pending': counts['pending'], 'methods_failed_audit': counts['failed_audit'],
        'paired_records': len(details), 'methods': reports,
        'formulas': {'seed42_correct': 'both_correct + seed42_only_correct',
            'seed1024_correct': 'both_correct + seed1024_only_correct',
            'correct_difference': 'seed1024_only_correct - seed42_only_correct',
            'choice_changed': 'Different normalized A/B/C/D/None choices; invalid native HippoMM answers normalize to None.',
            'native_prediction_changed': 'Different stored native predictions, preserving distinct invalid HippoMM answer strings.'},
        'outcome_policy': 'Preserve native statuses separately. Derived terminal outcomes use provider rejection, explicit model refusal, token limit, invalid choice, then ok in that priority order. Provider rejection remains in the 371 denominator; no replacement answers or hypothetical scores.',
        'pending_policy': 'No paired counts or new scores are reported until complete-source markers, hashes, canonical coverage and native raw-output grading pass.',
        'gold_use_boundary': 'Canonical gold is read only for this offline posthoc scoring audit. No request is built or sent and no source answer is modified.',
        'interpretation_limit': 'Same provider alias across dates does not establish fixed underlying weights. Full-flow methods can change memories/retrieval as well as answers; the paired deltas are not an isolated causal seed effect.',
        'source_sha256': {str(p): metrics.sha(p) for p in (Path(__file__), Path(metrics.__file__), metrics.BENCHMARK, metrics.REFERENCE / 'storage_summary.json')}}
    record_path = report_dir / 'paired_mcq_records.jsonl'
    metrics.write_rows(record_path, details)
    report['paired_records_path'] = str(record_path)
    report['paired_records_sha256'] = metrics.sha(record_path)
    metrics.dump(report_dir / 'paired_mcq_audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=metrics.OUT)
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args()
    result = audit(args.output_root, args.report_dir)
    print(json.dumps({key: result[key] for key in ('status', 'methods_complete', 'methods_pending', 'methods_failed_audit', 'paired_records')}, ensure_ascii=False))
    for method, value in result['methods'].items():
        print(json.dumps({'method': method, **{key: value.get(key) for key in ('status', 'seed42_correct', 'seed1024_correct', 'both_correct', 'seed42_only_correct', 'seed1024_only_correct', 'both_wrong', 'choice_changed', 'error')}}, ensure_ascii=False))
