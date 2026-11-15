"""Durable, serial GPU queue for the full MemEye seed-1024 replication.

Each stage runs its audited native driver in a separate process. Failures are
recorded; independent methods continue. The queue never treats partial answers
as a completed score. Invoke status to refresh the comparison during a long run.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/memeye_qwen_seed1024_full_20261105'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_sources(plan):
    for path, expected in plan['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Pinned queue implementation changed: ' + path)
    if plan['seed'] != 1024 or plan['judge_seed'] != 42:
        raise ValueError('Wrong replication/evaluator seeds')


def completed_marker(stage):
    marker = stage.get('marker')
    if not marker or not Path(marker).exists():
        return False
    item = read(marker)
    key = stage.get('marker_key', 'status')
    return item.get(key) == stage.get('marker_value', 'complete')


def snapshot_rows(path):
    """Ignore only an unfinished trailing write; validate every complete row."""
    data = Path(path).read_bytes()
    return [json.loads(line.decode('utf-8')) for line in data.splitlines(keepends=True)
            if line.endswith(b'\n') and line.strip()]


def snapshot_json(path):
    try:
        return read(path)
    except json.JSONDecodeError:
        return {'status': 'writing'}


def comparison(plan):
    table = []
    for method in plan['methods']:
        row = {k: method.get(k) for k in ('id', 'label', 'seed42_correct', 'seed42_recall',
                                         'seed42_judge', 'seed42_storage_mib')}
        row.update(seed42_accuracy=method['seed42_correct'] / 371, expected_questions=371,
                   seed1024_correct=None, seed1024_accuracy=None, accuracy_delta_pp=None,
                   seed1024_recall=None, seed1024_judge=None, seed1024_storage_mib=None,
                   two_seed_accuracy_mean=None, two_seed_accuracy_sample_sd=None,
                   answered_questions=0, mcq_status='pending', judge_status='pending')
        summary = Path(method['mcq_summary'])
        responses = Path(method['mcq_responses'])
        if responses.exists():
            values = snapshot_rows(responses)
            if method.get('condition'):
                values = [r for r in values if r.get('condition') == method['condition']]
            row['answered_questions'] = len(values)
            row['mcq_status'] = 'running' if values else 'pending'
        if summary.exists():
            data = snapshot_json(summary)
            if data.get('status', 'complete') == 'complete':
                metric = data['overall'][method['condition']] if method.get('condition') else data
                if metric.get('questions') == 371:
                    correct = metric.get('correct')
                    if correct is None and 'accuracy' in metric:
                        correct = round(metric['accuracy'] * 371)
                    if correct is not None:
                        accuracy = correct / 371
                        row.update(seed1024_correct=correct, seed1024_accuracy=accuracy,
                            accuracy_delta_pp=(accuracy - row['seed42_accuracy']) * 100,
                            two_seed_accuracy_mean=(accuracy + row['seed42_accuracy']) / 2,
                            two_seed_accuracy_sample_sd=abs(accuracy - row['seed42_accuracy']) / math.sqrt(2),
                            answered_questions=371, mcq_status='complete')
                    for key in ('recall_at_5_micro', 'recall_at5_micro', 'context_source_micro_recall'):
                        if key in metric:
                            row['seed1024_recall'] = metric[key]
                            break
        extra = Path(method['metric_summary']) if method.get('metric_summary') else None
        if extra and extra.exists():
            metric = snapshot_json(extra)
            if 'rows' in metric:
                metric = next((r for r in metric['rows'] if r.get('method') == method['id']), {})
            if metric.get('storage_mib') is not None:
                row['seed1024_storage_mib'] = metric['storage_mib']
            if metric.get('recall_at5_micro') is not None:
                row['seed1024_recall'] = metric['recall_at5_micro']
            elif metric.get('recall_at_5_micro') is not None:
                row['seed1024_recall'] = metric['recall_at_5_micro']
        judge = Path(method['open_root']) / 'judge/llm_judge_summary.json'
        if judge.exists():
            metric = snapshot_json(judge)
            row['judge_status'] = metric['status']
            if metric['status'] == 'complete' and metric['questions_scored'] == 371:
                row['seed1024_judge'] = metric['llm_judge']
        table.append(row)
    result = {'updated_at_utc': now(), 'seed': 1024, 'reference_seed': 42, 'judge_seed': 42,
        'questions_per_method': 371, 'mcq_methods_complete': sum(r['mcq_status'] == 'complete' for r in table),
        'judge_methods_complete': sum(r['judge_status'] == 'complete' for r in table), 'methods': table,
        'notes': ['Accuracy is fixed-option MCQ, and Judge uses the official Open questions.',
                  'Native UUID, wall-clock and batching effects can accompany the seed change.',
                  'Two seeds provide a limited stability check; they do not establish general robustness.',
                  'Only full 371-question scores are published. Empty cells are pending measurements.']}
    dump(OUT / 'comparison.json', result)
    csv_path = OUT / ('comparison.csv.tmp-' + str(os.getpid()))
    with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    csv_path.replace(OUT / 'comparison.csv')
    return result


def run(plan):
    validate_sources(plan)
    state_path = OUT / 'progress.json'
    state = read(state_path) if state_path.exists() else {'stages': {}}
    state.update(status='running', pid=os.getpid(), seed=1024, judge_seed=42,
                 plan_sha256=sha(OUT / 'plan.json'), started_or_resumed_at_utc=now())
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    failures = []
    for stage in plan['stages']:
        validate_sources(plan)
        name = stage['name']
        previous = state['stages'].get(name, {})
        if previous.get('status') == 'complete' and (not stage.get('marker') or completed_marker(stage)):
            continue
        unit = stage.get('external_unit')
        if unit:
            while subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0:
                state.update(current_stage=name, updated_at_utc=now())
                state['stages'][name] = {'status': 'running', 'external_unit': unit}
                dump(state_path, state)
                comparison(plan)
                time.sleep(15)
        if completed_marker(stage):
            state['stages'][name] = {'status': 'complete', 'adopted_existing_run': True,
                                     'marker_sha256': sha(stage['marker']), 'finished_at_utc': now()}
            dump(state_path, state)
            comparison(plan)
            continue
        missing = [dep for dep in stage.get('depends_on', [])
                   if state['stages'].get(dep, {}).get('status') != 'complete']
        if missing:
            state['stages'][name] = {'status': 'skipped_dependency', 'dependencies': missing}
            dump(state_path, state)
            continue
        log = logs / (name + '.log')
        state.update(current_stage=name, updated_at_utc=now())
        state['stages'][name] = {'status': 'running', 'started_at_utc': now(),
                                 'argv': stage['argv'], 'log': str(log)}
        dump(state_path, state)
        print(json.dumps({'stage': name, 'status': 'started', 'log': str(log)}), flush=True)
        with log.open('ab') as stream:
            child = subprocess.Popen(stage['argv'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            state['stages'][name]['child_pid'] = child.pid
            dump(state_path, state)
            while child.poll() is None:
                comparison(plan)
                state['updated_at_utc'] = now()
                dump(state_path, state)
                time.sleep(15)
            code = child.returncode
        okay = code == 0 and (not stage.get('marker') or completed_marker(stage))
        state['stages'][name].update(status='complete' if okay else 'failed', returncode=code,
                                     finished_at_utc=now())
        if okay and stage.get('marker'):
            state['stages'][name]['marker_sha256'] = sha(stage['marker'])
        if not okay:
            failures.append(name)
        dump(state_path, state)
        comparison(plan)
        print(json.dumps({'stage': name, 'status': state['stages'][name]['status'], 'returncode': code}), flush=True)
    report = comparison(plan)
    pending = [s['name'] for s in plan['stages'] if state['stages'].get(s['name'], {}).get('status') != 'complete']
    metric_pending = [row['id'] for row in report['methods']
        if row['seed1024_storage_mib'] is None or
        (row['seed42_recall'] is not None and row['seed1024_recall'] is None)]
    state.update(status='partial_failed' if pending else 'awaiting_metric_review' if metric_pending else 'complete', current_stage=None,
                 unfinished_stages=pending, updated_at_utc=now(),
                 unfinished_metric_methods=metric_pending,
                 mcq_methods_complete=report['mcq_methods_complete'], judge_methods_complete=report['judge_methods_complete'])
    dump(state_path, state)
    if pending:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run', 'status'), required=True)
    args = parser.parse_args()
    plan = read(OUT / 'plan.json')
    if args.phase == 'status':
        report = comparison(plan)
        print(json.dumps({'mcq_methods_complete': report['mcq_methods_complete'],
                          'judge_methods_complete': report['judge_methods_complete'],
                          'progress': read(OUT / 'progress.json') if (OUT / 'progress.json').exists() else None}), flush=True)
        return
    with (OUT / 'suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(plan)
        except BaseException as exc:
            if not isinstance(exc, SystemExit):
                dump(OUT / 'runner_failure.json', {'at_utc': now(), 'error': str(exc), 'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
