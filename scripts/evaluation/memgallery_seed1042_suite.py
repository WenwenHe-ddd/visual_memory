"""Persistent single-GPU queue for the audited MemGallery seed1042 replication."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/memgallery_qwen_seed1042_full_20261106'
PYTHON = '/usr/bin/python3'


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


def verify_plan(plan):
    if plan['seed'] != 1042 or plan['reference_seed'] != 42 or plan['judge_seed'] != 42:
        raise ValueError('Incorrect generation/reference/evaluation seeds')
    for path, expected in plan['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen replication implementation changed: ' + path)


def marker_complete(stage):
    path = Path(stage['marker'])
    if not path.exists():
        return False
    try:
        data = read(path)
    except json.JSONDecodeError:
        return False
    return data.get(stage.get('marker_key', 'status')) == stage.get('marker_value', 'complete')


def snapshot_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
            if line.endswith(b'\n') and line.strip()]


def status(plan):
    reference = read(RUN / 'reference_seed42.json')
    old = {r['id']: r for r in reference['methods']}
    values = []
    for item in plan['methods']:
        mid = item['id']; first = old[mid]
        row = {'id': mid, 'method': item['label'], 'questions': 1711,
               'seed42_correct': first['correct'], 'seed1042_correct': None,
               'generation_status': 'pending', 'judge_status': 'pending'}
        for key in ('em', 'f1', 'bleu_1', 'llm_judge'):
            row['seed42_' + key] = first.get(key)
            row['seed1042_' + key] = None
            row[key + '_delta'] = None
            row[key + '_two_seed_mean'] = None
            row[key + '_two_seed_sample_sd'] = None
        old_judge = first.get('judge_source_path')
        if row['seed42_llm_judge'] is None and old_judge and Path(old_judge).exists():
            data = read(old_judge)
            if data.get('status') == 'complete' and data.get('questions_scored') == 1711:
                row['seed42_llm_judge'] = data['llm_judge']
        path = RUN / 'evaluations' / mid / 'metrics.json'
        if path.exists():
            data = read(path)
            if data.get('status') == 'complete' and data.get('seed') == 1042 and data.get('questions') == 1711:
                row.update(generation_status='complete', seed1042_correct=data['correct'])
                for key in ('em', 'f1', 'bleu_1'):
                    row['seed1042_' + key] = data[key]
        path = RUN / 'judge' / mid / 'methods' / mid / 'summary.json'
        if path.exists():
            data = read(path)
            row['judge_status'] = data.get('status', 'pending')
            if data.get('status') == 'complete' and data.get('questions_scored') == 1711:
                row['seed1042_llm_judge'] = data['llm_judge']
        for key in ('em', 'f1', 'bleu_1', 'llm_judge'):
            a, b = row['seed42_' + key], row['seed1042_' + key]
            if a is not None and b is not None:
                row[key + '_delta'] = b - a
                row[key + '_two_seed_mean'] = (a + b) / 2
                row[key + '_two_seed_sample_sd'] = abs(b - a) / (2 ** .5)
        values.append(row)
    caption = sum(len(snapshot_rows(RUN / 'captions' / f'captions.worker{i}.jsonl')) for i in (0, 1))
    result = {'updated_at_utc': now(), 'seed': 1042, 'reference_seed': 42, 'judge_seed': 42,
              'captions_generated': caption, 'captions_expected': 1393,
              'methods_generation_complete': sum(r['generation_status'] == 'complete' for r in values),
              'methods_judge_complete': sum(r['judge_status'] == 'complete' for r in values),
              'methods': values, 'notes': plan['notes']}
    dump(RUN / 'comparison.json', result)
    temporary = RUN / ('comparison.csv.tmp-' + str(os.getpid()))
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)
    temporary.replace(RUN / 'comparison.csv')
    return result


def run(plan):
    verify_plan(plan)
    state_path = RUN / 'progress.json'
    state = read(state_path) if state_path.exists() else {'stages': {}}
    state.update(status='running', pid=os.getpid(), seed=1042, judge_seed=42,
                 plan_sha256=sha(RUN / 'plan.json'), resumed_at_utc=now())
    for stage in plan['stages']:
        verify_plan(plan)
        name = stage['name']; previous = state['stages'].get(name, {})
        if previous.get('status') == 'complete' and marker_complete(stage):
            continue
        unit = stage.get('external_unit')
        if unit:
            while subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0:
                state.update(current_stage=name, updated_at_utc=now())
                state['stages'][name] = {'status': 'running', 'external_unit': unit}
                dump(state_path, state); status(plan); time.sleep(15)
        if marker_complete(stage):
            state['stages'][name] = {'status': 'complete', 'adopted_existing_run': True,
                'marker_sha256': sha(stage['marker']), 'finished_at_utc': now()}
            dump(state_path, state)
            continue
        missing = [d for d in stage.get('depends_on', []) if state['stages'].get(d, {}).get('status') != 'complete']
        if missing:
            state['stages'][name] = {'status': 'skipped_dependency', 'dependencies': missing}
            dump(state_path, state)
            continue
        log_path = RUN / 'logs' / (name + '.log')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        if stage.get('needs_gpu', False):
            while True:
                free = subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.free',
                                                '--format=csv,noheader,nounits'], text=True).strip()
                free_mib = int(free)
                if free_mib >= plan.get('minimum_free_gpu_mib', 55000):
                    break
                state.update(current_stage=name, updated_at_utc=now())
                state['stages'][name] = {'status': 'waiting_gpu', 'physical_gpu': 1, 'free_mib': free_mib}
                dump(state_path, state); status(plan); time.sleep(15)
        state.update(current_stage=name, updated_at_utc=now())
        state['stages'][name] = {'status': 'running', 'started_at_utc': now(), 'argv': stage['argv'], 'log': str(log_path)}
        dump(state_path, state)
        print(json.dumps({'stage': name, 'status': 'started'}), flush=True)
        with log_path.open('ab') as stream:
            child = subprocess.Popen(stage['argv'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            state['stages'][name]['child_pid'] = child.pid
            dump(state_path, state)
            while child.poll() is None:
                state['updated_at_utc'] = now()
                dump(state_path, state); status(plan); time.sleep(15)
            code = child.returncode
        okay = code == 0 and marker_complete(stage)
        state['stages'][name].update(status='complete' if okay else 'failed', returncode=code, finished_at_utc=now())
        if okay:
            state['stages'][name]['marker_sha256'] = sha(stage['marker'])
        dump(state_path, state); status(plan)
        print(json.dumps({'stage': name, 'status': state['stages'][name]['status']}), flush=True)
    summary = status(plan)
    all_stages_done = all(state['stages'].get(s['name'], {}).get('status') == 'complete' for s in plan['stages'])
    state.update(status='generation_complete' if all_stages_done and summary['methods_generation_complete'] == 9 else 'partial_failure',
                 finished_at_utc=now(), current_stage=None)
    dump(state_path, state)
    dump(RUN / 'generation_completion.json', state)
    return 0 if state['status'] == 'generation_complete' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run', 'status'), required=True)
    args = parser.parse_args()
    plan = read(RUN / 'plan.json')
    if args.phase == 'status':
        print(json.dumps(status(plan), ensure_ascii=False, indent=2))
        return
    (RUN / 'logs').mkdir(exist_ok=True)
    with (RUN / 'queue.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run(plan))


if __name__ == '__main__':
    main()
