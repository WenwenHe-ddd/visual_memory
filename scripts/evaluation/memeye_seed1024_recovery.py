"""Resume audited incomplete stages on two GPUs without rerunning completed work."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import time
import uuid

from scripts.evaluation import memeye_seed1024_suite as suite

ROOT, OUT = suite.ROOT, suite.OUT
RECOVERY = OUT / 'recovery_20261106'


def dump(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    tmp.write_text(json.dumps(item, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(plan):
    for path, expected in plan['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Recovery source changed: ' + path)
    suite.validate_sources(read(OUT / 'plan.json'))


def lane(name, jobs, updates, initial, plan):
    """A lane owns one GPU; API-only judges follow their own completed answers."""
    status = copy.deepcopy(initial)
    for job in jobs:
        verify(plan)
        identity = job['name']
        dependencies = job.get('depends_on', [])
        missing = [key for key in dependencies if status.get(key, {}).get('status') != 'complete']
        if missing:
            entry = {'status': 'skipped_dependency', 'dependencies': missing, 'recovery_lane': name}
            updates.put((identity, entry))
            status[identity] = entry
            continue
        if job.get('marker') and suite.completed_marker(job):
            entry = {'status': 'complete', 'recovery_lane': name, 'adopted_existing_run': True,
                     'marker_sha256': sha(job['marker']), 'finished_at_utc': suite.now()}
            updates.put((identity, entry))
            status[identity] = entry
            continue
        log = RECOVERY / 'logs' / (identity + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        entry = {'status': 'running', 'recovery_lane': name, 'argv': job['argv'],
                 'log': str(log), 'started_at_utc': suite.now()}
        with log.open('ab') as stream:
            child = subprocess.Popen(job['argv'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            entry['child_pid'] = child.pid
            updates.put((identity, dict(entry)))
            code = child.wait()
        okay = code == 0 and (not job.get('marker') or suite.completed_marker(job))
        entry.update(status='complete' if okay else 'failed', returncode=code,
                     finished_at_utc=suite.now())
        if okay and job.get('marker'):
            entry['marker_sha256'] = sha(job['marker'])
        status[identity] = entry
        updates.put((identity, dict(entry)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run', 'verify'), required=True)
    args = parser.parse_args()
    plan = read(RECOVERY / 'plan.json')
    verify(plan)
    if args.phase == 'verify':
        print(json.dumps({'sources_verified': True, 'lanes': list(plan['lanes']),
                          'jobs': sum(map(len, plan['lanes'].values()))}), flush=True)
        return
    with (RECOVERY / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_plan = read(OUT / 'plan.json')
        state = read(OUT / 'progress.json')
        initial = copy.deepcopy(state['stages'])
        state.update(status='running_recovery', recovery_pid=os.getpid(),
                     recovery_plan_sha256=sha(RECOVERY / 'plan.json'),
                     recovery_started_at_utc=suite.now())
        updates = queue.Queue()
        with ThreadPoolExecutor(max_workers=len(plan['lanes'])) as pool:
            futures = [pool.submit(lane, name, jobs, updates, initial, plan)
                       for name, jobs in plan['lanes'].items()]
            while True:
                while not updates.empty():
                    identity, entry = updates.get_nowait()
                    prior = state['stages'].get(identity, {})
                    if prior.get('status') in ('failed', 'skipped_dependency'):
                        state.setdefault('recovery_prior_status', {}).setdefault(identity, prior)
                    state['stages'][identity] = entry
                    print(json.dumps({'stage': identity, 'status': entry['status'],
                                      'lane': entry.get('recovery_lane')}), flush=True)
                state['running_stages'] = [k for k, v in state['stages'].items() if v['status'] == 'running']
                state['current_stage'] = ', '.join(state['running_stages']) or None
                state['updated_at_utc'] = suite.now()
                result = suite.comparison(original_plan)
                state['mcq_methods_complete'] = result['mcq_methods_complete']
                state['judge_methods_complete'] = result['judge_methods_complete']
                dump(OUT / 'progress.json', state)
                if all(f.done() for f in futures) and updates.empty():
                    for future in futures:
                        future.result()
                    break
                time.sleep(10)
        pending = [s['name'] for s in original_plan['stages']
                   if state['stages'].get(s['name'], {}).get('status') != 'complete']
        metric_pending = [r['id'] for r in result['methods'] if r['seed1024_storage_mib'] is None
                          or (r['seed42_recall'] is not None and r['seed1024_recall'] is None)]
        state.update(status='partial_failed' if pending else 'awaiting_metric_review' if metric_pending else 'complete',
                     unfinished_stages=pending, unfinished_metric_methods=metric_pending,
                     current_stage=None, running_stages=[], updated_at_utc=suite.now())
        dump(OUT / 'progress.json', state)
        dump(RECOVERY / 'completion.json', {'status': state['status'], 'unfinished_stages': pending,
                                           'unfinished_metric_methods': metric_pending,
                                           'mcq_methods_complete': result['mcq_methods_complete'],
                                           'judge_methods_complete': result['judge_methods_complete']})
        if pending:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
