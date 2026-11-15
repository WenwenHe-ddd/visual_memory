"""Precompute tail tasks for an existing immutable M2A queue on another GPU.

Each output is reserved exclusively. The main queue's existing resume path will
accept a completed attempt only after its independent audit. Existing attempts
are never overwritten. No model, prompt, retrieval or evaluation settings change.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.m2a_suite import read, sha, write, now


def run(root, tasks, device):
    plan = read(root / 'plan.json')
    available = {t['task']: t for t in plan['datasets']['memeye']['tasks']}
    state = dict(pid=os.getpid(), started_at=now(), status='running', tasks=[],
                 physical_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 model_and_retrieval_policy='unchanged from signed plan',
                 script_sha256=sha(__file__))
    status_path = root / 'parallel_status.json'

    def save():
        state['updated_at'] = now()
        write(status_path, state)

    try:
        for task_name in tasks:
            for path, expected in plan['implementation_sha256'].items():
                if sha(path) != expected:
                    raise ValueError('Queued implementation changed: ' + path)
            task = available[task_name]
            if sha(task['source']) != task['source_sha256']:
                raise ValueError('Task source changed: ' + task_name)
            target = root / 'memeye' / task_name
            target.mkdir(exist_ok=True)
            if list(target.glob('attempt_*')):
                state['tasks'].append(dict(task=task_name, status='skipped_existing_attempt'))
                save()
                continue
            attempt = target / 'attempt_001'
            # Reserve before the main queue inspects this task. If it reaches an
            # unfinished reservation, it will choose a distinct attempt number.
            try:
                attempt.mkdir(exist_ok=False)
            except FileExistsError:
                state['tasks'].append(dict(task=task_name, status='skipped_race'))
                save()
                continue
            cmd = [sys.executable, '-u', str(ROOT/'scripts/evaluation/m2a_single_pass_run.py'),
                   '--dataset', 'memeye', '--task', task_name, '--mandatory-query',
                   '--device', device, '--output', str(attempt)]
            state['current'] = dict(task=task_name, output=str(attempt), command=cmd)
            with (target/'attempt_001.log').open('x') as log:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['current']['pid'] = child.pid
                save()
                while child.poll() is None:
                    progress = attempt/'progress.json'
                    if progress.exists():
                        state['current']['progress'] = read(progress)
                    save()
                    time.sleep(5)
            rc = child.returncode
            if rc == 0:
                with (attempt/'parallel_audit.log').open('w') as log:
                    rc = subprocess.call([sys.executable, str(ROOT/'scripts/evaluation/audit_memeye_m2a.py'),
                        str(attempt)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if rc != 0:
                state['status'] = 'stopped_on_failure'
                state['tasks'].append(dict(task=task_name, status='failed', exit_code=rc, output=str(attempt)))
                save()
                return 1
            summary = read(attempt/'summary.json')
            if not summary['is_complete_task'] or summary['completed_questions'] != task['questions']:
                raise ValueError('Incomplete precomputed task')
            state['tasks'].append(dict(task=task_name, status='passed', output=str(attempt),
                questions=summary['completed_questions'], correct=summary['correct'],
                completion_sha256=sha(attempt/'completion_manifest.json')))
            state.pop('current', None)
            save()
        state['status'] = 'completed'
        save()
        return 0
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raise SystemExit(run(args.output.resolve(), args.tasks, args.device))
