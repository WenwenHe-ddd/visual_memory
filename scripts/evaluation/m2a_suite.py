"""Durable per-dataset M2A queue; isolated tasks, strict audits, explicit failures.

Resume skips only independently audited completed tasks. Failed/interrupted tasks
are retained and retried in a fresh attempt directory. No partially built memory
is accepted as a completed benchmark.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.m2a_inputs import task_names


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(root, runner='original', smoke_first=False, stop_on_failure=False, datasets_to_run=None):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        existing = read(root / 'plan.json')
        for key, value in dict(runner=runner, smoke_first=smoke_first, stop_on_failure=stop_on_failure).items():
            if existing['configuration'].get(key, 'original' if key == 'runner' else False) != value:
                raise ValueError('Existing plan differs; use a fresh output root: ' + key)
        return existing
    datasets = {}
    for dataset in (datasets_to_run or (('memeye',) if runner == 'single_pass' else ('memeye', 'memgallery'))):
        tasks = []
        for task in task_names(dataset):
            folder = 'MemEye' if dataset == 'memeye' else 'Mem-Gallery'
            suffix = '_Open.json' if dataset == 'memeye' else '.json'
            source = ROOT / 'data' / folder / 'data/dialog' / (task + suffix)
            data = read(source)
            rounds = [t for s in data['multi_session_dialogues'] for t in s['dialogues']]
            qs = data['human-annotated QAs']
            tasks.append(dict(task=task, rounds=len(rounds), questions=len(qs),
                source=str(source), source_sha256=sha(source),
                memory_source=str(ROOT / 'results/m2a_qwen_bge_card_20261112/run_v2')
                    if runner == 'original' and dataset == 'memeye' and task == 'Card_Playlog_Test' else None))
        # Verify the previously completed Card memory first; then shorter tasks.
        tasks.sort(key=lambda t: (not bool(t['memory_source']), t['rounds']))
        if runner == 'repaired':
            first = 'Social_Chat_Memory_Test' if dataset == 'memeye' else 'Fashion_Personal_Care_Lifestyle_Shopping'
            tasks.sort(key=lambda t: (t['task'] != first, t['rounds']))
        datasets[dataset] = dict(tasks=tasks, rounds=sum(t['rounds'] for t in tasks),
                                 questions=sum(t['questions'] for t in tasks))
    code = [ROOT / p for p in (
        'scripts/evaluation/m2a_suite.py', 'scripts/evaluation/memeye_m2a.py',
        'scripts/evaluation/m2a_inputs.py', 'scripts/evaluation/audit_memeye_m2a.py',
        'scripts/evaluation/mem_gallery_master_reader.py', 'scripts/memory/m2a_local.py',
        'scripts/memory/m2a_vendor.py', 'scripts/memory/m2a_search_budget.py')]
    if runner in ('repaired', 'single_pass'):
        code += [ROOT / p for p in ('scripts/evaluation/m2a_evidence_run.py',
            'scripts/memory/m2a_evidence.py', 'scripts/memory/m2a_length_prompt.py')]
    if runner == 'single_pass':
        code += [ROOT / p for p in ('scripts/evaluation/m2a_single_pass_run.py',
                                   'scripts/memory/m2a_single_pass.py')]
    frozen = {}
    for p in code:
        target = root / 'code_snapshot' / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
        frozen[str(p)] = sha(p)
    plan = dict(created_at=now(), datasets=datasets, implementation_sha256=frozen,
        configuration=dict(model='Qwen3-VL-8B-Instruct', text_encoder='BAAI/bge-small-en-v1.5',
            runner=runner, smoke_first=smoke_first, stop_on_failure=stop_on_failure,
            visual_encoder='google/siglip2-base-patch16-384', seed=42, do_sample=False,
            mandatory_initial_query=True, image_max_pixels=262144, max_new_tokens=1200,
            max_additional_search_rounds=2, max_search_calls_per_question=3,
            memeye_metric='fixed-permutation MCQ accuracy',
            memgallery_metric='normalized open-answer EM/F1/BLEU; no judge',
            history='original dialogue and image pixels; no precomputed captions',
            workers='one sequential task queue per dataset; each subprocess has its own models and DB',
            failure_policy=('retain failed attempt and stop this dataset queue at its first failure'
                if stop_on_failure else 'retain failed attempt, continue other tasks, exclude failures from completed coverage')))
    write(root / 'plan.json', plan)
    return plan


def run(root, dataset, device):
    import fcntl
    plan = read(root / 'plan.json')
    directory = root / dataset
    directory.mkdir(exist_ok=True)
    lock = (directory / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(dataset=dataset, pid=os.getpid(), started_at=now(), status='running', tasks=[])
    started = time.time()
    entrypoint = ROOT / 'scripts/evaluation' / {
        'original': 'memeye_m2a.py', 'repaired': 'm2a_evidence_run.py',
        'single_pass': 'm2a_single_pass_run.py'}[plan['configuration'].get('runner', 'original')]

    def verify_code():
        for p, expected in plan['implementation_sha256'].items():
            if sha(p) != expected:
                raise ValueError('Queued implementation changed: ' + p)

    def save():
        done = [x for x in state['tasks'] if x['status'] == 'passed']
        state.update(updated_at=now(), elapsed_seconds=time.time()-started,
            completed_tasks=len(done), expected_tasks=len(plan['datasets'][dataset]['tasks']),
            completed_questions=sum(x['questions'] for x in done),
            expected_questions=plan['datasets'][dataset]['questions'],
            correct=sum(x['correct'] for x in done))
        write(directory / 'status.json', state)

    save()

    def execute(command, log_path):
        with log_path.open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['current']['pid'] = child.pid
            save()
            try:
                while child.poll() is None:
                    progress = Path(state['current']['output']) / 'progress.json'
                    if progress.exists():
                        state['current']['progress'] = read(progress)
                    save()
                    time.sleep(5)
            except BaseException:
                child.terminate()
                child.wait()
                raise
        return child.returncode

    def independent_audit(attempt, filename='audit.log'):
        with (attempt / filename).open('w') as log:
            return subprocess.call([sys.executable, str(ROOT / 'scripts/evaluation/audit_memeye_m2a.py'),
                                    str(attempt)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    try:
        verify_code()
        if plan['configuration'].get('smoke_first'):
            from scripts.evaluation.m2a_inputs import load_task
            task = plan['datasets'][dataset]['tasks'][0]
            _, questions, _ = load_task(dataset, task['task'])
            question = next((q for q in questions if q['query_image_paths']), questions[0])
            target = directory / 'smoke'
            target.mkdir(exist_ok=True)
            attempts = sorted(target.glob('attempt_*'))
            accepted = None
            for attempt in attempts:
                if (attempt/'completion_manifest.json').exists() and independent_audit(attempt, 'reaudit.log') == 0:
                    accepted = attempt
                    break
            if accepted is None:
                attempt = target / f'attempt_{len(attempts)+1:03d}'
                command = [sys.executable, '-u', str(entrypoint), '--dataset', dataset, '--task', task['task'],
                    '--mandatory-query', '--device', device, '--output', str(attempt),
                    '--limit-rounds', '4', '--question-id', question['id']]
                state['current'] = dict(phase='smoke', task=task['task'], output=str(attempt), command=command)
                rc = execute(command, target/f'{attempt.name}.log')
                if rc == 0:
                    rc = independent_audit(attempt)
                    if rc == 0 and read(attempt/'summary.json')['semantic_memories'] == 0:
                        rc = 1
                if rc != 0:
                    state.update(status='stopped_on_smoke_failure', smoke=dict(status='failed', output=str(attempt), exit_code=rc))
                    save()
                    return False
                accepted = attempt
            state['smoke'] = dict(status='passed', output=str(accepted), excluded_from_full_metrics=True)
            state.pop('current', None)
            save()
        for task in plan['datasets'][dataset]['tasks']:
            verify_code()
            if sha(task['source']) != task['source_sha256']:
                raise ValueError('Queued data changed: ' + task['source'])
            target = directory / task['task']
            target.mkdir(exist_ok=True)
            attempts = sorted(target.glob('attempt_*'))
            accepted = None
            for attempt in attempts:
                if (attempt / 'completion_manifest.json').exists():
                    with (attempt / 'reaudit.log').open('w') as log:
                        rc = subprocess.call([sys.executable, str(ROOT / 'scripts/evaluation/audit_memeye_m2a.py'),
                                              str(attempt)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    if rc == 0 and read(attempt / 'protocol.json')['is_complete_task']:
                        accepted = attempt
                        break
            if accepted is None:
                attempt = target / f'attempt_{len(attempts)+1:03d}'
                command = [sys.executable, '-u', str(entrypoint),
                    '--dataset', dataset, '--task', task['task'], '--mandatory-query',
                    '--device', device, '--output', str(attempt)]
                if task['memory_source']:
                    command += ['--memory-source', task['memory_source']]
                state['current'] = dict(phase='full_task', task=task['task'], output=str(attempt), command=command)
                save()
                rc = execute(command, target / f'{attempt.name}.log')
                if rc == 0:
                    with (attempt / 'audit.log').open('w') as log:
                        rc = subprocess.call([sys.executable, str(ROOT / 'scripts/evaluation/audit_memeye_m2a.py'),
                                              str(attempt)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    if rc == 0:
                        accepted = attempt
                if accepted is None:
                    state['tasks'].append(dict(task=task['task'], status='failed', exit_code=rc, output=str(attempt)))
                    if plan['configuration'].get('stop_on_failure'):
                        state['status'] = 'stopped_on_failure'
                        save()
                        return False
                    save()
                    continue
            summary = read(accepted / 'summary.json')
            assert summary['completed_questions'] == task['questions'] and summary['is_complete_task']
            state['tasks'].append(dict(task=task['task'], status='passed', output=str(accepted),
                questions=summary['completed_questions'], correct=summary['correct'],
                metrics=summary.get('metrics'), completion_sha256=sha(accepted / 'completion_manifest.json')))
            save()
        state.pop('current', None)
        state['status'] = 'completed' if all(x['status'] == 'passed' for x in state['tasks']) else 'completed_with_failures'
        save()
        if state['status'] == 'completed':
            result = dict(state, accuracy=state['correct'] / state['completed_questions'],
                          plan_sha256=sha(root / 'plan.json'))
            if dataset == 'memgallery':
                result['metrics'] = {k: sum(t['metrics'][k]*t['questions'] for t in state['tasks']) /
                    state['completed_questions'] for k in ('em','f1','bleu','bleu_1','bleu_2')}
            write(directory / 'completion_manifest.json', result)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['memeye','memgallery'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--runner', choices=['original','repaired','single_pass'], default='original')
    parser.add_argument('--smoke-first', action='store_true')
    parser.add_argument('--stop-on-failure', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.stage == 'prepare':
        p = prepare(root, args.runner, args.smoke_first, args.stop_on_failure)
        print(json.dumps({k: {n:v for n,v in d.items() if n != 'tasks'} for k,d in p['datasets'].items()}))
    else:
        if not args.dataset:
            parser.error('--dataset is required for run')
        if run(root, args.dataset, args.device) is False:
            raise SystemExit(1)
