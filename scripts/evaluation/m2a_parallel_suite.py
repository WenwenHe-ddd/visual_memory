"""Complete a dataset suite on two GPUs, reusing only audited complete tasks.

The inference protocol is unchanged. The source run's already valid empty
no-image captions satisfy the new deterministic caption-presence normalization.
"""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.m2a_suite import prepare as prepare_plan, read, write, sha, now
from scripts.evaluation.audit_memeye_m2a import audit


def prepare(root, source):
    if (root/'plan.json').exists():
        raise ValueError('Use a fresh aggregate root')
    plan = prepare_plan(root, runner='single_pass')
    prior = read(source/'memeye/status.json')
    inherited = {}
    for item in prior['tasks']:
        if item['status'] != 'passed':
            continue
        out = Path(item['output'])
        assert sha(out/'completion_manifest.json') == item['completion_sha256']
        checked = audit(out)
        assert checked['passed'] and checked['is_complete_task'], checked
        inherited[item['task']] = item
    plan['completed_sources'] = inherited
    plan['normalization_change'] = ('For rounds without an image, discard unsupported image_caption '
        'and log it. No extra model calls. Existing passed tasks already have empty captions on all '
        'no-image rounds and were re-audited under this same effective rule.')
    plan['configuration']['workers'] = 'Two local GPU subprocesses; longest unfinished tasks first'
    plan['configuration']['physical_gpus'] = [0, 1]
    plan['configuration']['failure_policy'] = 'Retain failed attempt; finish other tasks; never count failures as completed'
    for path in [Path(__file__), ROOT/'scripts/evaluation/audit_m2a_full_memeye.py']:
        dest = root/'code_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        plan['implementation_sha256'][str(path)] = sha(path)
    write(root/'plan.json', plan)
    return plan


def run(root):
    import fcntl
    plan = read(root/'plan.json')
    dataset = plan.get('dataset', 'memeye')
    directory = root/dataset
    directory.mkdir(exist_ok=True)
    lock = (directory/'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = plan['datasets'][dataset]['tasks']
    done = dict(plan.get('completed_sources', {}))
    pending = sorted([t for t in tasks if t['task'] not in done], key=lambda t: -t['rounds'])
    active, failures = {}, []
    started = time.time()
    state = dict(pid=os.getpid(), started_at=now(), status='running')

    def save():
        state.update(updated_at=now(), elapsed_seconds=time.time()-started,
            tasks=[done[t['task']] for t in tasks if t['task'] in done] + failures,
            active=[{k:v for k,v in item.items() if k not in ('process','log')} for item in active.values()],
            completed_tasks=len(done), expected_tasks=len(tasks),
            completed_questions=sum(t['questions'] for t in done.values()),
            expected_questions=plan['datasets'][dataset]['questions'],
            correct=sum(t['correct'] for t in done.values()))
        write(directory/'status.json', state)

    def verify():
        for path, expected in plan['implementation_sha256'].items():
            if sha(path) != expected:
                raise ValueError('Frozen implementation changed: ' + path)

    save()
    try:
        while pending or active:
            for gpu in (0, 1):
                if gpu in active or not pending:
                    continue
                verify()
                task = pending.pop(0)
                assert sha(task['source']) == task['source_sha256']
                target = directory/task['task']
                target.mkdir(exist_ok=True)
                attempts = sorted(p for p in target.glob('attempt_*') if p.is_dir())
                accepted = None
                for attempt in attempts:
                    if (attempt/'completion_manifest.json').exists() and audit(attempt)['passed']:
                        accepted = attempt
                        break
                if accepted is not None:
                    summary = read(accepted/'summary.json')
                    done[task['task']] = dict(task=task['task'], status='passed', output=str(accepted),
                        questions=summary['completed_questions'], correct=summary['correct'], metrics=summary.get('metrics'),
                        completion_sha256=sha(accepted/'completion_manifest.json'))
                    continue
                attempt = target/f'attempt_{len(attempts)+1:03d}'
                cmd = [sys.executable,'-u',str(ROOT/'scripts/evaluation/m2a_single_pass_run.py'),
                    '--dataset',dataset,'--task',task['task'],'--mandatory-query',
                    '--device','cuda:0','--output',str(attempt)]
                cmd += ['--extraction-attempts', str(plan['configuration'].get('extraction_max_attempts', 1))]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
                log = (target/f'{attempt.name}.log').open('w')
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = dict(task=task['task'], output=str(attempt), gpu=gpu,
                    pid=proc.pid, process=proc, log=log, command=cmd)
                save()
            for gpu, item in list(active.items()):
                out, proc = Path(item['output']), item['process']
                if (out/'progress.json').exists():
                    item['progress'] = read(out/'progress.json')
                if (out/'failure.json').exists() and proc.poll() is None:
                    # A recorded terminal failure can otherwise stall in model/library teardown.
                    proc.terminate()
                rc = proc.poll()
                if rc is None:
                    continue
                item['log'].close()
                if (out/'completion_manifest.json').exists():
                    checked = audit(out)
                    if checked['passed'] and checked['is_complete_task']:
                        summary = read(out/'summary.json')
                        done[item['task']] = dict(task=item['task'], status='passed', output=str(out),
                            questions=summary['completed_questions'], correct=summary['correct'], metrics=summary.get('metrics'),
                            completion_sha256=sha(out/'completion_manifest.json'))
                    else:
                        failures.append(dict(task=item['task'],status='failed',output=str(out),audit=checked))
                else:
                    failures.append(dict(task=item['task'],status='failed',output=str(out),exit_code=rc))
                del active[gpu]
            save()
            if pending or active:
                time.sleep(5)
        state['status'] = 'completed' if len(done) == len(tasks) else 'completed_with_failures'
        save()
        if state['status'] == 'completed':
            if dataset == 'memgallery':
                if plan.get('scope') == 'retry_subset':
                    state['scope'] = 'retry_subset; not a complete benchmark result'
                    state['metrics'] = {k:sum(x['metrics'][k]*x['questions'] for x in done.values()) /
                        state['completed_questions'] for k in ('em','f1','bleu','bleu_1','bleu_2')}
                else:
                    from scripts.evaluation.audit_m2a_full_memgallery import audit_full
                    checked = audit_full(root)
                    if not checked['passed']:
                        state['status'] = 'audit_failed'
                        save()
                        return False
                    state['metrics'] = checked['metrics']
            write(directory/'completion_manifest.json',dict(state,
                accuracy=state['correct']/state['completed_questions'],plan_sha256=sha(root/'plan.json')))
        return state['status'] == 'completed'
    except BaseException as exc:
        for item in active.values():
            item['process'].terminate()
        state.update(status='failed',error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--dataset',choices=['memeye','memgallery'],default='memeye')
    args=parser.parse_args()
    if args.stage == 'prepare':
        if args.dataset == 'memgallery':
            root = args.output.resolve()
            if (root/'plan.json').exists():
                parser.error('Use a fresh aggregate root')
            plan = prepare_plan(root, runner='single_pass', datasets_to_run=('memgallery',))
            plan.update(dataset='memgallery', completed_sources={})
            plan['configuration'].update(workers='Two local GPU subprocesses; longest unfinished tasks first',
                                         physical_gpus=[0, 1])
            for path in [Path(__file__), ROOT/'scripts/evaluation/audit_m2a_full_memgallery.py']:
                dest = root/'code_snapshot'/path.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(path.read_bytes())
                plan['implementation_sha256'][str(path)] = sha(path)
            write(root/'plan.json', plan)
        elif args.source is None:
            parser.error('--source is required for prepare')
        else:
            prepare(args.output.resolve(),args.source.resolve())
    else:
        raise SystemExit(0 if run(args.output.resolve()) else 1)
