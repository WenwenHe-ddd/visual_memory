"""Resume the unfinished HippoMM stage without editing its signed pipeline.

The independent rewrite adapter applies the same disclosed formatting recovery
as the seed-42 reference. This wrapper archives the interrupted queue state,
then invokes the original visual, generation, Judge and report entry points.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'results/memgallery_qwen_seed1042_full_20261106'
REWRITE = ROOT / 'scripts/evaluation/memgallery_seed1042_hippomm_recovery.py'
VISUAL = ROOT / 'scripts/evaluation/memgallery_seed1042_visual.py'
SUITE = ROOT / 'scripts/evaluation/memgallery_seed1042_suite.py'
EVALUATE = ROOT / 'scripts/evaluation/memgallery_seed1042_evaluation_queue.py'
REPORT = ROOT / 'scripts/analysis/report_memgallery_seed1042_stability.py'
PYTHON = '/usr/bin/python3'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        if read(path) != value:
            raise ValueError('Recovery contract changed: ' + str(path))
        return
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def directory(run):
    return run / 'recovery/hippomm_completion_20261113'


def fixed_artifacts(run):
    """The eight completed methods stay byte-for-byte unchanged."""
    plan = read(run / 'plan.json')
    paths = []
    for method in plan['methods']:
        mid = method['id']
        if mid == 'hippomm':
            continue
        metric_path = run / 'evaluations' / mid / 'metrics.json'
        metric = read(metric_path)
        if (metric.get('status') != 'complete' or metric.get('seed') != 1042
                or metric.get('questions') != 1711):
            raise ValueError('An earlier completed method is no longer complete: ' + mid)
        paths += [metric_path, metric_path.with_name('records.jsonl')]
        if mid in ('semrag', 'ours_1', 'ours_1_ours_2'):
            paths.append(run / 'caption_methods/stages/audit.json')
        elif mid == 'clip_image_dialogue':
            paths.append(run / 'visual_methods' / mid / 'completion.json')
        else:
            paths.append(run / 'text_methods' / mid / 'completion_manifest.json')
        judge = run / 'judge' / mid
        if read(judge / 'summary.json').get('status') != 'complete':
            raise ValueError('An earlier method Judge is incomplete: ' + mid)
        for name in ('protocol.json', 'summary.json', 'requests.jsonl', 'question_map.jsonl',
                     'scores.jsonl', 'reused_scores.jsonl', 'attempts.jsonl',
                     'methods/' + mid + '/summary.json', 'methods/' + mid + '/per_question.jsonl'):
            path = judge / name
            if path.is_file():
                paths.append(path)
    return sorted(set(paths))


def verify_fixed(run, plan):
    for name, expected in plan['source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Pinned recovery or experiment source changed: ' + name)
    for name, expected in plan['completed_eight_sha256'].items():
        if sha(name) != expected:
            raise ValueError('A previously completed method artifact changed: ' + name)
    answer = Path(plan['baseline_prefix']['path'])
    with answer.open('rb') as stream:
        prefix = stream.read(plan['baseline_prefix']['bytes'])
    if hashlib.sha256(prefix).hexdigest() != plan['baseline_prefix']['sha256']:
        raise ValueError('The 1699 original baseline answers changed')


def prepare(run):
    target = directory(run)
    plan_path = target / 'resume_plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        verify_fixed(run, plan)
        return plan
    if not REWRITE.is_file():
        raise FileNotFoundError('The reviewed rewrite recovery driver is not ready: ' + str(REWRITE))
    gpu = read(run / 'plan.json')
    evaluation = read(run / 'evaluation_plan.json')
    source_hashes = dict(gpu['source_sha256'])
    for name, expected in evaluation['source_sha256'].items():
        if name in source_hashes and source_hashes[name] != expected:
            raise ValueError('Generation and evaluation plans disagree on ' + name)
        source_hashes[name] = expected
    for path in (Path(__file__).resolve(), REWRITE, VISUAL, SUITE, EVALUATE, REPORT,
                 run / 'plan.json', run / 'evaluation_plan.json'):
        expected = sha(path)
        if str(path) in source_hashes and source_hashes[str(path)] != expected:
            raise ValueError('Original signed source changed before recovery: ' + str(path))
        source_hashes[str(path)] = expected
    baseline = run / 'visual_methods/hippomm/baseline/mem_gallery/online/qa_responses.jsonl'
    original = rows(baseline)
    if len(original) != 1699 or len({row['id'] for row in original}) != 1699:
        raise ValueError('First recovery preparation requires exactly 1699 original baseline rows')
    archived = {}
    for name in ('progress.json', 'generation_completion.json', 'evaluation_progress.json',
                 'comparison.json', 'comparison.csv', 'plan.json', 'evaluation_plan.json'):
        path = run / name
        if path.is_file():
            copied = target / 'before' / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copied)
            if sha(path) != sha(copied):
                raise ValueError('Interrupted state changed during its archival copy: ' + name)
            archived[str(copied)] = sha(copied)
    plan = {'status': 'frozen_recovery_plan', 'created_at_utc': now(),
            'seed': 1042, 'reference_seed': 42, 'judge_seed': 42,
            'physical_gpu': 0, 'CUDA_VISIBLE_DEVICES': '0', 'logical_device': 'cuda:0',
            'source_sha256': source_hashes,
            'completed_eight_sha256': {str(path): sha(path) for path in fixed_artifacts(run)},
            'archived_state_sha256': archived,
            'baseline_prefix': {'path': str(baseline), 'rows': 1699,
                                'bytes': baseline.stat().st_size, 'sha256': sha(baseline)},
            'policy': 'Append only the 12 missing baseline answers using the seed42 rewrite-format recovery; invoke the original signed native pipeline, generation queue, Judge and report.'}
    verify_fixed(run, plan)
    write(plan_path, plan, frozen=True)
    return plan


def marker(path, key='status', value='complete'):
    return path.is_file() and read(path).get(key) == value


def commands(run):
    recovery = run / 'recovery/hippomm_20261113'
    common = [PYTHON, '-u', str(REWRITE), '--run-root', str(run), '--device', 'cuda:0']
    return [
        ('rewrite-prepare', common + ['--phase', 'prepare'], recovery / 'rewrite_prepare.json', 'status', 'complete'),
        ('rewrite-run', common + ['--phase', 'run'], recovery / 'recovery_completion.json', 'status', 'complete'),
        # Always invoke the independent adapter audit, including after a restart.
        ('rewrite-audit', common + ['--phase', 'audit'], None, None, None),
        ('native-controlled-answer-and-export',
         [PYTHON, '-u', '-m', 'scripts.evaluation.memgallery_seed1042_visual',
          '--method', 'hippomm', '--phase', 'run', '--run-root', str(run), '--device', 'cuda:0'],
         run / 'visual_methods/hippomm/completion.json', 'status', 'complete'),
        ('generation-seal', [PYTHON, '-u', str(SUITE), '--phase', 'run'],
         run / 'generation_completion.json', 'status', 'generation_complete'),
        ('judge-and-workbook', [PYTHON, '-u', str(EVALUATE), '--phase', 'run'],
         run / 'completion.json', 'status', 'complete')]


def audit(run, plan):
    verify_fixed(run, plan)
    for name, expected in plan['archived_state_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Archived interrupted state changed: ' + name)
    paths = [run / 'visual_methods/hippomm/completion.json', run / 'evaluations/hippomm/metrics.json',
             run / 'judge/hippomm/summary.json', run / 'completion.json']
    if not all(marker(path) for path in paths):
        raise ValueError('HippoMM answers, metrics, Judge and final report must all finish before completion')
    if not marker(run / 'generation_completion.json', 'status', 'generation_complete'):
        raise ValueError('The original generation queue has not sealed all nine methods')
    final = read(run / 'completion.json')
    if final.get('method_count') != 9 or final.get('paired_questions') != 9 * 1711:
        raise ValueError('Final completion does not cover all nine paired cohorts')
    for name, expected in final['source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Final completion artifact hash changed: ' + name)
    result = {'status': 'passed', 'audited_at_utc': now(), 'seed': 1042,
              'original_baseline_rows_unchanged': 1699, 'previous_complete_methods_unchanged': 8,
              'questions_per_method': 1711, 'methods_complete': 9,
              'final_completion_path': str(run / 'completion.json'),
              'final_completion_sha256': sha(run / 'completion.json'),
              'resume_plan_sha256': sha(directory(run) / 'resume_plan.json')}
    write(directory(run) / 'audit.json', result)
    return result


def run_queue(run, plan):
    target = directory(run)
    state_path = target / 'progress.json'
    state = read(state_path) if state_path.exists() else {'stages': {}}
    state.update(status='running', pid=os.getpid(), resumed_at_utc=now())
    write(state_path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    try:
        for name, argv, seal, key, value in commands(run):
            verify_fixed(run, plan)
            if seal is not None and marker(seal, key, value):
                state['stages'][name] = {'status': 'complete', 'adopted_existing': True,
                                         'marker': str(seal), 'marker_sha256': sha(seal)}
                write(state_path, state)
                continue
            log = target / 'logs' / (name + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            stage = {'status': 'running', 'argv': argv, 'log': str(log), 'started_at_utc': now()}
            state['stages'][name] = stage
            state['current_stage'] = name
            write(state_path, state)
            print(json.dumps({'stage': name, 'status': 'started', 'at_utc': now()}), flush=True)
            with log.open('ab') as stream:
                process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                stage['child_pid'] = process.pid
                write(state_path, state)
                code = process.wait()
            stage.update(returncode=code, finished_at_utc=now())
            if code or (seal is not None and not marker(seal, key, value)):
                stage['status'] = 'failed'
                raise RuntimeError(f'{name} did not complete; exit={code}; inspect {log}')
            stage['status'] = 'complete'
            if seal is not None:
                stage.update(marker=str(seal), marker_sha256=sha(seal))
            write(state_path, state)
            print(json.dumps({'stage': name, 'status': 'complete', 'at_utc': now()}), flush=True)
        proof = audit(run, plan)
        state.update(status='complete', current_stage=None, finished_at_utc=now(), audit=proof)
        write(state_path, state)
        write(target / 'completion.json', {'status': 'complete', 'completed_at_utc': now(),
              'audit_path': str(target / 'audit.json'), 'audit_sha256': sha(target / 'audit.json'),
              'final_completion_path': str(run / 'completion.json'),
              'final_completion_sha256': sha(run / 'completion.json')})
        return 0
    except BaseException as exc:
        state.update(status='failed', error=type(exc).__name__ + ': ' + str(exc), failed_at_utc=now())
        write(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'audit'), required=True)
    parser.add_argument('--run-root', type=Path, default=RUN)
    args = parser.parse_args()
    run = args.run_root.resolve()
    if run != RUN:
        parser.error('The original generation and evaluation queues require their signed default run root')
    target = directory(run)
    target.mkdir(parents=True, exist_ok=True)
    with (target / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(run)
        if args.phase == 'run':
            raise SystemExit(run_queue(run, plan))
        result = audit(run, plan) if args.phase == 'audit' else {
            'status': 'complete', 'resume_plan': str(target / 'resume_plan.json')}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
