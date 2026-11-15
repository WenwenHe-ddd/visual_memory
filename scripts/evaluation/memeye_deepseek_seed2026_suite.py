"""Run seven fresh MemEye methods with at most two complete methods in flight.

Each slot includes native memory construction, retrieval, MCQ, Open answering,
Judge(seed42), and its evidence audit. All frozen inputs stay read-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis import report_memeye_deepseek_seed2026 as report

RUN = report.DEFAULT
MAX_METHODS = 2
# Start a text method alongside the common caption build, rather than occupying
# both slots with methods that wait on the same construction lock.
ORDER = ('semrag', 'clip_image_dialogue', 'gen_agents', 'ours_1_ours_2', 'memoryos', 'simplemem', 'a_mem')
CAPTION = ROOT / 'scripts/evaluation/memeye_deepseek_seed2026_caption.py'
TEXT = ROOT / 'scripts/evaluation/memeye_deepseek_seed2026_text_methods.py'
VISUAL = ROOT / 'scripts/evaluation/memeye_deepseek_seed2026_visual.py'
TRANSPORT = ROOT / 'scripts/evaluation/memeye_deepseek_seed2026_transport.py'


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    report.dump(path, value)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def event(output, kind, **values):
    record = dict(time=utc(), event=kind, **values)
    path = output / 'scheduler/events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(record, ensure_ascii=False), flush=True)


def command(output, method, device, workers):
    driver = CAPTION if method in ('semrag', 'ours_1_ours_2') else VISUAL if method == 'clip_image_dialogue' else TEXT
    args = [sys.executable, '-u', str(driver), '--phase', 'run', '--run-root', str(output),
            '--method', method, '--workers', str(workers)]
    if method != 'clip_image_dialogue':
        args += ['--device', 'cpu' if method == 'simplemem' else device]
    return args


def driver_ready(output, method):
    driver = CAPTION if method in ('semrag', 'ours_1_ours_2') else VISUAL if method == 'clip_image_dialogue' else TEXT
    path = output / 'scheduler/ready' / (driver.stem + '.json')
    if not path.exists():
        return False
    seal = report.read(path)
    report.require(seal['status'] == 'ready' and seal['path'] == str(driver), 'Driver ready seal differs')
    report.require(report.sha(driver) == seal['sha256'], 'Reviewed driver changed: ' + str(driver))
    return seal


def plan(output, device, workers):
    """Pin code and reference cohort before the first paid request."""
    # Drivers are independently sealed only after their CPU preflight is complete.
    # This allows the first two reviewed methods to run while another is prepared.
    code = [Path(__file__), Path(report.__file__), TRANSPORT,
            ROOT / 'scripts/evaluation/deepseek_flash_seed42_audited_transport.py',
            ROOT / 'scripts/evaluation/memeye_deepseek_open_judge.py',
            ROOT / 'scripts/evaluation/memeye_open_judge_engine.py']
    code += sorted((ROOT / 'scripts/evaluation').glob('memeye_deepseek_seed1024_*.py'))
    code += [ROOT / 'scripts/memory/memeye_backbone_retrieval.py', report.BENCHMARK, report.OPEN_GOLD, report.REFERENCE]
    sources = {str(p): report.sha(p) for p in code}
    value = dict(dataset='MemEye', model='deepseek-v4.1-flash', seed=2026, caption_seed=42, judge_seed=42,
        questions=371, methods=list(ORDER), maximum_active_methods=MAX_METHODS, api_workers_per_method=workers,
        device=device, lambda_value=.03, alpha=1., source_sha256=sources,
        caption_policy='Freeze original seed42 captions; rebuild new memories and retrieval.',
        visrag_policy='Original CLIP vectors + original images and dialogue; no caption conversion.',
        method_slot='Construction -> retrieval -> MCQ -> Open -> Judge42 -> method audit.',
        output_root=str(output),
        driver_policy='Require reviewed SHA in scheduler/ready and freeze it per method before launch.',
        authorization='User explicitly requested MemEye DeepSeek Flash seed2026 experiment, maximum two methods concurrently.',
        commands={m: command(output, m, device, workers) for m in ORDER})
    destination = output / 'plan.json'
    if destination.exists():
        report.require(report.read(destination) == value, 'Frozen seed2026 plan differs')
    else:
        dump(destination, value)
    return value


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def state(output, methods, running, started):
    snapshot = dict(status='running' if running else 'prepared', seed=2026, model='deepseek-v4.1-flash',
        maximum_active_methods=MAX_METHODS, active_methods=list(running), methods=methods,
        started_at=started, updated_at=utc(), supervisor_pid=os.getpid())
    dump(output / 'progress.json', snapshot)
    return snapshot


def run(output, device, workers, retry_failed=False):
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / 'scheduler/suite.lock'):
        contract = plan(output, device, workers)
        report.hashes(contract['source_sha256'])
        canonical = report.indexed(report.rows(report.BENCHMARK))
        old_progress = report.read(output / 'progress.json') if (output / 'progress.json').exists() else {}
        events = report.rows(output / 'scheduler/events.jsonl') if (output / 'scheduler/events.jsonl').exists() else []
        event_active = {}
        for item in events:
            if item['event'] == 'method_started':
                event_active[item['method']] = item.get('pid')
            elif item['event'] in ('method_finished', 'method_interrupted'):
                event_active.pop(item['method'], None)
        # Reconcile the durable event log, including a crash between event and progress.
        for method, pid in event_active.items():
            report.require(not pid or not alive(pid), 'Previous method process still alive; refusing to exceed two slots')
            event(output, 'method_interrupted', method=method, pid=pid)
        for method in old_progress.get('active_methods', []):
            pid = old_progress['methods'][method].get('pid')
            report.require(not pid or not alive(pid), 'Previous method process still alive; refusing to exceed two slots')
        methods = old_progress.get('methods', {m: {'status': 'pending'} for m in ORDER})
        for method in ORDER:
            checked = report.audit_method(output, method, canonical)
            if checked['status'] == 'complete':
                methods[method] = dict(status='complete', audit=checked)
            elif methods[method]['status'] != 'failed' or retry_failed:
                methods[method] = {'status': 'pending'}
        started = old_progress.get('started_at', utc())
        pending = [m for m in ORDER if methods[m]['status'] == 'pending']
        running = {}
        stopping = False

        def stop(signum, frame):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        event(output, 'supervisor_started', pid=os.getpid(), maximum_active_methods=MAX_METHODS)
        try:
            while pending or running:
                while pending and len(running) < MAX_METHODS and not stopping:
                    ready_methods = [m for m in pending if driver_ready(output, m)]
                    if not ready_methods:
                        break
                    method = ready_methods[0]
                    pending.remove(method)
                    report.hashes(contract['source_sha256'])
                    driver = driver_ready(output, method)
                    driver_path = output / 'scheduler/method_drivers' / (method + '.json')
                    if driver_path.exists():
                        report.require(report.read(driver_path) == driver, 'Pinned method driver differs')
                    else:
                        dump(driver_path, driver)
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    log_path = output / 'logs' / f'{method}-{stamp}.log'
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    stream = log_path.open('ab', buffering=0)
                    cmd = contract['commands'][method]
                    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
                    process = subprocess.Popen(cmd, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    running[method] = (process, stream)
                    methods[method] = dict(status='running', pid=process.pid, started_at=utc(), command=cmd, log=str(log_path))
                    event(output, 'method_started', method=method, pid=process.pid, active_methods=list(running), log=str(log_path))
                    state(output, methods, running, started)
                for method, (process, stream) in list(running.items()):
                    exit_code = process.poll()
                    if exit_code is None:
                        continue
                    stream.close()
                    del running[method]
                    methods[method].update(exit_code=exit_code, ended_at=utc())
                    try:
                        report.require(exit_code == 0, f'Method exited {exit_code}; inspect native log')
                        report.hashes(contract['source_sha256'])
                        checked = report.audit_method(output, method, canonical)
                        report.require(checked['status'] == 'complete', 'Process exited without complete371 MCQ/Open/Judge evidence')
                        methods[method].update(status='complete', audit=checked)
                    except Exception as error:
                        methods[method].update(status='failed', error=f'{type(error).__name__}: {error}')
                    event(output, 'method_finished', method=method, pid=process.pid, exit_code=exit_code,
                          status=methods[method]['status'], active_methods=list(running), error=methods[method].get('error'))
                    state(output, methods, running, started)
                    # A failed evidence check must remain visible, never become a zero score.
                    try:
                        report.export(output)
                    except Exception as error:
                        dump(output / 'audit/report_error.json', dict(time=utc(), error=f'{type(error).__name__}: {error}'))
                if stopping:
                    for method, (process, stream) in running.items():
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                    break
                if running or pending:
                    state(output, methods, running, started)
                    time.sleep(5)
        finally:
            for method, (process, stream) in running.items():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                stream.close()
                methods[method].update(status='interrupted', ended_at=utc(), exit_code=process.returncode)
                event(output, 'method_interrupted', method=method, pid=process.pid)
            running.clear()
            snapshot = state(output, methods, running, started)
            snapshot['status'] = 'interrupted' if stopping else 'complete' if all(
                r['status'] == 'complete' for r in methods.values()) else 'incomplete'
            dump(output / 'progress.json', snapshot)
        event(output, 'supervisor_finished', status=snapshot['status'])
        if snapshot['status'] == 'complete':
            result = report.export(output, require_complete=True)
            evidence = [output / 'comparison.json', output / 'comparison.csv', output / 'audit/independent_results.json',
                        output / 'audit/paired_mcq_vs42.json', output / 'plan.json', output / 'scheduler/events.jsonl']
            evidence += list((output / 'scheduler/method_drivers').glob('*.json'))
            drivers = [report.read(p) for p in (output / 'scheduler/method_drivers').glob('*.json')]
            dump(output / 'completion_manifest.json', dict(status='complete', seed=2026, caption_seed=42, judge_seed=42,
                model='deepseek-v4.1-flash', methods=7, mcq_records=2597, open_records=2597, judge_records=2597,
                maximum_active_methods=result['concurrency']['maximum_active_methods'],
                source_sha256={**contract['source_sha256'], **{r['path']: r['sha256'] for r in drivers}},
                artifact_sha256={str(p): report.sha(p) for p in evidence},
                completed_at=utc()))
        return 0 if snapshot['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'status', 'audit'), default='status')
    parser.add_argument('--run-root', type=Path, default=RUN)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    output = args.run_root.resolve()
    if args.phase == 'prepare':
        value = plan(output, args.device, args.workers)
        print(json.dumps(dict(status='prepared', methods=value['methods'], maximum_active_methods=MAX_METHODS)))
    elif args.phase == 'run':
        raise SystemExit(run(output, args.device, args.workers, args.retry_failed))
    elif args.phase == 'audit':
        print(json.dumps(report.export(output), ensure_ascii=False))
    else:
        value = report.read(output / 'progress.json') if (output / 'progress.json').exists() else {'status': 'not_started'}
        print(json.dumps(value, ensure_ascii=False, indent=2))
