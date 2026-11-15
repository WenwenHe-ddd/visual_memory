"""Run the next independent MCQ while only the suite dispatcher is held.

The active A-Mem child keeps running. The held dispatcher cannot launch a
second SimpleMem process; after completion it adopts the native marker.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.evaluation import memeye_deepseek_seed1024_suite as suite


def main():
    plan = suite.read(suite.OUT / 'plan.json')
    suite.verify(plan)
    if not suite.export_allowed():
        raise PermissionError('Explicit export authorization required')
    stages = {s['name']: s for s in plan['stages']}
    stage = stages['simplemem_mcq']
    if suite.marker(stage):
        return
    if not suite.marker(stages['simplemem_prepare']):
        raise ValueError('SimpleMem inputs not prepared')
    current = suite.read(suite.OUT / 'progress.json')
    if current.get('status') != 'running' or current.get('current_stage') != 'a_mem_mcq':
        raise ValueError('Independent A-Mem child must be the current stage')
    pid = current['pid']
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if command[1:5] != [b'-m', b'scripts.evaluation.memeye_deepseek_seed1024_suite', b'--phase', b'run']:
        raise ValueError('Dispatcher identity differs')
    if Path(f'/proc/{pid}/cwd').resolve() != suite.ROOT:
        raise ValueError('Dispatcher working directory differs')
    work = suite.OUT / 'parallel'
    work.mkdir(exist_ok=True)
    path = work / 'simplemem_mcq.json'
    with (work / 'simplemem_mcq.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='starting', dispatcher_pid=pid,
                     active_independent_child=current['stages']['a_mem_mcq']['child_pid'],
                     helper_sha256=suite.sha(Path(__file__)), started_at_utc=suite.now())
        suite.dump(path, state)
        held = False
        child = None
        try:
            os.kill(pid, signal.SIGSTOP)
            held = True
            for _ in range(50):
                status = Path(f'/proc/{pid}/status').read_text()
                if next(s for s in status.splitlines() if s.startswith('State:')).split()[1] == 'T':
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('Dispatcher did not enter stopped state')
            state.update(status='running', dispatcher_held=True)
            suite.dump(path, state)
            # Recheck after holding, before the only network-capable child.
            suite.verify(plan)
            if not suite.export_allowed():
                raise PermissionError('Export authorization changed')
            if not suite.marker(stages['simplemem_prepare']):
                raise ValueError('SimpleMem preparation changed before launch')
            if suite.read(suite.OUT / 'progress.json').get('current_stage') != 'a_mem_mcq':
                raise ValueError('Dispatcher advanced before hold')
            argv = suite.resolve(stage['argv'])
            state['argv'] = argv
            with (work / 'simplemem_mcq.log').open('ab') as log:
                child = subprocess.Popen(argv, cwd=suite.ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                suite.dump(path, state)
                code = child.wait()
            okay = code == 0 and suite.marker(stage)
            state.update(status='complete' if okay else 'failed', returncode=code,
                         finished_at_utc=suite.now())
            if okay:
                state['marker_sha256'] = suite.sha(stage['marker'])
            suite.dump(path, state)
            if not okay:
                raise SystemExit(2)
        except BaseException:
            if state['status'] not in ('complete', 'failed'):
                state.update(status='failed', finished_at_utc=suite.now())
            raise
        finally:
            safe_to_resume = child is None or child.poll() is not None
            if not safe_to_resume:
                try:
                    try:
                        child.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        try:
                            child.kill()
                        except ProcessLookupError:
                            pass
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            state['requires_external_child_cleanup'] = True
                except BaseException as error:
                    state.update(status='failed', cleanup_error=type(error).__name__ + ': ' + str(error),
                                 requires_external_child_cleanup=True)
                safe_to_resume = child.poll() is not None
            if held and safe_to_resume:
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    state['dispatcher_exited'] = True
                state['dispatcher_resumed_at_utc'] = suite.now()
            state['dispatcher_held'] = held and not safe_to_resume
            suite.dump(path, state)


if __name__ == '__main__':
    main()
