"""Run prepared local attribution while only the suite dispatcher is held.

The independent SimpleMem Open retrieval child continues.  The dispatcher
cannot start a duplicate attribution process and later adopts its native
stage marker.  Run under systemd KillMode=control-group with an ExecStopPost
SIGCONT of the validated dispatcher PID, as for the MCQ parallel helper.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.evaluation import memeye_deepseek_seed1024_suite as suite

STAGE = 'simplemem_attribution_run'
PREPARE = 'simplemem_attribution_prepare'
MCQ = 'simplemem_mcq'
HOLD = 'simplemem_open_retrieve'


def prerequisites(stages):
    if not suite.marker(stages[MCQ]):
        raise ValueError('Complete SimpleMem MCQ source is required')
    completed = suite.read(stages[MCQ]['marker'])
    if (completed.get('questions') != 371 or completed.get('passed') is not True
            or completed.get('seed') != 1024 or completed.get('model') != 'deepseek-v4.1-flash'):
        raise ValueError('SimpleMem MCQ must be audited complete for all 371 questions')
    if not suite.marker(stages[PREPARE]):
        raise ValueError('SimpleMem attribution inputs are not prepared')
    protocol_path = Path(stages[PREPARE]['marker']).parent / 'protocol.json'
    protocol = suite.read(protocol_path)
    if (protocol.get('seed') != 42 or protocol.get('memory_generation_seed') != 1024
            or protocol.get('max_new_tokens') != 4096 or protocol.get('generation_batch_size') != 8
            or protocol.get('reader_device_map') != {'': 1} or protocol.get('gold_blind') is not True):
        raise ValueError('Prepared local attribution evaluator differs from fixed Qwen seed42/cuda1/4096/batch8')
    return protocol_path


def main():
    plan = suite.read(suite.OUT / 'plan.json')
    suite.verify(plan)
    if not suite.export_allowed():
        raise PermissionError('Explicit export authorization required')
    stages = {s['name']: s for s in plan['stages']}
    stage = stages[STAGE]
    if stage['kind'] != 'local_after_api':
        raise ValueError('Attribution must retain its original local stage kind')
    if suite.marker(stage):
        return
    attribution_protocol = prerequisites(stages)
    current = suite.read(suite.OUT / 'progress.json')
    if current.get('status') != 'running' or current.get('current_stage') != HOLD:
        raise ValueError('Independent SimpleMem Open retrieval must be the current stage')
    pid = current['pid']
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if command[1:5] != [b'-m', b'scripts.evaluation.memeye_deepseek_seed1024_suite', b'--phase', b'run']:
        raise ValueError('Dispatcher identity differs')
    if Path(f'/proc/{pid}/cwd').resolve() != suite.ROOT:
        raise ValueError('Dispatcher working directory differs')
    work = suite.OUT / 'parallel'
    work.mkdir(exist_ok=True)
    path = work / (STAGE + '.json')
    with (work / (STAGE + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='starting', stage=STAGE, dispatcher_pid=pid,
                     active_independent_stage=HOLD,
                     active_independent_child=current['stages'][HOLD]['child_pid'],
                     helper_sha256=suite.sha(Path(__file__)), plan_sha256=suite.sha(suite.OUT / 'plan.json'),
                     attribution_protocol_sha256=suite.sha(attribution_protocol), started_at_utc=suite.now())
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
            suite.verify(plan)
            if not suite.export_allowed():
                raise PermissionError('Export authorization changed')
            prerequisites(stages)
            confirmed = suite.read(suite.OUT / 'progress.json')
            if (confirmed.get('status') != 'running' or confirmed.get('current_stage') != HOLD
                    or confirmed.get('pid') != pid):
                raise ValueError('Dispatcher advanced or changed before hold')
            argv = suite.resolve(stage['argv'])
            state['argv'] = argv
            with (work / (STAGE + '.log')).open('ab') as log:
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
