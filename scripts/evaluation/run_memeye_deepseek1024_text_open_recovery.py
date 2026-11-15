"""Run original answer/collect commands after audited CPU manifest recovery.

The original dispatcher has already skipped these stages in its current pass.
This helper never changes its progress and refuses a concurrent dispatcher
restart or an unfinished original retrieval. Rerun the suite after all helpers
finish so it adopts the native completion markers.
"""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess

from scripts.evaluation import memeye_deepseek_seed1024_suite as suite


def guard(method, expected_pid):
    progress = suite.read(suite.OUT / 'progress.json')
    if progress.get('pid') != expected_pid:
        raise RuntimeError('Dispatcher changed; do not overlap a restarted queue')
    states = progress['stages']
    prefix = method + '_open_'
    if states[prefix + 'retrieve']['status'] != 'failed':
        raise RuntimeError('Only a finished original retrieval failure can be recovered')
    for suffix in ('answers', 'collect'):
        if states[prefix + suffix]['status'] != 'skipped_dependency':
            raise RuntimeError('Original dispatcher must already have skipped this branch')
    if (progress.get('current_stage') or '').startswith(prefix):
        raise RuntimeError('Original dispatcher is executing this branch')
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True,
                        choices=('gen_agents', 'memoryos', 'a_mem', 'simplemem'))
    parser.add_argument('--dispatcher-pid', type=int, default=2825379)
    args = parser.parse_args()
    plan = suite.read(suite.OUT / 'plan.json')
    suite.verify(plan)
    stages = {item['name']: item for item in plan['stages']}
    prefix = args.method + '_open_'
    guard(args.method, args.dispatcher_pid)
    if not suite.marker(stages[prefix + 'retrieve']):
        raise RuntimeError('Audited CPU recovery must have produced the native retrieval marker')
    work = suite.OUT / 'parallel'
    path = work / ('text_open_recovery_' + args.method + '.json')
    with (work / ('text_open_recovery_' + args.method + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='running', method=args.method, dispatcher_pid=args.dispatcher_pid,
                     started_at_utc=suite.now(), helper_sha256=suite.sha(Path(__file__)),
                     plan_sha256=suite.sha(suite.OUT / 'plan.json'), stages={})
        suite.dump(path, state)
        try:
            for suffix in ('answers', 'collect'):
                name = prefix + suffix
                stage = stages[name]
                suite.verify(plan)
                guard(args.method, args.dispatcher_pid)
                if not suite.export_allowed():
                    raise PermissionError('Explicit export authorization is required')
                if stage.get('marker') and suite.marker(stage):
                    state['stages'][name] = dict(status='complete', adopted_existing_artifact=True)
                else:
                    argv = suite.resolve(stage['argv'])
                    log = work / ('text_open_recovery_' + args.method + '.log')
                    with log.open('ab') as stream:
                        code = subprocess.call(argv, cwd=suite.ROOT, stdout=stream,
                                               stderr=subprocess.STDOUT)
                    if code != 0 or (stage.get('marker') and not suite.marker(stage)):
                        raise RuntimeError('Original command failed: ' + name + ' exit=' + str(code))
                    state['stages'][name] = dict(status='complete', argv=argv, returncode=code)
                suite.dump(path, state)
            state.update(status='complete', finished_at_utc=suite.now())
        except BaseException as error:
            state.update(status='failed', error=type(error).__name__ + ': ' + str(error),
                         finished_at_utc=suite.now())
            raise
        finally:
            suite.dump(path, state)


if __name__ == '__main__':
    main()
