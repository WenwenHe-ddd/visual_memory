"""Run one prepared independent stage with the same export and source gates.

This helper does not alter the persistent suite's state.  The suite adopts the
native completion artifact when it reaches this stage.  Native driver locks
remain responsible for excluding concurrent runs of the same method.
"""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess

from scripts.evaluation import memeye_deepseek_seed1024_suite as suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('clip_image_dialogue_mcq', 'hippomm_mcq'), required=True)
    args = parser.parse_args()
    plan = suite.read(suite.OUT / 'plan.json')
    suite.verify(plan)
    if not suite.export_allowed():
        raise PermissionError('Explicit export authorization required')
    stages = {s['name']: s for s in plan['stages']}
    stage = stages[args.stage]
    for name in stage.get('depends_on', []):
        if not suite.marker(stages[name]):
            raise ValueError('Native prerequisite is not complete: ' + name)
    work = suite.OUT / 'parallel'
    work.mkdir(exist_ok=True)
    with (work / (args.stage + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = dict(stage=args.stage, helper_sha256=suite.sha(Path(__file__)),
                      plan_sha256=suite.sha(suite.OUT / 'plan.json'),
                      authorization_sha256=suite.sha(suite.OUT / 'export_authorization.json'),
                      started_at_utc=suite.now())
        if suite.marker(stage):
            report.update(status='complete', adopted_existing_artifact=True)
        else:
            state = suite.read(suite.OUT / 'progress.json')
            if state.get('current_stage') == args.stage:
                raise ValueError('Persistent suite is already running this stage')
            argv = suite.resolve(stage['argv'])
            report.update(status='running', argv=argv)
            suite.dump(work / (args.stage + '.json'), report)
            with (work / (args.stage + '.log')).open('ab') as stream:
                child = subprocess.Popen(argv, cwd=suite.ROOT, stdout=stream, stderr=subprocess.STDOUT)
                report['child_pid'] = child.pid
                suite.dump(work / (args.stage + '.json'), report)
                code = child.wait()
            report.update(status='complete' if code == 0 and suite.marker(stage) else 'failed',
                          returncode=code, finished_at_utc=suite.now())
        if report['status'] == 'complete':
            report['marker_sha256'] = suite.sha(stage['marker'])
        suite.dump(work / (args.stage + '.json'), report)
        if report['status'] != 'complete':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
