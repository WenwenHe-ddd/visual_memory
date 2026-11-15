"""Execute an independent prepared visual Open/Judge branch, with export gates."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess

from scripts.evaluation import memeye_deepseek_seed1024_suite as suite

BRANCHES = {
    'clip_image_dialogue': ('clip_image_dialogue_open_answers',
                           'clip_image_dialogue_open_collect', 'clip_image_dialogue_judge42'),
    'hippomm': ('hippomm_open_prepare', 'hippomm_open_answers', 'hippomm_judge42'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(BRANCHES), required=True)
    args = parser.parse_args()
    plan = suite.read(suite.OUT / 'plan.json')
    stages = {s['name']: s for s in plan['stages']}
    suite.verify(plan)
    if not suite.export_allowed():
        raise PermissionError('Explicit export authorization required')
    work = suite.OUT / 'parallel'
    work.mkdir(exist_ok=True)
    path = work / (args.method + '_open_branch.json')
    with (path.with_suffix('.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = suite.read(path) if path.exists() else {'stages': {}}
        state.update(status='running', method=args.method, helper_sha256=suite.sha(Path(__file__)),
                     plan_sha256=suite.sha(suite.OUT / 'plan.json'), started_at_utc=suite.now())
        for name in BRANCHES[args.method]:
            suite.verify(plan)
            if not suite.export_allowed():
                raise PermissionError('Export authorization changed')
            stage = stages[name]
            for dep in stage.get('depends_on', []):
                if state['stages'].get(dep, {}).get('status') != 'complete' and not suite.marker(stages[dep]):
                    raise ValueError('Native prerequisite is not complete: ' + dep)
            if suite.marker(stage):
                state['stages'][name] = {'status': 'complete', 'adopted_existing_artifact': True}
            elif state['stages'].get(name, {}).get('status') != 'complete' or stage.get('marker'):
                current = suite.read(suite.OUT / 'progress.json').get('current_stage')
                if current == name:
                    raise ValueError('Persistent suite already runs this stage')
                argv = suite.resolve(stage['argv'])
                entry = dict(status='running', argv=argv, started_at_utc=suite.now())
                state['stages'][name] = entry
                state['current_stage'] = name
                suite.dump(path, state)
                with (work / (name + '.log')).open('ab') as log:
                    child = subprocess.Popen(argv, cwd=suite.ROOT, stdout=log, stderr=subprocess.STDOUT)
                    entry['child_pid'] = child.pid
                    suite.dump(path, state)
                    code = child.wait()
                okay = code == 0 and (not stage.get('marker') or suite.marker(stage))
                entry.update(status='complete' if okay else 'failed', returncode=code,
                             finished_at_utc=suite.now())
                if not okay:
                    state.update(status='failed', finished_at_utc=suite.now())
                suite.dump(path, state)
                if not okay:
                    raise SystemExit(2)
            if stage.get('marker'):
                state['stages'][name]['marker_sha256'] = suite.sha(stage['marker'])
            suite.dump(path, state)
        state.update(status='complete', current_stage=None, finished_at_utc=suite.now())
        suite.dump(path, state)


if __name__ == '__main__':
    main()
