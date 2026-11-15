"""Run fixed-seed42 judge stages from the sealed plan, using native root locks."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess

from scripts.evaluation import memeye_deepseek_seed1024_suite as suite

METHODS = ('semrag', 'ours_1', 'ours_2', 'ours_1_ours_2', 'gen_agents',
           'memoryos', 'simplemem', 'a_mem', 'clip_image_dialogue', 'hippomm')


def inputs(stage):
    argv = stage['argv']
    if (stage['kind'] != 'api' or argv[1:4] !=
            ['-m', 'scripts.evaluation.memeye_deepseek_seed1024_open', '--phase']
            or argv[4] != 'judge'):
        raise ValueError('Only the original fixed evaluator command is supported')
    root = Path(argv[argv.index('--root') + 1])
    path = root / 'judge_inputs.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 371 or len({row['id'] for row in rows}) != 371:
        raise ValueError('Complete unique371-question judge inputs are required')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=METHODS, required=True)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        raise ValueError('Duplicate methods')
    plan = suite.read(suite.OUT / 'plan.json')
    stages = {stage['name']: stage for stage in plan['stages']}
    work = suite.OUT / 'parallel'
    for method in args.methods:
        stage = stages[method + '_judge42']
        suite.verify(plan)
        if not suite.export_allowed():
            raise PermissionError('Explicit export authorization is required')
        source = inputs(stage)
        path = work / ('judge_' + method + '.json')
        with (work / ('judge_' + method + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = dict(status='running', method=method, started_at_utc=suite.now(),
                         helper_sha256=suite.sha(Path(__file__)),
                         plan_sha256=suite.sha(suite.OUT / 'plan.json'),
                         judge_inputs_sha256=suite.sha(source), argv=stage['argv'])
            suite.dump(path, state)
            try:
                if suite.marker(stage):
                    state['adopted_existing_artifact'] = True
                else:
                    with (work / ('judge_' + method + '.log')).open('ab') as log:
                        code = subprocess.call(stage['argv'], cwd=suite.ROOT, stdout=log,
                                               stderr=subprocess.STDOUT)
                    state['returncode'] = code
                    if code != 0 or not suite.marker(stage):
                        raise RuntimeError('Original fixed evaluator did not complete')
                if suite.sha(source) != state['judge_inputs_sha256']:
                    raise ValueError('Judge inputs changed during execution')
                state.update(status='complete', marker_sha256=suite.sha(stage['marker']))
            except BaseException as error:
                state.update(status='failed', error=type(error).__name__ + ': ' + str(error))
                raise
            finally:
                state['finished_at_utc'] = suite.now()
                suite.dump(path, state)


if __name__ == '__main__':
    main()
