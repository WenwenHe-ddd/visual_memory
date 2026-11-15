"""Recover failed tasks, reuse audited successes, preserve the initial run."""
import argparse
import copy
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import m2a_deepseek_suite as suite
from scripts.memory import m2a_deepseek_recovery as images


def verify_inherited(entry):
    out = Path(entry['output'])
    if suite.sha(out/'completion_manifest.json') != entry['completion_sha256']:
        raise ValueError('Parent completion manifest changed: '+str(out))
    manifest = suite.read(out/'completion_manifest.json')
    for name, expected in manifest['artifacts'].items():
        if suite.sha(out/name) != expected:
            raise ValueError('Parent artifact changed: '+str(out/name))
    if not entry['audit']['passed']:
        raise ValueError('Cannot inherit a failed parent audit')


def prepare(parent, root):
    parent_plan = suite.verify(parent)
    root.mkdir(parents=True, exist_ok=False)
    plan = copy.deepcopy(parent_plan)
    plan['recovery'] = dict(parent=str(parent), parent_plan_sha256=suite.sha(parent/'plan.json'),
                            policy=images.POLICY, successful_tasks='Reuse audited immutable parent outputs',
                            cache_policy='Reuse exact request hashes from same-task parent cache only',
                            inherited_audits='Parent passing audits with manifest and all sealed artifact hashes reverified',
                            content_rejections='One fresh unchanged retry; never rewrite content or substitute models')
    plan['configuration']['image_policy'] = images.POLICY
    shutil.copytree(parent/'code_snapshot', root/'code_snapshot')
    for path in [Path(__file__), ROOT/'scripts/evaluation/m2a_deepseek_recovery_run.py',
                 Path(images.__file__)]:
        path = path.resolve()
        plan['implementation_sha256'][str(path)] = suite.sha(path)
        target = root/'code_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    inherited, retry = [], []
    for entry in suite.read(parent/'status.json')['tasks']:
        if entry['status'] == 'passed':
            out = Path(entry['output'])
            verify_inherited(entry)
            target = root/entry['dataset']/entry['task']
            target.mkdir(parents=True)
            (target/'attempt_001').symlink_to(out, target_is_directory=True)
            inherited.append(entry['task'])
            print('INHERITED', entry['task'], flush=True)
        else:
            retry.append(entry['task'])
    suite.write(root/'plan.json', plan)
    suite.write(root/'recovery_preparation.json', dict(inherited=inherited, retry=retry,
                original_probe_sha256=suite.sha(parent/'probe.json')))
    suite.write(root/'probe.json', dict(passed=True,
                kind='Inherited baseline image/tools probe; amended preprocessing covered by local tests',
                parent_probe=str(parent/'probe.json'), parent_probe_sha256=suite.sha(parent/'probe.json'),
                plan_sha256=suite.sha(root/'plan.json')))


def run(parent, root):
    if not (root/'plan.json').exists():
        prepare(parent, root)
    if suite.read(root/'plan.json')['recovery']['parent'] != str(parent):
        raise ValueError('Recovery parent mismatch')
    audit_cache = {}
    for entry in suite.read(parent/'status.json')['tasks']:
        if entry['status'] == 'passed':
            verify_inherited(entry)
            audit_cache[(str(Path(entry['output']).resolve()), entry['completion_sha256'])] = entry['audit']
    original_audit = suite.audit

    def cached_audit(out):
        key = (str(Path(out).resolve()), suite.sha(Path(out)/'completion_manifest.json'))
        if key not in audit_cache:
            audit_cache[key] = original_audit(out)
        return copy.deepcopy(audit_cache[key])

    suite.audit = cached_audit
    original_command = suite.command

    def command(task, out, cache, env_file):
        cmd = original_command(task, out, parent/task['dataset']/task['task']/'api_cache', env_file)
        cmd[2] = str(ROOT/'scripts/evaluation/m2a_deepseek_recovery_run.py')
        return cmd

    suite.command = command
    suite.run(root, suite.DEFAULT_ENV)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.parent.resolve(), args.output.resolve())
