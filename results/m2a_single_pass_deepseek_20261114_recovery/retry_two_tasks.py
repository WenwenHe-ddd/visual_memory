"""One unchanged retry of the two remaining tasks; never replay empty output."""
from pathlib import Path
import copy
import json
import shutil
import sys

BASE = Path('/storage/hewenwen/VisualMemory')
sys.path.insert(0, str(BASE))
from scripts.evaluation import m2a_deepseek_recovery_suite as recovery

suite = recovery.suite
root = Path(__file__).resolve().parent
parent = BASE/'results/m2a_single_pass_deepseek_20261113'
state = suite.read(root/'status.json')
if state.get('active'):
    raise ValueError('An existing worker is active')
if (root/'retry_two_tasks_amendment.json').exists():
    raise ValueError('This one-time retry has already been launched')
suite.verify(root)
suite.write(root/'status_before_retry_two_tasks.json', state)
verified = {}
for entry in state['tasks']:
    if entry['status'] == 'passed':
        recovery.verify_inherited(entry)
        verified[str(Path(entry['output']).resolve())] = entry['audit']

caches, excluded = {}, []
for entry in state['tasks']:
    if entry['status'] != 'failed':
        continue
    target = root/entry['dataset']/entry['task']/'retry_valid_cache'
    target.mkdir()
    source = parent/entry['dataset']/entry['task']/'api_cache'
    for folder in source.iterdir():
        if not folder.is_dir():
            continue
        response = folder/'response.json'
        if response.exists():
            value = suite.read(response)
            valid = (value['returned_model'] == suite.MODEL
                     and value['finish_reason'] in ('stop', 'tool_calls')
                     and not value.get('refusal')
                     and bool(value['raw_response'] or value['tool_calls']))
            if not valid:
                # Keep the original response and its referring failure logs intact.
                fresh = target/folder.name
                fresh.mkdir()
                shutil.copyfile(folder/'request.json.gz', fresh/'request.json.gz')
                excluded.append(dict(task=entry['task'], request_sha256=folder.name,
                                     original_response=str(response), reason='invalid/empty cached response'))
                continue
        (target/folder.name).symlink_to(folder, target_is_directory=True)
    caches[entry['task']] = target

suite.write(root/'retry_two_tasks_amendment.json', dict(
    created_at=suite.now(), driver_sha256=suite.sha(Path(__file__)),
    tasks=list(caches), excluded_cached_responses=excluded,
    request_policy='Same model, prompts, images, tools and token limits; one fresh attempt per task',
    prior_artifacts='Original invalid responses and failure logs retained unchanged',
    gpt_started=False))
original_audit, original_command = suite.audit, suite.command

def audit(out):
    resolved = str(Path(out).resolve())
    return copy.deepcopy(verified[resolved]) if resolved in verified else original_audit(out)

def command(task, out, cache, env_file):
    return original_command(task, out, caches.get(task['task'], cache), env_file)

suite.audit, suite.command = audit, command
recovery.run(parent, root)
