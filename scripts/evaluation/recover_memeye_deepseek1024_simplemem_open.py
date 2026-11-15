"""CPU recovery for an exited SimpleMem child while its dispatcher is held."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import types

from scripts.evaluation import recover_memeye_deepseek1024_text_open as core

CORE_SHA256 = '55f1bf5799068ec9e11e4fabcc3d53bfcb5228392c2b514fb8e243e77c596569'
DISPATCHER_PID = 2825379
NATIVE_CHILD_PID = 2897726
STAGE = 'simplemem_open_retrieve'


def proc_state(pid):
    values = {}
    for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            values[key] = value.strip()
    return values


def held_zombie_evidence(progress):
    entry = progress['stages'][STAGE]
    if (progress.get('pid') != DISPATCHER_PID or progress.get('status') != 'running'
            or progress.get('current_stage') != STAGE or entry.get('status') != 'running'
            or entry.get('child_pid') != NATIVE_CHILD_PID):
        raise ValueError('Only the exact held dispatcher and exited original SimpleMem child are allowed')
    parent, child = proc_state(DISPATCHER_PID), proc_state(NATIVE_CHILD_PID)
    if parent['State'].split()[0] != 'T' or child['State'].split()[0] != 'Z' or int(child['PPid']) != DISPATCHER_PID:
        raise ValueError('Dispatcher must be stopped and its exact native child must be a zombie')
    command = (Path('/proc') / str(DISPATCHER_PID) / 'cmdline').read_bytes().split(b'\0')
    if command[1:5] != [b'-m', b'scripts.evaluation.memeye_deepseek_seed1024_suite', b'--phase', b'run']:
        raise ValueError('Held dispatcher identity differs')
    hold = core.suite.read(core.OUT / 'parallel/simplemem_attribution_run.json')
    if (hold.get('dispatcher_pid') != DISPATCHER_PID or hold.get('dispatcher_held') is not True
            or hold.get('active_independent_child') != NATIVE_CHILD_PID
            or hold.get('active_independent_stage') != STAGE):
        raise ValueError('The reviewed attribution helper must own this exact dispatcher hold')
    failure = core.suite.read(core.OUT / 'open/text_methods/simplemem/stage_failure.json')
    if failure.get('error') != 'Native Open retrieval changed its frozen question/context':
        raise ValueError('Native failure is not the isolated derived-query contract error')
    return {'dispatcher_pid': DISPATCHER_PID, 'dispatcher_state': 'T', 'native_child_pid': NATIVE_CHILD_PID,
        'native_child_state': 'Z', 'native_child_parent_pid': DISPATCHER_PID,
        'hold_receipt_sha256': core.suite.sha(core.OUT / 'parallel/simplemem_attribution_run.json'),
        'native_failure_sha256': core.suite.sha(core.OUT / 'open/text_methods/simplemem/stage_failure.json')}


def process_check(method, progress):
    if method != 'simplemem':
        raise ValueError('This wrapper is exclusive to SimpleMem')
    held = held_zombie_evidence(progress)
    plan = core.suite.read(core.OUT / 'plan.json')
    attribution = next(s['argv'] for s in plan['stages'] if s['name'] == 'simplemem_attribution_run')
    active, independent = [], []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = [p.decode(errors='replace') for p in (proc / 'cmdline').read_bytes().split(b'\0') if p]
        except OSError:
            continue
        if not any('deepseek1024' in p or 'deepseek_seed1024' in p for p in args):
            continue
        same = any(flag in args and args.index(flag) + 1 < len(args)
                   and args[args.index(flag) + 1] == method for flag in ('--method', '--method-id'))
        if not same:
            continue
        if args[1:] == attribution[1:]:
            independent.append({'pid': int(proc.name), 'exact_original_attribution_argv': True})
        else:
            active.append({'pid': int(proc.name), 'command': args})
    if active:
        raise ValueError('Another live SimpleMem Open/API process exists: ' + str(active))
    return {'same_method_active_processes': [], 'host_dispatcher_pid_verified': DISPATCHER_PID,
        'held_parent_zombie_child': held, 'independent_mcq_attribution_processes': independent}


def recover():
    if core.suite.sha(core.__file__) != CORE_SHA256:
        raise ValueError('Executed base adapter changed')
    original = ast.parse(Path(core.__file__).read_text())
    audit = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
    removed = 0
    messages = ('Only an ended failed native retrieval stage is recoverable', 'Native retrieval stage is still current')
    body = []
    for node in audit.body:
        if isinstance(node, ast.If) and any(isinstance(n, ast.Constant) and n.value in messages for n in ast.walk(node)):
            removed += 1
        else:
            body.append(node)
    if removed != 2:
        raise ValueError('Unexpected base adapter dispatcher guard')
    audit.body = body
    index = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'paths' for t in n.targets))
    audit.body[index + 1:index + 1] = ast.parse(
        "paths = [path for path in paths if path.name != 'replayed_calls.jsonl' or path.is_file()]"
    ).body
    namespace = dict(vars(core), __file__=__file__, check_no_live_method=process_check)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[audit], type_ignores=[])), __file__, 'exec'), namespace)
    function = types.FunctionType(core.recover.__code__, namespace, 'simplemem_recover')
    result = function('simplemem')
    result.update(base_recovery_adapter_sha256=CORE_SHA256,
        dispatcher_guard_boundary='Only the recorded original native child is Z and the same reviewed attribution helper holds the unchanged dispatcher in T; no live Open/API writer is allowed. The independent exact-original-argv MCQ attribution reader is separate.')
    core.suite.dump(core.OUT / 'metrics/recovery_diagnostics/simplemem_open_failure_20261106/recovery_evidence.json', result)
    return result


if __name__ == '__main__':
    result = recover()
    print(json.dumps({k: result[k] for k in ('method', 'questions', 'different_frozen_fields', 'api_calls', 'gpu_model_loads')}))
