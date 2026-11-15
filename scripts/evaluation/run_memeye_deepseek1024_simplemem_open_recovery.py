"""Original SimpleMem Open answer/collect CLI with an exact held-parent guard.

No generation setting changes.  When the dispatcher resumes, the executed
base helper's original failed/skipped guard is required again.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from scripts.evaluation import run_memeye_deepseek1024_text_open_recovery as base

BASE_SHA256 = '808cac3a98b0eee843d30e2b5e7e2e50d5551d6c3193d3a1006d121219ba6fc9'
CPU_ADAPTER_SHA256 = '7dea1fcbffc7f7110bae3c2beb0d43ce4476a962583a20a250f97a3114264007'
DISPATCHER_PID, NATIVE_CHILD_PID = 2825379, 2897726
ORIGINAL_GUARD = base.guard
LAST_GUARD = {}


def proc_status(pid):
    path = Path('/proc') / str(pid) / 'status'
    if not path.exists():
        return None
    return {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
            for line in path.read_text().splitlines() if ':' in line}


def verify_recovery():
    suite = base.suite
    root = suite.OUT / 'open/text_methods/simplemem'
    path = suite.OUT / 'metrics/recovery_diagnostics/simplemem_open_failure_20261106/recovery_evidence.json'
    evidence = suite.read(path)
    if (evidence.get('status') != 'cpu_recovered' or evidence.get('method') != 'simplemem'
            or evidence.get('questions') != 371 or evidence.get('adapter_sha256') != CPU_ADAPTER_SHA256):
        raise RuntimeError('Complete exact-adapter CPU recovery evidence is required')
    for name, expected in {**evidence['preserved_input_sha256'], **evidence['output_sha256']}.items():
        if suite.sha(name) != expected:
            raise RuntimeError('CPU recovery source/output changed: ' + name)
    marker = suite.read(root / 'complete.json')
    if (marker.get('status') != 'complete' or marker.get('passed') is not True
            or marker.get('questions') != 371 or marker.get('seed') != 1024
            or marker['protocol_sha256'] != suite.sha(root / 'protocol.json')
            or marker['retrieval_sha256'] != suite.sha(root / 'retrieval.jsonl')
            or marker['reader_manifest_sha256'] != suite.sha(root / 'reader_manifest.jsonl')):
        raise RuntimeError('Native full Open retrieval completion hashes differ')
    protocol = suite.read(root / 'protocol.json')
    if protocol.get('model') != 'deepseek-v4.1-flash' or protocol.get('seed') != 1024:
        raise RuntimeError('Recovered reader source model/seed differs')
    return path, evidence


def guard(method, expected_pid):
    global LAST_GUARD
    if method != 'simplemem' or expected_pid != DISPATCHER_PID:
        raise RuntimeError('This wrapper permits only the exact SimpleMem dispatcher')
    progress = base.suite.read(base.suite.OUT / 'progress.json')
    if progress.get('pid') != expected_pid:
        raise RuntimeError('Dispatcher changed')
    evidence_path, evidence = verify_recovery()
    parent = proc_status(expected_pid)
    if parent is None or parent['State'].split()[0] != 'T':
        value = ORIGINAL_GUARD(method, expected_pid)
        LAST_GUARD = {'policy': 'original_failed_skipped_guard',
                      'cpu_recovery_evidence_sha256': base.suite.sha(evidence_path)}
        return value
    stage = 'simplemem_open_retrieve'
    entry = progress['stages'][stage]
    if (progress.get('status') != 'running' or progress.get('current_stage') != stage
            or entry.get('status') != 'running' or entry.get('child_pid') != NATIVE_CHILD_PID):
        raise RuntimeError('Only the exact held original retrieval state may bypass the base guard')
    command = (Path('/proc') / str(expected_pid) / 'cmdline').read_bytes().split(b'\0')
    if command[1:5] != [b'-m', b'scripts.evaluation.memeye_deepseek_seed1024_suite', b'--phase', b'run']:
        raise RuntimeError('Held dispatcher identity differs')
    hold_path = base.suite.OUT / 'parallel/simplemem_attribution_run.json'
    hold = base.suite.read(hold_path)
    if (hold.get('dispatcher_pid') != expected_pid or hold.get('dispatcher_held') is not True
            or hold.get('active_independent_child') != NATIVE_CHILD_PID
            or hold.get('active_independent_stage') != stage):
        raise RuntimeError('Exact reviewed attribution hold is required')
    child = proc_status(NATIVE_CHILD_PID)
    if child is not None and (child['State'].split()[0] != 'Z' or int(child['PPid']) != expected_pid):
        raise RuntimeError('Original native SimpleMem retrieval child is still active or changed')
    prior = evidence['held_parent_zombie_child']
    if (prior['dispatcher_pid'] != expected_pid or prior['native_child_pid'] != NATIVE_CHILD_PID
            or prior['native_child_state'] != 'Z'):
        raise RuntimeError('CPU recovery must have proved this exact child exited')
    LAST_GUARD = {'policy': 'held_original_dispatcher_and_exited_original_native_child',
        'dispatcher_pid': expected_pid, 'dispatcher_state': 'T', 'native_child_pid': NATIVE_CHILD_PID,
        'native_child_state': 'Z' if child is not None else 'absent_after_recorded_zombie',
        'cpu_recovery_evidence_sha256': base.suite.sha(evidence_path),
        'attribution_hold_receipt_sha256': base.suite.sha(hold_path)}
    return progress


def main():
    if base.suite.sha(base.__file__) != BASE_SHA256:
        raise RuntimeError('Executed base API helper source changed')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatcher-pid', type=int, default=DISPATCHER_PID)
    args = parser.parse_args()
    original_dump, original_guard, original_argv = base.suite.dump, base.guard, sys.argv[:]
    state_path = base.suite.OUT / 'parallel/text_open_recovery_simplemem.json'
    def stamped_dump(path, value):
        if Path(path) == state_path:
            value = dict(value, wrapper_sha256=base.suite.sha(Path(__file__)),
                         base_helper_sha256=BASE_SHA256, wrapper_guard=dict(LAST_GUARD))
        return original_dump(path, value)
    try:
        base.guard = guard
        base.suite.dump = stamped_dump
        sys.argv = [str(base.__file__), '--method', 'simplemem', '--dispatcher-pid', str(args.dispatcher_pid)]
        base.main()
    finally:
        base.guard, base.suite.dump, sys.argv = original_guard, original_dump, original_argv


if __name__ == '__main__':
    main()
