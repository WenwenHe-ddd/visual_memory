"""Resume the unchanged queue after its active child and original format repair.

The seed42 pipeline normalized three container-only candidate outputs after its
three native attempts. This recovery adds that same CPU step without modifying
the frozen generation code/plan or interrupting the active CLIP inference.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/memgallery_qwen_seed1042_full_20261106'
RECOVERY = RUN / 'recovery'
NEW_UNIT = 'visualmemory-memgallery-qwen1042-suite-recovered-20261106.service'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def process_state(pid):
    path = Path('/proc') / str(pid) / 'stat'
    try:
        raw = path.read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # Process names may contain spaces or parentheses.
    return raw.rsplit(')', 1)[1].strip().split()[0]


def verify(plan):
    for path, digest in plan['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Recovery source hash changed: ' + path)
    source = ROOT / 'scripts/evaluation/memgallery_seed1042_suite.py'
    spec = importlib.util.spec_from_file_location('frozen_seed1042_suite', source)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    suite.verify_plan(read(RUN / 'plan.json'))


def main():
    plan = read(RECOVERY / 'restart_plan.json')
    verify(plan)
    held = read(RECOVERY / 'coordinator_hold.json')
    coordinator, child = held['coordinator_pid'], held['child_pid']
    progress = {'status': 'waiting_active_child', 'coordinator_pid': coordinator,
                'child_pid': child, 'seed': 1042, 'started_at_utc': now()}
    proof_path = RECOVERY / 'candidate_format_completion.json'
    while True:
        progress.update(updated_at_utc=now(), child_state=process_state(child),
                        repair_complete=proof_path.exists() and read(proof_path).get('status') == 'complete')
        dump(RECOVERY / 'restart_progress.json', progress)
        if progress['repair_complete'] and progress['child_state'] in (None, 'Z'):
            break
        if process_state(coordinator) not in ('T', 't'):
            raise RuntimeError('Held coordinator unexpectedly resumed or exited')
        time.sleep(15)
    verify(plan)
    repair = read(proof_path)
    if (repair.get('seed') != 1042 or repair.get('total_candidates') != 1711
            or repair.get('new_generation_attempts') != 0
            or not repair.get('source_code_and_old_data_unchanged')
            or repair.get('raw_generation_attempts_before') != repair.get('raw_generation_attempts_after')):
        raise ValueError('Original format-only recovery contract not satisfied')
    # The pre-normalization candidates hash is historical; output hashes must
    # all remain current, as must the complete original generation attempts.
    for path, digest in repair['output_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Candidate repair output changed: ' + path)
    attempts_path = RUN / 'caption_methods/base/candidate_attempts.jsonl'
    if sha(attempts_path) != repair['input_sha256'][str(attempts_path)]:
        raise ValueError('Original candidate attempts changed during recovery')
    if sha(RUN / 'plan.json') != held['gpu_plan_sha256']:
        raise ValueError('Frozen GPU plan changed')
    current = int(subprocess.check_output(['systemctl', '--user', 'show', held['unit'],
                  '--property=MainPID', '--value'], text=True).strip())
    if current != coordinator or process_state(coordinator) not in ('T', 't'):
        raise RuntimeError('Unexpected coordinator identity/state before recovery')
    # The child has already released its CUDA context. Killing the held main
    # process releases only its queue lock; no active inference is interrupted.
    os.kill(coordinator, signal.SIGKILL)
    for _ in range(60):
        if process_state(coordinator) in (None, 'Z'):
            break
        time.sleep(1)
    else:
        raise RuntimeError('Old coordinator did not exit')
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', NEW_UNIT]).returncode == 0:
        raise RuntimeError('Recovery queue already active')
    log = RUN / 'logs/suite-recovered.log'
    argv = ['systemd-run', '--user', '--unit=' + NEW_UNIT, '--collect',
        '--property=WorkingDirectory=' + str(ROOT), '--property=Restart=no',
        '--property=StandardOutput=append:' + str(log),
        '--property=StandardError=append:' + str(log),
        '--setenv=CUDA_VISIBLE_DEVICES=1', '--setenv=HF_HUB_OFFLINE=1',
        '--setenv=TOKENIZERS_PARALLELISM=false', '/usr/bin/python3', '-u',
        str(ROOT / 'scripts/evaluation/memgallery_seed1042_suite.py'), '--phase', 'run']
    subprocess.run(argv, check=True)
    new_pid = int(subprocess.check_output(['systemctl', '--user', 'show', NEW_UNIT,
                  '--property=MainPID', '--value'], text=True).strip())
    if new_pid < 1:
        raise RuntimeError('Recovered suite did not start')
    progress.update(status='complete', new_unit=NEW_UNIT, new_coordinator_pid=new_pid,
        plan_sha256=sha(RUN / 'plan.json'), candidate_repair_sha256=sha(proof_path),
        active_inference_interrupted=False, completed_at_utc=now())
    dump(RECOVERY / 'restart_progress.json', progress)
    dump(RECOVERY / 'restart_completion.json', progress)
    print(json.dumps(progress), flush=True)


if __name__ == '__main__':
    main()
