"""Persistent seed1024 DeepSeek queue with an explicit dataset-export gate."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def verify(plan):
    for path, expected in plan['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Prepared implementation changed: ' + path)
    if plan['seed'] != 1024 or plan['judge_seed'] != 42 or plan['model'] != 'deepseek-v4.1-flash':
        raise ValueError('Model or seed policy differs')


def export_allowed():
    path = OUT / 'export_authorization.json'
    if not path.exists():
        return False
    value = read(path)
    return (value.get('approved') is True and value.get('source') == 'explicit_user_reply'
            and value.get('manifest_sha256') == sha(OUT / 'data_export_manifest.json')
            and value.get('model') == 'deepseek-v4.1-flash'
            and value.get('destination') == 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions')


def marker(stage):
    path = stage.get('marker')
    if not path or not Path(path).exists():
        return False
    data = read(path)
    return data.get(stage.get('marker_key', 'status')) in stage.get('marker_values', ['complete'])


def resolve(argv):
    result = []
    for item in argv:
        if item.startswith('@CORRECT:') and item.endswith('@'):
            path = item[len('@CORRECT:'):-1]
            data = read(path)
            if data.get('questions') != 371:
                raise ValueError('Reader reference count requires full MCQ completion')
            count = data.get('correct', round(data['accuracy'] * 371) if 'accuracy' in data else None)
            if type(count) is not int:
                raise ValueError('Missing measured correct count')
            result.append(str(count))
        else:
            result.append(item)
    return result


def run(plan, offline_only=False):
    verify(plan)
    if not offline_only and not export_allowed():
        raise PermissionError('Dataset export has not received explicit user approval; no API stages started')
    path = OUT / 'progress.json'
    state = read(path) if path.exists() else {'stages': {}}
    state.update(status='offline_preparation' if offline_only else 'running', seed=1024,
                 judge_seed=42, model=plan['model'], pid=os.getpid(), updated_at_utc=now())
    logs = OUT / 'logs'; logs.mkdir(exist_ok=True)
    for stage in plan['stages']:
        name = stage['name']
        if offline_only and stage['kind'] != 'offline':
            state['stages'].setdefault(name, {'status': 'awaiting_export_authorization'})
            continue
        verify(plan)
        previous = state['stages'].get(name, {})
        if previous.get('status') == 'complete' and (not stage.get('marker') or marker(stage)):
            continue
        if marker(stage):
            state['stages'][name] = {'status': 'complete', 'adopted_existing_artifact': True,
                                     'marker_sha256': sha(stage['marker'])}
            dump(path, state)
            continue
        missing = [d for d in stage.get('depends_on', []) if state['stages'].get(d, {}).get('status') != 'complete']
        if missing:
            state['stages'][name] = {'status': 'skipped_dependency', 'dependencies': missing}
            dump(path, state)
            continue
        if stage['kind'] == 'api' and not export_allowed():
            raise PermissionError('Export authorization changed; API queue stopped')
        argv = resolve(stage['argv'])
        log = logs / (name + '.log')
        entry = {'status': 'running', 'argv': argv, 'kind': stage['kind'],
                 'log': str(log), 'started_at_utc': now()}
        state['stages'][name] = entry
        state.update(current_stage=name, updated_at_utc=now())
        dump(path, state)
        print(json.dumps({'stage': name, 'status': 'started', 'kind': stage['kind']}), flush=True)
        with log.open('ab') as stream:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            entry['child_pid'] = child.pid
            while child.poll() is None:
                state['updated_at_utc'] = now(); dump(path, state)
                time.sleep(10)
            code = child.returncode
        okay = code == 0 and (not stage.get('marker') or marker(stage))
        entry.update(status='complete' if okay else 'failed', returncode=code, finished_at_utc=now())
        if okay and stage.get('marker'):
            entry['marker_sha256'] = sha(stage['marker'])
        dump(path, state)
        print(json.dumps({'stage': name, 'status': entry['status'], 'returncode': code}), flush=True)
    if offline_only:
        pending = [s['name'] for s in plan['stages'] if s['kind'] == 'offline' and state['stages'].get(s['name'], {}).get('status') != 'complete']
        state['status'] = 'offline_failed' if pending else 'ready_awaiting_export_authorization'
    else:
        pending = [s['name'] for s in plan['stages'] if state['stages'].get(s['name'], {}).get('status') != 'complete']
        state['status'] = 'partial_failed' if pending else 'generation_complete_pending_final_audit'
    state.update(current_stage=None, unfinished_stages=pending, updated_at_utc=now())
    dump(path, state)
    if pending:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('verify', 'offline', 'run'), required=True)
    args = parser.parse_args()
    plan = read(OUT / 'plan.json')
    verify(plan)
    if args.phase == 'verify':
        print(json.dumps({'sources_verified': True, 'stages': len(plan['stages']),
                          'dataset_export_authorized': export_allowed()}))
        return
    with (OUT / 'suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(plan, args.phase == 'offline')


if __name__ == '__main__':
    main()
