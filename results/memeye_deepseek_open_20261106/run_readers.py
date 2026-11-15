"""Run up to two independent DeepSeek readers as native Open contexts become ready."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/storage/hewenwen/VisualMemory')
RUN = Path(__file__).resolve().parent


def complete(root):
    path = root / 'reader/completion.json'
    return path.exists() and json.loads(path.read_text()).get('status') == 'complete'


def main():
    with (RUN / 'reader_dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        methods = json.loads((RUN / 'methods_manifest.json').read_text())['methods']
        pending = [m for m in methods if m['id'] != 'hippomm' and not complete(Path(m['root']))]
        active = {}
        while pending or active:
            for mid, (process, log, method) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                del active[mid]
                if status or not complete(Path(method['root'])):
                    raise RuntimeError(f'Reader failed: {mid}, status={status}')
                print(json.dumps({'event': 'reader_complete', 'method': mid}), flush=True)
            for method in pending[:]:
                if len(active) >= 2:
                    break
                root = Path(method['root'])
                ready = (root / 'ready_for_reader.json').exists()
                if method['id'] == 'clip_image_dialogue':
                    ready = (root / 'input_audit.json').exists() and (root / 'reader_manifest.jsonl').exists()
                if not ready:
                    continue
                log = (root / 'reader_run.log').open('a')
                command = [sys.executable, '-u', str(ROOT / 'scripts/evaluation/memeye_deepseek_open_reader.py'),
                    '--phase', 'run', '--root', str(root), '--method-id', method['id'],
                    '--reference-correct', str(method['source_mcq_correct_for_identification_only']), '--workers', '4']
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                active[method['id']] = process, log, method
                pending.remove(method)
                print(json.dumps({'event': 'reader_started', 'method': method['id'], 'pid': process.pid}), flush=True)
            if pending or active:
                time.sleep(10)
        print(json.dumps({'status': 'complete', 'generic_readers': 9}), flush=True)


if __name__ == '__main__':
    main()
