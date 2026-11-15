"""Start the two native text readers only after their complete input audits pass."""
from pathlib import Path
import fcntl
import json
import subprocess
import sys
import time

ROOT = Path('/storage/hewenwen/VisualMemory')
RUN = Path(__file__).resolve().parent


def main():
    lock = (RUN / 'remaining_readers.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = {'a_mem': 195, 'simplemem': 186}
    jobs, streams = {}, []
    while pending or jobs:
        for mid, correct in list(pending.items()):
            root = RUN / 'text_methods' / mid
            ready = root / 'ready_for_reader.json'
            if not ready.exists():
                continue
            # The reader itself validates all 371 canonical IDs and freezes the input hashes.
            command = [sys.executable, '-u', str(ROOT / 'scripts/evaluation/memeye_open_method_reader.py'),
                '--phase', 'run', '--root', str(root), '--method-id', mid,
                '--reference-correct', str(correct), '--url', 'http://127.0.0.1:18762',
                '--service-root', str(RUN / 'qwen_large_budget_service')]
            stream = (root / 'reader_run.log').open('a')
            streams.append(stream)
            jobs[mid] = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            del pending[mid]
            print(json.dumps({'event': 'reader_started', 'method': mid, 'pid': jobs[mid].pid,
                              'service_port': 18762, 'max_new_tokens': 512}), flush=True)
        for mid, process in list(jobs.items()):
            code = process.poll()
            if code is None:
                continue
            if code:
                raise RuntimeError(f'{mid} reader failed with exit code {code}')
            del jobs[mid]
            print(json.dumps({'event': 'reader_complete', 'method': mid}), flush=True)
        if pending or jobs:
            time.sleep(15)
    for stream in streams:
        stream.close()


if __name__ == '__main__':
    main()
