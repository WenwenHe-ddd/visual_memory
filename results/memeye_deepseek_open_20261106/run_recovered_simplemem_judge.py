"""Judge the sealed SimpleMem recovery projection; never rerun successful answers."""
from pathlib import Path
import fcntl
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path('/storage/hewenwen/VisualMemory')
RUN = Path(__file__).resolve().parent
METHOD = RUN / 'text_methods/simplemem'
RECOVERY = ROOT / 'scripts/evaluation/memeye_deepseek_open_reader_recovery.py'
JUDGE = ROOT / 'scripts/evaluation/memeye_deepseek_open_judge.py'


def read(path):
    return json.loads(path.read_text())


def complete(root):
    path = root / 'judge/summary.json'
    return path.exists() and read(path).get('status') == 'complete'


def call(command):
    print(json.dumps({'command': command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    lock = (RUN / 'simplemem_recovery_judge.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seal = METHOD / 'reader/evaluation_completion.json'
    while not seal.exists() or read(seal).get('status') != 'complete':
        time.sleep(10)
    call([sys.executable, str(RECOVERY), '--phase', 'collect', '--root', str(METHOD),
          '--method-id', 'simplemem', '--reference-correct', '217'])
    label = next(m['label'] for m in read(RUN / 'methods_manifest.json')['methods'] if m['id'] == 'simplemem')
    args = [sys.executable, '-u', str(JUDGE), '--method-id', 'simplemem', '--method-label', label,
            '--input', str(METHOD / 'judge_inputs.jsonl'), '--output', str(METHOD / 'judge'),
            '--source-protocol', str(METHOD / 'reader/evaluation_protocol.json')]
    if not (METHOD / 'judge/protocol.json').exists():
        manifests = [ROOT / 'results/memeye_qwen_other_open_20261105/methods_manifest.json',
                     ROOT / 'results/memeye_gpt5_open_20261105/methods_manifest.json',
                     RUN / 'methods_manifest.json']
        for manifest in manifests:
            for method in read(manifest)['methods']:
                previous = Path(method['root'])
                if previous != METHOD and complete(previous):
                    args += ['--cache-root', str(previous / 'judge')]
        call(args + ['--phase', 'run', '--limit', '9'])
    call(args + ['--phase', 'run'])
    if not complete(METHOD):
        raise ValueError('Recovered SimpleMem Judge is incomplete')
    sources = [Path(__file__), RECOVERY, JUDGE, METHOD / 'reader/evaluation_completion.json',
               METHOD / 'judge/summary.json']
    result = {'status': 'complete', 'method': 'simplemem', 'questions': 371,
              'policy': 'Original successful answers plus explicit provider rejections; actual five-grade Judge for all 371 records.',
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (RUN / 'simplemem_recovery_judge_completion.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
