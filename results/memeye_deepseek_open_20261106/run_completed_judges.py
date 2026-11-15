"""Score sealed method answers as they finish, keeping compatible cache roots frozen."""
from pathlib import Path
import fcntl
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path('/storage/hewenwen/VisualMemory')
RUN = Path(__file__).resolve().parent
MANIFEST = RUN / 'methods_manifest.json'
QWEN_MANIFEST = ROOT / 'results/memeye_qwen_other_open_20261105/methods_manifest.json'
GPT_MANIFEST = ROOT / 'results/memeye_gpt5_open_20261105/methods_manifest.json'
ENGINE = ROOT / 'scripts/evaluation/memeye_open_judge_engine.py'
READER = ROOT / 'scripts/evaluation/memeye_deepseek_open_reader.py'
JUDGE = ROOT / 'scripts/evaluation/memeye_deepseek_open_judge.py'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(root):
    path = root / 'judge/summary.json'
    if not path.exists() or read(path).get('status') != 'complete':
        return False
    if root.is_relative_to(RUN):
        public = root / 'judge/llm_judge_summary.json'
        return public.exists() and read(public).get('status') == 'complete'
    return True


def call(command, log):
    print(json.dumps({'event': 'subprocess', 'command': command, 'log': str(log)}, ensure_ascii=False), flush=True)
    with log.open('a') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    lock = (RUN / 'judge_dispatch.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = {'source_sha256': {str(p): sha(p) for p in (Path(__file__), MANIFEST, QWEN_MANIFEST, GPT_MANIFEST, READER, JUDGE, ENGINE)},
        'policy': 'Wait for complete 371-answer seals, collect original answers, sequential exact-wire-cached Judge jobs.',
        'cache_order': 'All completed Qwen and GPT Open methods followed by completed DeepSeek methods; exact-wire-only reuse; frozen at first preparation.',
        'first_api_batch_limit': 9, 'workers': 8, 'max_attempts': 5}
    protocol_path = RUN / 'judge_dispatch_protocol.json'
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError('Dispatcher contract changed')
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    methods = read(MANIFEST)['methods']
    remaining = list(methods)
    cache_methods = read(QWEN_MANIFEST)['methods'] + read(GPT_MANIFEST)['methods'] + methods
    last_status = 0
    while remaining:
        for method in remaining[:]:
            root = Path(method['root'])
            if complete(root):
                remaining.remove(method)
                continue
            native = method['id'] == 'hippomm'
            seal_path = root / ('completion.json' if native else 'reader/completion.json')
            if not seal_path.exists() or read(seal_path).get('status') != 'complete':
                continue
            if native and not (root / 'judge_inputs.jsonl').exists():
                continue  # Native collector and final audit are owned by the visual worker.
            if not native:
                call([sys.executable, str(READER), '--phase', 'collect', '--root', str(root),
                      '--method-id', method['id'], '--reference-correct',
                      str(method['source_mcq_correct_for_identification_only'])], root / 'judge_collect.log')
            source_protocol = root / ('collection_protocol.json' if native else 'reader/protocol.json')
            args = [sys.executable, '-u', str(JUDGE), '--method-id', method['id'],
                    '--method-label', method['label'], '--input', str(root / 'judge_inputs.jsonl'),
                    '--output', str(root / 'judge'), '--source-protocol', str(source_protocol)]
            if not (root / 'judge/protocol.json').exists():
                for previous in cache_methods:
                    previous_root = Path(previous['root'])
                    if previous_root != root and complete(previous_root):
                        args.extend(['--cache-root', str(previous_root / 'judge')])
                call(args + ['--phase', 'run', '--limit', '9'], root / 'judge_run.log')
                # The first nine calls exercise the real provider before the remaining cohort.
            call(args + ['--phase', 'run'], root / 'judge_run.log')
            if not complete(root):
                raise ValueError('Judge returned without complete coverage: ' + method['id'])
            remaining.remove(method)
            summary = read(root / 'judge/llm_judge_summary.json')
            print(json.dumps({'event': 'method_complete', 'method_id': method['id'],
                'llm_judge': summary['llm_judge'], 'questions_scored': summary['questions_scored']}, ensure_ascii=False), flush=True)
        if time.monotonic() - last_status > 60:
            print(json.dumps({'event': 'waiting_for_sealed_answers', 'remaining': [m['id'] for m in remaining]}), flush=True)
            last_status = time.monotonic()
        if remaining:
            time.sleep(15)
    print(json.dumps({'status': 'complete', 'new_methods_judged': 10, 'questions_per_method': 371}), flush=True)


if __name__ == '__main__':
    main()
