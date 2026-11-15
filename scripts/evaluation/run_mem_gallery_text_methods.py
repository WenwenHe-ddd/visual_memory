"""Run isolated native-method processes sequentially on the selected CUDA GPU."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.evaluation.mem_gallery_qwen_text_methods import METHODS, OUTPUT, ROOT, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    status = dict(status='running', started=time.time(), pid=os.getpid(), device=args.device, methods={})
    for method in METHODS:
        status['active_method'] = method
        status['methods'][method] = dict(status='running', started=time.time())
        dump(output / 'execution_status.json', status)
        with (output / (method + '.log')).open('a') as log:
            result = subprocess.run([sys.executable, '-u', '-m',
                'scripts.evaluation.mem_gallery_qwen_text_methods', 'run',
                '--method', method, '--device', args.device], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status['methods'][method].update(status='generated' if result.returncode == 0 else 'failed',
                                         returncode=result.returncode, finished=time.time())
        dump(output / 'execution_status.json', status)
        print(method, status['methods'][method], flush=True)
    if any(v['returncode'] for v in status['methods'].values()):
        status.update(status='failed', active_method=None)
        dump(output / 'execution_status.json', status)
        raise SystemExit(1)
    status.update(status='generated_pending_audit', active_method=None)
    dump(output / 'execution_status.json', status)


if __name__ == '__main__':
    main()
