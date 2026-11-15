"""Run and audit each metadata-preserving Gallery method in an isolated process."""
import argparse
import os
import subprocess
import sys
import time
from scripts.evaluation.mem_gallery_qwen_text_methods_v2 import OUTPUT, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    status = dict(status='running', started=time.time(), pid=os.getpid(), device=args.device, methods={})
    for method in base.METHODS:
        status['active_method'] = method
        status['methods'][method] = dict(status='running', started=time.time())
        base.dump(OUTPUT / 'execution_status.json', status)
        with (OUTPUT / (method + '.log')).open('a') as log:
            cmd = [sys.executable, '-u', '-m', 'scripts.evaluation.mem_gallery_snapshot_runtime']
            process = subprocess.run(cmd + ['run', '--method', method, '--device', args.device],
                cwd=base.ROOT, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode == 0:
                process = subprocess.run(cmd + ['audit', '--method', method],
                    cwd=base.ROOT, stdout=log, stderr=subprocess.STDOUT)
        status['methods'][method].update(status='audited_complete' if process.returncode == 0 else 'failed',
            returncode=process.returncode, finished=time.time())
        base.dump(OUTPUT / 'execution_status.json', status)
        print(method, status['methods'][method], flush=True)
    if any(v['returncode'] for v in status['methods'].values()):
        status.update(status='failed', active_method=None)
        base.dump(OUTPUT / 'execution_status.json', status)
        raise SystemExit(1)
    subprocess.run([sys.executable, '-m', 'scripts.evaluation.mem_gallery_snapshot_runtime', 'export'],
        cwd=base.ROOT, check=True)


if __name__ == '__main__':
    main()
