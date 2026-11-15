"""Export only after every original answer and raw Judge grade has been verified."""
from pathlib import Path
import fcntl
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path('/storage/hewenwen/VisualMemory')
RUN = Path(__file__).resolve().parent
EXPORT = ROOT / 'scripts/analysis/export_memeye_deepseek_open_llm_judge.py'
AUDIT = ROOT / 'scripts/analysis/audit_memeye_deepseek_open_llm_judge_report.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = (RUN / 'final_export.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [Path(__file__), EXPORT, AUDIT, RUN / 'methods_manifest.json',
               ROOT / 'scripts/analysis/memeye_deepseek_open_rejection_provenance.py',
               ROOT / 'scripts/evaluation/memeye_deepseek_open_reader_recovery.py']
    contract = {'source_sha256': {str(p): sha(p) for p in sources},
                'policy': 'Require all ten independently verified methods and 3710 original grades before export; verify exported Excel again.'}
    path = RUN / 'final_export_protocol.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Final export contract differs')
    path.write_text(json.dumps(contract, indent=2) + '\n')
    proof = RUN / 'export_audits/completion.json'
    while True:
        for p, expected in contract['source_sha256'].items():
            if sha(Path(p)) != expected:
                raise ValueError('Frozen final exporter changed: ' + p)
        if proof.exists():
            try:
                status = json.loads(proof.read_text())
            except json.JSONDecodeError:
                time.sleep(5)
                continue
            if status.get('status') == 'complete':
                if (status['methods'] != 10 or status['method_questions'] != 3710
                        or status['export_script_sha256'] != sha(EXPORT)
                        or status['manifest_sha256'] != sha(RUN / 'methods_manifest.json')):
                    raise ValueError('Per-method export proof differs')
                break
        time.sleep(15)
    subprocess.run([sys.executable, '-u', str(EXPORT)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-u', str(AUDIT)], cwd=ROOT, check=True)
    print(json.dumps({'status': 'complete', 'methods': 10, 'method_questions': 3710}), flush=True)


if __name__ == '__main__':
    main()
