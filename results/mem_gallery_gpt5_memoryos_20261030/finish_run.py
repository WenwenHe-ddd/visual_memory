"""Continue the authorized experiment after its separate memory process finishes."""
import json
from pathlib import Path
import subprocess
import sys
import time

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
while True:
    status = json.loads((OUTPUT / 'execution_status.json').read_text())
    if status['status'] == 'stopped_before_completion':
        raise RuntimeError('Memory stage stopped: ' + status.get('error', 'unknown'))
    if status['status'] == 'stage_complete' and status['stage'] == 'memory':
        break
    time.sleep(5)
for label, command in [
    ('answers', [sys.executable, '-u', '-m', 'scripts.evaluation.mem_gallery_gpt5_memoryos', 'answers']),
    ('audit', [sys.executable, '-u', '-m', 'scripts.analysis.audit_mem_gallery_gpt5_memoryos']),
]:
    with (OUTPUT / (label + '.log')).open('a') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(label + ' failed; inspect ' + str(OUTPUT / (label + '.log')))
    print(label + ' completed', flush=True)
