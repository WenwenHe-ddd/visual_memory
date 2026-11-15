"""Wait for the original pool, recover the diagnosed format error, finish."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
ENV = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')


def read(path):
    return json.loads(path.read_text())


def status(stage, state, **extras):
    record = dict(stage=stage, status=state, pid=os.getpid(), updated=datetime.now(timezone.utc).isoformat(), **extras)
    (OUTPUT/'continuation_execution.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


status('waiting_for_original_pool', 'running')
while read(OUTPUT/'execution.json')['status'] == 'running':
    time.sleep(5)
assert read(OUTPUT/'execution.json')['status'] == 'failed'
errors = read(OUTPUT/'memory/run_errors.json')
assert {e['task'] for e in errors} == {'Baking_Dessert_Daily_Life_Skill'}, 'New failures need diagnosis'
for stage, module, args in [
    ('format_recovery', 'scripts.evaluation.recover_gallery_gpt5_simplemem030', []),
    ('answers', 'scripts.evaluation.mem_gallery_gpt5_simplemem030', ['answers']),
    ('audit', 'scripts.analysis.audit_mem_gallery_gpt5_simplemem030', []),
]:
    command = [sys.executable, '-u', '-m', module, *args, '--output', str(OUTPUT)]
    status(stage, 'running', command=command)
    print(stage, 'started', flush=True)
    with (OUTPUT/(stage+'_continued.log')).open('a') as log:
        code = subprocess.call(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    if code:
        status(stage, 'failed', returncode=code)
        sys.exit(code)
    print(stage, 'complete', flush=True)
status('final_comparison', 'running')
command = [sys.executable, '-u', str(ROOT/'results/mem_gallery_simplemem030_backbones_20261031/finalize.py')]
with (OUTPUT/'final_comparison.log').open('a') as log:
    code = subprocess.call(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
if code:
    status('final_comparison', 'failed', returncode=code)
    sys.exit(code)
record = status('complete', 'complete')
(OUTPUT/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
print('Full GPT-5 experiment and comparison verified', flush=True)
