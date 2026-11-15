"""Run the fixed DeepSeek comparison; stop and retain artifacts on any failure."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/storage/hewenwen/VisualMemory')
OUTPUT = Path(__file__).resolve().parent
STAGES = ('captions','candidates','offline','retrieve','answers','audit')
COMMANDS = {s: [sys.executable,'-u','-m',
    'scripts.analysis.audit_mem_gallery_deepseek_three_methods' if s=='audit'
    else 'scripts.evaluation.mem_gallery_deepseek_three_methods',
    *([] if s=='audit' else [s]), '--output',str(OUTPUT)] for s in STAGES}
(OUTPUT/'commands.json').write_text(json.dumps(COMMANDS,indent=2)+'\n')
env = dict(os.environ, OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')

def status(value):
    (OUTPUT/'execution_status.json').write_text(json.dumps(value,indent=2)+'\n')

def run(stage):
    path=OUTPUT/(stage+'.log')
    with path.open('a') as log:
        result=subprocess.run(COMMANDS[stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    print(stage, 'exit',result.returncode,flush=True)
    if result.returncode: raise RuntimeError(stage+' failed; inspect '+str(path))

try:
    status(dict(status='running',stage='captions_and_candidates',pid=os.getpid(),started=time.time()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(run,s) for s in ('captions','candidates')]
        for f in fs:f.result()
    for s in ('offline','retrieve','answers','audit'):
        status(dict(status='running',stage=s,pid=os.getpid(),updated=time.time()))
        run(s)
    assert json.loads((OUTPUT/'completion_manifest.json').read_text())['status']=='complete'
    status(dict(status='complete',updated=time.time()))
except BaseException as exc:
    status(dict(status='failed',error=str(exc),updated=time.time(),results_available=False))
    raise
