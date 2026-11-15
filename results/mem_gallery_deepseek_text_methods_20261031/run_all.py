"""Separate processes preserve each native memory core's global adapters."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[1]
METHODS=('gen_agents','memoryos','simplemem')
ENV=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
         TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
         PYTHONUNBUFFERED='1')


def stamp(path,**record):
    path.write_text(json.dumps(dict(updated=datetime.now(timezone.utc).isoformat(),**record),indent=2)+'\n')


def run(method):
    folder=OUTPUT/method
    for stage in ('memory','answers','audit'):
        command=[sys.executable,'-u','-m']
        command+=['scripts.analysis.audit_mem_gallery_deepseek_text_methods'] if stage=='audit' else ['scripts.evaluation.mem_gallery_deepseek_text_methods',stage]
        command+=['--method',method,'--output',str(OUTPUT)]
        stamp(folder/'execution.json',status='running',stage=stage,command=command)
        print(method,stage,'started',flush=True)
        with (folder/(stage+'.log')).open('a') as log:
            code=subprocess.call(command,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
        if code:
            stamp(folder/'execution.json',status='failed',stage=stage,returncode=code)
            print(method,stage,'FAILED',code,flush=True)
            return False
        print(method,stage,'completed',flush=True)
    stamp(folder/'execution.json',status='complete')
    return True


if __name__=='__main__':
    stamp(OUTPUT/'execution.json',status='running',pid=os.getpid(),methods=METHODS,
          driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(run,m):m for m in METHODS}
        success={futures[f]:f.result() for f in as_completed(futures)}
    if all(success.values()):
        subprocess.run([sys.executable,'-u','-m','scripts.analysis.audit_mem_gallery_deepseek_text_methods',
                        '--output',str(OUTPUT)],cwd=ROOT,env=ENV,check=True)
    stamp(OUTPUT/'execution.json',status='complete' if all(success.values()) else 'failed',methods=success)
    sys.exit(0 if all(success.values()) else 1)
