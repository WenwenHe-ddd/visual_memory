from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
root=Path(__file__).resolve().parent
project=root.parents[1]
env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",PYTHONUNBUFFERED="1")
def stage(name):
    module="scripts.analysis.audit_mem_gallery_gpt5_caption_visual_master" if name=="audit" else "scripts.evaluation.mem_gallery_gpt5_caption_visual_master"
    command=[sys.executable,"-u","-m",module]+([] if name=="audit" else [name])+["--output",str(root)]
    with (root/(name+".log")).open("a") as stream:
        subprocess.run(command,cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(name+" complete",flush=True)
try:
    (root/"execution_status.json").write_text(json.dumps(dict(status="running",started=time.time()))+"\n")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(stage,s) for s in ("candidates","offline")]
        for f in futures:f.result()
    for s in ("retrieve","answers","audit"):stage(s)
    (root/"execution_status.json").write_text(json.dumps(dict(status="complete",finished=time.time()))+"\n")
except BaseException as error:
    (root/"execution_status.json").write_text(json.dumps(dict(status="stopped_before_completion",error=str(error),updated=time.time()))+"\n")
    raise
