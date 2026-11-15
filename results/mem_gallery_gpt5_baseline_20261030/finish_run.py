import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
project=root.parents[1]
env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
for name, command in [
    ("answers", [sys.executable,"-u","-m","scripts.evaluation.mem_gallery_gpt5_baseline","answers","--output",str(root)]),
    ("audit", [sys.executable,"-u","-m","scripts.analysis.audit_mem_gallery_gpt5_baseline","--output",str(root)])
]:
    with (root/(name+".log")).open("a") as stream:
        subprocess.run(command,cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(name+" complete",flush=True)
