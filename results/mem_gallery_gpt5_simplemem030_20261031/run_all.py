"""Execute the prepared GPT-5 SimpleMem run and its independent audit."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[1]
ENV=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
         TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
for stage in ('memory','answers','audit'):
    command=[sys.executable,'-u','-m']
    command+=['scripts.analysis.audit_mem_gallery_gpt5_simplemem030'] if stage=='audit' else ['scripts.evaluation.mem_gallery_gpt5_simplemem030',stage]
    command+=['--output',str(OUTPUT)]
    status=dict(stage=stage,status='running',pid=os.getpid(),updated=datetime.now(timezone.utc).isoformat(),command=command)
    (OUTPUT/'execution.json').write_text(json.dumps(status,indent=2)+'\n')
    print(stage,'started',flush=True)
    with (OUTPUT/(stage+'.log')).open('a') as log:
        code=subprocess.call(command,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    if code:
        status.update(status='failed',returncode=code)
        (OUTPUT/'execution.json').write_text(json.dumps(status,indent=2)+'\n')
        sys.exit(code)
    print(stage,'complete',flush=True)
status.update(status='complete',updated=datetime.now(timezone.utc).isoformat())
(OUTPUT/'execution.json').write_text(json.dumps(status,indent=2)+'\n')
