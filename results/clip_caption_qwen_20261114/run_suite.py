"""Resume both frozen caption-only Qwen evaluations through the public pipeline CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/storage/hewenwen/VisualMemory')
OUT=Path(__file__).resolve().parent
os.environ.update(PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
sys.path.insert(0,str(ROOT))
from scripts.memory.clip_caption import dump, sha

def main():
    commands=[]
    for dataset in ('memeye','mem_gallery'):
        directory=OUT/dataset
        deadline=time.monotonic()+1800
        while not (directory/'memory'/'memory.json').is_file():
            dump(OUT/'suite_progress.json',dict(status='waiting_for_offline_memory',dataset=dataset))
            if time.monotonic()>deadline:
                raise TimeoutError('Offline construction did not finish: '+dataset)
            time.sleep(10)
        command=[sys.executable,str(ROOT/'scripts/visual_memory_pipeline.py'),'online',
                 '--memory-path',str(directory/'memory'),'--dataset',dataset,
                 '--output',str(directory/'evaluation'),'--stage','all','--top-k','5','--device','cuda:0']
        commands.append(command)
        dump(OUT/'suite_progress.json',dict(status='running',dataset=dataset,command=command))
        with (OUT/(dataset+'_reader.log')).open('a') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    summaries={}
    hashes={}
    for dataset in ('memeye','mem_gallery'):
        directory=OUT/dataset/'evaluation'
        summaries[dataset]=json.loads((directory/'summary.json').read_text())
        audit=json.loads((directory/'audit.json').read_text())
        if summaries[dataset]['status']!='complete' or audit['status']!='passed':
            raise RuntimeError('Incomplete dataset '+dataset)
        for name in ('summary.json','audit.json','protocol.json'):
            hashes[str(directory/name)]=sha(directory/name)
    dump(OUT/'completion_manifest.json',dict(status='complete',completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         commands=commands,results=summaries,artifact_sha256=hashes))
    dump(OUT/'suite_progress.json',dict(status='complete',datasets=list(summaries)))

try:
    main()
except Exception as error:
    dump(OUT/'suite_failure.json',dict(error=repr(error),time=time.time()))
    raise
