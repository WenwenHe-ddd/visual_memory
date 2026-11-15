import datetime, json, os, subprocess, sys
from pathlib import Path
ROOT=Path('/storage/hewenwen/VisualMemory')
OUT=Path(__file__).resolve().parent

def state(stage, **kwargs):
    value=dict(stage=stage,pid=os.getpid(),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs)
    target=OUT/'launch.json'
    tmp=target.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(target)

for stage in ['probe','run']:
    state(stage)
    with (OUT/(stage+'_authorized.log')).open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/evaluation/m2a_deepseek_suite.py'),stage,'--output',str(OUT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state(stage,child_pid=process.pid)
        rc=process.wait()
    if rc:
        state('failed',failed_stage=stage,returncode=rc)
        sys.exit(rc)
state('finished')
