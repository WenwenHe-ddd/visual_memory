"""Resume the entire four-method GPT-5 experiment from audited checkpoints."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'results/memeye_gpt5_seed42_text_methods_20261025'
METHODS=('gen_agents','memoryos','simplemem','a_mem')

def invoke(script,*args,log):
    with (OUTPUT/log).open('a') as f:
        return subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    code=invoke('scripts/evaluation/gpt5_seed42_text_captions.py',log='caption_run.log')
    if code:raise SystemExit('Caption generation stopped; see caption_source/attempts.jsonl and caption_run.log')
    code=invoke('scripts/evaluation/memeye_gpt5_text_methods.py','--stage','prepare',log='prepare.log')
    if code:raise SystemExit('Preparation failed; see prepare.log')
    jobs=[]
    try:
        for method in METHODS:
            f=(OUTPUT/(method+'_run.log')).open('a')
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/evaluation/memeye_gpt5_text_methods.py'),'--stage','run','--method',method],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            jobs.append((method,p,f))
        results={m:p.wait() for m,p,f in jobs}
    finally:
        for m,p,f in jobs:f.close()
    if any(results.values()):raise SystemExit('Some methods stopped: '+json.dumps(results))
    for script in ['audit_gpt5_text_methods.py','export_gpt5_text_methods.py']:
        if invoke('scripts/analysis/'+script,log='audit_export.log'):raise SystemExit('Audit/export failed; inspect audit_export.log')
    print('Complete: '+str(OUTPUT/'report.html'))

if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    status=OUTPUT/'execution_status.json'
    try:
        status.write_text(json.dumps(dict(status='running',requested_model='gpt-5',seed=42))+'\n')
        main()
    except BaseException as exc:
        status.write_text(json.dumps(dict(status='stopped_before_completion',requested_model='gpt-5',seed=42,error=str(exc),results_available=False))+'\n')
        raise
    else:
        status.write_text(json.dumps(dict(status='complete',requested_model='gpt-5',seed=42,completed_methods=4,questions_per_method=371,results_available=True))+'\n')
