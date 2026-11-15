"""Missing MemEye/Qwen text methods; preserve native stores and separate stages."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_memeye_qwen_caption import gpu,model as open_model
OLD=c.ROOT/'scripts/evaluation/memeye_seed1024_text_methods.py'
FRESH=c.ROOT/'scripts/evaluation/memeye_seed1024_open.py'

def runtime(job):
    root=Path(job['run_root']);seed=job['seed']
    replacement={'results/memeye_qwen_seed1024_full_20261105':str(root.relative_to(c.ROOT))}
    module=c.clone(OLD,'grid_qwen_text_native',seed,(26,),replacement)
    module.OUTPUT=root/'text_methods';module.OPEN_OUTPUT=root/'open/text_methods'
    fresh=c.clone(FRESH,'grid_qwen_text_open',seed,(29,),replacement)
    return module,fresh

def observed_factory(module,job):
    base=module.baseline;original=base.LocalMemoryQwen
    class Observed(original):
        def generate(self,*args,**kwargs):
            self.torch.manual_seed(job['seed'])
            result=super().generate(*args,**kwargs)
            actual=int(self.torch.initial_seed())
            if actual!=job['seed'] or self.protocol['seed']!=job['seed']:
                raise ValueError('Native Qwen actual memory/MCQ generation seed differs')
            return [dict(r,generation_seed=actual) for r in result]
    base.LocalMemoryQwen=Observed

def stage(job,phase,device):
    module,fresh=runtime(job);method=job['method'];root=Path(job['run_root'])
    if phase=='mcq':
        module.prepare(method,module.OUTPUT,device)
        observed_factory(module,job)
        return module.run_method(method,module.OUTPUT,device)
    if phase=='open-retrieval':
        observed_factory(module,job)
        return module.run_open(method,module.OUTPUT,device,module.OPEN_OUTPUT)
    opened=module.OPEN_OUTPUT/method
    if phase=='open-answers':
        value=open_model(fresh,device,job)
        fresh.run_manifest(opened,method,device,model=value)
        fresh.collect(opened)
        return
    if phase=='judge':
        fresh.judge(opened,method,c.LABELS[method],workers=job['workers']);return
    if phase!='audit':raise ValueError('Unknown native stage')
    source=module.target(method,module.OUTPUT)
    checked=module.audit(method,module.OUTPUT,device)
    if checked['status']!='complete':raise ValueError('Incomplete native MCQ')
    raw=c.rows(source/'qa_responses.jsonl')
    for call_path in (source/'calls').glob('*.json'):
        call=c.read(call_path)
        if call['request']['seed']!=job['seed'] or call['output'].get('generation_seed')!=job['seed']:
            raise ValueError('Native cached Qwen generation seed differs')
    raw_open=c.rows(opened/'reader/responses.jsonl')
    if any(r.get('generation_seed')!=job['seed'] or r['seed']!=job['seed'] for r in raw_open):
        raise ValueError('Open Qwen RNG evidence differs')
    normalized=[dict(r,native_outcome='ok' if r['predicted_choice'] else 'native_invalid_choice') for r in raw]
    c.complete(job,normalized,[dict(r,model_output=r['raw_response']) for r in raw_open],opened/'judge',
        [source/'qa_responses.jsonl',source/'summary.json',source/'completion_manifest.json',
         opened/'reader/responses.jsonl',opened/'reader/completion.json',Path(__file__),OLD,FRESH])

def run(job,job_path):
    if job['dataset']!='memeye' or job['backbone']!='qwen' or job['method'] not in ('gen_agents','memoryos','simplemem','a_mem'):
        raise ValueError('Wrong native text handler')
    c.job_protocol(job,[Path(__file__),Path(c.__file__),OLD,FRESH,
        c.ROOT/'scripts/evaluation/required_grid_memeye_qwen_caption.py'])
    with gpu(job) as device:
        c.freeze(Path(job['method_root'])/'runtime_device.json',{'device':device})
        for phase in ('mcq','open-retrieval','open-answers'):
            print({'id':job['id'],'stage':phase},flush=True)
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(job_path),
                '--stage',phase,'--device',device],cwd=c.ROOT,check=True)
    for phase in ('judge','audit'):
        print({'id':job['id'],'stage':phase},flush=True)
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(job_path),
            '--stage',phase,'--device',device],cwd=c.ROOT,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--device',default='cuda:0');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:
        module,fresh=runtime(j);print(module.prepare(j['method'],module.OUTPUT,a.device))
    elif a.stage:stage(j,a.stage,a.device)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
