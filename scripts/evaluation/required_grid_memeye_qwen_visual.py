"""Fresh native MemEye Qwen visual memories, MCQ/Open readers and Judge42."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_memeye_qwen_caption import gpu
from scripts.evaluation.required_grid_gallery_caption import observed_gallery
OLD=c.ROOT/'scripts/evaluation/memeye_seed1024_visual.py'
FRESH=c.ROOT/'scripts/evaluation/memeye_seed1024_open.py'

def runtime(job,device):
    fresh=c.clone(FRESH,'scripts.evaluation.memeye_seed1024_open',job['seed'],(29,),register=True)
    value=c.clone(OLD,'grid_memeye_qwen_visual',job['seed'],(36,))
    args=SimpleNamespace(method=job['method'],output_root=Path(job['run_root'])/'visual_methods',device=device)
    return value,fresh,args

def stage(job,phase,device):
    value,fresh,args=runtime(job,device);value.patch_runtime_seed();value.prepare(args)
    root=args.output_root/job['method'];observed_gallery(job,root/'native_raw_batches')
    if phase=='mcq':return value.run_mcq(args)
    if phase=='open-prepare':return value.open_prepare(args)
    if phase=='open-answers':return value.open_run(args)
    if phase=='evaluation':
        value.analyze(args);fresh.judge(root/'open',job['method'],c.LABELS[job['method']],workers=job['workers'])
        cfg=value.settings(args)
        src=value.dialogue.output_dir(cfg) if job['method']=='clip_image_dialogue' else root/'online'
        raw=c.rows(src/'qa_responses.jsonl');gold={q['id']:q for q in c.rows(value.BENCHMARK)};mcq=[]
        for r in raw:
            choice=r.get('predicted_choice',r.get('answer'));choice=choice if choice in ('A','B','C','D') else None
            mcq.append(dict(r,id=r['id'],predicted_choice=choice,gold_choice=gold[r['id']]['gold_choice'],correct=choice==gold[r['id']]['gold_choice']))
        opath=root/'open'/('reader/responses.jsonl' if job['method']=='clip_image_dialogue' else 'responses.jsonl')
        opened=c.rows(opath)
        c.complete(job,mcq,[dict(r,model_output=r['raw_response']) for r in opened],root/'open/judge',
            [src/'qa_responses.jsonl',root/'coverage_audit.json',opath,root/'open/protocol.json',Path(__file__),OLD,FRESH],
            dict(native_full_visual_pipeline=True))
        return
    raise ValueError('Unknown stage')

def run(job,path):
    if job['dataset']!='memeye' or job['backbone']!='qwen':raise ValueError('Wrong Qwen visual handler')
    c.job_protocol(job,[Path(__file__),OLD,FRESH,c.ROOT/'scripts/evaluation/required_grid_gallery_caption.py'])
    with gpu(job) as device:
        c.freeze(Path(job['method_root'])/'runtime_device.json',dict(device=device))
        for phase in ('mcq','open-prepare','open-answers'):
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage',phase,'--device',device],check=True,cwd=c.ROOT)
    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage','evaluation','--device',device],check=True,cwd=c.ROOT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--device',default='cuda:0');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:
        v,f,args=runtime(j,a.device);v.prepare(args)
    elif a.stage:stage(j,a.stage,a.device)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
