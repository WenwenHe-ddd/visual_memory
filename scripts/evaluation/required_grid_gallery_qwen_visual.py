"""Original Gallery Qwen visual methods with new seeds and fixed raw captions."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_memeye_qwen_caption import gpu
OLD=c.ROOT/'scripts/evaluation/memgallery_seed1042_visual.py'

def runtime(job,device):
    replacements={'Use the fresh seed-1042 shared inputs, including historical captions.':'Use the fixed original Qwen caption bytes; fresh memory, retrieval and answers.',
        'seed-1042':f"seed-{job['seed']}"}
    value=c.clone(OLD,'grid_gallery_qwen_visual',job['seed'],(34,),replacements,old_seed=1042)
    args=SimpleNamespace(method=job['method'],run_root=Path(job['run_root']),device=device,limit_run=0)
    source=value.SOURCE_INPUTS
    def inputs(_args):
        names=('memories.jsonl','questions.jsonl','gold.jsonl','protocol.json','artifact_manifest.json')
        return source,value.read(source/'memories.jsonl'),value.read(source/'questions.jsonl'),{str(source/n):c.sha(source/n) for n in names}
    value.new_inputs=inputs
    value.patch_native_seed()
    return value,args

def prepare(job,value,args):
    source=value.SOURCE_INPUTS
    for name in ('memories.jsonl','questions.jsonl','gold.jsonl','protocol.json','artifact_manifest.json'):
        dst=args.run_root/'inputs'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if c.sha(dst)!=c.sha(source/name):raise ValueError('Original shared input bytes differ')
        else:shutil.copyfile(source/name,dst)
    target=value.out(args);target.mkdir(parents=True,exist_ok=True)
    if not (target/'source42_audit.json').exists():value.source_audit(args)
    if not (target/'prepared.json').exists():value.prepare(args)
    return value.verify(args)

def stage(job,phase,device):
    value,args=runtime(job,device);prepare(job,value,args)
    if phase=='native':
        if job['method']=='clip_image_dialogue':value.run_clip(args)
        else:value.build_hip(args);value.run_hip(args)
        return value.analyze(args)
    if phase!='evaluation':raise ValueError('Unknown stage')
    value.analyze(args);root=value.out(args)
    path=root/('qa_responses.jsonl' if job['method']=='clip_image_dialogue' else 'online/qa_responses.jsonl')
    raw=c.rows(path)
    responses=[dict(r,raw_response=r['answer']) if job['method']=='hippomm' else r for r in raw]
    from scripts.evaluation.required_grid_gallery_common import publish
    return publish(job,responses,[path,root/'stability_protocol.json',root/'completion.json',root/'summary.json',Path(__file__),OLD],
        dict(caption_source_run_seed=42,deterministic_clip_features_reused=True,old_llm_outputs_imported=0))

def run(job,path):
    if job['dataset']!='memgallery' or job['backbone']!='qwen':raise ValueError('Wrong Qwen Gallery visual handler')
    c.job_protocol(job,[Path(__file__),OLD,c.ROOT/'scripts/evaluation/required_grid_gallery_common.py'])
    with gpu(job) as device:
        c.freeze(Path(job['method_root'])/'runtime_device.json',dict(device=device))
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage','native','--device',device],check=True,cwd=c.ROOT)
    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage','evaluation','--device',device],check=True,cwd=c.ROOT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--device',default='cuda:1');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:
        v,args=runtime(j,a.device);prepare(j,v,args);print(dict(status='prepared',id=j['id'],model_calls=0))
    elif a.stage:stage(j,a.stage,a.device)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
