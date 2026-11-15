"""New Gallery seeds using the frozen native per-method implementations."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_memeye_qwen_caption import gpu
OLD=c.ROOT/'scripts/evaluation/memgallery_seed1042_text_methods.py'

def runtime(job):
    helper=c.clone(OLD,'grid_gallery_qwen_text',job['seed'],(34,),old_seed=1042)
    helper.RUN=Path(job['run_root'])
    ctx=helper.load_native(job['method'])
    return helper,ctx

def prepare(job,helper,ctx,device):
    method=job['method'];run=Path(job['run_root']);output=run/'text_methods';root=output/method
    if (root/'protocol.json').exists():return helper.verify(ctx,method,run,device)
    reference=ctx.reference/method;p=copy.deepcopy(c.read(reference/'protocol.json'))
    if p['seed']!=42:raise ValueError('Expected original Qwen seed42 inputs')
    questions=c.rows(ctx.reference/'inputs/questions.jsonl');observations=c.read(ctx.reference/'observations.json')
    from scripts.evaluation.required_grid_gallery_common import verify_inputs
    verify_inputs(questions,observations)
    for key in ('experiment_signature','input_sha256'):p.pop(key,None)
    p.update(seed=job['seed'],reader_device=device,reader_model_path=str(ctx.gallery.MODEL),
        study='Required seed grid with fixed original Qwen captions and fresh native memory/retrieval/reader',
        questions_sha256=ctx.base.obj(questions),observations_sha256=ctx.base.obj(observations),
        original_baseline_root=str(run),no_external_api=True,
        caption_policy='Original model-matched seed42 history captions, verbatim; no regeneration.',
        caption_source_run_seed=42,query_caption_policy='Original official question image captions, verbatim.',
        stability_replication=dict(reference_root=str(reference),reference_seed=42,seed=job['seed'],
            reference_correct=helper.EXPECTED_CORRECT[method],judge_seed=42,
            construction_policy='Fresh native stores; no old memory/calls/readers imported',
            caption_policy='Fixed original seed42 cached captions',
            generation_seed_evidence='Observed actual Torch and native categorical sampler seed; durable ordered raw generation batches'))
    files=[Path(__file__),OLD,reference/'protocol.json',ctx.reference/'source_snapshot/manifest.json',
           ctx.reference/'observations.json',ctx.reference/'inputs/questions.jsonl',ctx.reference/'inputs/gold.jsonl']
    files += [Path(v['path']) for v in ctx.entries.values()]
    p['input_sha256']={str(f.resolve()):c.sha(f) for f in set(files)}
    p['snapshot_sources']=copy.deepcopy(ctx.entries);p['experiment_signature']=helper.signature(p)
    c.freeze(output/'observations.json',observations)
    c.write_rows(output/'inputs/questions.jsonl',questions)
    c.write_rows(output/'inputs/gold.jsonl',c.rows(ctx.reference/'inputs/gold.jsonl'))
    for name in ('questions.jsonl','gold.jsonl'):
        c.write_rows(run/'inputs'/name,c.rows(ctx.reference/'inputs'/name))
    c.freeze(root/'protocol.json',p);c.freeze(root/'source_snapshot_manifest.json',ctx.entries)
    return helper.verify(ctx,method,run,device)

def stage(job,phase,device):
    helper,ctx=runtime(job);run=Path(job['run_root']);method=job['method'];root=run/'text_methods'/method
    p=prepare(job,helper,ctx,device)
    if phase=='native':
        helper.install_runtime(ctx,method,run,p,device)
        ctx.base.run(run/'text_methods',method,device)
        helper.export(ctx,method,run,device)
        return
    if phase!='evaluation':raise ValueError('Unknown stage')
    helper.audit(ctx,method,run,device)
    from scripts.evaluation.required_grid_gallery_common import publish
    return publish(job,c.rows(root/'reader/responses.jsonl'),
        [root/n for n in ('protocol.json','reader/responses.jsonl','reader/manifest.jsonl','independent_audit.json','completion_manifest.json')],
        dict(caption_seed=42,caption_source_run_seed=42,original_native_snapshot=True))

def run(job,path):
    if job['dataset']!='memgallery' or job['backbone']!='qwen':raise ValueError('Wrong native Gallery handler')
    c.job_protocol(job,[Path(__file__),OLD,Path(c.__file__),c.ROOT/'scripts/evaluation/required_grid_gallery_common.py'])
    with gpu(job) as device:
        c.freeze(Path(job['method_root'])/'runtime_device.json',dict(device=device))
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage','native','--device',device],check=True,cwd=c.ROOT)
    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage','evaluation','--device',device],check=True,cwd=c.ROOT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--device',default='cuda:1');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:
        helper,ctx=runtime(j);prepare(j,helper,ctx,a.device);print(dict(status='prepared',id=j['id'],model_calls=0))
    elif a.stage:stage(j,a.stage,a.device)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
