"""Original Qwen Gallery CUDA build/retrieval and selected native reader branches."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_memeye_qwen_caption import gpu
from scripts.evaluation.required_grid_gallery_caption import observed_gallery
from scripts.evaluation import required_grid_gallery_common as shared
SOURCE=c.ROOT/'results/mem_gallery_master_answer_20261021/inputs'
ZSCORE=c.ROOT/'scripts/evaluation/mem_gallery_zscore.py'

def prepare(job):
    root=Path(job['run_root'])/'qwen_caption_methods';base=root/'base'
    files=[f for f in SOURCE.iterdir() if f.is_file()]
    for f in files:
        dest=base/'inputs'/f.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if c.sha(dest)!=c.sha(f):raise ValueError('Frozen original Gallery inputs differ')
        else:shutil.copyfile(f,dest)
    from scripts.data.mem_gallery_inputs import prepare as native_prepare
    native_prepare(base/'inputs')
    qs=c.rows(base/'inputs/questions.jsonl');shared.verify_inputs(qs)
    contract=dict(seed=job['seed'],questions=1711,top_k=5,caption_source_run_seed=42,
        memory_policy='Fresh original native caption-Ward build on CUDA; fixed original Qwen captions.',
        reader_policy='Original reader prompts, image processing and text16/image4 batches; cap256.',
        candidate_policy='Original Qwen gold-free candidates: batch32, cap384, three format attempts.',
        **{'lambda':.03,'alpha':1.},source_sha256={str(f):c.sha(f) for f in [Path(__file__),ZSCORE,*files]})
    c.freeze(root/'protocol.json',contract)
    # ZScore uses this file only as provenance for the shared final reader contract.
    # It is deliberately not named reader/protocol.json: no answers exist yet.
    c.freeze(base/'grid_reader_contract.json',dict(seed=job['seed'],system='Original shared Gallery reader',max_new_tokens=256,
        batch_size_text=16,batch_size_images=4,original_images=True,old_answers_reused=False))
    return root

def runtime(job):
    from scripts.memory import mem_gallery_master_answer as original
    from scripts.evaluation import mem_gallery_master_reader as reader
    zscore=c.clone(ZSCORE,'grid_gallery_qwen_zscore',replacements={'reader/protocol.json':'grid_reader_contract.json'})
    root=Path(job['run_root'])/'qwen_caption_methods';zscore.SOURCE=root/'base'
    original.DEFAULT_OUTPUT=root/'base';reader.SEED=job['seed']
    return original,reader,zscore

def audit_reader(job,source):
    from scripts.evaluation import mem_gallery_master_reader as reader
    responses=c.rows(source/'reader/responses.jsonl');manifest=c.rows(source/'reader/qa_manifest.jsonl')
    protocol=c.read(source/'reader/protocol.json')
    if protocol['contract']['seed']!=job['seed']:raise ValueError('Wrong final Qwen reader seed')
    m={r['record_id']:r for r in manifest};proof=set();batches=0
    for path in (source/'reader/raw_batches').glob('*.json'):
        raw=c.read(path);request=raw['request']
        if raw['actual_torch_seed']!=job['seed'] or request['seed']!=job['seed'] or request['cap']!=256:
            raise ValueError('Wrong actual raw-batch generation boundary')
        if len(request['rows'])!=len(raw['outputs']):raise ValueError('Raw batch alignment differs')
        for item,out in zip(request['rows'],raw['outputs']):
            proof.add((c.digest(item),out['raw_response']))
        batches+=1
    if len(responses)!=1711 or len({r['id'] for r in responses})!=1711:raise ValueError('Incomplete native Qwen reader')
    for row in responses:
        item=m[row['record_id']]
        candidate=dict(system=reader.READER_SYSTEM,user_text=item['reader_text'],query_image_paths=item['query_image_paths'])
        if (c.digest(candidate),row['raw_response']) not in proof:raise ValueError('Final response lacks actual new-seed raw generation')
        if row['payload_sha256']!=item['payload_sha256']:raise ValueError('Native final payload differs')
    c.dump(source/'reader/independent_audit.json',dict(status='passed',questions=1711,seed=job['seed'],raw_batches_checked=batches))
    return responses

def stage(job,phase,device):
    root=prepare(job);original,reader,zscore=runtime(job);base=root/'base'
    if phase=='common':
        original.build(base,device)
        observed_gallery(job,base/'candidate_raw_batches')
        reader.generate_candidates(base,device,32,3)
        original.retrieve(base,device,5)
        return
    selected='baseline_question_only' if job['method']=='semrag' else 'master_answer_query+zscore'
    source=base if job['method']=='semrag' else root/'zscore'
    if phase=='zscore':return zscore.prepare(source,device,.03,1.)
    if phase=='reader':
        reader.CONDITIONS=(selected,);build=reader.build_reader_manifest
        def manifest(questions,memories,retrieval,image_hash=reader.sha):
            values=copy.deepcopy(retrieval)
            for row in values:row['conditions']=[r for r in row['conditions'] if r['name']==selected]
            return build(questions,memories,values,image_hash=image_hash)
        reader.build_reader_manifest=manifest
        observed_gallery(job,source/'reader/raw_batches')
        reader.run_reader(source,device,16,4)
        audit_reader(job,source)
        return
    if phase!='evaluation':raise ValueError('Unknown native stage')
    rows=audit_reader(job,source)
    return shared.publish(job,[dict(r,seed=job['seed']) for r in rows],
        [root/'protocol.json',source/'retrieval_protocol.json',source/'retrieval_results.jsonl',source/'reader/protocol.json',
         source/'reader/responses.jsonl',source/'reader/qa_manifest.jsonl',source/'reader/independent_audit.json',base/'offline/protocol.json',Path(__file__)],
        dict(caption_source_run_seed=42,native_cuda_build_retrieval=True))

def run(job,path):
    if job['dataset']!='memgallery' or job['backbone']!='qwen':raise ValueError('Wrong Qwen Gallery caption handler')
    c.job_protocol(job,[Path(__file__),ZSCORE,c.ROOT/'scripts/memory/mem_gallery_master_answer.py',
        c.ROOT/'scripts/evaluation/mem_gallery_master_reader.py',Path(shared.__file__)])
    with gpu(job) as device:
        c.freeze(Path(job['method_root'])/'runtime_device.json',dict(device=device))
        root=Path(job['run_root'])/'qwen_caption_methods'
        with c.lock(root/'common.lock'):
            if not (root/'base/retrieval_protocol.json').exists():
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage','common','--device',device],check=True,cwd=c.ROOT)
        phases=(['zscore'] if job['method']!='semrag' else [])+['reader']
        for phase in phases:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage',phase,'--device',device],check=True,cwd=c.ROOT)
    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage','evaluation','--device',device],check=True,cwd=c.ROOT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--device',default='cuda:0');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:prepare(j);runtime(j);print(dict(status='prepared',id=j['id'],model_generations=0))
    elif a.stage:stage(j,a.stage,a.device)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
