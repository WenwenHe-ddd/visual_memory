"""Native Gallery VisRAG images/dialogue and provider-specific HippoMM roles."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation import required_grid_gallery_common as shared
from scripts.evaluation.required_grid_api import Gateway,ENV
from scripts.evaluation.required_grid_hippo_chat import chat_class
HELPER=c.ROOT/'scripts/evaluation/memgallery_seed1042_visual.py'
SOURCE=c.ROOT/'results/mem_gallery_master_answer_20261021/inputs'

def setup(job):
    from scripts.memory import hippomm_gallery_api as hip
    original=hip.base.contract_at
    def contract(path,value):
        if Path(path).resolve().is_relative_to(Path(job['run_root'])):
            value=dict(value,seed=job['seed'],judge_seed=42,grid_wrapper_sha256=c.sha(__file__))
        return original(path,value)
    hip.base.contract_at=contract;hip.transport.APIChat=chat_class(job)
    root=Path(job['method_root'])/'native'
    cfg=SimpleNamespace(dataset='mem_gallery',task=None,question_ids=None,limit_questions=0,device='cpu',top_k=5,
        hippomm_gallery_inputs=SOURCE,hippomm_backbone=job['backbone'],hippomm_api_workers=job['workers'],hippomm_api_env_file=ENV,
        hippomm_max_new_tokens=2048,hippomm_context_budget=120000,hippomm_stage='build',output=root/'offline',memory_dir=root/'offline')
    return hip,cfg,root

def prepare(job):
    hip,cfg,root=setup(job);g=Gateway(job)
    qs=c.rows(SOURCE/'questions.jsonl');shared.verify_inputs(qs)
    if job['method']=='hippomm':
        folder,cap=hip.caption_source(job['backbone'])
        caption=dict(source=str(folder),captions=979,sha256=c.sha(folder/'captions.jsonl'),caption_source_run_seed=42)
    else:caption=dict(policy='No captions are used; original images and source dialogue only')
    c.freeze(root/'input_protocol.json',dict(dataset='MemGallery',seed=job['seed'],method=job['method'],expected_model=g.expected,
        questions=1711,caption_policy=caption,old_llm_outputs_imported=0,
        source_sha256={str(SOURCE/n):c.sha(SOURCE/n) for n in ('questions.jsonl','memories.jsonl','protocol.json')}))
    return hip,cfg,root

def clip_prepare(job,root):
    value=c.clone(HELPER,'grid_gallery_clip_retrieval',job['seed'],(34,),old_seed=1042)
    args=SimpleNamespace(method='clip_image_dialogue',run_root=Path(job['run_root'])/'api_visual_retrieval',device='cpu')
    if not (value.out(args)/'protocol.json').exists():
        value.prepare_clip(args,c.rows(SOURCE/'memories.jsonl'),c.rows(SOURCE/'questions.jsonl'),SOURCE,
            {str(SOURCE/n):c.sha(SOURCE/n) for n in ('questions.jsonl','memories.jsonl','protocol.json')})
    source=value.out(args);p,manifest,old=value.clip.verify(source)
    if old:raise ValueError('Retrieval-only stage unexpectedly contains generated answers')
    c.write_rows(root/'reader_manifest.jsonl',manifest)
    c.freeze(root/'retrieval_audit.json',dict(status='passed',seed=job['seed'],questions=1711,rankings_and_payloads_equal_native_source=True,
        five_images_per_context=all(len(q['retrieved_images'])==5 for q in manifest),
        source_sha256={str(source/n):c.sha(source/n) for n in ('protocol.json','qa_manifest.jsonl','image_candidates.jsonl')}))

def clip_answers(job,root):
    from scripts.evaluation import mem_gallery_clip_image_dialogue_api as native
    g=Gateway(job);manifest=c.rows(root/'reader_manifest.jsonl')
    def one(q):
        body=g.wire_from_native(native.build_request(q['reader_messages'],q['reader_image_hashes'],g.MODEL))
        out=g.call(root/'wire_cache',body,attempts=3,allow_rejection=job['backbone']=='deepseek')
        return dict(id=q['id'],**out,payload_sha256=q['payload_sha256'])
    values={}
    with ThreadPoolExecutor(max_workers=job['workers']) as pool:
        for future in as_completed([pool.submit(one,q) for q in manifest]):
            r=future.result();values[r['id']]=r
            if len(values)%100==0 or len(values)==1711:print(dict(stage='reader',done=len(values),total=1711),flush=True)
    c.write_rows(root/'reader_responses.jsonl',[values[q['id']] for q in manifest])

def stage(job,phase):
    hip,cfg,root=prepare(job)
    if phase=='memory':
        if job['method']=='hippomm':hip.build(cfg)
        else:clip_prepare(job,root)
        return
    if phase=='answers':
        if job['method']=='clip_image_dialogue':return clip_answers(job,root)
        cfg.output=root/'online';hip.prepare(cfg);hip.run(cfg);hip.analyze(cfg)
        raw=c.rows(root/'online/qa_responses.jsonl')
        c.write_rows(root/'reader_responses.jsonl',[dict(r,raw_response=r['answer'],seed=job['seed']) for r in raw]);return
    if phase!='evaluation':raise ValueError('Unknown stage')
    values=c.rows(root/'reader_responses.jsonl');g=Gateway(job)
    if len(values)!=1711 or any(r['seed']!=job['seed'] for r in values):raise ValueError('Wrong native cohort/seed')
    count=0
    if job['method']=='hippomm':
        cfg.output=root/'online';hip.analyze(cfg)
        for path in root.rglob('call_*.json'):
            item=c.read(path)
            if 'response' not in item:continue
            out=item['response']
            if out['seed']!=job['seed'] or (out.get('status')!='provider_rejection' and out['returned_model']!=g.expected):raise ValueError('Wrong HippoMM actual role seed/model')
            count+=1
        if not count:raise ValueError('No native model provenance')
        evidence=[root/'offline/memory.json',root/'offline/events.jsonl',root/'online/qa_responses.jsonl',root/'online/summary.json']
    else:
        for r in values:
            if r.get('status')!='provider_rejection' and r['returned_model']!=g.expected:raise ValueError('Wrong VisRAG model')
        evidence=[root/'retrieval_audit.json',root/'reader_manifest.jsonl']
    return shared.publish(job,values,[*evidence,root/'reader_responses.jsonl',root/'input_protocol.json',Path(__file__)],
        dict(caption_source_run_seed=42,native_model_calls_checked=count,
            native_reader_cap=2048 if job['method']=='hippomm' or job['backbone']=='gpt5' else 256))

def run(job,path):
    if job['dataset']!='memgallery' or job['backbone'] not in ('gpt5','deepseek'):raise ValueError('Wrong API Gallery visual handler')
    c.job_protocol(job,[Path(__file__),HELPER,Path(shared.__file__),c.ROOT/'scripts/evaluation/required_grid_hippo_chat.py'])
    for phase in ('memory','answers','evaluation'):
        print(dict(id=job['id'],stage=phase),flush=True)
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage',phase],check=True,cwd=c.ROOT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:prepare(j);print(dict(status='prepared',id=j['id'],model_calls=0))
    elif a.stage:stage(j,a.stage)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
