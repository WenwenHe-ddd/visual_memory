"""Original GPT-5 VisRAG and all-role HippoMM pipelines with fresh native seeds."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_api import Gateway,ENV
from scripts.evaluation.required_grid_hippo_chat import chat_class
from scripts.evaluation import required_grid_memeye_gpt_common as shared
OLD=c.ROOT/'scripts/evaluation/memeye_seed1024_visual.py'

def setup(job):
    from scripts.memory import hippomm_api as hip
    from scripts.evaluation import memeye_gpt5_open_visual_methods as opened
    from scripts.memory import hippomm as base
    Chat=chat_class(job);hip.transport.APIChat=Chat;opened.AuditedChat=Chat
    original=base.contract_at
    def contract(path,value):
        if Path(path).resolve().is_relative_to(Path(job['run_root'])):
            value=dict(value,seed=job['seed'],judge_seed=42,grid_wrapper_sha256=c.sha(__file__))
        return original(path,value)
    base.contract_at=contract
    root=Path(job['method_root'])/'native'
    cfg=SimpleNamespace(dataset='memeye',natural_root=c.ROOT/'results/natural_observation',
        benchmark_manifest=c.ROOT/'results/mcq_top15_caption_qa/qa_manifest.jsonl',task=None,question_ids=None,limit_questions=0,
        top_k=5,device='cpu',hippomm_backbone='gpt5',hippomm_api_workers=job['workers'],hippomm_api_env_file=ENV,
        hippomm_max_new_tokens=2048,hippomm_context_budget=120000,hippomm_stage='build',
        hippomm_gallery_inputs=c.ROOT/'results/mem_gallery_master_answer_20261021/inputs',
        output=root/'offline',memory_dir=root/'offline')
    helper=c.clone(OLD,'grid_gpt_visual_clip_prepare',job['seed'],(36,))
    args=SimpleNamespace(method=job['method'],output_root=Path(job['run_root'])/'gpt_visual_inputs',device='cpu')
    helper.verify_preflight=lambda _args:dict(status='grid_inputs_validated')
    return hip,opened,base,cfg,helper,args,root

def prepare(job):
    hip,opened,base,cfg,helper,args,root=setup(job)
    source,captions=hip.caption_source('gpt5')
    if len(captions)!=437:raise ValueError('Wrong native GPT HippoMM caption source')
    qs=opened.questions()
    if len(qs)!=371:raise ValueError('Wrong Open cohort')
    c.freeze(root/'input_protocol.json',dict(seed=job['seed'],questions=371,caption_source=str(source),
        caption_source_run_seed=42,caption_sha256=c.sha(source/'captions.jsonl'),
        native_methods='Original VisRAG prompts/image order or native HippoMM role pipeline',old_llm_outputs_imported=0))
    return hip,opened,base,cfg,helper,args,root

def clip_prepare(job,helper,args,root):
    config=helper.settings(args);source=helper.prepare_clip_mcq(args,config)
    _,manifest=helper.dialogue.verify(source)
    c.write_rows(root/'mcq_manifest.jsonl',manifest)
    helper.open_prepare(args)
    old=args.output_root/'clip_image_dialogue/open'
    c.write_rows(root/'open/reader_manifest.jsonl',c.rows(old/'reader_manifest.jsonl'))
    c.freeze(root/'open/protocol.json',dict(dataset='MemEye Open',method='clip_image_dialogue',seed=job['seed'],top_k=5,
        source_sha256={str(old/'reader_manifest.jsonl'):c.sha(old/'reader_manifest.jsonl'),str(source/'qa_manifest.jsonl'):c.sha(source/'qa_manifest.jsonl')}))

def mcq(job,root,manifest):
    g=Gateway(job);gold={q['id']:q for q in c.rows(c.ROOT/'results/mcq_top15_caption_qa/qa_manifest.jsonl')}
    def one(q):
        messages=copy.deepcopy(q['reader_messages'])
        for message in messages:
            if isinstance(message['content'],list):
                for part in message['content']:
                    if part['type']=='image':part['sha256']=c.sha(part['path'])
        wire=g.build_message_request(messages,16,0);out=g.call(root/'mcq_api',wire,attempts=3)
        match=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',out['raw_response'].strip(),re.I)
        choice=match.group(1).upper() if match else None
        return dict(id=q['id'],**out,predicted_choice=choice,gold_choice=gold[q['id']]['gold_choice'],correct=choice==gold[q['id']]['gold_choice'])
    with ThreadPoolExecutor(max_workers=job['workers']) as pool:values=list(pool.map(one,manifest))
    c.write_rows(root/'mcq_records.jsonl',values)

def hip_open(job,hip,opened,base,cfg,root,correct):
    memory=root/'offline';_,events,vectors,dialogues,images=hip.load_memory(memory,'gpt5')
    qs=opened.questions();encoder=base.ClipText('cpu')
    import threading
    mutex=threading.Lock()
    def encode(text):
        with mutex:return encoder(text)
    def one(index,q):
        row=opened.native_answer(root/'open',index,q,memory,events,vectors,dialogues,images,encode)
        return dict(row,seed=job['seed'],source_mcq_correct=correct)
    values={}
    with ThreadPoolExecutor(max_workers=job['workers']) as pool:
        for f in as_completed([pool.submit(one,i,q) for i,q in enumerate(qs,1)]):
            row=f.result();values[row['id']]=row;print(dict(stage='open',done=len(values),total=371),flush=True)
    c.write_rows(root/'open/native_responses.jsonl',[values[q['id']] for q in qs])

def stage(job,phase):
    hip,opened,base,cfg,helper,args,root=prepare(job)
    if phase=='prepare-retrieval':
        if job['method']=='clip_image_dialogue':clip_prepare(job,helper,args,root)
        else:hip.build(cfg);cfg.output=root/'online';hip.prepare(cfg)
        return
    if phase=='mcq':
        if job['method']=='clip_image_dialogue':return mcq(job,root,c.rows(root/'mcq_manifest.jsonl'))
        cfg.output=root/'online';hip.run(cfg);hip.analyze(cfg)
        gold={q['id']:q for q in c.rows(cfg.benchmark_manifest)}
        c.write_rows(root/'mcq_records.jsonl',[dict(r,predicted_choice=r['answer'] if r['answer'] in ('A','B','C','D') else None,
            gold_choice=gold[r['id']]['gold_choice'],correct=r['answer']==gold[r['id']]['gold_choice']) for r in c.rows(root/'online/qa_responses.jsonl')]);return
    mcq_values=c.rows(root/'mcq_records.jsonl');correct=sum(r['correct'] for r in mcq_values);target=root/'open'
    if phase=='open-answers':
        if job['method']=='clip_image_dialogue':return shared.answers(job,target,correct)
        hip_open(job,hip,opened,base,cfg,root,correct)
        qs=opened.questions();gmap={q['id']:q for q in c.rows(c.ROOT/'results/memeye_qwen_semrag_open_20261105/inputs/gold.jsonl')}
        rmap={r['id']:r for r in c.rows(target/'native_responses.jsonl')}
        c.write_rows(target/'judge_inputs.jsonl',[dict(id=q['id'],task=q['task'],point=q['point'],question=q['question'],
            ground_truth=gmap[q['id']]['answer'],model_output=rmap[q['id']]['raw_response'],answer_model='gpt-5',source_seed=job['seed']) for q in qs])
        c.freeze(target/'protocol.json',dict(dataset='MemEye Open',seed=job['seed'],method='hippomm',questions=371,
            memory_sha256=c.sha(root/'offline/memory.json'),source_mcq_correct=correct));return
    if phase=='judge':return shared.judge(job,target)
    if phase!='audit':raise ValueError('Unknown stage')
    opath=target/('reader/responses.jsonl' if job['method']=='clip_image_dialogue' else 'native_responses.jsonl')
    raw=c.rows(opath);calls=0
    for path in root.rglob('call_*.json'):
        item=c.read(path)
        if 'response' in item:
            if item['response']['seed']!=job['seed'] or item['response']['returned_model']!='gpt-5-2025-08-07':raise ValueError('Wrong HippoMM role seed/revision')
            calls+=1
    c.complete(job,mcq_values,[dict(r,model_output=r['raw_response']) for r in raw],target/'judge',
        [root/'mcq_records.jsonl',opath,target/'protocol.json',root/'input_protocol.json',Path(__file__)],
        dict(caption_source_run_seed=42,native_model_calls_checked=calls,old_llm_outputs_imported=0))

def run(job,path):
    if job['dataset']!='memeye' or job['backbone']!='gpt5':raise ValueError('Wrong GPT visual handler')
    c.job_protocol(job,[Path(__file__),OLD,Path(shared.__file__),c.ROOT/'scripts/evaluation/required_grid_hippo_chat.py'])
    for phase in ('prepare-retrieval','mcq','open-answers','judge','audit'):
        print(dict(id=job['id'],stage=phase),flush=True)
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage',phase],check=True,cwd=c.ROOT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:prepare(j);print(dict(status='prepared',id=j['id'],model_calls=0))
    elif a.stage:stage(j,a.stage)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
