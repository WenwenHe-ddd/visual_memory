"""Fresh caption-Ward GPT-5 memories and native MCQ/Open retrieval at .03/1."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import gc
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_api import Gateway
from scripts.evaluation import required_grid_memeye_gpt_common as shared
from scripts.memory import memeye_backbone_retrieval as native
from scripts.memory.memeye_backbone_long_text import encode_complete,POLICY
OLD=c.ROOT/'scripts/evaluation/memeye_deepseek_seed1024_ours.py'
OPEN=c.ROOT/'scripts/evaluation/memeye_deepseek_seed1024_ours_open.py'
SOURCE=c.ROOT/'results/memeye_gpt5_base_seed42_full_20261024'
OPEN_SOURCE=c.ROOT/'results/memeye_gpt5_open_20261105/caption_methods/inputs/questions.jsonl'
COND={'semrag':'baseline_question_only','ours_1_ours_2':'master_answer_query_zscore'}

def runtime(job):
    main=c.clone(OLD,'grid_gpt_caption_rank',job['seed'],(40,159,313,326,397,409))
    opened=c.clone(OPEN,'grid_gpt_open_rank',job['seed'],(70,94,117,130,232,256,278,295,329,331))
    main.MODEL='gpt-5';opened.main=main
    return main,opened

def prepare(job):
    root=Path(job['run_root'])/'caption_methods'
    if (root/'protocol.json').exists():
        p=c.read(root/'protocol.json');c.hashes(p['source_sha256']);return p
    ms,qs,model=native.attach_captions(c.rows(SOURCE/'inputs/source_memories.jsonl'),c.rows(SOURCE/'inputs/source_questions.jsonl'),c.rows(SOURCE/'captions.jsonl'))
    if model!='gpt-5-2025-08-07' or len(ms)!=438 or len(qs)!=371:raise ValueError('Wrong GPT caption corpus')
    oq=c.rows(OPEN_SOURCE);qmap={q['id']:q for q in qs}
    for q in oq:
        if (q['query_image_captions']!=qmap[q['id']]['query_image_captions'] or q['query_image_paths']!=qmap[q['id']]['query_image_paths']
            or set(q)&{'options','gold_choice','gold_ids','clue','answer'}):raise ValueError('GPT Open caption alignment changed')
    files=[Path(__file__),OLD,OPEN,Path(shared.__file__),Path(native.__file__),
        SOURCE/'protocol.json',SOURCE/'captions.jsonl',SOURCE/'inputs/source_memories.jsonl',SOURCE/'inputs/source_questions.jsonl',
        SOURCE/'inputs/protocol.json',OPEN_SOURCE]
    meta=c.read(SOURCE/'inputs/protocol.json')
    for b in meta['basis_by_domain'].values():files += [Path(b['path']),Path(b['protocol_path'])]
    p=dict(dataset='MemEye',model='gpt-5',expected_model=model,seed=job['seed'],caption_seed=None,caption_source_run_seed=42,
        judge_seed=42,top_k=5,**{'lambda':.03,'alpha':1.},long_text_policy=POLICY,
        source_sha256={str(f):c.sha(f) for f in files},historical_graphs_or_answers_reused=False)
    p['experiment_signature']=c.digest(p)
    c.write_rows(root/'inputs/memories.jsonl',ms);c.write_rows(root/'inputs/mcq_questions.jsonl',qs)
    c.write_rows(root/'inputs/open_questions.jsonl',oq);c.freeze(root/'inputs/encoding.json',meta);c.freeze(root/'protocol.json',p)
    return p

def build(job,root,p):
    if (root/'memory_manifest.json').exists():c.hashes(c.read(root/'memory_manifest.json')['artifact_sha256']);return
    import torch
    torch.manual_seed(job['seed']);torch.set_num_threads(4)
    meta=c.read(root/'inputs/encoding.json');ms=c.rows(root/'inputs/memories.jsonl')
    image=native.ImageEncoder(meta['image_encoder']['model_name'],revision=meta['image_encoder']['revision'],device='cpu',local_files_only=True)
    text=native.load_text_encoder(meta['text_encoder'],'cpu');artifacts={}
    for task in sorted({m['task'] for m in ms}):
        selected=[m for m in ms if m['task']==task]
        graph,sums,proof=native._build_domain(selected,root/'domains'/task,np.load(meta['basis_by_domain'][task]['path']),image,text,p['experiment_signature'])
        artifacts.update(proof['artifact_sha256']);print(dict(stage='build',task=task,leaves=len(selected),masters=len(graph['masters'])),flush=True)
    c.freeze(root/'memory_manifest.json',dict(status='complete',seed=job['seed'],leaves=438,artifact_sha256=artifacts))

def candidates(job,root,questions):
    from transformers import AutoTokenizer
    from scripts.evaluation import mem_gallery_master_reader as parser
    g=Gateway(job);tokenizer=AutoTokenizer.from_pretrained(str(parser.CLIP),local_files_only=True)
    def one(q):
        clean=dict(id=q['id'],question=q['question'],query_image_caption='\n'.join(q['query_image_captions']))
        for attempt in range(1,4):
            payload=dict(system=parser.CANDIDATE_SYSTEM,user_text=parser.candidate_user_text(clean,attempt),query_images=[])
            wire=g.build_request(payload,2048);out=g.call(root/'candidate_calls',wire,attempts=3)
            try:
                if out['finish_reason']!='stop' or out.get('refusal'):raise ValueError('Invalid completion')
                statements,lengths,fmt=parser.parse_candidates(out['raw_response'],tokenizer,return_format=True)
                return dict(id=q['id'],candidate_statements=statements,clip_token_lengths=lengths,format_attempt=attempt,
                    response_id=out['response_id'],request_sha256=c.digest(wire),seed=job['seed'])
            except ValueError as e:c.freeze(root/'candidate_errors'/(c.digest(wire)+'.json'),dict(error=str(e),raw=out))
        raise ValueError('Original three candidate-format attempts exhausted')
    if (root/'candidates.jsonl').exists():return {r['id']:r for r in c.rows(root/'candidates.jsonl')}
    values={}
    with ThreadPoolExecutor(max_workers=job['workers']) as pool:
        for f in as_completed([pool.submit(one,q) for q in questions]):
            r=f.result();values[r['id']]=r
    c.write_rows(root/'candidates.jsonl',[values[q['id']] for q in questions]);return values

def retrieve(job,root,p,kind):
    target=root/kind
    if (target/'retrieval_complete.json').exists():c.hashes(c.read(target/'retrieval_complete.json')['artifact_sha256']);return
    main,opened=runtime(job);ms=c.rows(root/'inputs/memories.jsonl');qs=c.rows(root/'inputs'/('mcq_questions.jsonl' if kind=='mcq' else 'open_questions.jsonl'))
    fresh=candidates(job,root,qs) if kind=='open' else None
    meta=c.read(root/'inputs/encoding.json');bge=native.load_text_encoder(meta['text_encoder'],'cpu')
    clip,tok,limit=native.load_clip_text_encoder(meta['image_encoder'],'cpu');results=[];manifest=[];audits=[]
    from scripts.evaluation.memeye_open_semrag_inputs import SYSTEM,payload_hash
    for task in sorted({m['task'] for m in ms}):
        memories=[m for m in ms if m['task']==task];questions=[q for q in qs if q['task']==task]
        graph=c.read(root/'domains'/task/'memory/graph.json');sums=np.load(root/'domains'/task/'memory/master_sums.npy')
        leaves,leaf_info=encode_complete(bge,[m['retrieval_text'] for m in memories],ids=[m['id'] for m in memories])
        statements=[s for q in questions for s in (list(fresh[q['id']]['candidate_statements']) if fresh else [q['candidate_statements'][letter] for letter in 'ABCD'])]
        textqueries=[t for i,q in enumerate(questions) for t in [q['retrieval_query'],*[q['retrieval_query']+'\nCandidate answer: '+s for s in statements[4*i:4*i+4]]]]
        qv,qinfo=encode_complete(bge,textqueries,meta['query_prefix']);qv=qv.reshape(len(questions),5,-1)
        cv,cinfo=native._clip_candidates(clip,tok,limit,statements);cv=cv.reshape(len(questions),4,-1)
        for i,q in enumerate(questions):
            if kind=='mcq':
                result=main.rank_four(q,memories,graph,sums,qv[i],leaves,cv[i]);results.append(result)
                for condition in result['conditions']:
                    texts={m['id']:m['retrieval_text'] for m in memories};ids=condition['top5_memory_ids']
                    context='\n\n'.join(f'Memory {j}: {texts[mid]}' for j,mid in enumerate(ids,1))
                    text='Retrieved dialogue and image descriptions (ranked by relevance):\n\n'+context+'\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(f'{l}. {q["options"][l]}' for l in 'ABCD')+'\n\nAnswer with one letter only.'
                    manifest.append(dict(id=q['id'],condition=condition['name'],top_memory_ids=ids,system=meta['reader_system_prompt'],user_text=text,query_images=q['query_images']))
            else:
                scores,details=opened.rank(q,memories,graph,sums,qv[i],leaves,cv[i]);audits.append(dict(id=q['id'],masters=details))
                for method,values in scores.items():
                    order=np.argsort(-values,kind='stable');ids=[memories[j]['id'] for j in order[:5]]
                    context='Retrieved dialogue and image descriptions (ranked by relevance):\n\n'+'\n\n'.join(f'Memory {j}: {memories[ix]["retrieval_text"]}' for j,ix in enumerate(order[:5],1))
                    text=context+'\n\nQuestion:\n'+q['question']+'\n\nAnswer concisely.'
                    manifest.append(dict(id=q['id'],method=method,task=task,point=q['point'],question=q['question'],query_image_paths=q['query_image_paths'],
                        top_memory_ids=ids,system=SYSTEM,context=context,user_text=text,payload_sha256=payload_hash(SYSTEM,text,q['query_image_paths'])))
        audits.append(dict(task=task,leaf=leaf_info,query=qinfo,candidate=cinfo));print(dict(stage=kind+'_retrieve',task=task),flush=True)
    c.write_rows(target/'manifest.jsonl',manifest);c.write_rows(target/'retrieval.jsonl',results if kind=='mcq' else audits)
    c.freeze(target/'retrieval_complete.json',dict(status='complete',artifact_sha256={str(target/n):c.sha(target/n) for n in ('manifest.jsonl','retrieval.jsonl')}))

def mcq(job,root):
    g=Gateway(job);method=job['method'];items=[r for r in c.rows(root/'mcq/manifest.jsonl') if r['condition']==COND[method]]
    gold={q['id']:q for q in c.rows(c.ROOT/'results/mcq_top15_caption_qa/qa_manifest.jsonl')}
    def one(r):
        wire=g.build_request(r,16);out=g.call(Path(job['method_root'])/'mcq_api',wire,attempts=3)
        match=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',out['raw_response'].strip(),re.I)
        choice=match.group(1).upper() if match else None
        return dict(id=r['id'],**out,predicted_choice=choice,gold_choice=gold[r['id']]['gold_choice'],correct=choice==gold[r['id']]['gold_choice'])
    with ThreadPoolExecutor(max_workers=job['workers']) as pool:values=list(pool.map(one,items))
    c.write_rows(Path(job['method_root'])/'native_mcq.jsonl',values)

def stage(job,phase):
    root=Path(job['run_root'])/'caption_methods';p=prepare(job);methodroot=Path(job['method_root']);opened=methodroot/'open'
    if phase=='memory-retrieval':
        with c.lock(root/'common.lock'):
            build(job,root,p);retrieve(job,root,p,'mcq');retrieve(job,root,p,'open')
        return
    if phase=='mcq':return mcq(job,root)
    values=c.rows(methodroot/'native_mcq.jsonl');correct=sum(r['correct'] for r in values)
    manifests=[{k:v for k,v in r.items() if k!='method'} for r in c.rows(root/'open/manifest.jsonl') if r['method']==job['method']]
    c.write_rows(opened/'reader_manifest.jsonl',manifests);c.freeze(opened/'protocol.json',dict(dataset='MemEye Open',method=job['method'],seed=job['seed'],top_k=5,
        source_sha256={str(root/'protocol.json'):c.sha(root/'protocol.json'),str(root/'open/manifest.jsonl'):c.sha(root/'open/manifest.jsonl')}))
    if phase=='open-answers':return shared.answers(job,opened,correct)
    if phase=='judge':return shared.judge(job,opened)
    if phase!='audit':raise ValueError('Unknown stage')
    responses=c.rows(opened/'reader/responses.jsonl')
    if any(r['seed']!=job['seed'] or r['returned_model']!='gpt-5-2025-08-07' for r in values+responses):raise ValueError('Wrong actual generation seed/revision')
    c.complete(job,values,[dict(r,model_output=r['raw_response']) for r in responses],opened/'judge',
        [methodroot/'native_mcq.jsonl',opened/'reader/responses.jsonl',root/'memory_manifest.json',root/'mcq/retrieval_complete.json',root/'open/retrieval_complete.json',Path(__file__)],
        dict(caption_seed=None,caption_source_run_seed=42))

def run(job,path):
    if job['dataset']!='memeye' or job['backbone']!='gpt5' or job['method'] not in COND:raise ValueError('Wrong GPT caption handler')
    c.job_protocol(job,[Path(__file__),OLD,OPEN,Path(shared.__file__),c.ROOT/'scripts/evaluation/required_grid_api.py'])
    for phase in ('memory-retrieval','mcq','open-answers','judge','audit'):
        print(dict(id=job['id'],stage=phase),flush=True)
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage',phase],check=True,cwd=c.ROOT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:prepare(j);runtime(j);print(dict(status='prepared',id=j['id'],model_calls=0))
    elif a.stage:stage(j,a.stage)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
