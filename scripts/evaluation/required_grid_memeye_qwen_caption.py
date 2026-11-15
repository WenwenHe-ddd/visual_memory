"""Missing Qwen caption cells: fresh native graph, retrieval, MCQ/Open and Judge42."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import gc
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c

OLD=c.ROOT/'scripts/evaluation/memeye_seed1024_ours.py'
FRESH=c.ROOT/'scripts/evaluation/memeye_seed1024_open.py'
OPEN=c.ROOT/'scripts/evaluation/memeye_seed1024_ours_open.py'
CONDITIONS={'semrag':'baseline_question_only','ours_1_ours_2':'master_answer_query__l0p03_a1'}

def runtime(job):
    seed=job['seed'];root=Path(job['run_root'])
    replacement={'results/memeye_qwen_seed1024_full_20261105':str(root.relative_to(c.ROOT))}
    rebuilt=c.clone(OLD,'scripts.evaluation.memeye_seed1024_ours',seed,(46,),replacement,True)
    rebuilt.OUTPUT=root/'ours'
    fresh=c.clone(FRESH,'scripts.evaluation.memeye_seed1024_open',seed,(29,),replacement,True)
    opened=c.clone(OPEN,'scripts.evaluation.memeye_seed1024_ours_open',seed,
        (83,108,111,129,230,242,257,281),replacement,True)
    opened.rebuilt=rebuilt;opened.fresh=fresh;opened.OUTPUT=rebuilt.OUTPUT/'open'
    return rebuilt,fresh,opened

@contextmanager
def gpu(job):
    directory=c.RUN/'gpu';directory.mkdir(parents=True,exist_ok=True)
    assignment=Path(job['run_root'])/'qwen_gpu_assignment.json'
    while True:
        indices=[c.read(assignment)['index']] if assignment.exists() else [0,1]
        for idx in indices:
            stream=(directory/f'gpu{idx}.lock').open('a')
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:stream.close();continue
            free=int(subprocess.check_output(['nvidia-smi',f'--id={idx}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<55000:stream.close();continue
            with c.lock(Path(job['run_root'])/'gpu_assignment.lock'):
                if assignment.exists() and c.read(assignment)['index']!=idx:
                    stream.close();continue
                if not assignment.exists():c.freeze(assignment,{'index':idx,'minimum_free_mib':55000})
            try:yield f'cuda:{idx}'
            finally:stream.close()
            return
        print({'stage':'waiting_gpu','id':job['id']},flush=True);time.sleep(20)

def model(fresh,device,job):
    value=fresh.SeededMultimodalQwen(device,seed=job['seed'])
    original=value.generate
    def generate(*args,**kwargs):
        result=original(*args,**kwargs)
        actual=int(value.torch.initial_seed())
        if value.seed!=job['seed'] or actual!=job['seed']:
            raise ValueError('Actual Qwen generation RNG seed differs')
        return [dict(r,generation_seed=actual,seeded_transport=c.sha(FRESH)) for r in result]
    value.generate=generate
    return value

def common(job,rebuilt,fresh,opened,device):
    root=Path(job['run_root'])
    marker=root/'ours/grid_common_completion.json'
    with c.lock(root/'ours/grid_common.lock'):
        if marker.exists():
            result=c.read(marker);c.hashes(result['artifact_sha256']);return None
        protocol=rebuilt.prepare(rebuilt.OUTPUT,device)
        rebuilt.rebuild_memories(rebuilt.OUTPUT,protocol)
        rebuilt.retrieve(rebuilt.OUTPUT,protocol,device)
        opened.prepare(opened.OUTPUT)
        value=model(fresh,device,job)
        opened.candidates(opened.OUTPUT,device,model=value)
        opened.retrieve(opened.OUTPUT,'cpu')
        paths=[rebuilt.OUTPUT/n for n in ('protocol.json','memory_manifest.json','retrieval_audit.json','reader/qa_manifest.jsonl')]
        paths += [opened.OUTPUT/n for n in ('preparation_protocol.json','candidate_validation.json','candidates.jsonl','retrieval_validation.json')]
        c.freeze(marker,dict(status='complete',seed=job['seed'],artifact_sha256={str(p):c.sha(p) for p in paths}))
        return value

def run(job):
    if job['dataset']!='memeye' or job['backbone']!='qwen' or job['method'] not in CONDITIONS:
        raise ValueError('Wrong caption handler')
    c.job_protocol(job,[Path(__file__),Path(c.__file__),OLD,FRESH,OPEN])
    rebuilt,fresh,opened=runtime(job)
    output=Path(job['method_root']);native_method='semantic_rag' if job['method']=='semrag' else job['method']
    with gpu(job) as device:
        value=common(job,rebuilt,fresh,opened,device)
        records=[r for r in c.rows(rebuilt.OUTPUT/'reader/qa_manifest.jsonl') if r['condition']==CONDITIONS[job['method']]]
        if len(records)!=371:raise ValueError('MCQ request coverage differs')
        settings=c.read(rebuilt.OUTPUT/'protocol.json')['reader_settings']
        c.write_rows(output/'mcq_manifest.jsonl',records)
        result={r['id']:r for r in c.rows(output/'mcq_responses.jsonl')} if (output/'mcq_responses.jsonl').exists() else {}
        for q in records:
            if q['id'] in result:continue
            messages=[{'role':'system','content':settings['reader_system_prompt']},{'role':'user','content':
                [{'type':'image','path':i['path'],'sha256':i['sha256']} for i in q['query_images']]+[{'type':'text','text':q['reader_text']}]}]
            request=dict(messages=messages,seed=job['seed'],cap=16)
            path=output/'mcq_calls'/(c.digest(request)+'.json')
            if path.exists():
                call=c.read(path)
                if call['request']!=request:raise ValueError('MCQ cache differs')
                answer=call['output']
            else:
                if value is None:value=model(fresh,device,job)
                answer=value.generate([messages],16)[0]
                c.freeze(path,dict(request=request,output=answer))
            if answer['generation_seed']!=job['seed']:raise ValueError('MCQ raw seed differs')
            match=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',answer['raw_response'].strip(),re.I)
            choice=match.group(1).upper() if match else None
            result[q['id']]=dict(answer,id=q['id'],task=q['task'],question=q['question'],
                predicted_choice=choice,gold_choice=q['gold_choice'],correct=choice==q['gold_choice'],
                native_outcome='ok' if choice else 'native_invalid_choice',payload_sha256=q['payload_sha256'])
            # Append-only recovery retains the original response order.
            import json
            with (output/'mcq_responses.jsonl').open('a') as stream:
                stream.write(json.dumps(result[q['id']],ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
            if len(result)%20==0:print({'stage':'mcq','id':job['id'],'done':len(result)},flush=True)
        open_root=opened.OUTPUT/native_method
        if not (open_root/'reader/completion.json').exists() and value is None:value=model(fresh,device,job)
        fresh.run_manifest(open_root,native_method,device,model=value)
        fresh.collect(open_root)
        del value;gc.collect()
        import torch
        torch.cuda.empty_cache()
    fresh.judge(open_root,native_method,c.LABELS[job['method']],workers=job['workers'])
    native_open=c.rows(open_root/'reader/responses.jsonl')
    if any(r.get('generation_seed')!=job['seed'] or r['seed']!=job['seed'] for r in native_open):
        raise ValueError('Open actual RNG provenance differs')
    opened_records=[dict(r,model_output=r['raw_response']) for r in native_open]
    c.complete(job,[result[q['id']] for q in records],opened_records,open_root/'judge',
        [output/'mcq_responses.jsonl',open_root/'reader/responses.jsonl',open_root/'reader/completion.json',
         rebuilt.OUTPUT/'grid_common_completion.json',Path(__file__),OLD,FRESH,OPEN])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:
        rebuilt,fresh,opened=runtime(j);rebuilt.prepare(rebuilt.OUTPUT,'cuda:0')
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j)
