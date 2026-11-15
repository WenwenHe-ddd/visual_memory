"""Official SimpleMem 0.3.0 on Mem-Gallery with the complete GPT-5 backbone.

Reuse verified GPT-5 history captions; freshly construct, plan, reflect and answer.
Preserve the official supplied query captions and original question-image bytes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from scripts.evaluation import mem_gallery_gpt5_gen_agents as ga
from scripts.evaluation import mem_gallery_gpt5_memoryos as mos
from scripts.evaluation import gallery_gpt5_transport as transport
from scripts.evaluation import mem_gallery_gpt5_image_formats as images
from scripts.evaluation import gpt5_text_memory_adapters as adapters
from scripts.evaluation.mem_gallery_simplemem_official import GalleryMemory
from scripts.memory import simplemem_official as official

ROOT, SOURCE = ga.ROOT, ga.DEFAULT_OUTPUT
OUTPUT = ROOT/'results/mem_gallery_gpt5_simplemem030_20261031'
REFERENCE = ROOT/'results/simplemem_official_qwen_memgallery_k10_rrf5_reflection_20261029/simplemem/protocol.json'
read, rows, sha, obj, dump, write_rows, freeze = ga.read, ga.rows, ga.sha, ga.obj, ga.dump, ga.write_rows, ga.freeze
verify = mos.verify


def prepare(output):
    if (output/'protocol.json').exists():
        return verify(output)
    parent=mos.verify_ga(SOURCE)
    reference=read(REFERENCE)
    upstream=official.verify_installation()
    assert upstream['version']=='0.3.0' and upstream==reference['upstream']
    output.mkdir(parents=True,exist_ok=True)
    names=('observations.json','inputs/questions.jsonl','inputs/gold.jsonl')
    for name in names:
        dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dest)
    qs=rows(output/'inputs/questions.jsonl');obs=read(output/'observations.json')
    assert len(qs)==len({q['id'] for q in qs})==1711 and sum(map(len,obs.values()))==3962
    assert all(set(q)=={'id','task','point','question','retrieval_query','query_image_paths','query_image_caption'} for q in qs)
    for q in qs:
        assert q['retrieval_query']==q['question']+("\nquestion's image:\nimage_caption: "+q['query_image_caption'] if q['query_image_caption'] else '')
    p={k:copy.deepcopy(reference[k]) for k in ('method_defaults','embedding_model_path','memory_max_new_tokens',
        'simplemem_backend','context_fusion','context_memory_limit','fusion_rank_constant','preserve_empty_context')}
    assert p['method_defaults']['SEMANTIC_TOP_K']==10 and p['method_defaults']['ENABLE_REFLECTION'] is True
    assert p['method_defaults']['MAX_REFLECTION_ROUNDS']==2 and p['context_memory_limit']==5
    p.update(schema='mem_gallery_gpt5_simplemem030_v1',mode='simplemem',method='SimpleMem 0.3.0',
        requested_model='gpt-5',expected_model=parent['expected_model'],endpoint=parent['endpoint'],
        seed=42,reasoning_effort='minimal',temperature_wire_policy='omitted',
        max_new_tokens=2048,image_detail='high',api_workers=8,task_workers=8,embedding_device='cpu',
        memory_sampling='Native requested temperatures recorded; GPT-5 wire omits temperature, seed42, minimal reasoning.',
        completion_budget_policy='max(2048,native requested cap); reasoning tokens included',
        tasks=parent['tasks'],questions=1711,memory_rounds=3962,context_budget=128000,
        upstream=upstream,reference_config=str(REFERENCE),reference_sha256=sha(REFERENCE),
        caption_source=str(SOURCE),caption_signature=parent['experiment_signature'],
        caption_policy='Reuse all 979 verified GPT-5 history captions (1003 original image occurrences).',
        image_input_policy='Original image bytes, MIME determined from signature; no resizing or transcoding.',
        query_policy=parent['query_policy'],reader_policy=parent['reader_policy'],
        reader_system_prompt=parent['reader_system_prompt'],task_instructions=parent['task_instructions'],
        context_policy=reference['retrieval_count_policy'],
        source_recall_policy='Unavailable without exact entry-to-fact attribution; construction windows are not source hits.',
        gold_policy='Gold copied and hashed only; labels first opened by independent audit after all answers are sealed.',
        failure_policy='Terminal API failures, refusals, empty or truncated responses stop the run; no model fallback.',
        generated_memory_reused=False,prior_answers_reused=False)
    paths={Path(__file__),ROOT/'scripts/analysis/audit_mem_gallery_gpt5_simplemem030.py',
        REFERENCE,SOURCE/'protocol.json',SOURCE/'audit.json',SOURCE/'independent_audit.json',
        SOURCE/'image_format_amendment.json',SOURCE/'caption_source/complete.json',SOURCE/'caption_source/captions.jsonl',
        *official.source_files(),*[output/name for name in names]}
    paths.update(f for f in Path(p['embedding_model_path']).rglob('*') if f.is_file())
    for module in tuple(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT/'scripts')) and str(f).endswith('.py'):paths.add(Path(f))
    hashes=dict(parent['source_sha256']);hashes.update({str(path):sha(path) for path in paths})
    snapshots={}
    for filename,digest in hashes.items():
        if Path(filename).suffix=='.py':
            dest=output/'source_snapshot'/(digest[:16]+'_'+Path(filename).name)
            dest.parent.mkdir(exist_ok=True);shutil.copyfile(filename,dest)
            snapshots[filename]=dict(path=str(dest),sha256=digest)
    freeze(output/'source_snapshot/manifest.json',snapshots)
    p['source_sha256']=hashes;p['experiment_signature']=obj(p)
    freeze(output/'protocol.json',p)
    print('Prepared GPT-5 SimpleMem 0.3.0: 1711 questions / 3962 rounds',flush=True)
    return p


class SimpleMem(GalleryMemory):
    def __init__(self,p,workdir):
        super().__init__('simplemem',p,workdir)
        self.core.llm_client.model='gpt-5'
        cfg=read(workdir/'resolved_config.json');cfg.update(llm_model='gpt-5',actual_model=p['expected_model'])
        dump(workdir/'resolved_config.json',cfg)


def memory_stage(output,env_file,task=None):
    p=verify(output);method=p['mode'];root=output/'memory';root.mkdir(exist_ok=True)
    obs,qs=read(output/'observations.json'),rows(output/'inputs/questions.jsonl')
    pending=[t for t in p['tasks'] if (task is None or task==t) and not (root/t/'complete.json').exists()]
    if pending:
        model=mos.Model(root,env_file,p['expected_model']);model.torch.set_num_threads(2)
        broker=adapters.make_broker(method,model,root,p)
        try:
            encoder=adapters.install_runtime(method,broker,p)
            dump(root/'generation_environment.json',model.environment|dict(embedding_device=str(encoder.device)))
            def one(name):
                broker.local.owner=name;dest=root/name;runtime=dest/'runtime'
                if runtime.exists():runtime.rename(dest/('runtime_incomplete_'+str(time.time_ns())))
                runtime.mkdir(parents=True)
                memory=SimpleMem(p,runtime)
                try:
                    for i,row in enumerate(obs[name],1):
                        if model.fatal_errors:raise RuntimeError('Terminal memory API failure')
                        memory.store(row)
                        if model.fatal_errors:raise RuntimeError('Native core swallowed a terminal API failure')
                        dump(dest/'progress.json',dict(phase='ingest',done=i,total=len(obs[name])))
                        if i%25==0:print(method,name,'ingest',i,'/',len(obs[name]),flush=True)
                    if method=='simplemem':memory.finalize()
                    selected=[q for q in qs if q['task']==name];retrieved=[]
                    for i,q in enumerate(selected,1):
                        context,detail=memory.recall(q['retrieval_query'])
                        if model.fatal_errors:raise RuntimeError('Terminal retrieval API failure')
                        r=dict(id=q['id'],task=name,retrieval_query=q['retrieval_query'],retrieved_context=context,
                            retrieval_detail=detail,experiment_signature=p['experiment_signature'])
                        retrieved.append(r)
                        dump(dest/'retrieval_checkpoints'/(obj(q['id'])+'.json'),r)
                        dump(dest/'progress.json',dict(phase='retrieval',done=i,total=len(selected)))
                    write_rows(dest/'retrieval.jsonl',retrieved);memory.export(dest)
                    files=[dest/'retrieval.jsonl',dest/'memory_stats.json',*runtime.rglob('*.json')]
                    if (dest/'final_memory.json').exists():files.append(dest/'final_memory.json')
                    freeze(dest/'complete.json',dict(experiment_signature=p['experiment_signature'],stored_rounds=memory.stats['stored_rounds'],
                        questions=len(selected),artifact_sha256={str(f):sha(f) for f in files}))
                    dump(dest/'progress.json',dict(phase='complete',done=len(selected),total=len(selected)))
                    print(method,name,'complete',flush=True)
                finally:
                    if method=='memoryos':memory.core.client.shutdown()
            errors=[]
            with ThreadPoolExecutor(max_workers=p['task_workers']) as pool:
                fs={pool.submit(one,t):t for t in pending}
                for f in as_completed(fs):
                    try:f.result()
                    except Exception:
                        errors.append(dict(task=fs[f],traceback=traceback.format_exc()));dump(root/'run_errors.json',errors)
                        print(errors[-1]['traceback'],flush=True)
            if errors:raise RuntimeError(str(len(errors))+' memory tasks failed; diagnostics retained')
        finally:broker.close()
    if task:return
    combined=[];artifacts={}
    for name in p['tasks']:
        done=read(root/name/'complete.json');assert done['experiment_signature']==p['experiment_signature']
        for filename,digest in done['artifact_sha256'].items():assert sha(filename)==digest,filename
        combined.extend(rows(root/name/'retrieval.jsonl'));artifacts.update(done['artifact_sha256'])
        artifacts[str(root/name/'complete.json')]=sha(root/name/'complete.json')
    rmap={r['id']:r for r in combined};assert len(combined)==len(rmap)==len(qs)==1711
    write_rows(root/'retrieval.jsonl',[rmap[q['id']] for q in qs])
    for f in [root/'retrieval.jsonl',*root.glob('calls/*.json'),*root.glob('*.jsonl'),root/'generation_environment.json']:
        artifacts[str(f)]=sha(f)
    freeze(root/'complete.json',dict(status='complete',questions=1711,artifact_sha256=artifacts,experiment_signature=p['experiment_signature']))


def answers(output,env_file):
    p=verify(output);root=output/'reader';model=transport.Model(root,env_file,p['expected_model'])
    if (root/'complete.json').exists():return
    memory_done=read(output/'memory/complete.json')
    for filename,digest in memory_done['artifact_sha256'].items():assert sha(filename)==digest,filename
    qs=rows(output/'inputs/questions.jsonl');rmap={r['id']:r for r in rows(output/'memory/retrieval.jsonl')}
    payloads=[ga.reader_payload(q,rmap[q['id']]['retrieved_context']) for q in qs]
    if (root/'manifest.jsonl').exists():assert rows(root/'manifest.jsonl')==payloads
    else:write_rows(root/'manifest.jsonl',payloads)
    def one(payload):
        path=root/'responses'/(obj(payload['record_id'])+'.json');wire=images.wire_for(payload,p['max_new_tokens'])
        if path.exists():
            saved=read(path);assert saved['payload']==payload and saved['experiment_signature']==p['experiment_signature'];out=saved['output']
        else:
            out=model.invoke(wire,'answer',payload['record_id'])
            saved=dict(payload=payload,output=out,experiment_signature=p['experiment_signature'])
            if out.get('status')!='provider_rejection' and (out['finish_reason']!='stop' or out['refusal'] or not out['raw_response'].strip()):
                dump(root/'invalid'/path.name,saved);raise RuntimeError('Invalid answer archived')
            dump(path,saved)
        assert out['request_sha256']==transport.request_sha256(wire)
        return dict(id=payload['record_id'],**out,experiment_signature=p['experiment_signature'])
    outputs={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,f in enumerate(as_completed([pool.submit(one,payload) for payload in payloads]),1):
            result=f.result();outputs[result['id']]=result
            dump(root/'progress.json',dict(done=i,total=1711))
            if i%100==0 or i==1711:print(p['mode'],'answers',i,'/1711',flush=True)
    write_rows(root/'responses.jsonl',[outputs[q['id']] for q in qs])
    files=[root/'manifest.jsonl',root/'responses.jsonl',root/'api_responses.jsonl',root/'api_attempts.jsonl',*root.glob('responses/*.json')]
    freeze(root/'complete.json',dict(status='complete',count=1711,artifact_sha256={str(f):sha(f) for f in files},experiment_signature=p['experiment_signature']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','memory','answers'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--env-file',type=Path,default=ga.DEFAULT_ENV)
    parser.add_argument('--task')
    args=parser.parse_args();output=args.output.resolve()
    status=output/'stages'/(args.stage+'.json')
    dump(status,dict(status='running',pid=os.getpid(),updated=time.time()))
    try:
        if args.stage=='prepare':prepare(output)
        elif args.stage=='memory':memory_stage(output,args.env_file,args.task)
        else:answers(output,args.env_file)
    except BaseException as exc:
        dump(status,dict(status='failed',error=str(exc),updated=time.time()));raise
    dump(status,dict(status='complete',updated=time.time()))


if __name__=='__main__':main()
