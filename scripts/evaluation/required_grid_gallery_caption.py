"""Native Natural/visual-master Gallery retrieval on frozen captions, fresh seeds."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation import required_grid_gallery_common as shared
from scripts.evaluation.required_grid_api import Gateway,APIModel,ENV
from scripts.evaluation.required_grid_memeye_qwen_caption import gpu
BASE=c.ROOT/'scripts/evaluation/mem_gallery_gpt5_baseline.py'
MASTER=c.ROOT/'scripts/evaluation/mem_gallery_gpt5_caption_visual_master.py'
SOURCES=dict(qwen=c.ROOT/'results/mem_gallery_master_answer_20261021',
    gpt5=c.ROOT/'results/mem_gallery_gpt5_baseline_20261030',deepseek=c.ROOT/'results/mem_gallery_deepseek_v41flash_20261030')

def runtime(job):
    base=c.clone(BASE,'grid_gallery_natural');master=c.clone(MASTER,'grid_gallery_master')
    base.verify=shared.verify_protocol;master.verify=shared.verify_protocol;master.base=base
    master.SOURCE=Path(job['run_root'])/'caption_methods/natural'
    if job['backbone']!='qwen':
        g=Gateway(job)
        if job['backbone']=='deepseek':
            from scripts.evaluation import gallery_deepseek_bounded_images as images
        else:
            from scripts.evaluation import mem_gallery_gpt5_image_formats as images
        original_wire=images.wire_for
        class Images:
            @staticmethod
            def wire_for(payload,cap=2048):return g.wire_from_native(original_wire(payload,cap))
        from types import SimpleNamespace
        master.images=Images
        master.transport=SimpleNamespace(Model=lambda root,env,expected:APIModel(root,job,job['workers'],expected))
    return base,master

def prepare(job):
    group=Path(job['run_root'])/'caption_methods';source=SOURCES[job['backbone']]
    ref=c.read(c.ROOT/'results/mem_gallery_gpt5_caption_visual_master_20261030/protocol.json')
    meta=c.read(c.ROOT/'results/mem_gallery_master_answer_20261021/offline/protocol.json')
    if job['backbone']=='qwen':memories=c.rows(source/'inputs/memories.jsonl')
    else:memories=c.rows(source/'memories.jsonl')
    qs=c.rows(source/'inputs/questions.jsonl');shared.verify_inputs(qs)
    if len(memories)!=3962 or sum(m['is_visual'] for m in memories)!=1003:raise ValueError('Wrong frozen raw round/caption inputs')
    if job['backbone']=='qwen':
        parent_path=c.ROOT/'results/mem_gallery_master_answer_20261021/reader/protocol.json'
        expected='Qwen3-VL-8B-Instruct'
    else:
        parent_path=source/'protocol.json';expected=Gateway(job).expected
    original={m['id']:m for m in c.rows(c.ROOT/'results/mem_gallery_master_answer_20261021/inputs/memories.jsonl')}
    historical=[dict(id=m['id'],node_id=m['id']+'#image1',task=m['task'],path=original[m['id']]['image_path'],sha256=original[m['id']]['image_sha256']) for m in memories if m['is_visual']]
    paths=[Path(__file__),BASE,MASTER,Path(shared.__file__),source/('inputs/memories.jsonl' if job['backbone']=='qwen' else 'memories.jsonl'),
        source/'inputs/questions.jsonl',source/'inputs/gold.jsonl',parent_path,
        c.ROOT/'scripts/evaluation/required_grid_api.py',c.ROOT/'scripts/evaluation/mem_gallery_zscore.py',
        c.ROOT/'scripts/memory/mem_gallery_master_answer.py',c.ROOT/'scripts/graph/caption_ward_visual.py']
    if job['backbone']=='qwen':paths.append(c.ROOT/'scripts/evaluation/mem_gallery_master_reader.py')
    for name in ('natural','master'):
        root=group/name
        if (root/'protocol.json').exists():shared.verify_protocol(root);continue
        p=dict(ref);p.pop('experiment_signature',None);p.pop('source_sha256',None)
        p.update(seed=job['seed'],expected_model=expected,requested_model=c.MODELS[job['backbone']],
            max_new_tokens=256 if job['backbone']=='qwen' else 2048,candidate_max_tokens=384 if job['backbone']=='qwen' else 2048,
            api_workers=job['workers'],text_encoder=meta['text_encoder'],image_encoder=meta['image_encoder'],query_prefix=meta['query_prefix'],
            caption_policy='Original same-backbone caption bytes fixed, with complete source dialogue.',caption_source_run_seed=42,
            memory_policy='Fresh chronological native caption-Ward build; original fixed external 2D basis.',
            study='Requested full seed grid with fresh native memory/retrieval/reader and Judge42',
            source_sha256={str(f):c.sha(f) for f in paths})
        p['experiment_signature']=c.digest(p)
        c.write_rows(root/'memories.jsonl',memories);c.write_rows(root/'inputs/questions.jsonl',qs)
        c.write_rows(root/'inputs/gold.jsonl',c.rows(source/'inputs/gold.jsonl'))
        c.write_rows(root/'inputs/sanitized_candidate_inputs.jsonl',[{k:q[k] for k in ('id','question','query_image_caption')} for q in qs])
        c.write_rows(root/'history_image_inputs.jsonl',historical);c.freeze(root/'protocol.json',p)
    return group

def observed_gallery(job,rawroot):
    from scripts.evaluation import mem_gallery_master_reader as native
    native.SEED=job['seed'];load=native.load_runtime;generate=native.generate_batch
    def loaded(device):
        runtime=load(device);processor,model,env=runtime;actual=model.generate
        def checked(*args,**kw):
            import torch
            if int(torch.initial_seed())!=job['seed'] or kw.get('do_sample') is not False:raise ValueError('Wrong actual Qwen sampler boundary')
            return actual(*args,**kw)
        model.generate=checked;return processor,model,dict(env,seed=job['seed'])
    def generated(runtime,rows,cap):
        request=dict(rows=rows,cap=cap,seed=job['seed']);path=rawroot/(c.digest(request)+'.json')
        if path.exists():
            saved=c.read(path)
            if saved['request']!=request:raise ValueError('Raw native batch mismatch')
            return saved['outputs']
        out=generate(runtime,rows,cap)
        import torch
        actual=int(torch.initial_seed())
        if actual!=job['seed']:raise ValueError('Wrong Qwen RNG seed')
        out=[dict(r,seed=actual,generation_seed=actual) for r in out]
        c.freeze(path,dict(request=request,outputs=out,actual_torch_seed=actual))
        return out
    native.load_runtime=loaded;native.generate_batch=generated
    return native

def qwen_answers(job,questions,retrieval,root,device):
    from scripts.evaluation import mem_gallery_qwen_text_methods as native_text
    native=observed_gallery(job,root/'raw_batches');rmap={r['id']:r for r in retrieval}
    manifest=[dict(id=q['id'],task=q['task'],point=q['point'],system=native.READER_SYSTEM,
        user_text=native_text.reader_text(q,rmap[q['id']]['retrieved_context']),query_image_paths=q['query_image_paths']) for q in questions]
    c.write_rows(root/'manifest.jsonl',manifest)
    outputs={r['id']:r for r in c.rows(root/'responses.jsonl')} if (root/'responses.jsonl').exists() else {}
    if len(outputs)==1711:return list(outputs.values())
    runtime=native.load_runtime(device)
    for visual in (False,True):
        selected=[r for r in manifest if bool(r['query_image_paths'])==visual and r['id'] not in outputs]
        size=4 if visual else 16
        for start in range(0,len(selected),size):
            batch=selected[start:start+size];results=native.generate_batch(runtime,batch,256)
            for item,out in zip(batch,results):outputs[item['id']]=dict(id=item['id'],payload_sha256=c.digest(item),**out)
            c.dump(root/'responses.jsonl.checkpoint.json',dict(done=len(outputs),total=1711))
            # Only this raw-batch-backed checkpoint is mutable; final records are sealed once.
            c.dump(root/'checkpoint.json',outputs)
            print(dict(stage='reader',done=len(outputs),total=1711),flush=True)
    values=[outputs[q['id']] for q in questions];c.write_rows(root/'responses.jsonl',values);return values

def stage(job,phase,device):
    group=prepare(job);base,master=runtime(job);natural=group/'natural';target=group/'master'
    if phase=='natural':return base.retrieve(natural)
    if phase=='master-build':return master.offline(target)
    if phase=='candidates':
        if job['backbone']=='qwen':
            native=observed_gallery(job,target/'candidate_raw_batches');native.generate_candidates(target,device,32,3)
            p=shared.verify_protocol(target)
            master.seal(target,'candidates_complete.json',[target/'candidates.jsonl',target/'candidate_attempts.jsonl',target/'candidate_validation.json'],p,questions=1711)
        else:master.candidates(target,ENV)
        return
    if phase=='master-retrieve':return master.retrieve(target)
    root=Path(job['method_root'])/'native';source=natural if job['method']=='semrag' else target
    questions=c.rows(source/'inputs/questions.jsonl');retrieval=c.rows(source/'retrieval.jsonl')
    if job['method']!='semrag':
        retrieval=[dict(id=r['id'],task=r['task'],**next(v for v in r['conditions'] if v['name']=='caption_visual_master+zscore')) for r in retrieval]
    c.write_rows(root/'retrieval.jsonl',retrieval)
    if phase=='reader':
        if job['backbone']=='qwen':return qwen_answers(job,questions,retrieval,root/'reader',device)
        return shared.api_answers(job,questions,retrieval,root/'reader')
    if phase!='evaluation':raise ValueError('Unknown stage')
    responses=c.rows(root/'reader/responses.jsonl')
    if any(r['seed']!=job['seed'] for r in responses):raise ValueError('Native reader seed differs')
    files=[source/'protocol.json',source/'retrieval.jsonl',source/'retrieval_complete.json',root/'retrieval.jsonl',root/'reader/responses.jsonl',root/'reader/manifest.jsonl',Path(__file__)]
    if job['method']!='semrag':files += [source/'offline_complete.json',source/'candidates_complete.json']
    return shared.publish(job,responses,files,dict(caption_source_run_seed=42))

def run(job,path):
    if job['dataset']!='memgallery' or job['method'] not in ('semrag','ours_1_ours_2'):raise ValueError('Wrong Gallery caption handler')
    c.job_protocol(job,[Path(__file__),BASE,MASTER,Path(shared.__file__)])
    phases=['natural']+(['master-build','candidates','master-retrieve'] if job['method']!='semrag' else [])+['reader']
    for phase in phases:
        context=gpu(job) if job['backbone']=='qwen' and phase in ('candidates','reader') else nullcontext('cpu')
        with context as device:
            with c.lock(Path(job['run_root'])/'caption_methods'/(phase+'.lock')):
                print(dict(id=job['id'],stage=phase),flush=True)
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage',phase,'--device',device],check=True,cwd=c.ROOT)
    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage','evaluation'],check=True,cwd=c.ROOT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--device',default='cpu');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:prepare(j);runtime(j);print(dict(status='prepared',id=j['id'],model_calls=0))
    elif a.stage:stage(j,a.stage,a.device)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
