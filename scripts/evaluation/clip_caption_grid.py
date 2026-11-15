"""Model-matched caption-only CLIP evaluation, frozen captions, real reader seeds.

python3 -m scripts.evaluation.clip_caption_grid --stage prepare|smoke|run|report
The historical seed42 Qwen implementation and artifacts remain byte-identical.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import gc
import json
from pathlib import Path
import re
import shutil
import time
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.memory import clip_caption as method
from scripts.evaluation import required_grid_common as common
from scripts.evaluation.required_grid_api import Gateway

SOURCE=ROOT/'results/clip_caption_qwen_20261114'
OUTPUT=ROOT/'results/clip_caption_backbones_seeds_20261114'
DATASETS=('memeye','mem_gallery')
BACKBONES=('qwen','gpt5','deepseek')
SEEDS=(42,1024,2026)
CAPTIONS={
 ('memeye','gpt5'):ROOT/'results/memeye_gpt5_seed42_text_methods_20261025/caption_source',
 ('memeye','deepseek'):ROOT/'results/memeye_deepseek_flash_seed42_full_20261025',
 ('mem_gallery','gpt5'):ROOT/'results/mem_gallery_gpt5_gen_agents_20261028',
 ('mem_gallery','deepseek'):ROOT/'results/mem_gallery_deepseek_v41flash_20261030',
}
read,rows,dump,sha,digest,write_rows=method.read,method.rows,method.dump,method.sha,method.digest,method.write_rows


def signature(value,key='experiment_signature'):
    return digest({k:v for k,v in value.items() if k!=key})


def caption_cache(dataset,backbone):
    root=CAPTIONS[dataset,backbone]
    expected='gpt-5-2025-08-07' if backbone=='gpt5' else 'deepseek-v4.1-flash'
    caption_file=root/('caption_source/captions.jsonl' if dataset=='mem_gallery' else 'captions.jsonl')
    caps=rows(caption_file);files=[caption_file];outputs={}
    if dataset=='mem_gallery':
        marker=root/'caption_source/complete.json';m=read(marker);files.append(marker)
        if backbone=='gpt5':
            if sha(caption_file)!=m['captions_sha256']:raise ValueError('Caption cache hash changed')
        else:
            if m['artifact_sha256'].get(str(caption_file))!=sha(caption_file):raise ValueError('Caption cache hash changed')
        for r in caps:
            path=root/'caption_source/responses'/(r['id']+'.json');files.append(path)
            outputs[r['id']]=read(path)['output']
    elif backbone=='gpt5':
        marker=root/'completion_manifest.json';m=read(marker);files.append(marker)
        if not m['complete'] or m['caption_count']!=437:raise ValueError('Incomplete GPT caption source')
        if m['artifacts_sha256'].get(str(caption_file))!=sha(caption_file):raise ValueError('Caption cache hash changed')
        for r in caps:
            path=root/'responses'/(r['id']+'.json');files.append(path);outputs[r['id']]=read(path)
    else:
        marker=root/'completion_manifest.json';m=read(marker);files.append(marker)
        if m['status'] not in ('complete','complete_with_explicit_failed_records'):
            raise ValueError('Incomplete DeepSeek source; caption records are checked individually below')
        if m['artifacts_sha256'].get(str(caption_file))!=sha(caption_file):raise ValueError('Caption cache hash changed')
        path=root/'caption_generation/fresh_responses.jsonl';files.append(path)
        outputs={r['record_id']:r for r in rows(path)}
    count=437 if dataset=='memeye' else 979
    if len(caps)!=count or len({r['id'] for r in caps})!=count:raise ValueError('Caption coverage differs')
    rejected=[]
    for r in caps:
        out=outputs[r['id']]
        if out.get('status')=='provider_rejection':
            if dataset!='mem_gallery' or backbone!='deepseek' or r['caption']!='':raise ValueError('Unexpected empty caption')
            rejected.append(r['id']);continue
        if (out['returned_model']!=expected or out['seed']!=42 or out['finish_reason']!='stop'
                or out.get('refusal') or not r['caption'].strip() or r['caption']!=out['raw_response'].strip()):
            raise ValueError('Caption provenance invalid: '+r['id'])
    if len(rejected)!=(2 if (dataset,backbone)==('mem_gallery','deepseek') else 0):raise ValueError('Caption rejection count changed')
    return {r['id']:r['caption'] for r in caps},dict(model=expected,seed=42,count=count,
        rejected_image_hashes=rejected,source_sha256={str(p):sha(p) for p in files})


def memory(output,dataset,backbone):
    original=SOURCE/dataset/'memory'
    if backbone=='qwen':return original,dict(model='Qwen3-VL-8B-Instruct',seed=42,rejected_image_hashes=[])
    caps,audit=caption_cache(dataset,backbone)
    target=output/'memories'/backbone/dataset
    if (target/'memory.json').exists():
        method.load_memory(target)
        if read(target.parent/(dataset+'_caption_audit.json'))!=audit:raise ValueError('Caption source changed')
        return target,audit
    manifest,records,_=method.load_memory(original)
    new=[]
    for r in records:
        item=copy.deepcopy(r);item['caption']=caps[r['image_sha256']];new.append(item)
    target.mkdir(parents=True,exist_ok=True)
    write_rows(target/'records.jsonl',new)
    shutil.copyfile(original/'image_embeddings.npy',target/'image_embeddings.npy')
    manifest.update(caption_source_model=audit['model'],caption_source_seed=42,
        missing_caption_count=sum(not r['caption'] for r in new),
        missing_caption_policy='Retain provider-rejected captions as empty; no substitution, no dropped candidates.')
    manifest['files']={n:sha(target/n) for n in ('records.jsonl','image_embeddings.npy')}
    manifest['memory_signature']=signature(manifest,'memory_signature');dump(target/'memory.json',manifest)
    dump(target.parent/(dataset+'_caption_audit.json'),audit);method.load_memory(target)
    return target,audit


def code_hashes():
    from scripts.evaluation import required_grid_api,gpt5_text_memory_transport,memeye_deepseek_seed2026_transport
    from scripts.evaluation import deepseek_flash_seed42_audited_transport,openai_gpt5_base_transport
    from scripts.evaluation import openai_backbone_transport,openai_chat_transport,mem_gallery_master_reader
    mods=[sys.modules[__name__],method,common,required_grid_api,gpt5_text_memory_transport,
          memeye_deepseek_seed2026_transport,deepseek_flash_seed42_audited_transport,
          openai_gpt5_base_transport,openai_backbone_transport,openai_chat_transport,mem_gallery_master_reader]
    return {str(Path(m.__file__).resolve()):sha(m.__file__) for m in mods}


def request(payload,backbone,seed,gateway):
    if set(payload)!={'system','user_text'} or any(not isinstance(v,str) for v in payload.values()):
        raise ValueError('Only text system/user payload allowed')
    body=gateway.build_request(payload,max_output_tokens=2048 if backbone=='gpt5' else 256)
    if body['seed']!=seed or body['model']!=common.MODELS[backbone]:raise ValueError('Wrong seed/model')
    if any(not isinstance(m['content'],str) for m in body['messages']):raise ValueError('Only string contents allowed on wire')
    return body


def prepare(output):
    sources={ds:method.verify(SOURCE/ds/'evaluation')[0] for ds in DATASETS}
    code=code_hashes();cells=[]
    for backbone in BACKBONES:
        for ds in DATASETS:
            path,caption_audit=memory(output,ds,backbone)
            m,records,_=method.load_memory(path);by_id={r['id']:r for r in records}
            source=SOURCE/ds/'evaluation';qs=rows(source/'questions.jsonl');hits=rows(source/'retrieval.jsonl')
            for seed in SEEDS:
                if backbone=='qwen' and seed==42:continue
                target=output/backbone/f'seed{seed}'/ds;target.mkdir(parents=True,exist_ok=True)
                manifests=[]
                gateway=None if backbone=='qwen' else Gateway(dict(backbone=backbone,seed=seed))
                for q,r in zip(qs,hits):
                    for kind in sources[ds]['answer_formats']:
                        payload=method.reader_payload(q,[by_id[i] for i in r['selected_ids']],kind)
                        entry=dict(id=q['id'],kind=kind,payload=payload,payload_sha256=digest(payload))
                        if gateway:
                            wire=request(payload,backbone,seed,gateway)
                            entry.update(wire=wire,request_sha256=digest(wire))
                        manifests.append(entry)
                common.write_rows(target/'reader_manifest.jsonl',manifests)
                inputs={**code,**caption_audit.get('source_sha256',{}),**sources[ds]['input_sha256']}
                for f in [path/n for n in ('memory.json','records.jsonl','image_embeddings.npy')]+[target/'reader_manifest.jsonl',source/'protocol.json']:
                    inputs[str(f.resolve())]=sha(f)
                p=dict(method='clip_caption',dataset=ds,backbone=backbone,seed=seed,caption_seed=42,
                    requested_model=common.MODELS[backbone],expected_model=caption_audit['model'],
                    source=str(source),memory_path=str(path),memory_signature=m['memory_signature'],
                    questions=len(qs),answer_requests=len(manifests),top_k=5,answer_formats=sources[ds]['answer_formats'],
                    caption_source=caption_audit,caption_policy='Reuse existing same-backbone seed42 captions for all reader seeds.',
                    ranking_policy='Frozen original CLIP question-to-image Top5; caption cannot change ranking.',
                    reader_policy='Only retrieved caption text and image IDs, question and MCQ choices; no image pixels or history dialogue.',
                    reader_settings=dict(batch_size=8,max_output_tokens=2048 if backbone=='gpt5' else 256,
                        reasoning_effort='minimal' if backbone=='gpt5' else None,
                        temperature=None if backbone=='gpt5' else 0,do_sample=False if backbone=='qwen' else None),
                    seed_policy='Actual per-batch torch seed for Qwen; actual wire seed for APIs; captions fixed at42. Provider determinism not guaranteed.',
                    failures='Retain provider rejections/refusals/empty/truncated/invalid-format answers in full denominator; no quality-based retries.',
                    input_sha256=inputs)
                p['experiment_signature']=digest(p);common.freeze(target/'protocol.json',p)
                verify(target);cells.append(dict(backbone=backbone,dataset=ds,seed=seed,path=str(target)))
    common.freeze(output/'suite_protocol.json',dict(method='clip_caption',seeds=list(SEEDS),new_cells=cells,
        existing_qwen_seed42=str(SOURCE),caption_seed=42,order='seed42 GPT/DeepSeek first, then seed1024 all backbones, then seed2026 all backbones'))
    print('Prepared',len(cells),'new full-cohort runs',flush=True)


def verify(target):
    p=read(target/'protocol.json')
    if signature(p)!=p['experiment_signature']:raise ValueError('Protocol signature changed')
    common.hashes(p['input_sha256'])
    m,records,vectors=method.load_memory(Path(p['memory_path']))
    if m['memory_signature']!=p['memory_signature']:raise ValueError('Memory identity changed')
    source=Path(p['source']);qs=rows(source/'questions.jsonl');hits=rows(source/'retrieval.jsonl')
    query_vectors=__import__('numpy').load(source/'query_embeddings.npy',allow_pickle=False)
    manifests=rows(target/'reader_manifest.jsonl');expected=[]
    for q,r,v in zip(qs,hits,query_vectors):
        selected=[records[i] for i,_ in method.rank(records,vectors,q['task'],v,5)]
        if [x['id'] for x in selected]!=r['selected_ids']:raise ValueError('Pure visual ranking changed')
        for kind in p['answer_formats']:
            payload=method.reader_payload(q,selected,kind)
            expected.append(dict(id=q['id'],kind=kind,payload=payload,payload_sha256=digest(payload)))
    if len(qs)!=p['questions'] or len(manifests)!=p['answer_requests'] or len(expected)!=len(manifests):raise ValueError('Coverage differs')
    for a,b in zip(expected,manifests):
        if {k:b[k] for k in a}!=a:raise ValueError('Context differs from caption-only reconstruction')
        if p['backbone']!='qwen':
            wire=b['wire']
            if b['request_sha256']!=digest(wire) or wire['seed']!=p['seed']:raise ValueError('Wire identity changed')
            messages=[{'role':'system','content':a['payload']['system']},{'role':'user','content':a['payload']['user_text']}]
            if wire['messages']!=messages:raise ValueError('Non-text or extra content in wire')
    return p,manifests


def save_response(target,p,item,out):
    key=digest([item['id'],item['kind']])
    dump(target/'responses'/(key+'.json'),dict(request=item,output=out,experiment_signature=p['experiment_signature']))


def pending(target,p,items):
    result=[]
    for item in items:
        file=target/'responses'/(digest([item['id'],item['kind']])+'.json')
        if file.exists():
            saved=read(file)
            if saved['experiment_signature']!=p['experiment_signature'] or saved['request']!=item:raise ValueError('Cached answer identity differs')
        else:result.append(item)
    return result


def api_answers(target,workers=8,smoke=False):
    p,items=verify(target);todo=pending(target,p,items)
    if smoke:todo=todo[:1]
    gateway=Gateway(dict(backbone=p['backbone'],seed=p['seed']))
    def one(item):
        wire=request(item['payload'],p['backbone'],p['seed'],gateway)
        if wire!=item['wire']:raise ValueError('Actual request differs from prepared wire')
        out=gateway.call(target/'api_cache',wire,attempts=3,allow_rejection=p['backbone']=='deepseek')
        out.update(reader_image_count=0,processor_keys=['text_messages'],actual_wire_seed=wire['seed'])
        save_response(target,p,item,out)
        return out
    done=len(items)-len(pending(target,p,items))
    with ThreadPoolExecutor(max_workers=1 if smoke else workers) as pool:
        for future in as_completed([pool.submit(one,r) for r in todo]):
            future.result();done+=1
            dump(target/'progress.json',dict(done=done,total=len(items),backbone=p['backbone'],dataset=p['dataset'],seed=p['seed']))
            if done%100==0 or done==len(items) or smoke:print(p['backbone'],p['seed'],p['dataset'],done,'/',len(items),flush=True)


def qwen_answers(target,device):
    import torch
    p,items=verify(target);todo=pending(target,p,items)
    if not todo:return
    # Isolate the per-batch RNG literal, leaving the audited seed42 implementation untouched.
    tree=ast.parse(Path(method.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='text_batch')
    locations={n.lineno for n in ast.walk(fn) if isinstance(n,ast.Constant) and type(n.value)is int and n.value==42}
    if len(locations)!=1:raise ValueError('Qwen batch seed boundary changed')
    clone=common.clone(Path(method.__file__),'clip_caption_reader_seed'+str(p['seed']),p['seed'],locations,old_seed=42)
    source=read(Path(p['source'])/'protocol.json')
    for name,value in source['reader_model']['files'].items():
        if sha(Path(source['reader_model']['path'])/name)!=value['sha256']:raise ValueError('Qwen weights changed')
    runtime=method.shared.load_runtime(device)
    dump(target/'generation_environment.json',dict(runtime[2],actual_batch_seed=p['seed'],do_sample=False))
    visual=getattr(runtime[1],'visual',None)
    if visual is None:visual=getattr(getattr(runtime[1],'model',None),'visual',None)
    if visual is None:raise ValueError('Cannot enforce vision-forward prohibition')
    def reject(*args):raise RuntimeError('Caption-only reader must not execute vision tower')
    hook=visual.register_forward_pre_hook(reject)
    try:
        done=len(items)-len(todo)
        for i in range(0,len(todo),8):
            batch=todo[i:i+8];outputs=clone.text_batch(runtime,batch,256)
            if len(batch)!=len(outputs) or torch.initial_seed()!=p['seed']:raise ValueError('Actual batch seed or output count differs')
            for item,out in zip(batch,outputs):
                out.update(seed=p['seed'],requested_model=p['requested_model'],returned_model=p['expected_model'])
                save_response(target,p,item,out)
            done+=len(batch);dump(target/'progress.json',dict(done=done,total=len(items),seed=p['seed']))
            if done%104==0 or done==len(items):print('qwen',p['seed'],p['dataset'],done,'/',len(items),flush=True)
    finally:
        hook.remove();del visual,hook,runtime;gc.collect();torch.cuda.empty_cache()


def analyze(target):
    p,items=verify(target)
    if pending(target,p,items):raise ValueError('Answers incomplete')
    gold={r['id']:r for r in rows(Path(p['source'])/'gold.jsonl')};scored=[];hashes={};models=Counter()
    for item in items:
        f=target/'responses'/(digest([item['id'],item['kind']])+'.json');hashes[str(f)]=sha(f)
        out=read(f)['output'];g=gold[item['id']]
        if out['reader_image_count']!=0 or out.get('actual_wire_seed',out.get('seed'))!=p['seed']:raise ValueError('Images or wrong seed in reader')
        models[out.get('returned_model') or 'provider_rejection']+=1
        if out.get('returned_model') not in (None,p['expected_model']):raise ValueError('Unexpected model revision')
        rejected=out.get('status')=='provider_rejection' or bool(out.get('refusal'))
        row=dict(id=item['id'],kind=item['kind'],raw_response=out['raw_response'].strip(),
                 provider_rejection=rejected,hit_token_limit=out['hit_token_limit'])
        if item['kind']=='mcq':
            m=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',row['raw_response'],re.I)
            choice=m.group(1).upper() if m else None
            row.update(predicted_choice=choice,gold_choice=g['gold_choice'],correct=bool(not rejected and choice==g['gold_choice']),format_error=choice is None)
        else:
            answer=g['gold_answer_text'] if p['dataset']=='memeye' else g['answer']
            metrics=method.shared.score_answer(row['raw_response'],answer)
            if rejected:metrics={k:0.0 for k in metrics}
            row.update(answer=answer,**metrics)
        scored.append(row)
    metrics={}
    for kind in p['answer_formats']:
        group=[r for r in scored if r['kind']==kind];n=len(group)
        result=dict(questions=n,provider_rejections=sum(r['provider_rejection'] for r in group),
            empty_answers=sum(not r['raw_response'] for r in group),token_limit_hits=sum(r['hit_token_limit'] for r in group))
        if kind=='mcq':result.update(correct=sum(r['correct'] for r in group),accuracy=sum(r['correct'] for r in group)/n,format_errors=sum(r['format_error'] for r in group))
        else:result.update(correct=sum(int(r['em']) for r in group),**{k:sum(r[k] for r in group)/n for k in ('em','f1','bleu_1','bleu_2','bleu')})
        metrics[kind]=result
    storage=sum(f.stat().st_size for f in Path(p['memory_path']).iterdir())
    summary=dict(status='complete',dataset=p['dataset'],backbone=p['backbone'],seed=p['seed'],caption_seed=42,
        questions=p['questions'],answer_requests=len(items),metrics=metrics,returned_models=dict(models),
        storage_bytes=storage,storage_mib=storage/1048576,storage_includes_image_vectors=True,
        missing_caption_count=len(p['caption_source']['rejected_image_hashes']))
    write_rows(target/'scored_questions.jsonl',scored);dump(target/'summary.json',summary)
    hashes.update({str(target/n):sha(target/n) for n in ['protocol.json','reader_manifest.jsonl','scored_questions.jsonl','summary.json']})
    dump(target/'audit.json',dict(status='passed',questions=p['questions'],answer_requests=len(items),
        pure_visual_rankings_recomputed=True,caption_only_payloads_reconstructed=True,actual_seed_verified=True,
        no_images_in_reader=True,model_matched_captions=True,artifact_sha256=hashes))
    print('COMPLETE',p['backbone'],p['seed'],p['dataset'],json.dumps(metrics),flush=True)
    return summary


def report(output):
    protocol=read(output/'suite_protocol.json');records=[];missing=[];artifacts={}
    for ds in DATASETS:
        original=SOURCE/ds/'evaluation/summary.json';record=read(original)
        common.hashes(read(SOURCE/ds/'evaluation/audit.json')['artifact_sha256'])
        records.append(dict(backbone='qwen',seed=42,**record));artifacts[str(original)]=sha(original)
    for cell in protocol['new_cells']:
        target=Path(cell['path'])
        if not (target/'audit.json').exists():missing.append(cell);continue
        audit=read(target/'audit.json');common.hashes(audit['artifact_sha256'])
        records.append(read(target/'summary.json'))
        for n in ['summary.json','audit.json']:artifacts[str(target/n)]=sha(target/n)
    table=[]
    for r in records:
        for kind,metrics in r['metrics'].items():
            table.append(dict(backbone=r['backbone'],seed=r['seed'],dataset=r['dataset'],format=kind,
                questions=metrics['questions'],storage_mib=r['storage_mib'],
                **{k:metrics.get(k) for k in ['accuracy','em','f1','bleu_1','bleu_2','bleu','correct','provider_rejections','format_errors','token_limit_hits']}))
    with (output/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    status=dict(status='complete' if not missing else 'partial',complete_cells=len(records),required_cells=18,
        pending_cells=missing,results=records,artifact_sha256=artifacts)
    dump(output/'results.json',status)
    if not missing:dump(output/'completion_manifest.json',status)
    return status


def suite(output,workers,device):
    for seed in SEEDS:
        def run_backbone(backbone):
            if seed==42 and backbone=='qwen':return
            for ds in DATASETS:
                target=output/backbone/f'seed{seed}'/ds
                if (target/'audit.json').exists():
                    common.hashes(read(target/'audit.json')['artifact_sha256']);continue
                if backbone=='qwen':qwen_answers(target,device)
                else:api_answers(target,workers)
                analyze(target)
        dump(output/'suite_progress.json',dict(status='running',seed=seed))
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(run_backbone,b) for b in BACKBONES]
            for f in as_completed(futures):f.result()
        report(output)
        dump(output/'suite_progress.json',dict(status='seed_complete',seed=seed))
    final=report(output)
    if final['status']!='complete':raise RuntimeError('Missing grid cells')
    dump(output/'suite_progress.json',dict(status='complete',cells=18))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','smoke','run','report'],required=True)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();args.output=args.output.resolve()
    if args.workers<1:parser.error('Positive worker count required')
    if args.stage=='prepare':prepare(args.output)
    elif args.stage=='smoke':
        for seed in (42,):
            for b in ['gpt5','deepseek']:
                api_answers(args.output/b/f'seed{seed}'/'memeye',workers=1,smoke=True)
    elif args.stage=='report':print(json.dumps(report(args.output),ensure_ascii=False))
    else:
        try:suite(args.output,args.workers,args.device)
        except Exception as error:
            dump(args.output/'suite_failure.json',dict(error_type=type(error).__name__,error=str(error),time=time.time()))
            raise


if __name__=='__main__':main()
