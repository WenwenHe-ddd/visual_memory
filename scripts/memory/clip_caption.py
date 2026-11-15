"""CLIP text-to-image retrieval over image vectors; caption-only Qwen answers.

The portable memory contains no original images, image paths, dialogue, or text
embeddings. Raw source images are read only by offline construction. Evaluation
inputs/provenance and outputs are kept outside that memory directory.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import re
import time

import numpy as np

from scripts.evaluation import mem_gallery_master_reader as shared

ROOT = shared.ROOT
METHOD = 'clip_caption'
MEMEYE_SOURCE = ROOT/'results/natural_observation'
MEMEYE_QUESTIONS = ROOT/'results/mcq_top15_caption_qa/qa_manifest.jsonl'
GALLERY_SOURCE = ROOT/'results/mem_gallery_master_answer_20261021/inputs'
sha, digest, dump = shared.sha, shared.digest, shared.dump
READ_SYSTEM = ('Answer the question using only the retrieved image captions and their image ID labels. '
               'Do not add unsupported facts. Keep the answer concise and complete. '
               'If the requested information is absent, answer "Not mentioned."')
MCQ_SYSTEM = ('Choose the best answer using the retrieved image captions. '
              'Return only one uppercase option letter: A, B, C, or D.')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False)+'\n' for v in values))


def read(path):
    return json.loads(Path(path).read_text())


def source_images(dataset, source):
    """Construction-only inputs. Return caption records and temporary image paths."""
    grouped = {}
    if dataset == 'memeye':
        images = {r['id']:r for r in rows(source/'images.jsonl')}
        captions = {r['id']:r['caption'] for r in rows(source/'captions.jsonl')}
        memories = rows(source/'memories.jsonl')
        expected = 438
        for m in memories:
            for ident in m['images']:
                key = (m['task'], ident)
                record = grouped.setdefault(key, dict(id=m['task']+'/'+ident, task=m['task'],
                    image_sha256=ident, caption=captions[ident], image_ids=[], source_memory_ids=[],
                    source_path=images[ident]['path']))
                record['source_memory_ids'].append(m['id'])
        inputs = [source/f for f in ('images.jsonl','captions.jsonl','memories.jsonl','protocol.json')]
    else:
        manifest = read(source/'artifact_manifest.json')
        for name in ('memories.jsonl','protocol.json'):
            if sha(source/name) != manifest[name]:
                raise ValueError('Gallery source changed: '+name)
        memories = [r for r in rows(source/'memories.jsonl') if r['is_visual']]
        expected = 1003
        for m in memories:
            key = (m['task'],m['image_sha256'])
            record = grouped.setdefault(key, dict(id=m['task']+'/'+m['image_sha256'],task=m['task'],
                image_sha256=m['image_sha256'],caption=m['caption'],image_ids=[],source_memory_ids=[],
                source_path=m['image_path']))
            if record['caption'] != m['caption']:
                raise ValueError('One historical image has inconsistent captions')
            record['source_memory_ids'].append(m['id'])
            if m['image_id'] not in record['image_ids']:
                record['image_ids'].append(m['image_id'])
        inputs = [source/f for f in ('memories.jsonl','protocol.json','artifact_manifest.json')]
    records = list(grouped.values())
    if sum(len(r['source_memory_ids']) for r in records) != expected:
        raise ValueError('Historical image occurrence count changed')
    if len(records) != (433 if dataset=='memeye' else 1001):
        raise ValueError('Historical candidate count changed')
    if any(not r['caption'].strip() for r in records):
        raise ValueError('Missing cached caption')
    return records, inputs


def persistent_record(record):
    return {k:record[k] for k in ('id','task','image_sha256','caption','image_ids','source_memory_ids')}


def offline(args):
    from scripts.graph.visual_image_encoder import ImageEncoder
    import torch
    target = args.output.resolve()
    if (target/'memory.json').exists():
        existing,_,_=load_memory(target)
        if existing['dataset']!=args.dataset:
            raise ValueError('Existing memory belongs to a different dataset')
        print('Verified existing caption/vector memory:',target,flush=True)
        return
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('Use an empty output directory for construction')
    source = args.clip_caption_source_root or (MEMEYE_SOURCE if args.dataset=='memeye' else GALLERY_SOURCE)
    records, sources = source_images(args.dataset,source)
    torch.set_num_threads(8)
    encoder = ImageEncoder(str(shared.CLIP),device=args.clip_caption_encoding_device,local_files_only=True)
    cache = {}; image_provenance = {}
    for i,r in enumerate(records,1):
        path = Path(r['source_path'])
        if str(path) not in image_provenance:
            image_provenance[str(path)] = sha(path)
        if image_provenance[str(path)] != r['image_sha256']:
            raise ValueError('History image hash mismatch: '+str(path))
        if r['image_sha256'] not in cache:
            cache[r['image_sha256']] = encoder.encode(path)
        if i%100==0:
            print('CLIP image encoding',i,'/',len(records),flush=True)
    vectors = np.asarray([cache[r['image_sha256']] for r in records],dtype=np.float32)
    vectors /= np.linalg.norm(vectors,axis=1,keepdims=True)
    metadata = encoder.metadata
    del encoder; gc.collect()
    target.mkdir(parents=True,exist_ok=True)
    write_rows(target/'records.jsonl',[persistent_record(r) for r in records])
    np.save(target/'image_embeddings.npy',vectors,allow_pickle=False)
    checkpoint_hashes={str(f.resolve()):sha(f) for f in shared.CLIP.iterdir() if f.is_file()}
    manifest=dict(schema='clip_caption_memory_v1',offline_method=METHOD,dataset=args.dataset,
        candidates=len(records),source_rounds=sum(len(r['source_memory_ids']) for r in records),
        dimensions=int(vectors.shape[1]),dtype=str(vectors.dtype),normalization='L2',
        clip_model_path=str(shared.CLIP),clip_model_sha256=checkpoint_hashes,encoder=metadata,
        caption_source_model='Qwen3-VL-8B-Instruct',
        storage_policy='Image embeddings + cached image captions + image/source IDs. No pixels, image paths, dialogues or caption embeddings.',
        files={name:sha(target/name) for name in ('records.jsonl','image_embeddings.npy')})
    manifest['memory_signature']=digest(manifest);dump(target/'memory.json',manifest)
    # Audit paths are deliberately outside the deployable memory directory.
    dump(target.parent/(target.name+'_construction_audit.json'),dict(status='passed',
        source_sha256={str(f.resolve()):sha(f) for f in sources},source_image_sha256=image_provenance,
        memory_signature=manifest['memory_signature'],original_images_copied=0))
    load_memory(target)
    print('Built',args.dataset,len(records),'image/caption memories',flush=True)


def load_memory(path):
    path = Path(path)
    m = read(path/'memory.json')
    if digest({k:v for k,v in m.items() if k!='memory_signature'})!=m['memory_signature']:
        raise ValueError('Memory manifest signature differs')
    if m['offline_method']!=METHOD or set(f.name for f in path.iterdir())!={'memory.json','records.jsonl','image_embeddings.npy'}:
        raise ValueError('Unexpected memory contents; pixels and auxiliary files are not allowed')
    for name,h in m['files'].items():
        if sha(path/name)!=h:
            raise ValueError('Memory file changed: '+name)
    data=rows(path/'records.jsonl');vectors=np.load(path/'image_embeddings.npy',allow_pickle=False)
    allowed={'id','task','image_sha256','caption','image_ids','source_memory_ids'}
    if any(set(r)!=allowed for r in data) or len({r['id'] for r in data})!=len(data):
        raise ValueError('Invalid memory record schema')
    if vectors.shape!=(len(data),m['dimensions']) or len(data)!=m['candidates'] or vectors.dtype!=np.float32:
        raise ValueError('Memory embedding shape/type mismatch')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):
        raise ValueError('Invalid normalized image embeddings')
    return m,data,vectors


def encode_queries(texts, memory, device):
    import torch
    from transformers import AutoConfig,AutoTokenizer,CLIPTextModelWithProjection
    for f,h in memory['clip_model_sha256'].items():
        if sha(f)!=h:
            raise ValueError('CLIP model changed')
    checkpoint=memory['clip_model_path']
    torch.set_num_threads(8)
    config=AutoConfig.from_pretrained(checkpoint,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    model,loading=CLIPTextModelWithProjection.from_pretrained(checkpoint,config=config.text_config,
        local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):
        raise ValueError('Incomplete pretrained CLIP text tower')
    model.to(device=device,dtype=torch.float32).eval()
    limit=config.text_config.max_position_embeddings
    lengths=[len(tokenizer.encode(t,truncation=False)) for t in texts]
    result=[]
    for start in range(0,len(texts),32):
        inputs=tokenizer(texts[start:start+32],padding=True,truncation=True,max_length=limit,return_tensors='pt').to(device)
        with torch.inference_mode():
            v=model(**inputs).text_embeds.float().cpu().numpy()
        result.extend(v/np.linalg.norm(v,axis=1,keepdims=True))
    del model;gc.collect()
    return np.asarray(result,dtype=np.float32),lengths,limit


def rank(records,vectors,task,query,top_k):
    indices=np.asarray([i for i,r in enumerate(records) if r['task']==task])
    if len(indices)<top_k:
        raise ValueError('Not enough distinct historical images for Top-K')
    scores=vectors[indices] @ query
    order=np.argsort(-scores,kind='stable')[:top_k]
    return [(int(indices[i]),float(scores[i])) for i in order]


def reader_payload(question,selected,kind):
    evidence=[]
    for i,r in enumerate(selected,1):
        label='Caption '+str(i)
        if r['image_ids']:
            label+=' [image_id: '+', '.join(r['image_ids'])+']'
        evidence.append(label+':\n'+r['caption'])
    text='Retrieved image captions (ranked by relevance):\n\n'+'\n\n'.join(evidence)
    text+='\n\nQuestion:\n'+question['question']
    if kind=='mcq':
        text+='\n\nOptions:\n'+'\n'.join(k+'. '+question['options'][k] for k in 'ABCD')
        text+='\n\nReturn only one uppercase option letter.'
    elif question.get('point','').upper() in shared.TASK_INSTRUCTIONS:
        text+='\n\n'+shared.TASK_INSTRUCTIONS[question['point'].upper()]
    return dict(system=MCQ_SYSTEM if kind=='mcq' else READ_SYSTEM,user_text=text)


def prepare(args):
    target=args.output.resolve();memory_path=args.memory_path.resolve()
    if (target/'protocol.json').exists():
        p,_=verify(target)
        if (p['memory_path']!=str(memory_path) or p['top_k']!=args.top_k
                or p['max_new_tokens']!=args.clip_caption_max_new_tokens
                or p['batch_size']!=args.clip_caption_batch_size):
            raise ValueError('Resume arguments differ from frozen protocol; use a fresh output directory')
        return
    m,records,vectors=load_memory(memory_path)
    dataset=m['dataset'];source=args.clip_caption_source_root or (MEMEYE_SOURCE if dataset=='memeye' else GALLERY_SOURCE)
    target.mkdir(parents=True,exist_ok=True)
    if dataset=='memeye':
        raw=rows(args.benchmark_manifest)
        qs=[{**{k:q[k] for k in ('id','task','question','options')},'has_query_image':bool(q['query_image_paths'])} for q in raw]
        gold=[{k:q[k] for k in ('id','gold_ids','gold_choice','gold_answer_text')} for q in raw]
        write_rows(target/'gold.jsonl',gold)
        sources=[args.benchmark_manifest];formats=['mcq','open']
    else:
        source_manifest=read(source/'artifact_manifest.json')
        for name in ('questions.jsonl','gold.jsonl'):
            if sha(source/name)!=source_manifest[name]:
                raise ValueError('Gallery input changed: '+name)
        qs=[{**{k:q[k] for k in ('id','task','question','point')},'has_query_image':bool(q['query_image_paths'])} for q in rows(source/'questions.jsonl')]
        (target/'gold.jsonl').write_bytes((source/'gold.jsonl').read_bytes())
        sources=[source/'questions.jsonl',source/'gold.jsonl',source/'artifact_manifest.json'];formats=['open']
    if len(qs)!=(371 if dataset=='memeye' else 1711) or len({q['id'] for q in qs})!=len(qs):
        raise ValueError('Benchmark coverage differs')
    queries,lengths,limit=encode_queries([q['question'] for q in qs],m,args.clip_caption_encoding_device)
    retrieval=[];payloads=[]
    for q,vector,length in zip(qs,queries,lengths):
        hits=rank(records,vectors,q['task'],vector,args.top_k)
        selected=[records[i] for i,_ in hits]
        retrieval.append(dict(id=q['id'],retrieval_query=q['question'],query_tokens=length,query_truncated=length>limit,
            selected_ids=[r['id'] for r in selected],scores=[s for _,s in hits],
            source_memory_ids=list(dict.fromkeys(mid for r in selected for mid in r['source_memory_ids']))))
        for kind in formats:
            payload=reader_payload(q,selected,kind)
            payloads.append(dict(id=q['id'],kind=kind,payload=payload,payload_sha256=digest(payload)))
    write_rows(target/'questions.jsonl',qs);write_rows(target/'retrieval.jsonl',retrieval)
    write_rows(target/'reader_manifest.jsonl',payloads);np.save(target/'query_embeddings.npy',queries,allow_pickle=False)
    memory_files=[memory_path/name for name in ('memory.json','records.jsonl','image_embeddings.npy')]
    files=[*memory_files,*sources,Path(__file__),Path(shared.__file__),ROOT/'scripts/visual_memory_pipeline.py']
    files += [target/name for name in ('questions.jsonl','retrieval.jsonl','reader_manifest.jsonl','query_embeddings.npy','gold.jsonl')]
    p=dict(method=METHOD,dataset=dataset,questions=len(qs),answer_formats=formats,top_k=args.top_k,seed=42,
        memory_path=str(memory_path),memory_signature=m['memory_signature'],reader_model=shared.model_provenance(),
        max_new_tokens=args.clip_caption_max_new_tokens,batch_size=args.clip_caption_batch_size,
        dtype='bfloat16',attention='sdpa',do_sample=False,reader_images=0,
        retrieval_policy='Question text only -> CLIP text projection -> cosine with stored image vectors; same scenario; stable Top-K. No caption/text score.',
        answer_policy='Retrieved captions plus image-ID labels, question and MCQ options where applicable. No history dialogue, dates, images or query-image captions.',
        query_image_questions=sum(q['has_query_image'] for q in qs),clip_token_limit=limit,
        query_truncated_count=sum(n>limit for n in lengths),
        input_sha256={str(f.resolve()):sha(f) for f in set(files)})
    p['experiment_signature']=digest(p);dump(target/'protocol.json',p)
    verify(target)
    print('Prepared',dataset,len(qs),'questions;',len(payloads),'text-only answer requests',flush=True)


def verify(target):
    p=read(target/'protocol.json')
    if digest({k:v for k,v in p.items() if k!='experiment_signature'})!=p['experiment_signature']:
        raise ValueError('Protocol changed')
    for f,h in p['input_sha256'].items():
        if sha(f)!=h:
            raise ValueError('Frozen input changed: '+f)
    for name,info in p['reader_model']['files'].items():
        if sha(Path(p['reader_model']['path'])/name)!=info['sha256']:
            raise ValueError('Qwen checkpoint changed')
    m,records,vectors=load_memory(Path(p['memory_path']))
    if m['memory_signature']!=p['memory_signature']:
        raise ValueError('Memory identity changed')
    qs=rows(target/'questions.jsonl');retrieval=rows(target/'retrieval.jsonl');payloads=rows(target/'reader_manifest.jsonl')
    gold=rows(target/'gold.jsonl')
    if len(gold)!=len(qs) or {g['id'] for g in gold}!={q['id'] for q in qs}:
        raise ValueError('Gold coverage does not match question coverage')
    query_vectors=np.load(target/'query_embeddings.npy',allow_pickle=False)
    if (len(qs)!=p['questions'] or len({q['id'] for q in qs})!=len(qs)
            or query_vectors.shape!=(len(qs),m['dimensions'])
            or not np.isfinite(query_vectors).all()
            or not np.allclose(np.linalg.norm(query_vectors,axis=1),1,atol=1e-5)
            or len(retrieval)!=len(qs) or len(payloads)!=len(qs)*len(p['answer_formats'])):
        raise ValueError('Missing benchmark inputs')
    expected=[]
    for q,r,v in zip(qs,retrieval,query_vectors):
        hits=rank(records,vectors,q['task'],v,p['top_k']);selected=[records[i] for i,_ in hits]
        if r['id']!=q['id'] or r['retrieval_query']!=q['question'] or r['selected_ids']!=[x['id'] for x in selected]:
            raise ValueError('Retrieval does not match image-only scores')
        if not np.allclose(r['scores'],[s for _,s in hits],atol=1e-7,rtol=0):
            raise ValueError('Retrieval scores changed')
        if r['source_memory_ids']!=list(dict.fromkeys(mid for x in selected for mid in x['source_memory_ids'])):
            raise ValueError('Source attribution changed')
        for kind in p['answer_formats']:
            payload=reader_payload(q,selected,kind)
            expected.append(dict(id=q['id'],kind=kind,payload=payload,payload_sha256=digest(payload)))
    if expected!=payloads:
        raise ValueError('Reader includes content beyond selected captions/question/options')
    return p,payloads


def text_batch(runtime,batch,cap):
    import torch
    processor,model,_=runtime
    messages=[[dict(role='system',content=r['payload']['system']),dict(role='user',content=r['payload']['user_text'])] for r in batch]
    rendered=[processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
    encoded=processor(text=rendered,padding=True,truncation=False,return_tensors='pt')
    if any('pixel' in k or 'image' in k or 'video' in k for k in encoded):
        raise ValueError('Text-only reader received vision tensors')
    if encoded['input_ids'].shape[1]+cap>model.config.text_config.max_position_embeddings:
        raise ValueError('Reader context would truncate')
    keys=list(encoded);lengths=encoded['attention_mask'].sum(dim=1).tolist()
    encoded=encoded.to(next(model.parameters()).device);torch.manual_seed(42)
    start=time.monotonic()
    with torch.inference_mode():
        generated=model.generate(**encoded,do_sample=False,max_new_tokens=cap,pad_token_id=processor.tokenizer.pad_token_id)
    tails=generated[:,encoded['input_ids'].shape[1]:]
    decoded=processor.batch_decode(tails,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    eos=model.generation_config.eos_token_id
    stops=set(eos if isinstance(eos,list) else [eos]);stops.add(processor.tokenizer.eos_token_id)
    result=[]
    for text,tokens,length in zip(decoded,tails.tolist(),lengths):
        stop=next((i for i,t in enumerate(tokens) if t in stops),None)
        result.append(dict(raw_response=text.strip(),input_tokens=int(length),generated_tokens=stop+1 if stop is not None else len(tokens),
            hit_token_limit=stop is None and len(tokens)>=cap,reader_image_count=0,processor_keys=keys,
            batch_seconds=time.monotonic()-start))
    return result


def run_answers(args):
    target=args.output.resolve();p,payloads=verify(target)
    response_dir=target/'responses';response_dir.mkdir(exist_ok=True)
    pending=[]
    for r in payloads:
        path=response_dir/(digest([r['id'],r['kind']])+'.json')
        if path.exists():
            saved=read(path)
            if saved['request']!=r or saved['experiment_signature']!=p['experiment_signature']:
                raise ValueError('Existing answer provenance differs')
        else:
            pending.append(r)
    if not pending:return
    runtime=shared.load_runtime(args.device)
    dump(target/'generation_environment.json',runtime[2]|dict(reader_mode='text only; vision forward prohibited'))
    visual=getattr(runtime[1],'visual',None)
    if visual is None:visual=getattr(getattr(runtime[1],'model',None),'visual',None)
    def reject_vision(*unused):
        raise RuntimeError('Vision tower must not execute in caption-only answering')
    if visual is None:
        raise RuntimeError('Cannot locate Qwen vision tower to enforce caption-only inference')
    hook=visual.register_forward_pre_hook(reject_vision)
    try:
        done=len(payloads)-len(pending)
        for start in range(0,len(pending),p['batch_size']):
            batch=pending[start:start+p['batch_size']]
            outputs=text_batch(runtime,batch,p['max_new_tokens'])
            for r,out in zip(batch,outputs):
                dump(response_dir/(digest([r['id'],r['kind']])+'.json'),dict(request=r,output=out,experiment_signature=p['experiment_signature']))
            done+=len(batch);dump(target/'progress.json',dict(done=done,total=len(payloads)))
            print('text-only answers',done,'/',len(payloads),flush=True)
    finally:
        if hook is not None:hook.remove()


def analyze(args):
    target=args.output.resolve();p,payloads=verify(target)
    gold={r['id']:r for r in rows(target/'gold.jsonl')}
    retrieval={r['id']:r for r in rows(target/'retrieval.jsonl')};scored=[];answer_files=[]
    for request in payloads:
        f=target/'responses'/(digest([request['id'],request['kind']])+'.json');saved=read(f);answer_files.append(f)
        if saved['request']!=request or saved['experiment_signature']!=p['experiment_signature']:
            raise ValueError('Answer signature or payload differs')
        out=saved['output']
        if out['reader_image_count']!=0 or any('pixel' in k or 'image' in k or 'video' in k for k in out['processor_keys']):
            raise ValueError('Non-text input reached reader')
        g=gold[request['id']];sources=set(retrieval[request['id']]['source_memory_ids']);gids=set(g['gold_ids'])
        r=dict(id=request['id'],kind=request['kind'],raw_response=out['raw_response'],
            gold_hits=len(sources&gids),gold_count=len(gids),hit_token_limit=out['hit_token_limit'])
        if request['kind']=='mcq':
            match=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',out['raw_response'],re.I)
            choice=match.group(1).upper() if match else None
            r.update(predicted_choice=choice,gold_choice=g['gold_choice'],correct=choice==g['gold_choice'],format_error=choice is None)
        else:
            answer=g['gold_answer_text'] if p['dataset']=='memeye' else g['answer']
            r.update(answer=answer,**shared.score_answer(out['raw_response'],answer))
        scored.append(r)
    results={}
    for kind in p['answer_formats']:
        group=[r for r in scored if r['kind']==kind];evidence=[r for r in group if r['gold_count']]
        v=dict(questions=len(group),source_recall_macro=sum(r['gold_hits']/r['gold_count'] for r in evidence)/len(evidence),
            source_recall_micro=sum(r['gold_hits'] for r in evidence)/sum(r['gold_count'] for r in evidence),
            token_limit_hits=sum(r['hit_token_limit'] for r in group),empty_answers=sum(not r['raw_response'] for r in group))
        if kind=='mcq':
            v.update(correct=sum(r['correct'] for r in group),accuracy=sum(r['correct'] for r in group)/len(group),format_errors=sum(r['format_error'] for r in group))
        else:
            v.update(correct=sum(int(r['em']) for r in group),**{k:sum(r[k] for r in group)/len(group) for k in ('em','f1','bleu_1','bleu_2','bleu')})
        results[kind]=v
    memory_path=Path(p['memory_path']);storage=sum(f.stat().st_size for f in memory_path.iterdir())
    write_rows(target/'scored_questions.jsonl',scored)
    dump(target/'summary.json',dict(status='complete',method=METHOD,dataset=p['dataset'],questions=p['questions'],
        top_k=p['top_k'],metrics=results,query_truncated_count=p['query_truncated_count'],
        query_image_questions_without_visual_input=p['query_image_questions'],
        storage_bytes=storage,storage_mib=storage/1048576,storage_includes_image_vectors=True))
    files=[target/'summary.json',target/'scored_questions.jsonl',*answer_files]
    dump(target/'audit.json',dict(status='passed',questions=p['questions'],answers=len(payloads),
        no_images_in_memory=True,no_images_in_reader=True,caption_only_contexts_reconstructed=True,
        image_only_rankings_recomputed=True,source_ids_verified=True,
        artifact_sha256={str(f):sha(f) for f in files}))
    print(json.dumps(results),flush=True)


def add_arguments(parser):
    parser.add_argument('--clip-caption-source-root',type=Path,help='construction/benchmark source; dataset defaults use the original Qwen caption caches')
    parser.add_argument('--clip-caption-encoding-device',default='cpu',help='offline image and online text CLIP encoding device')
    parser.add_argument('--clip-caption-batch-size',type=int,default=8)
    parser.add_argument('--clip-caption-max-new-tokens',type=int,default=256)


def pipeline(args,parser):
    if args.clip_caption_batch_size<1 or args.clip_caption_max_new_tokens<1:
        parser.error('clip_caption requires positive reader batch and token limits')
    if getattr(args,'reader_backend','local')!='local':
        parser.error('clip_caption currently uses the local Qwen text-only reader')
    if getattr(args,'task',None) or getattr(args,'limit',0) or getattr(args,'limit_questions',0):
        parser.error('clip_caption evaluates full benchmark cohorts; omit task/limit filters')
    if getattr(args,'query_image_captions',False) or getattr(args,'online_weighting',None) is not None:
        parser.error('clip_caption uses question-only CLIP retrieval without caption score or weighting')
    if (getattr(args,'retrieval',None) is not None or getattr(args,'retrieval_scoring',None)=='z_reweight'
            or getattr(args,'whitening_lambda',None) is not None or getattr(args,'reweight_alpha',None) is not None
            or getattr(args,'encode_queries',False) or getattr(args,'question_ids',None)):
        parser.error('clip_caption has fixed full-scan CLIP cosine ranking; omit legacy retrieval options')
    if args.output == ROOT/'results/debug_caption_master_answer'/args.phase:
        parser.error('clip_caption requires an explicit --output directory')
    if args.phase=='offline':
        offline(args)
    else:
        if args.memory_path is None:
            parser.error('clip_caption online requires --memory-path from its offline stage')
        args.top_k=5 if args.top_k is None else args.top_k
        if args.top_k<1:parser.error('Top-K must be positive')
        for stage,fn in [('prepare',prepare),('run',run_answers),('analyze',analyze)]:
            if args.stage in (stage,'all'):fn(args)
