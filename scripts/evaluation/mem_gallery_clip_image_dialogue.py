"""Mem-Gallery: question-only CLIP Top5, then native images and source dialogue.

Gold annotations are read only by analyze(). Prepare/run never use captions,
answers, clues, or candidate-answer hypotheses in retrieval or reader inputs.
"""
import argparse
from collections import Counter
import copy
import gc
import json
import os
from pathlib import Path
import time

from scripts.evaluation import mem_gallery_master_reader as common
from scripts.memory.clip_image_memory import rank_images
import numpy as np

ROOT = common.ROOT
SOURCE = common.DEFAULT_OUTPUT / 'inputs'
OUTPUT = ROOT / 'results/mem_gallery_clip_image_dialogue_qwen_20261102'
sha, digest, dump, rows = common.sha, common.digest, common.dump, common.read_lines


def candidates_from(memories):
    grouped = {}
    for memory in memories:
        if not memory['is_visual']:
            continue
        key = (memory['task'], memory['image_sha256'])
        entry = grouped.setdefault(key, dict(task=memory['task'], image_id=memory['image_sha256'],
            path=memory['image_path'], source_memory_ids=[], source_rounds=[]))
        entry['source_memory_ids'].append(memory['id'])
        entry['source_rounds'].append({k:memory[k] for k in
            ('id','round','timestamp','image_id','dialogue')})
    return list(grouped.values())


def messages_for(question, selected):
    content = [dict(type='text', text='The retrieved memory contents are as follows:')]
    for i, candidate in enumerate(selected, 1):
        content.append(dict(type='text', text=f'Historical image {i}:'))
        content.append(dict(type='image', path=candidate['path']))
        evidence = []
        for memory in candidate['source_rounds']:
            evidence.append(f"Date: {memory['timestamp']}\nRound: {memory['round']}\n"
                f"Image ID: {memory['image_id']}\n{memory['dialogue']}")
        content.append(dict(type='text',text=f'Dialogue associated with historical image {i}:\n'+'\n\n'.join(evidence)))
    for i,path in enumerate(question['query_image_paths'],1):
        content.extend([dict(type='text',text=f'Question image {i} (part of the question):'),dict(type='image',path=path)])
    suffix = ('Please answer the question about the conversation using the retrieved memories. '
              'Return only the answer content, concisely but completely. For dates or times, '
              'provide a specific date or time whenever possible.\n\nQuestion:\n'+question['question'])
    if question['point'].upper() in common.TASK_INSTRUCTIONS:
        suffix += '\n\n'+common.TASK_INSTRUCTIONS[question['point'].upper()]
    content.append(dict(type='text',text=suffix))
    return [dict(role='system',content=common.READER_SYSTEM),dict(role='user',content=content)]


def prepare(args):
    import torch
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection
    from scripts.graph.visual_image_encoder import ImageEncoder
    out = args.output
    if (out/'protocol.json').exists():
        raise FileExistsError('Prepared protocol exists; use --stage run or a new output')
    source_manifest = json.loads((args.source/'artifact_manifest.json').read_text())
    for name in ('memories.jsonl','questions.jsonl','protocol.json'):
        if sha(args.source/name) != source_manifest[name]:
            raise ValueError('Prepared source changed: '+name)
    source_questions, memories = rows(args.source/'questions.jsonl'), rows(args.source/'memories.jsonl')
    questions = [{k:q[k] for k in ('id','task','point','question','query_image_paths')} for q in source_questions]
    if len(questions)!=1711 or len({q['id'] for q in questions})!=1711 or len(memories)!=3962:
        raise ValueError('Unexpected Mem-Gallery cohort')
    candidates = candidates_from(memories)
    if sum(len(c['source_memory_ids']) for c in candidates)!=1003:
        raise ValueError('Unexpected image-memory coverage')
    original_hashes = {m['image_path']:m['image_sha256'] for m in memories if m['is_visual']}
    for path,expected in original_hashes.items():
        if sha(path)!=expected:raise ValueError('History image changed: '+path)
    checkpoint = common.CLIP
    torch.set_num_threads(8)
    torch.manual_seed(42)
    encoder = ImageEncoder(str(checkpoint),device=args.device,local_files_only=True)
    cache={}
    for i,c in enumerate(candidates,1):
        if c['image_id'] not in cache:cache[c['image_id']]=encoder.encode(c['path'])
        if i%100==0:print(f'CLIP history {i}/{len(candidates)}',flush=True)
    image_vectors=np.asarray([cache[c['image_id']] for c in candidates],dtype=np.float64)
    image_metadata=encoder.metadata
    del encoder
    gc.collect();torch.cuda.empty_cache()
    config=AutoConfig.from_pretrained(checkpoint,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    model,loading=CLIPTextModelWithProjection.from_pretrained(checkpoint,config=config.text_config,
        local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):
        raise ValueError('Incomplete CLIP text tower')
    model.to(device=args.device,dtype=torch.float32).eval()
    limit=config.text_config.max_position_embeddings
    texts=[q['question'] for q in questions]
    lengths=[len(tokenizer.encode(s,truncation=False)) for s in texts]
    query_vectors=[]
    for start in range(0,len(texts),32):
        encoded=tokenizer(texts[start:start+32],padding=True,truncation=True,max_length=limit,return_tensors='pt').to(args.device)
        with torch.inference_mode():v=model(**encoded,return_dict=True).text_embeds.double().cpu().numpy()
        query_vectors.extend(v/np.linalg.norm(v,axis=1,keepdims=True))
    query_vectors=np.asarray(query_vectors)
    if not np.isfinite(query_vectors).all() or not np.isfinite(image_vectors).all():raise ValueError('Nonfinite vectors')
    del model,encoded;gc.collect();torch.cuda.empty_cache()
    manifest=[]
    for q,v,n in zip(questions,query_vectors,lengths):
        selected=rank_images(q,candidates,image_vectors,v,5)
        messages=messages_for(q,selected)
        paths=[p['path'] for p in messages[1]['content'] if p['type']=='image']
        hashes={p:sha(p) for p in paths}
        manifest.append(dict(**q,retrieval_query=q['question'],query_token_count=n,query_truncated=n>limit,
            retrieved_images=selected,top_memory_ids=list(dict.fromkeys(mid for c in selected for mid in c['source_memory_ids'])),
            reader_messages=messages,reader_image_paths=paths,reader_image_hashes=hashes,
            payload_sha256=digest(dict(messages=messages,image_hashes=hashes))))
    out.mkdir(parents=True,exist_ok=True)
    for name,values in [('qa_manifest.jsonl',manifest),('image_candidates.jsonl',candidates)]:
        (out/name).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
    np.savez(out/'clip_embeddings.npz',image_vectors=image_vectors,query_vectors=query_vectors,
        image_ids=np.array([c['image_id'] for c in candidates]),query_ids=np.array([q['id'] for q in questions]))
    files=[Path(__file__),Path(common.__file__),ROOT/'scripts/memory/clip_image_memory.py',
           ROOT/'scripts/graph/visual_image_encoder.py',args.source/'questions.jsonl',args.source/'memories.jsonl',
           args.source/'artifact_manifest.json',args.source/'protocol.json',args.source/'gold.jsonl',
           out/'qa_manifest.jsonl',out/'image_candidates.jsonl',out/'clip_embeddings.npz']
    files+=list(checkpoint.glob('*'))
    files+=[Path(p) for q in manifest for p in q['reader_image_paths']]
    print('Hashing frozen reader model and inputs',flush=True)
    provenance=common.model_provenance()
    contract=dict(dataset='Mem-Gallery',questions=1711,scenarios=20,top_k=5,seed=42,
        method='clip_image_dialogue',reader_model=provenance,clip_image_encoder=image_metadata,
        source=str(args.source.resolve()),device=args.device,max_new_tokens=256,batch_size=2,
        max_padded_input_tokens=24000,do_sample=False,dtype='bfloat16',attention='sdpa',
        image_max_pixels=common.QUERY_IMAGE_MAX_PIXELS,candidate_images=len(candidates),historical_image_rounds=1003,
        query_image_questions=sum(bool(q['query_image_paths']) for q in questions),
        clip_token_limit=limit,query_truncated_count=sum(n>limit for n in lengths),
        query_policy='question text only; CLIP right truncation; no query caption/image, options, candidate answers or gold',
        retrieval_policy='same scenario; five content-deduplicated historical images; stable ties; original source order',
        answer_policy='five original historical images + all their associated dialogues with date/round/image ID + question + original question image; no captions',
        excluded_history='2959 text-only dialogue rounds are outside image retrieval; all gold clues retained in evaluation denominator',
        input_hashes={str(p.resolve()):sha(p) for p in set(files) if p.is_file()})
    contract['experiment_signature']=digest(contract)
    dump(out/'protocol.json',contract)
    print(json.dumps(dict(stage='prepared',questions=1711,candidates=len(candidates),query_truncated_count=contract['query_truncated_count'])),flush=True)


def verify(out):
    protocol=json.loads((out/'protocol.json').read_text())
    if digest({k:v for k,v in protocol.items() if k!='experiment_signature'})!=protocol['experiment_signature']:
        raise ValueError('Protocol changed')
    for name,expected in protocol['input_hashes'].items():
        if sha(name)!=expected:raise ValueError('Input changed: '+name)
    manifest=rows(out/'qa_manifest.jsonl')
    if len(manifest)!=1711 or len({r['id'] for r in manifest})!=1711:raise ValueError('Incomplete/duplicate questions')
    for r in manifest:
        selected=r['retrieved_images']
        if len(selected)!=5 or len({c['image_id'] for c in selected})!=5:raise ValueError('Not five distinct images')
        if any(c['task']!=r['task'] for c in selected):raise ValueError('Cross-scenario retrieval')
        if messages_for(r,selected)!=r['reader_messages']:raise ValueError('Prompt changed')
        if digest(dict(messages=r['reader_messages'],image_hashes=r['reader_image_hashes']))!=r['payload_sha256']:
            raise ValueError('Payload changed')
    previous=rows(out/'qa_responses.jsonl') if (out/'qa_responses.jsonl').exists() else []
    by_id=common.by_id(manifest)
    common.by_id(previous)
    for r in previous:
        if r['experiment_signature']!=protocol['experiment_signature'] or r['payload_sha256']!=by_id[r['id']]['payload_sha256']:
            raise ValueError('Response provenance mismatch')
    return protocol,manifest,previous


def generate(runtime,batch,protocol):
    import torch
    from PIL import Image
    processor,model,_=runtime
    rendered=[processor.apply_chat_template(q['reader_messages'],tokenize=False,add_generation_prompt=True) for q in batch]
    images=[]
    for q in batch:
        for path in q['reader_image_paths']:
            with Image.open(path) as image:images.append(image.convert('RGB'))
    try:
        encoded=processor(text=rendered,images=images,padding=True,truncation=False,return_tensors='pt')
    finally:
        for image in images:image.close()
    width=int(encoded['input_ids'].shape[1])
    if width+protocol['max_new_tokens']>model.config.text_config.max_position_embeddings:raise ValueError('Reader would truncate')
    if width*len(batch)>protocol['max_padded_input_tokens'] and len(batch)>1:
        del encoded
        return sum([generate(runtime,[q],protocol) for q in batch],[])
    if len(encoded['image_grid_thw'])!=sum(len(q['reader_image_paths']) for q in batch):raise ValueError('Image count mismatch')
    lengths=encoded['attention_mask'].sum(dim=1).tolist()
    grids=encoded['image_grid_thw'].tolist()
    encoded=encoded.to(next(model.parameters()).device)
    torch.manual_seed(42)
    start=time.perf_counter()
    with torch.inference_mode():
        generated=model.generate(**encoded,do_sample=False,max_new_tokens=protocol['max_new_tokens'],pad_token_id=processor.tokenizer.pad_token_id)
    elapsed=time.perf_counter()-start
    eos=processor.tokenizer.eos_token_id
    result=[];offset=0
    for q,seq,length in zip(batch,generated[:,width:].tolist(),lengths):
        actual=seq[:seq.index(eos)+1] if eos in seq else seq
        raw=processor.decode(actual,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
        count=len(q['reader_image_paths'])
        result.append(dict(id=q['id'],task=q['task'],point=q['point'],raw_response=raw,
            status='ok' if raw else 'empty_response',input_tokens=length,generated_tokens=len(actual),
            hit_token_limit=len(actual)>=protocol['max_new_tokens'] and eos not in actual,
            reader_image_count=count,image_grid_thw=grids[offset:offset+count],
            dialogue_round_count=sum(len(c['source_rounds']) for c in q['retrieved_images']),
            top_memory_ids=q['top_memory_ids'],batch_size=len(batch),batch_elapsed_seconds=elapsed,
            payload_sha256=q['payload_sha256'],experiment_signature=protocol['experiment_signature']))
        offset+=count
    del encoded,generated
    return result


def run(args):
    protocol,manifest,previous=verify(args.output)
    if args.device!=protocol['device']:raise ValueError('Device differs from protocol')
    done={q['id'] for q in previous}
    pending=[q for q in manifest if q['id'] not in done]
    if args.limit_run:pending=pending[:args.limit_run]
    if not pending:return
    for name,item in protocol['reader_model']['files'].items():
        if sha(common.MODEL/name)!=item['sha256']:raise ValueError('Reader model changed')
    runtime=common.load_runtime(args.device)
    runtime[2]['experiment_signature']=protocol['experiment_signature']
    dump(args.output/f'generation_environment_{os.getpid()}.json',runtime[2])
    for start in range(0,len(pending),protocol['batch_size']):
        batch=pending[start:start+protocol['batch_size']]
        answers=generate(runtime,batch,protocol)
        common.append(args.output/'qa_responses.jsonl',answers)
        done.update(r['id'] for r in answers)
        progress=dict(completed=len(done),total=1711,last_id=answers[-1]['id'],pid=os.getpid(),
            last_batch_seconds=answers[-1]['batch_elapsed_seconds'])
        dump(args.output/'progress.json',progress)
        print(json.dumps(progress),flush=True)


def analyze(args):
    p,manifest,answers=verify(args.output)
    by_id=common.by_id(answers)
    gold=common.by_id(rows(Path(p['source'])/'gold.jsonl'))
    if set(by_id)!=set(gold) or len(by_id)!=1711:raise ValueError('Incomplete results')
    detail=[]
    for q in manifest:
        a,g=by_id[q['id']],gold[q['id']]
        if a['reader_image_count']!=5+len(q['query_image_paths']) or a['top_memory_ids']!=q['top_memory_ids']:
            raise ValueError('Actual context mismatch')
        scores=common.score_answer(a['raw_response'],g['answer'])
        detail.append(dict(id=q['id'],task=q['task'],point=q['point'],answer=g['answer'],prediction=a['raw_response'],
            **scores,gold_hits=len(set(g['gold_ids'])&set(q['top_memory_ids'])),gold_count=len(g['gold_ids']),
            hit_token_limit=a['hit_token_limit'],empty=not a['raw_response']))
    def stats(group):
        annotated=[r for r in group if r['gold_count']]
        total=sum(r['gold_count'] for r in annotated)
        return dict(questions=len(group),exact_match_correct=sum(int(r['em']) for r in group),
            **{k:sum(r[k] for r in group)/len(group) for k in ('em','f1','bleu','bleu_1','bleu_2')},
            gold_hits=sum(r['gold_hits'] for r in annotated),gold_references=total,
            recall_at5_micro=sum(r['gold_hits'] for r in annotated)/total if total else None,
            nonempty_clue_questions=len(annotated),
            recall_at5_macro_nonempty=sum(r['gold_hits']/r['gold_count'] for r in annotated)/len(annotated) if annotated else None,
            token_limit_hits=sum(r['hit_token_limit'] for r in group),empty_responses=sum(r['empty'] for r in group))
    summary=dict(status='complete',model='Qwen3-VL-8B-Instruct',method='clip_image_dialogue',top_k=5,
        overall=stats(detail),by_scenario={t:stats([r for r in detail if r['task']==t]) for t in sorted({r['task'] for r in detail})},
        by_question_type={t:stats([r for r in detail if r['point']==t]) for t in sorted({r['point'] for r in detail})},
        official_scoring_revision=common.OFFICIAL_REVISION,query_truncated_count=p['query_truncated_count'],
        denominator_policy='All 1711 QAs for answer metrics; all 4109 clues including text-only clues for micro recall; no-clue questions excluded from macro recall',
        experiment_signature=p['experiment_signature'],responses_sha256=sha(args.output/'qa_responses.jsonl'))
    (args.output/'per_question.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in detail))
    dump(args.output/'summary.json',summary)
    print(json.dumps(summary['overall']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','run','analyze','all'),required=True)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--limit-run',type=int,default=0,help='Smoke-test prefix; full cohort remains fixed')
    args=parser.parse_args()
    if args.limit_run and args.stage=='all':parser.error('Use run for a limited smoke test')
    for stage in ('prepare','run','analyze'):
        if args.stage in (stage,'all'):globals()[stage](args)


if __name__=='__main__':main()
