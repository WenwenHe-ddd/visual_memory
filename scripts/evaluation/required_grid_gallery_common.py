"""Original Gallery prompts, model inputs and five-grade evaluator for new seeds."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import copy
import time
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_api import Gateway, ENV

CANONICAL=c.ROOT/'results/mem_gallery_master_answer_20261021/inputs'

def verify_inputs(questions, observations=None):
    original={q['id']:q for q in c.rows(CANONICAL/'questions.jsonl')}
    if len(questions)!=1711 or len({q['id'] for q in questions})!=1711:
        raise ValueError('Expected 1711 distinct original questions')
    for q in questions:
        if any(q[k]!=original[q['id']][k] for k in ('task','point','question','query_image_paths','query_image_caption','retrieval_query')):
            raise ValueError('Original Gallery query/caption/pixels changed')
        if set(q)&{'answer','gold_choice','gold_ids','clue'}:raise ValueError('Labels in model inputs')
    if observations is not None and (len(observations)!=20 or sum(map(len,observations.values()))!=3962):
        raise ValueError('Expected 3962 rounds in 20 isolated tasks')

def verify_protocol(root):
    p=c.read(root/'protocol.json');c.hashes(p['source_sha256'])
    if p['experiment_signature']!=c.digest({k:v for k,v in p.items() if k!='experiment_signature'}):
        raise ValueError('New signed protocol changed')
    return p

def judge(job, records, output):
    """Same official rubric and exact fixed-seed wire; raw attempts are retained."""
    from scripts.evaluation import memgallery_multi_method_judge as native
    from scripts.evaluation.memgallery_five_grade_judge import parse_grade
    records=list(records)
    canonical={q['id']:q for q in c.rows(CANONICAL/'questions.jsonl')}
    gold={q['id']:q['answer'] for q in c.rows(CANONICAL/'gold.jsonl')}
    if len(records)!=1711 or {r['id'] for r in records}!=set(canonical):raise ValueError('Judge cohort incomplete')
    requests=[]
    for r in records:
        q=canonical[r['id']]
        row={**{k:q[k] for k in ('id','task','point','question')},'ground_truth':gold[r['id']], 'model_output':r['model_output']}
        wire=native.build_request(row,native.PROMPT.read_text())
        requests.append(dict(row,request=wire,request_sha256=native.transport.request_sha256(wire)))
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(dataset='MemGallery',answer_model=c.MODELS[job['backbone']],answer_generation_seed=job['seed'],
        method=job['method'],questions=1711,**native.judge_settings(),
        source_sha256={str(p):c.sha(p) for p in [Path(__file__),Path(native.__file__),native.PROMPT]})
    c.freeze(output/'protocol.json',protocol);c.write_rows(output/'requests.jsonl',requests)
    config=native.transport.load_config(ENV)
    def one(row):
        key=row['request_sha256'];path=output/'responses'/(key+'.json')
        with c.lock(output/'locks'/(key+'.lock')):
            if path.exists():
                result=c.read(path);native.validate_score(result,key);return dict(row,**{k:result[k] for k in ('score','reasoning','response_id','raw_response')})
            previous=c.rows(output/'attempts'/(key+'.jsonl')) if (output/'attempts'/(key+'.jsonl')).exists() else []
            for attempt in range(len(previous)+1,6):
                raw=None
                try:
                    raw=native.transport.chat_request(config,row['request'],timeout=60)
                    grade=parse_grade(raw['raw_response']);result=dict(raw,**grade)
                    native.validate_score(result,key)
                    Gateway.append(output/'attempts'/(key+'.jsonl'),dict(attempt=attempt,status='ok',**result))
                    c.freeze(path,result)
                    return dict(row,**{k:result[k] for k in ('score','reasoning','response_id','raw_response')})
                except native.transport.APIError as error:
                    Gateway.append(output/'attempts'/(key+'.jsonl'),dict(attempt=attempt,status='error',kind=error.kind,retryable=error.retryable))
                    if not error.retryable:raise
                except (ValueError,TypeError) as error:
                    Gateway.append(output/'attempts'/(key+'.jsonl'),dict(attempt=attempt,status='invalid_grade',raw=raw,error=str(error)))
                if attempt<5:time.sleep(min(2**attempt,16))
            raise RuntimeError('Five-grade Judge attempts exhausted; no substitute score')
    results={}
    with ThreadPoolExecutor(max_workers=job['workers']) as pool:
        for i,future in enumerate(as_completed([pool.submit(one,row) for row in requests]),1):
            r=future.result();results[r['id']]=r
            if i%100==0 or i==1711:print({'stage':'judge','done':i,'total':1711},flush=True)
    c.write_rows(output/'per_question.jsonl',[results[r['id']] for r in requests])
    c.dump(output/'summary.json',dict(status='complete',questions=1711,seed=42,
        llm_judge=sum(r['score'] for r in results.values())/1711,
        artifacts_sha256={str(p):c.sha(p) for p in output.rglob('*.json') if p.name!='summary.json'}))

def api_answers(job, questions, retrieval, root, cap=2048):
    from scripts.evaluation import mem_gallery_gpt5_gen_agents as ga
    if job['backbone']=='deepseek':
        from scripts.evaluation import gallery_deepseek_bounded_images as images
    else:
        from scripts.evaluation import mem_gallery_gpt5_image_formats as images
    verify_inputs(questions);g=Gateway(job);rmap={r['id']:r for r in retrieval}
    if len(retrieval)!=1711 or set(rmap)!={q['id'] for q in questions}:raise ValueError('Retrieval coverage differs')
    manifest=[dict(id=q['id'],payload=ga.reader_payload(q,rmap[q['id']]['retrieved_context'])) for q in questions]
    c.write_rows(root/'manifest.jsonl',manifest)
    def one(item):
        request=g.wire_from_native(images.wire_for(item['payload'],cap))
        output=g.call(root/'wire_cache',request,attempts=3,allow_rejection=job['backbone']=='deepseek')
        if output.get('status')!='provider_rejection' and (output['finish_reason']!='stop' or output.get('refusal') or not output['raw_response'].strip()):
            c.freeze(root/'invalid'/(c.digest(request)+'.json'),output)
            raise ValueError('Original Gallery API reader output invalid; raw retained')
        return dict(id=item['id'],payload_sha256=c.digest(item['payload']),**output)
    outputs={}
    with ThreadPoolExecutor(max_workers=job['workers']) as pool:
        for i,future in enumerate(as_completed([pool.submit(one,m) for m in manifest]),1):
            row=future.result();outputs[row['id']]=row
            c.dump(root/'progress.json',dict(done=i,total=1711))
            if i%100==0 or i==1711:print({'stage':'reader','done':i,'total':1711},flush=True)
    values=[outputs[q['id']] for q in questions];c.write_rows(root/'responses.jsonl',values)
    c.dump(root/'complete.json',dict(status='complete',seed=job['seed'],questions=1711,
        responses_sha256=c.sha(root/'responses.jsonl'),manifest_sha256=c.sha(root/'manifest.jsonl')))
    return values

def publish(job, responses, evidence, extra=None):
    from scripts.evaluation.mem_gallery_master_reader import score_answer
    qs={q['id']:q for q in c.rows(CANONICAL/'questions.jsonl')};gold={q['id']:q['answer'] for q in c.rows(CANONICAL/'gold.jsonl')}
    values=[];metrics=[]
    for r in responses:
        metrics.append(score_answer(r['raw_response'],gold[r['id']]))
        values.append(dict(id=r['id'],model_output=r['raw_response'],**{k:qs[r['id']][k] for k in ('task','point','question')}))
    root=Path(job['method_root']);judge(job,values,root/'judge')
    numbers=dict(em_correct=sum(int(r['em']) for r in metrics),
        **{k:sum(r[k] for r in metrics)/1711 for k in ('em','f1','bleu_1')})
    return c.complete(job,None,values,root/'judge',evidence,dict(**numbers,**(extra or {})))
