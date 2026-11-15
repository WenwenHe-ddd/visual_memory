"""Complete the fixed-caption 2x2 DeepSeek retrieval experiment.

New conditions: question-only full scan + online z-score (one whole domain
per group), and unweighted master-specific answer queries. All reader requests
are fresh in this run and keep the previous explicit seed42 generation settings.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import html
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'.vendor'), str(ROOT)]
for key,value in {'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                  'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}.items():
    os.environ.setdefault(key,value)
import numpy as np
from scripts.analysis import memeye_deepseek_flash_seed42_zscore as prior
from scripts.analysis.deepseek_seed42_audit_helpers import wire_request, validate_response
from scripts.evaluation import deepseek_flash_seed42_audited_transport as transport
from scripts.evaluation.memeye_gpt5_reader import read, lines, sha, digest, dump, append, write_lines, indexed

SOURCE = prior.SOURCE
OUTPUT = SOURCE/'complementary_evaluation'
ENV = prior.ENV
MODEL = prior.MODEL
BASE = 'baseline_question_only_zscore'
MASTER = 'master_answer_query'
CONDITIONS = (BASE, MASTER)
RIDGE, ALPHA = .03, 1.
choice = prior.choice
checked_hashes = prior.checked_hashes
freeze = prior.freeze


def retrieve(output=OUTPUT, source=SOURCE, device='cuda:1'):
    output, source = Path(output).resolve(), Path(source).resolve()
    marker = output/'retrieval_protocol.json'
    if marker.exists():
        p=read(marker);checked_hashes(p['source_sha256']);checked_hashes(p['output_sha256'])
        assert p['memory_path']==str(source) and p['conditions']==list(CONDITIONS)
        return p
    proof=prior.verify_source(source)
    proof.update(captions_regenerated_this_extension=False, graph_rebuilt_this_extension=False)
    from scripts.memory import memeye_backbone_retrieval as original
    from scripts.memory.memeye_backbone_long_text import encode_complete
    inp=read(source/'inputs/protocol.json')
    memories,questions,model=original.attach_captions(lines(source/'inputs/source_memories.jsonl'),
        lines(source/'inputs/source_questions.jsonl'),lines(source/'captions.jsonl'))
    assert model==MODEL
    previous=indexed(lines(source/'retrieval_results.jsonl'),'question_id')
    args=SimpleNamespace(device=device);original.configure_device(args)
    encoder=original.load_text_encoder(inp['text_encoder'],args.device)
    output.mkdir(parents=True,exist_ok=True);dump(output/'source_validation.json',proof)
    results=[];encoding=[];max_raw_error=max_inverse_error=0.;checked_ranks=0
    for domain in sorted({m['task'] for m in memories}):
        ms=[m for m in memories if m['task']==domain];qs=[q for q in questions if q['task']==domain]
        e,ei=encode_complete(encoder,[m['retrieval_text'] for m in ms],ids=[m['id'] for m in ms],field='dialogue+caption')
        texts=[];ids=[]
        for q in qs:
            texts.extend([q['retrieval_query']]+[q['retrieval_query']+'\nCandidate answer: '+q['candidate_statements'][k] for k in 'ABCD'])
            ids.extend(q['id']+suffix for suffix in ('/baseline','/A','/B','/C','/D'))
        qv,qi=encode_complete(encoder,texts,inp['query_prefix'],ids=ids,field='query')
        qv=qv.reshape(len(qs),5,-1)
        for ix,q in enumerate(qs):
            old=previous[q['id']];raw_conditions=indexed(old['conditions'],'name')
            assert [l['memory_id'] for l in old['leaves']]==[m['id'] for m in ms]
            raw=e@qv[ix,0]
            master_raw=np.empty(len(ms))
            for g in old['masters']:
                member_indices=[i for i,l in enumerate(old['leaves']) if l['master_id']==g['master_id']]
                assert g['augmented_query']==q['retrieval_query']+'\nCandidate answer: '+q['candidate_statements'][g['selected_choice']]
                master_raw[member_indices]=e[member_indices]@qv[ix,1+'ABCD'.index(g['selected_choice'])]
            for values,name,field in ((raw,'baseline_question_only','baseline_score'),(master_raw,MASTER,'expanded_score')):
                err=float(np.max(np.abs(values-[l[field] for l in old['leaves']])))
                assert err<2e-6;max_raw_error=max(max_raw_error,err)
                order=sorted(range(len(ms)),key=lambda i:-values[i])
                assert [ms[i]['id'] for i in order]==raw_conditions[name]['full_rank_memory_ids']
                checked_ranks+=1
            scores,err=prior.checked_weighting(qv[ix,0],e)
            max_inverse_error=max(max_inverse_error,err)
            np.testing.assert_allclose(raw,scores['raw'],atol=2e-15,rtol=0)
            gid=domain+'::all_memory'
            leaves=[{'memory_id':m['id'],'leaf_index':i,'group_id':gid,
                     **{k:float(scores[k][i]) for k in ('raw','white','z','correction','weighted')}} for i,m in enumerate(ms)]
            order=sorted(range(len(ms)),key=lambda i:-leaves[i]['weighted'])
            weighted={'name':BASE,'query':q['retrieval_query'],
                'groups':[{'group_id':gid,'size':len(ms),**{k:scores[k] for k in ('raw_std','white_std','raw_status','z_status')}}],
                'leaves':leaves,'top5_memory_ids':[ms[i]['id'] for i in order[:5]],
                'top5_scores':[leaves[i]['weighted'] for i in order[:5]],'full_rank_memory_ids':[ms[i]['id'] for i in order]}
            master={k:raw_conditions[MASTER][k] for k in ('name','top5_memory_ids','top5_scores','full_rank_memory_ids')}
            results.append({'question_id':q['id'],'task':domain,'conditions':[weighted,master]})
        encoding.append({'domain':domain,'leaf':ei,'query':qi})
        print(json.dumps({'stage':'retrieve','domain':domain,'questions_done':len(results)}),flush=True)
    assert len(results)==371 and checked_ranks==742
    write_lines(output/'retrieval_results.jsonl',results)
    paths=[Path(__file__),Path(prior.__file__),Path(original.__file__),
        ROOT/'scripts/memory/memeye_backbone_long_text.py',ROOT/'scripts/memory/online_zscore.py',ROOT/'scripts/memory/ridge_whitening.py']
    sources={**proof['source_sha256'],**{str(p):sha(p) for p in paths}}
    checked_hashes(sources)
    p={'status':'complete','memory_path':str(source),'caption_model':MODEL,'caption_seed':42,
       'conditions':list(CONDITIONS),'lambda':RIDGE,'alpha':ALPHA,'questions':371,'memory_leaves':438,'masters':86,
       'baseline_zscore_group':'All candidate memories of the question domain, before Top5 selection',
       'master_online_weighting':'none','raw_rankings_reproduced':checked_ranks,'inverse_ridge_groups_checked':371,
       'max_raw_score_difference':max_raw_error,'max_inverse_ridge_difference':max_inverse_error,
       'device':args.device,'encoding':encoding,'gold_loaded_for_ranking':False,
       'captions_regenerated_this_extension':False,'graph_modified':False,'vectors_persisted':False,
       'source_sha256':sources,'output_sha256':{str(output/n):sha(output/n) for n in ('source_validation.json','retrieval_results.jsonl')}}
    dump(marker,p);return p


def annotations(source,system):
    old=prior.annotations(source,system)
    base=copy.deepcopy(old['baseline_question_only']);master=copy.deepcopy(old['master_answer_query_zscore'])
    base['3. 检索方式'].update(online_weighting='z_score',lambda_=RIDGE,alpha=ALPHA,
        group='当前 domain 全部候选记忆组成一个簇，在 Top5 选择之前计算',
        final_score='raw + alpha * std_pop(raw) * tanh(z_white)')
    base['3. 检索方式']['lambda']=base['3. 检索方式'].pop('lambda_')
    master['3. 检索方式'].update(online_weighting='none',alpha=None,
        final_score='BGE cosine between each leaf and its master-specific query')
    master['3. 检索方式']['lambda']=None
    return {BASE:base,MASTER:master}


def prepare(output=OUTPUT,source=SOURCE,workers=3,env_file=ENV):
    output,source=Path(output).resolve(),Path(source).resolve();folder=output/'reader'
    if (folder/'protocol.json').exists():return verify(output)[0]
    ret=read(output/'retrieval_protocol.json');checked_hashes(ret['source_sha256']);checked_hashes(ret['output_sha256'])
    assert ret['memory_path']==str(source) and 1<=workers<=8
    cfg=transport.load_config(env_file);system=read(source/'inputs/protocol.json')['reader_system_prompt']
    questions=indexed(lines(source/'inputs/source_questions.jsonl'),'id')
    memory={m['id']:m['retrieval_text'] for m in lines(source/'offline/memories.jsonl')}
    raw=indexed(lines(source/'retrieval_results.jsonl'),'question_id')
    records=[];master_matches=0
    for i,result in enumerate(lines(output/'retrieval_results.jsonl')):
        q=questions[result['question_id']];pair=[]
        for condition in result['conditions']:
            name=condition['name'];top=condition['top5_memory_ids']
            assert len(top)==len(set(top))==5 and all(mid.split('/',1)[0]==q['task'] for mid in top)
            context='\n\n'.join(f'Memory {j}: {memory[mid]}' for j,mid in enumerate(top,1))
            if name==MASTER:
                original=next(c for c in raw[q['id']]['conditions'] if c['name']==MASTER)
                assert top==original['top5_memory_ids'] and context==original['reader_context'];master_matches+=1
            text=('Retrieved dialogue and image descriptions (ranked by relevance):\n\n'+context+
                  '\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(f'{k}. {q["options"][k]}' for k in 'ABCD')+
                  '\n\nAnswer with one letter only.')
            payload={'system':system,'user_text':text,'query_images':q['query_images']}
            wire=transport.build_request(payload,model=MODEL,max_output_tokens=16,image_detail='high')
            assert wire==wire_request(payload,MODEL,16,'high') and wire['seed']==42
            pair.append({'record_id':q['id']+'::'+name,'question_id':q['id'],'task':q['task'],'condition':name,
                'top_memory_ids':top,**payload,'payload_sha256':digest(payload),'request_sha256':digest(wire)})
        records.extend(pair if i%2==0 else reversed(pair))
    assert len(records)==742 and master_matches==371
    folder.mkdir(parents=True,exist_ok=True);write_lines(folder/'requests.jsonl',records)
    write_lines(folder/'gold.jsonl',lines(source/'inputs/evaluation.jsonl'))
    previous=source/'weighted_evaluation'
    audit=read(previous/'external_audit.json');assert audit['passed']
    checked_hashes({str(previous/n):h for n,h in audit['artifact_sha256'].items()})
    paths=[Path(__file__),Path(transport.__file__),ROOT/'scripts/analysis/audit_deepseek_seed42_complementary.py',
        ROOT/'scripts/analysis/deepseek_seed42_audit_helpers.py',ROOT/'scripts/analysis/audit_memeye_deepseek_flash_alias.py',
        output/'retrieval_protocol.json',folder/'requests.jsonl',folder/'gold.jsonl',
        previous/'summary.json',previous/'reader/fresh_responses.jsonl',previous/'external_audit.json']
    sources={**ret['source_sha256'],**ret['output_sha256'],**{str(p):sha(p) for p in paths}}
    c={'model':MODEL,'provider':'DashScope International','api_config':cfg.public_dict(),'api_env_file':str(Path(env_file).resolve()),
       'memory_path':str(source),'conditions':list(CONDITIONS),'lambda':RIDGE,'alpha':ALPHA,'seed':42,'caption_seed':42,
       'temperature':0,'enable_thinking':False,'reader_max_output_tokens':16,'image_detail':'high',
       'workers':workers,'max_attempts':3,'timeout_seconds':60,'questions':371,'logical_records':742,
       'unique_requests':len({r['request_sha256'] for r in records}),'gold_sent_to_api':False,
       'historical_answers_reused':False,'same_run_full_request_deduplication':True,
       'failure_policy':{'input_inspection_rejection':'Retry unchanged at most three times, then retain as incorrect.',
           'invalid_response':'Retain first original response as incorrect; no changed prompt or token budget.',
           'other_api_errors':'Retry retryable errors up to three attempts, then stop; never invent an answer.',
           'denominator':'All 371 questions remain in each condition.'},
       'same_run_master_context_matches':master_matches,'method_annotations':annotations(source,system),'source_sha256':sources}
    p={'run_id':str(uuid.uuid4()),'experiment_signature':digest(c),'contract':c,
       'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(folder/'protocol.json',p);verify(output)
    print(json.dumps({'stage':'prepared','records':742,'unique_requests':c['unique_requests'],'seed':42}),flush=True)
    return p


def verify(output=OUTPUT):
    output=Path(output);p=read(output/'reader/protocol.json');c=p['contract']
    assert digest(c)==p['experiment_signature'] and c['seed']==42 and c['model']==MODEL
    assert c['conditions']==list(CONDITIONS) and c['lambda']==RIDGE and c['alpha']==ALPHA
    checked_hashes(c['source_sha256'])
    records=lines(output/'reader/requests.jsonl');assert len(records)==len(indexed(records,'record_id'))==742
    assert len({r['request_sha256'] for r in records})==c['unique_requests']
    for row in records:
        payload={k:row[k] for k in ('system','user_text','query_images')}
        wire=wire_request(payload,MODEL,16,'high')
        assert digest(wire)==row['request_sha256'] and digest(payload)==row['payload_sha256']
        assert wire==transport.build_request(payload,model=MODEL,max_output_tokens=16,image_detail='high')
    return p,records


def valid_answer(response):
    return (response['returned_model']==MODEL and response['finish_reason']=='stop'
            and not response.get('refusal') and choice(response['raw_response']) is not None)


def validate_outcome(ans,row,p):
    for k,v in {'record_id':row['record_id'],'request_sha256':row['request_sha256'],
                'payload_sha256':row['payload_sha256'],'run_id':p['run_id'],
                'experiment_signature':p['experiment_signature'],'seed':42,'historical_response_reuse':False}.items():
        assert ans[k]==v,(k,row['record_id'])
    if ans['status']=='ok':
        validate_response(ans,row,p,'reader');assert ans['predicted_choice']==choice(ans['raw_response'])
    elif ans['status']=='invalid_response':
        assert ans['returned_model']==ans['requested_model']==MODEL and not valid_answer(ans)
        assert ans['predicted_choice'] is None and ans['response_id']
        assert ans['enable_thinking'] is False and ans['reasoning_content_present'] is False and ans['reasoning_tokens'] in (None,0)
        if row['query_images']:assert ans['usage']['prompt_tokens_details']['image_tokens']>0
    else:
        assert ans['status']=='provider_input_rejected' and ans['provider_error_code']=='data_inspection_failed'
        assert ans['attempts']==3 and ans['response_id'] is None and ans['raw_response'] is None
        assert ans['predicted_choice'] is None


def run(output=OUTPUT,limit=0):
    output=Path(output);folder=output/'reader';p,records=verify(output);c=p['contract']
    cfg=transport.load_config(c['api_env_file']);assert cfg.public_dict()==c['api_config']
    reps={}
    for row in records:reps.setdefault(row['request_sha256'],row)
    existing=lines(folder/'outcomes.jsonl') if (folder/'outcomes.jsonl').exists() else []
    completed=indexed(existing,'request_sha256')
    for h,a in completed.items():validate_outcome(a,reps[h],p)
    pending=[r for h,r in reps.items() if h not in completed]
    if limit:pending=pending[:limit]
    execution_id=str(uuid.uuid4())
    def execute(row):
        payload={k:row[k] for k in ('system','user_text','query_images')}
        body=transport.build_request(payload,model=MODEL,max_output_tokens=16,image_detail='high')
        assert digest(body)==row['request_sha256'] and body['seed']==42
        provenance={k:row[k] for k in ('record_id','question_id','condition','payload_sha256','request_sha256')}
        provenance.update(run_id=p['run_id'],experiment_signature=p['experiment_signature'],execution_id=execution_id,
                          seed=42,historical_response_reuse=False)
        attempts=[]
        for attempt in range(1,c['max_attempts']+1):
            event={'record_id':row['record_id'],'request_sha256':row['request_sha256'],'execution_id':execution_id,'attempt':attempt}
            try:response=transport.chat_request(cfg,body,timeout=c['timeout_seconds'])
            except transport.APIError as error:
                code=getattr(error,'provider_error_code',None)
                attempts.append({**event,'status':'api_error','kind':error.kind,'http_status':error.http_status,
                                 'message':error.message,'provider_error_code':code})
                rejection=error.http_status==400 and code=='data_inspection_failed'
                if rejection and attempt==c['max_attempts']:
                    return {**provenance,'status':'provider_input_rejected','provider_error_code':code,
                        'http_status':400,'attempts':attempt,'retry_count':attempt-1,'raw_response':None,
                        'response_id':None,'predicted_choice':None},attempts
                if not (error.retryable or rejection) or attempt==c['max_attempts']:return None,attempts
                time.sleep(min(2**attempt,8));continue
            valid=valid_answer(response)
            ans={**response,**provenance,'attempts':attempt,'retry_count':attempt-1,
                 'status':'ok' if valid else 'invalid_response','predicted_choice':choice(response['raw_response']) if valid else None}
            attempts.append({**event,'status':ans['status'],'response_id':ans['response_id']})
            return ans,attempts
    with ThreadPoolExecutor(max_workers=c['workers']) as pool:
        for start in range(0,len(pending),c['workers']):
            futures={pool.submit(execute,r):r for r in pending[start:start+c['workers']]};blocked=[]
            for future in as_completed(futures):
                row=futures[future];ans,attempts=future.result()
                for event in attempts:append(folder/'api_attempts.jsonl',event)
                if ans is None:blocked.append(row['record_id']);continue
                validate_outcome(ans,row,p)
                if ans['response_id'] is not None:append(folder/'fresh_responses.jsonl',ans)
                else:append(folder/'terminal_rejections.jsonl',ans)
                append(folder/'outcomes.jsonl',ans);completed[row['request_sha256']]=ans
            counts=dict(Counter(a['status'] for a in completed.values()))
            status={'status':'stopped_on_error' if blocked else 'running','completed_unique':len(completed),
                    'expected_unique':len(reps),'outcome_counts':counts,'blocked_records':blocked}
            dump(output/'execution_status.json',status);print(json.dumps({'stage':'reader',**status}),flush=True)
            if blocked:raise RuntimeError('Unexpected API failure; completed outputs retained without changing inputs.')
    if set(completed)==set(reps):return analyze(output)
    return {'status':'partial','completed_unique':len(completed)}


def metrics(rows,name):
    n=len(rows);total=sum(r['gold_count'] for r in rows);hits=sum(r[name+'_gold_hits'] for r in rows)
    correct=sum(r[name+'_correct'] for r in rows)
    return {'correct':correct,'questions':n,'accuracy':correct/n,'gold_hits_top5':hits,'gold_references':total,
        'recall_at_5_micro':hits/total,'recall_at_5_macro':sum(r[name+'_gold_hits']/r['gold_count'] for r in rows)/n,
        'questions_with_any_gold':sum(r[name+'_gold_hits']>0 for r in rows),
        'failed_records':sum(r[name+'_status']!='ok' for r in rows)}


def analyze(output=OUTPUT):
    output=Path(output);folder=output/'reader';p,records=verify(output);c=p['contract'];source=Path(c['memory_path'])
    outcomes=indexed(lines(folder/'outcomes.jsonl'),'request_sha256');reps={}
    for r in records:reps.setdefault(r['request_sha256'],r)
    assert set(outcomes)==set(reps)
    for h,a in outcomes.items():validate_outcome(a,reps[h],p)
    fresh=lines(folder/'fresh_responses.jsonl')
    assert {a['request_sha256'] for a in fresh}=={h for h,a in outcomes.items() if a['response_id'] is not None}
    assert len({a['response_id'] for a in fresh})==len(fresh)
    old=lines(source/'weighted_evaluation/reader/fresh_responses.jsonl')
    assert not {a['response_id'] for a in old}&{a['response_id'] for a in fresh}
    gold=indexed(lines(folder/'gold.jsonl'),'id');questions=indexed(lines(source/'inputs/source_questions.jsonl'),'id')
    assert set(gold)==set(questions) and len(gold)==371
    logical=[];by_key={}
    for r in records:
        ans=outcomes[r['request_sha256']];qid=r['question_id'];pred=ans['predicted_choice']
        final={**ans,'record_id':r['record_id'],'question_id':qid,'condition':r['condition'],'top_memory_ids':r['top_memory_ids'],
               'correct':ans['status']=='ok' and pred==gold[qid]['gold_choice'],
               'fresh_inference_executed_for_this_record':ans['response_id'] is not None and ans['record_id']==r['record_id'],
               'same_run_shared_inference_record_id':None if ans['record_id']==r['record_id'] else ans['record_id']}
        logical.append(final);by_key[qid,r['condition']]=final
    write_lines(folder/'responses.jsonl',logical)
    per_question=[]
    for qid,g in gold.items():
        row={'question_id':qid,'domain':questions[qid]['task'],'gold_choice':g['gold_choice'],'gold_count':len(set(g['gold_ids']))}
        for name in CONDITIONS:
            a=by_key[qid,name]
            row.update({name+'_answer':a['predicted_choice'],name+'_correct':a['correct'],name+'_status':a['status'],
                        name+'_gold_hits':len(set(a['top_memory_ids'])&set(g['gold_ids'])),name+'_top5':a['top_memory_ids']})
        per_question.append(row)
    assert sum(r['gold_count'] for r in per_question)==1041
    overall={n:metrics(per_question,n) for n in CONDITIONS}
    ann=copy.deepcopy(c['method_annotations'])
    for n in CONDITIONS:ann[n]['5. 回答方式'].update(overall[n])
    attempts=lines(folder/'api_attempts.jsonl')
    for h,a in outcomes.items():
        events=[e for e in attempts if e['request_sha256']==h]
        assert len(events)==a['attempts']
        assert sum(e['status']==a['status'] for e in events)==(0 if a['status']=='provider_input_rejected' else 1)
    summary={'status':'complete','model':MODEL,'provider':c['provider'],'seed':42,'caption_seed':42,'lambda':RIDGE,'alpha':ALPHA,
        'caption_source':str(source),'captions_regenerated_this_extension':False,'graph_modified':False,'caption_images':437,
        'memory_leaves':438,'masters':86,'overall':overall,
        'by_domain':{d:{n:metrics([r for r in per_question if r['domain']==d],n) for n in CONDITIONS} for d in sorted({r['domain'] for r in per_question})},
        'logical_records':742,'unique_requests':len(outcomes),'fresh_model_responses':len(fresh),
        'same_run_shared_records':742-len(outcomes),'api_attempts':len(attempts),'historical_answers_reused':False,
        'unique_outcome_counts':dict(Counter(a['status'] for a in outcomes.values())),
        'failed_records':[{k:r[k] for k in ('record_id','condition','status','raw_response')} for r in logical if r['status']!='ok'],
        'all_failures_count_as_incorrect':True,'generation_parameters':{k:c[k] for k in ('seed','temperature','enable_thinking','reader_max_output_tokens')},
        'method_annotations':ann,'usage':{k:sum((r.get('usage') or {}).get(k,0) for r in fresh) for k in ('prompt_tokens','completion_tokens','total_tokens')},
        'limitations':['Previous two conditions were answered in an earlier batch; shared seed does not make those historical responses a simultaneous reader control.']}
    dump(output/'summary.json',summary);dump(output/'paired_results.json',per_question)
    for name,rows in (('metrics.csv',[{'condition':n,**m} for n,m in overall.items()]),('paired_results.csv',per_question)):
        with (output/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    previous=read(source/'weighted_evaluation/summary.json')
    ordered=['baseline_question_only',BASE,MASTER,'master_answer_query_zscore']
    combined={n:{**(overall[n] if n in overall else previous['overall'][n]),
                  'evaluation_batch':'new_complementary' if n in overall else 'previous_seed42'} for n in ordered}
    dump(output/'four_condition_comparison.json',{'model':MODEL,'seed':42,'same_caption_sha256':sha(source/'captions.jsonl'),
        'lambda':RIDGE,'alpha':ALPHA,'conditions':combined,'prior_summary':str(source/'weighted_evaluation/summary.json'),
        'new_summary':str(output/'summary.json'),'reader_batches_differ':True})
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>DeepSeek seed42 四组检索比较</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 18px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
        '<h1>DeepSeek V4.1 Flash：四组检索比较</h1><p>同一份 seed=42 caption、438 条记忆和 86 个 master。zscore 参数 λ=0.03、α=1；回答 seed=42、temperature=0、关闭 thinking、预算 16 tokens。</p>',
        '<p>本轮补跑 baseline_question_only + zscore（当前 domain 全量记忆为一个簇）以及不加权 master_answer_query。新两组全部重新回答，未复用上一批回答；完全相同的请求仅在本轮内共享新回答。</p>',
        '<table><tr><th>方法</th><th>准确率</th><th>Recall@5 micro</th><th>Recall@5 macro</th><th>回答批次</th></tr>']
    for n,m in combined.items():
        body.append(f'<tr><td>{n}</td><td>{m["correct"]}/371 ({m["accuracy"]:.4%})</td><td>{m["gold_hits_top5"]}/1041 ({m["recall_at_5_micro"]:.4%})</td><td>{m["recall_at_5_macro"]:.4%}</td><td>{m["evaluation_batch"]}</td></tr>')
    body+=['</table>',f'<p>本轮 {len(outcomes)} 个唯一请求、{len(fresh)} 个实际模型响应、{742-len(outcomes)} 条同轮共享；共 {len(summary["failed_records"])} 条失败，全部按错误保留在各组 371 题分母中。上一批有一条 baseline 输出截断和一条 master+zscore 平台拒绝，同样按错误计分。</p>',
        '<p>四组共享 caption、检索输入和回答参数；两批回答时间不同，因此准确率差异仍可能包含接口回答波动，不能全部归因于检索方式。</p>',
        '<p><a href="summary.json">本轮统计与五项标注</a> · <a href="four_condition_comparison.json">四组比较</a> · <a href="paired_results.csv">逐题 CSV</a> · <a href="external_audit.json">独立核验</a></p>',
        '<details><summary>方法标注</summary><pre>'+html.escape(json.dumps(ann,ensure_ascii=False,indent=2))+'</pre></details></html>']
    (output/'report.html').write_text('\n'.join(body))
    dump(output/'execution_status.json',{'status':'complete','completed_unique':len(outcomes),'expected_unique':len(outcomes),
        'outcome_counts':summary['unique_outcome_counts'],'logical_records':742})
    print(json.dumps({'status':'complete','overall':overall,'outcome_counts':summary['unique_outcome_counts']}),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--memory',type=Path,default=SOURCE)
    parser.add_argument('--stage',choices=['retrieve','prepare','run','analyze','all'],default='all')
    parser.add_argument('--device',default='cuda:1');parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--env-file',type=Path,default=ENV);parser.add_argument('--limit',type=int,default=0)
    a=parser.parse_args()
    if a.stage in ('retrieve','all'):retrieve(a.output,a.memory,a.device)
    if a.stage in ('prepare','all'):prepare(a.output,a.memory,a.workers,a.env_file)
    if a.stage in ('run','all'):run(a.output,a.limit)
    if a.stage=='analyze':analyze(a.output)
