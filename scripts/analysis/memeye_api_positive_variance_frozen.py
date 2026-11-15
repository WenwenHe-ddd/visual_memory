"""MemEye CVM + positive variance with frozen GPT-5/DeepSeek inputs and answers.

Use the canonical saved controls, not freshly answered controls. Only novel
complete reader requests are sent. Historical data hashes remain mandatory;
old runtime hashes are recorded separately because the pipeline has evolved.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for key, value in {'HF_HUB_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false',
                   'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4'}.items():
    os.environ.setdefault(key, value)
import numpy as np
from scripts.evaluation.memeye_gpt5_reader import read, lines, sha, digest, dump, append, write_lines, indexed, parse_choice
from scripts.evaluation import openai_gpt5_base_transport as gpt
from scripts.evaluation import deepseek_flash_seed42_audited_transport as ds
from scripts.memory.online_positive_variance import PositiveVarianceWeighting

RAW = 'master_answer_query'
PV = RAW + '+positive_variance'
RIDGE, ALPHA = .03, 1.
DEFAULT = ROOT / 'results/memeye_api_pv_frozen_l0p03_a1_20261108'
SPECS = {
    'gpt5': {
        'source': ROOT/'results/memeye_gpt5_base_seed42_full_20261024',
        'reference': ROOT/'results/memeye_gpt5_qwen_five_validation_20261024',
        'audit': 'independent_audit.json', 'requested_model':'gpt-5',
        'returned_model':'gpt-5-2025-08-07', 'caption_seed':None, 'raw_correct':221},
    'deepseek': {
        'source': ROOT/'results/memeye_deepseek_flash_seed42_full_20261025',
        'reference': ROOT/'results/memeye_deepseek_flash_seed42_full_20261025/complementary_evaluation',
        'audit':'external_audit.json', 'requested_model':'deepseek-v4.1-flash',
        'returned_model':'deepseek-v4.1-flash', 'caption_seed':42, 'raw_correct':209},
}


def hashes(values):
    for path, expected in values.items():
        assert sha(path) == expected, f'Changed artifact: {path}'


def transport(name):
    return gpt if name == 'gpt5' else ds


def wire(name, payload, caption=False):
    if name == 'gpt5':
        return gpt.build_request(payload, seed=None if caption else 42,
                                 max_output_tokens=2048, image_detail='high')
    return ds.build_request(payload, model=SPECS[name]['requested_model'],
                            max_output_tokens=1024 if caption else 16, image_detail='high')


def verify_data_source(name):
    spec = SPECS[name]; source = spec['source']; ref = spec['reference']
    pinned, changed_code = {}, {}
    def collect(mapping, base=ROOT):
        for path, expected in mapping.items():
            p = Path(path); p = p if p.is_absolute() else base/p
            actual = sha(p)
            if p.suffix == '.py' and p.is_relative_to(ROOT/'scripts'):
                if actual != expected:
                    changed_code[str(p)] = {'historical':expected, 'current':actual}
                continue
            assert actual == expected, str(p)
            if str(p) in pinned: assert pinned[str(p)] == expected
            pinned[str(p)] = expected
    for folder in [source, ref]:
        for file in ['protocol.json', 'retrieval_protocol.json', 'reader/protocol.json']:
            p = folder/file
            if not p.exists(): continue
            doc = read(p); c = doc.get('contract', doc)
            if 'contract' in doc: assert digest(c) == doc['experiment_signature']
            for key in ['source_sha256', 'output_sha256', 'artifact_sha256']:
                collect(doc.get(key, {}), folder)
                if c is not doc: collect(c.get(key, {}), folder)
            pinned[str(p)] = sha(p)
    inp = read(source/'inputs/protocol.json')
    collect(inp['source_sha256']); collect(inp['artifact_sha256'],source/'inputs')
    audit = read(ref/spec['audit']); assert audit['passed']
    collect(audit['artifact_sha256'],ref)
    pinned[str(ref/spec['audit'])] = sha(ref/spec['audit'])
    cp = read(source/'caption_generation/protocol.json')
    assert sha(source/'caption_generation/requests.jsonl') == cp['requests_sha256']
    root = read(source/'protocol.json')
    assert cp['run_id'] == root['run_id'] and cp['experiment_signature'] == root['experiment_signature']
    reqs = indexed(lines(source/'caption_generation/requests.jsonl'),'record_id')
    answers = indexed(lines(source/'caption_generation/fresh_responses.jsonl'),'request_sha256')
    caps = indexed(lines(source/'captions.jsonl'),'id')
    assert len(reqs) == len(answers) == len(caps) == 437
    for cap in caps.values():
        r = reqs[cap['source_response_record_id']]; a = answers[cap['request_sha256']]
        payload = {k:r[k] for k in ['system','user_text','query_images'] if k in r}
        assert digest(payload) == r['payload_sha256'] == a['payload_sha256']
        assert digest(wire(name,payload,True)) == r['request_sha256'] == cap['request_sha256']
        assert a['requested_model'] == spec['requested_model']
        assert a['returned_model'] == cap['model'] == spec['returned_model']
        assert cap['caption'] == a['raw_response'].strip()
        assert a['status'] == 'ok' and a['finish_reason'] == 'stop' and not a.get('refusal')
        assert a['run_id'] == root['run_id'] and a['experiment_signature'] == root['experiment_signature']
        if name == 'deepseek': assert cap['seed'] == a['seed'] == 42
    for f in ['captions.jsonl','offline/memories.jsonl','caption_generation/requests.jsonl',
              'caption_generation/fresh_responses.jsonl','caption_generation/protocol.json']:
        pinned[str(source/f)] = sha(source/f)
    return {'caption_images':437,'caption_model':spec['returned_model'],'caption_seed':spec['caption_seed'],
            'source_sha256':pinned,'historical_runtime_changes':changed_code,
            'runtime_policy':'Verify historical data exactly; pin current runtime separately; reproduce all raw scores and rankings.'}


def retrieve(name, out, device):
    target = out/'retrieval_protocol.json'
    if target.exists():
        p = read(target); hashes(p['source_sha256']); hashes(p['output_sha256']); return p
    spec = SPECS[name]; source = spec['source']; proof = verify_data_source(name)
    from scripts.memory import memeye_backbone_retrieval as original
    from scripts.memory.memeye_backbone_long_text import encode_complete
    memories, questions, model = original.attach_captions(lines(source/'inputs/source_memories.jsonl'),
        lines(source/'inputs/source_questions.jsonl'),lines(source/'captions.jsonl'))
    assert model == spec['returned_model'] and len(memories) == 438 and len(questions) == 371
    assert [{k:v for k,v in m.items() if k not in ['image_sha256','image_path']} for m in memories] == lines(source/'offline/memories.jsonl')
    prior = indexed(lines(source/'retrieval_results.jsonl'),'question_id')
    inp = read(source/'inputs/protocol.json')
    args = SimpleNamespace(device=device); original.configure_device(args)
    encoder = original.load_text_encoder(inp['text_encoder'],args.device)
    rows, audits, masters = [], [], {}
    raw_error = inverse_error = 0.; group_checks = rank_checks = 0
    out.mkdir(parents=True,exist_ok=True)
    for domain in sorted({m['task'] for m in memories}):
        ms = [m for m in memories if m['task']==domain]; qs = [q for q in questions if q['task']==domain]
        graph = read(source/'domains'/domain/'memory/graph.json')
        nodes = indexed(graph['image_nodes'],'id'); groups = indexed(graph['masters'],'id')
        masters[domain] = len(groups)
        assert len(nodes) == len(ms)
        for m in ms:
            assert all(nodes[m['image_id']][k] == m[k] for k in ['caption','dialogue','timestamp'])
        e, ei = encode_complete(encoder,[m['retrieval_text'] for m in ms],ids=[m['id'] for m in ms],field='dialogue+caption')
        texts, ids = [], []
        for q in qs:
            texts.extend([q['retrieval_query']]+[q['retrieval_query']+'\nCandidate answer: '+q['candidate_statements'][k] for k in 'ABCD'])
            ids.extend(q['id']+s for s in ['/baseline','/A','/B','/C','/D'])
        qv, qi = encode_complete(encoder,texts,inp['query_prefix'],ids=ids,field='query')
        qv = qv.reshape(len(qs),5,-1)
        for ix,q in enumerate(qs):
            old = prior[q['id']]; leaves = [None]*len(ms); details = []
            assert old['baseline_retrieval_query'] == q['retrieval_query']
            assert [l['memory_id'] for l in old['leaves']] == [m['id'] for m in ms]
            for g in old['masters']:
                gi = [i for i,l in enumerate(old['leaves']) if l['master_id']==g['master_id']]
                assert [ms[i]['image_id'] for i in gi] == groups[g['master_id']]['member_ids']
                assert g['augmented_query'] == q['retrieval_query']+'\nCandidate answer: '+q['candidate_statements'][g['selected_choice']]
                assert g['selected_statement'] == q['candidate_statements'][g['selected_choice']]
                query = qv[ix,1+'ABCD'.index(g['selected_choice'])]; emb = e[gi]
                scores = PositiveVarianceWeighting(RIDGE,ALPHA).apply(query,emb)
                centered = emb-emb.mean(0)
                # Independent dual solve checks the positive sign without SVD.
                check = emb@query if len(gi)==1 else emb@(query+centered.T@np.linalg.solve(
                    centered@centered.T+(len(gi)-1)*RIDGE*np.eye(len(gi)),centered@query))
                np.testing.assert_allclose(check,scores['positive_variance'],atol=2e-12,rtol=0)
                inverse_error=max(inverse_error,float(np.max(np.abs(check-scores['positive_variance']))))
                error=float(np.max(np.abs(scores['raw']-[old['leaves'][i]['expanded_score'] for i in gi])))
                assert error<2e-6; raw_error=max(raw_error,error)
                details.append({**g, 'size':len(gi), **{k:scores[k] for k in ['raw_std','positive_variance_std','raw_status','z_status']}})
                for j,i in enumerate(gi):
                    assert leaves[i] is None
                    leaves[i]={'memory_id':ms[i]['id'],'group_id':g['master_id'],'leaf_index':i,
                              **{k:float(scores[k][j]) for k in ['raw','positive_variance','z','correction','weighted']}}
                group_checks+=1
            assert all(l is not None for l in leaves)
            raw_order=sorted(range(len(ms)),key=lambda i:-leaves[i]['raw'])
            raw=next(c for c in old['conditions'] if c['name']==RAW)
            assert [ms[i]['id'] for i in raw_order] == raw['full_rank_memory_ids']; rank_checks+=1
            order=sorted(range(len(ms)),key=lambda i:-leaves[i]['weighted'])
            rows.append({'question_id':q['id'],'task':domain,'groups':details,'leaves':leaves,'conditions':[
                {k:raw[k] for k in ['name','top5_memory_ids','top5_scores','full_rank_memory_ids']},
                {'name':PV,'top5_memory_ids':[ms[i]['id'] for i in order[:5]],
                 'top5_scores':[leaves[i]['weighted'] for i in order[:5]],'full_rank_memory_ids':[ms[i]['id'] for i in order]}]})
        audits.append({'domain':domain,'leaf':ei,'query':qi})
        print(json.dumps({'model':name,'stage':'retrieved_domain','domain':domain,'questions':len(rows)}),flush=True)
    assert len(rows)==rank_checks==371
    write_lines(out/'retrieval_results.jsonl',rows)
    code=[Path(__file__),Path(original.__file__),ROOT/'scripts/memory/memeye_backbone_long_text.py',
          ROOT/'scripts/memory/online_positive_variance.py',ROOT/'scripts/memory/ridge_whitening.py',
          ROOT/'scripts/memory/caption_visual_master.py',ROOT/'scripts/evaluation/memeye_gpt5_reader.py',
          *[ROOT/'scripts/evaluation'/f for f in ['openai_gpt5_base_transport.py','openai_backbone_transport.py',
             'openai_chat_transport.py','deepseek_flash_seed42_audited_transport.py','openai_gpt54_seeded_transport.py']]]
    pinned={**proof['source_sha256'],**{str(p):sha(p) for p in code}}
    hashes(pinned)
    p={**proof,'source_sha256':pinned,'model':name,'memory_path':str(source),
       'lambda':RIDGE,'alpha':ALPHA,'online_weighting':'positive_variance','svd_term_sign':'+',
       'intermediate_score':'raw + sum_j v_j/(v_j+lambda)*(q dot u_j)*(e dot u_j)',
       'final_score':'raw + alpha*std_pop(raw in master)*tanh(z_positive_variance)',
       'raw_full_rankings_verified':371,'groups_verified_by_independent_dual_solve':group_checks,
       'max_raw_score_error':raw_error,'max_dual_solve_error':inverse_error,'master_counts':masters,
       'questions':371,'leaves':438,'encoding_audits':audits,'device':device,
       'captions_regenerated':False,'memory_modified':False,'vectors_persisted':False,
       'master_choices_reused':True,'gold_used_for_ranking':False,
       'output_sha256':{str(out/'retrieval_results.jsonl'):sha(out/'retrieval_results.jsonl')}}
    dump(target,p);return p


def reader_text(q,ids,memory):
    return ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n'+
        '\n\n'.join(f'Memory {j}: {memory[mid]}' for j,mid in enumerate(ids,1))+
        '\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+
        '\n'.join(f'{k}. {q["options"][k]}' for k in 'ABCD')+'\n\nAnswer with one letter only.')


def prepare_reader(name,out):
    marker=out/'reader/protocol.json'
    if marker.exists():return verify(out)[0]
    spec=SPECS[name];ref=spec['reference'];source=spec['source'];folder=out/'reader'
    ret=read(out/'retrieval_protocol.json');hashes(ret['source_sha256']);hashes(ret['output_sha256'])
    oldp=read(ref/'reader/protocol.json');c=oldp['contract'];assert c['seed']==42 and c['model']==spec['requested_model']
    qmap=indexed(lines(source/'inputs/source_questions.jsonl'),'id')
    memory={m['id']:m['retrieval_text'] for m in lines(source/'offline/memories.jsonl')}
    system=read(source/'inputs/protocol.json')['reader_system_prompt']
    reqpath=ref/('reader/inputs/requests.jsonl' if name=='gpt5' else 'reader/requests.jsonl')
    oldreq=indexed(lines(reqpath),'record_id');oldans=indexed(lines(ref/'reader/responses.jsonl'),'record_id')
    assert set(oldreq)==set(oldans)
    cache={}
    for r in oldreq.values():
        payload={'system':r.get('system',system),'user_text':r.get('reader_text',r.get('user_text')),'query_images':r['query_images']}
        body=wire(name,payload);h=digest(body);a=oldans[r['record_id']]
        assert digest(payload)==r['payload_sha256']==a['payload_sha256'] and h==a['request_sha256']
        assert a['run_id']==oldp['run_id'] and a['experiment_signature']==oldp['experiment_signature']
        if a.get('response_id'):
            assert a['requested_model']==spec['requested_model'] and a['returned_model']==spec['returned_model']
        else: assert a['status']=='provider_input_rejected'
        if h in cache: assert cache[h]['answer']['raw_response']==a['raw_response']
        cache[h]={'source_path':str(ref/'reader/responses.jsonl'),'source_record_id':r['record_id'],'answer':a}
    records=[];reused=[];pending={}
    for item in lines(out/'retrieval_results.jsonl'):
        q=qmap[item['question_id']]
        for condition in item['conditions']:
            ids=condition['top5_memory_ids'];assert len(set(ids))==len(ids)==5
            payload={'system':system,'user_text':reader_text(q,ids,memory),'query_images':q['query_images']}
            h=digest(wire(name,payload));rid=q['id']+'::'+condition['name']
            row={'record_id':rid,'question_id':q['id'],'task':q['task'],'condition':condition['name'],
                 'top_memory_ids':ids,**payload,'payload_sha256':digest(payload),'request_sha256':h}
            if condition['name']==RAW:
                a=oldans[rid]; assert h==a['request_sha256']
                entry={'source_path':str(ref/'reader/responses.jsonl'),'source_record_id':rid,'answer':a}
            else: entry=cache.get(h)
            if entry: reused.append({'record_id':rid,'request_sha256':h,**entry})
            else: pending.setdefault(h,row)
            records.append(row)
    assert len(records)==742 and all(r['condition']==PV for r in pending.values())
    folder.mkdir(parents=True,exist_ok=True)
    write_lines(folder/'requests.jsonl',records);write_lines(folder/'reused_answers.jsonl',reused)
    write_lines(folder/'pending_requests.jsonl',list(pending.values()))
    write_lines(folder/'gold.jsonl',lines(source/'inputs/evaluation.jsonl'))
    cfg=transport(name).load_config(c['api_env_file']); assert cfg.public_dict()==c['api_config']
    pinned={**ret['source_sha256'],**ret['output_sha256']}
    for path in [out/'retrieval_protocol.json',marker.parent/'requests.jsonl',marker.parent/'reused_answers.jsonl',
                 marker.parent/'pending_requests.jsonl',marker.parent/'gold.jsonl',reqpath,ref/'reader/responses.jsonl',ref/'reader/protocol.json']:
        pinned[str(path)]=sha(path)
    contract={'name':name,'model':spec['requested_model'],'expected_returned_model':spec['returned_model'],
        'caption_seed':spec['caption_seed'],'seed':42,'lambda':RIDGE,'alpha':ALPHA,'questions':371,
        'conditions':[RAW,PV],'online_weighting':'positive_variance','memory_path':str(source),
        'control_reference':str(ref),'expected_control_correct':spec['raw_correct'],
        'reader_settings':{k:c.get(k) for k in ['seed','temperature','reasoning_effort','max_completion_tokens','reader_max_output_tokens','image_detail','enable_thinking']},
        'api_env_file':c['api_env_file'],'api_config':cfg.public_dict(),'workers':3,'timeout_seconds':90,'max_attempts':3,
        'logical_records':742,'reused_records':len(reused),'new_unique_requests':len(pending),
        'answer_reuse_policy':'Canonical reference full-wire hash match; no other historical runs; preserve original raw controls exactly.',
        'cache_source':str(ref),'gold_sent_to_api':False,'source_sha256':pinned,
        'method_annotations':{
            '1. 候选记忆结构':{'offline_method':'caption_visual_master','memory_path':str(source),'caption_model':spec['returned_model'],'leaf_count':438,'master_counts':ret['master_counts']},
            '2. 查询内容':'保留每个 master 原有视觉匹配选出的完整候选答案陈述，与原问题和题图 caption 拼接',
            '3. 检索方式':{'retrieval':'multi_path','weighting':'positive_variance','lambda':RIDGE,'alpha':ALPHA,'final_score':ret['final_score'],'top_k':5,'control':'Unweighted raw cosine; original answers reused'},
            '4. 编码方式':{'text':'BGE-small-en-v1.5, L2 normalized, 384 dimensions','memory':'dialogue + full caption','query_prefix':read(source/'inputs/protocol.json')['query_prefix'],'encoding_limit':512,'online_vectors_persisted':False},
            '5. 回答方式':{'model':spec['requested_model'],'seed':42,'input':'Ordered Top-5 dialogue/caption, original question, options and query images','historical_identical_input_answers_reused':True,'new_requests':len(pending)}}}
    p={'run_id':str(uuid.uuid4()),'contract':contract,'experiment_signature':digest(contract)}
    dump(marker,p);verify(out)
    print(json.dumps({'stage':'prepared','model':name,'reused':len(reused),'new_unique_requests':len(pending)}),flush=True)
    return p


def verify(out):
    p=read(out/'reader/protocol.json');c=p['contract'];assert digest(c)==p['experiment_signature']
    hashes(c['source_sha256']);records=lines(out/'reader/requests.jsonl')
    assert len(records)==len(indexed(records,'record_id'))==742
    for r in records:
        payload={k:r[k] for k in ['system','user_text','query_images']}
        assert digest(payload)==r['payload_sha256'] and digest(wire(c['name'],payload))==r['request_sha256']
    return p,records


def run(out):
    p,records=verify(out);c=p['contract'];name=c['name'];folder=out/'reader'
    tr=transport(name);cfg=tr.load_config(c['api_env_file']);assert cfg.public_dict()==c['api_config']
    pending=lines(folder/'pending_requests.jsonl')
    done=indexed(lines(folder/'outcomes.jsonl'),'request_sha256') if (folder/'outcomes.jsonl').exists() else {}
    for h,a in done.items():assert a['run_id']==p['run_id'] and a['experiment_signature']==p['experiment_signature']
    todo=[r for r in pending if r['request_sha256'] not in done]
    def execute(row):
        body=wire(name,{k:row[k] for k in ['system','user_text','query_images']})
        assert digest(body)==row['request_sha256'] and body['seed']==42
        provenance={k:row[k] for k in ['record_id','question_id','condition','payload_sha256','request_sha256']}
        provenance.update(run_id=p['run_id'],experiment_signature=p['experiment_signature'],seed=42)
        attempts=[]
        for attempt in range(1,c['max_attempts']+1):
            try: answer=tr.chat_request(cfg,body,timeout=c['timeout_seconds'])
            except tr.APIError as error:
                code=getattr(error,'provider_error_code',None)
                attempts.append({'request_sha256':row['request_sha256'],'attempt':attempt,'status':'error',
                                 'kind':error.kind,'http_status':error.http_status,'provider_error_code':code,'message':error.message})
                rejection=error.http_status==400 and code=='data_inspection_failed'
                if rejection and attempt==c['max_attempts']:
                    return {**provenance,'status':'provider_input_rejected','attempts':attempt,
                            'raw_response':None,'predicted_choice':None,'response_id':None,'provider_error_code':code},attempts
                if not (error.retryable or rejection) or attempt==c['max_attempts']:return None,attempts
                time.sleep(min(2**attempt,8));continue
            assert answer['requested_model']==c['model'] and answer['returned_model']==c['expected_returned_model']
            pred=parse_choice(answer['raw_response'])
            valid=pred is not None and answer['finish_reason']=='stop' and not answer.get('refusal')
            ans={**answer,**provenance,'status':'ok' if valid else 'invalid_response',
                 'predicted_choice':pred if valid else None,'attempts':attempt}
            attempts.append({'request_sha256':row['request_sha256'],'attempt':attempt,'status':ans['status'],'response_id':ans['response_id']})
            return ans,attempts
    with ThreadPoolExecutor(max_workers=c['workers']) as pool:
        for start in range(0,len(todo),c['workers']):
            futures={pool.submit(execute,r):r for r in todo[start:start+c['workers']]};blocked=[]
            for future in as_completed(futures):
                row=futures[future];ans,events=future.result()
                for event in events:append(folder/'api_attempts.jsonl',event)
                if ans is None:blocked.append(row['record_id']);continue
                append(folder/'outcomes.jsonl',ans);done[row['request_sha256']]=ans
            progress={'model':name,'completed_new_requests':len(done),'planned_new_requests':len(pending),
                      'outcomes':dict(Counter(a['status'] for a in done.values())),'blocked':blocked}
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            if blocked:raise RuntimeError('API execution stopped; saved progress retained without changing inputs.')
    assert len(done)==len(pending)
    reuse=indexed(lines(folder/'reused_answers.jsonl'),'record_id');responses=[]
    for row in records:
        r=reuse.get(row['record_id']);a=r['answer'] if r else done[row['request_sha256']]
        responses.append({**row,'raw_response':a['raw_response'],'predicted_choice':a['predicted_choice'],
            'status':a['status'],'response_id':a['response_id'],'returned_model':a.get('returned_model'),
            'reuse_source':{'path':r['source_path'],'record_id':r['source_record_id']} if r else None,
            'origin':'historical' if r else 'new','source_request_sha256':a['request_sha256']})
    write_lines(folder/'responses.jsonl',responses)
    return summarize(out)


def summarize(out):
    p,req=verify(out);c=p['contract'];gold=indexed(lines(out/'reader/gold.jsonl'),'id')
    responses=indexed(lines(out/'reader/responses.jsonl'),'record_id');assert set(responses)=={r['record_id'] for r in req}
    pairs=[];overall={}
    for qid,g in gold.items():
        row={'question_id':qid,'task':qid.split('/')[0],'gold_choice':g['gold_choice'],'gold_ids':g['gold_ids']}
        for method in [RAW,PV]:
            a=responses[qid+'::'+method];pred=a['predicted_choice']
            assert pred==(parse_choice(a['raw_response']) if a['status']=='ok' else None)
            row[method]={'prediction':pred,'correct':pred==g['gold_choice'],'top5':a['top_memory_ids'],
                         'gold_hits':len(set(g['gold_ids'])&set(a['top_memory_ids'])),'gold_count':len(g['gold_ids']),
                         'origin':a['origin'],'status':a['status']}
        pairs.append(row)
    for method in [RAW,PV]:
        rr=[p0[method] for p0 in pairs];hits=sum(r['gold_hits'] for r in rr);ngold=sum(r['gold_count'] for r in rr)
        assert len(rr)==371 and ngold==1041
        overall[method]={'correct':sum(r['correct'] for r in rr),'questions':371,'accuracy':sum(r['correct'] for r in rr)/371,
            'gold_hits_top5':hits,'gold_references':ngold,'recall_at_5_micro':hits/ngold,
            'recall_at_5_macro':float(np.mean([r['gold_hits']/r['gold_count'] for r in rr])),
            'questions_with_any_gold':sum(r['gold_hits']>0 for r in rr),
            'failed_records':sum(r['status']!='ok' for r in rr),'reused_answers':sum(r['origin']=='historical' for r in rr)}
    assert overall[RAW]['correct']==c['expected_control_correct'] and overall[RAW]['reused_answers']==371
    comparison={'wrong_to_right':sum(not r[RAW]['correct'] and r[PV]['correct'] for r in pairs),
        'right_to_wrong':sum(r[RAW]['correct'] and not r[PV]['correct'] for r in pairs),
        'changed_ordered_top5':sum(r[RAW]['top5']!=r[PV]['top5'] for r in pairs),
        'answer_changes':sum(r[RAW]['prediction']!=r[PV]['prediction'] for r in pairs)}
    result={'status':'complete_pending_independent_audit','model':c['model'],'caption_model':c['expected_returned_model'],
        'lambda':RIDGE,'alpha':ALPHA,'caption_seed':c['caption_seed'],'reader_seed':42,'overall':overall,
        'comparison':comparison,'new_unique_requests':c['new_unique_requests'],'controls_rerun':False,
        'memory_path':c['memory_path'],'control_reference':c['control_reference'],'method_annotations':c['method_annotations']}
    dump(out/'summary.json',result);dump(out/'paired_results.json',pairs)
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        data=[{'condition':k,**v} for k,v in overall.items()];w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps({'stage':'summarized','model':c['model'],'overall':overall,'comparison':comparison}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','run','summarize'])
    p.add_argument('--model',choices=list(SPECS),required=True)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();out=args.output.resolve()/args.model
    if args.stage=='prepare':retrieve(args.model,out,args.device);prepare_reader(args.model,out)
    elif args.stage=='run':run(out)
    else:summarize(out)


if __name__=='__main__':main()
