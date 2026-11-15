"""Independent standard-library audit of the complementary seed42 conditions.

No runner, transport, weighting implementation, encoders, or model API is called.
Rebuild requests from original image bytes and text; regrade every answer and
terminal error; compare unweighted master ranking and full-domain score updates.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'results/memeye_deepseek_flash_seed42_full_20261025/complementary_evaluation'
MODEL='deepseek-v4.1-flash'
BASE='baseline_question_only_zscore'
MASTER='master_answer_query'
NAMES=(BASE,MASTER)

def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def index(rows,key):
    result={r[key]:r for r in rows};assert len(result)==len(rows);return result
def choice(text):
    m=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',text.strip(),re.I)
    return m.group(1).upper() if m else None
def hashes(mapping):
    for p,h in mapping.items():assert sha(p)==h,p

def wire(payload,limit):
    assert set(payload) in ({'system','user_text','query_images'},{'user_text','query_images'})
    content=[]
    for im in payload['query_images']:
        assert set(im)=={'path','sha256'}
        data=Path(im['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==im['sha256']
        if data.startswith(b'\x89PNG\r\n\x1a\n'):mime='image/png'
        else:assert data.startswith(b'\xff\xd8\xff');mime='image/jpeg'
        content.append({'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(data).decode(),'detail':'high'}})
    content.append({'type':'text','text':payload['user_text']})
    messages=[{'role':'system','content':payload['system']}] if 'system' in payload else []
    messages.append({'role':'user','content':content})
    return {'model':MODEL,'messages':messages,'temperature':0,'max_tokens':limit,'n':1,'stream':False,'enable_thinking':False,'seed':42}


def audit(output=DEFAULT):
    output=Path(output).resolve();folder=output/'reader'
    p=read(folder/'protocol.json');c=p['contract'];assert digest(c)==p['experiment_signature']
    assert c['model']==MODEL and c['seed']==c['caption_seed']==42 and c['conditions']==list(NAMES)
    assert c['lambda']==.03 and c['alpha']==1 and c['temperature']==0 and c['enable_thinking'] is False
    assert c['reader_max_output_tokens']==16 and c['historical_answers_reused'] is False and c['gold_sent_to_api'] is False
    assert c['api_config']['base_url']=='https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
    assert c['api_config']['credential_variable']=='DEEPSEEKV4_API_KEY'
    hashes(c['source_sha256'])
    source=Path(c['memory_path']);cap_protocol=read(source/'protocol.json')
    assert cap_protocol['contract']['seed']==42 and digest(cap_protocol['contract'])==cap_protocol['experiment_signature']
    caps=index(lines(source/'captions.jsonl'),'id');history=index(lines(source/'inputs/source_memories.jsonl'),'id')
    qs=index(lines(source/'inputs/source_questions.jsonl'),'id');gold=index(lines(folder/'gold.jsonl'),'id')
    original=index(lines(ROOT/'data/independent_visual_inputs/qa_manifest.jsonl'),'id')
    assert len(caps)==437 and len(history)==438 and set(gold)==set(qs)==set(original) and len(qs)==371
    for qid,g in gold.items():
        assert g['gold_choice']==original[qid]['gold_choice'] and g['gold_ids']==original[qid]['gold_ids']
    needed={m['image_sha256'] for m in history.values()}|{im['sha256'] for q in qs.values() for im in q['query_images']}
    assert set(caps)==needed
    cap_reqs=index(lines(source/'caption_generation/requests.jsonl'),'record_id')
    cap_ans=index(lines(source/'caption_generation/fresh_responses.jsonl'),'request_sha256')
    for cid,cap in caps.items():
        r=cap_reqs[cap['source_response_record_id']];a=cap_ans[cap['request_sha256']]
        payload={k:r[k] for k in ('system','user_text','query_images') if k in r}
        assert digest(wire(payload,1024))==cap['request_sha256']==r['request_sha256']==a['request_sha256']
        assert a['run_id']==cap_protocol['run_id'] and a['experiment_signature']==cap_protocol['experiment_signature']
        assert cap['seed']==a['seed']==42 and cap['caption']==a['raw_response'].strip()
        assert cap['model']==a['returned_model']==MODEL and a['status']=='ok' and a['finish_reason']=='stop'
    master_counts={}
    for domain in sorted({m['task'] for m in history.values()}):
        graph=read(source/'domains'/domain/'memory/graph.json');ms=[m for m in history.values() if m['task']==domain]
        assert [n['id'] for n in graph['image_nodes']]==[m['image_id'] for m in ms]
        for n,m in zip(graph['image_nodes'],ms):assert n['caption']==caps[m['image_sha256']]['caption'] and n['dialogue']==m['dialogue']
        master_counts[domain]=len(graph['masters'])
    assert sum(master_counts.values())==86
    raw=index(lines(source/'retrieval_results.jsonl'),'question_id');ret=index(lines(output/'retrieval_results.jsonl'),'question_id')
    assert set(ret)==set(qs);max_score_error=0.;group_checks=0
    for qid,row in ret.items():
        cc=index(row['conditions'],'name');assert set(cc)==set(NAMES)
        old=index(raw[qid]['conditions'],'name');w=cc[BASE];m=cc[MASTER]
        for key in ('top5_memory_ids','top5_scores','full_rank_memory_ids'):assert m[key]==old[MASTER][key]
        assert w['query']==raw[qid]['baseline_retrieval_query'] and len(w['groups'])==1
        ls=w['leaves'];g=w['groups'][0]
        expected=[x['id'] for x in history.values() if x['task']==qs[qid]['task']]
        assert [l['memory_id'] for l in ls]==expected and g['size']==len(expected)
        assert {l['group_id'] for l in ls}=={g['group_id']}=={qs[qid]['task']+'::all_memory'}
        assert all(abs(a['raw']-b['baseline_score'])<2e-6 for a,b in zip(ls,raw[qid]['leaves']))
        std=statistics.pstdev(l['white'] for l in ls);mean=statistics.mean(l['white'] for l in ls)
        raw_std=statistics.pstdev(l['raw'] for l in ls)
        wt=32*sys.float_info.epsilon*max(1.,max(abs(l['white']) for l in ls))
        rt=32*sys.float_info.epsilon*max(1.,max(abs(l['raw']) for l in ls))
        for l in ls:
            z=(l['white']-mean)/std if std>wt else 0.
            corr=raw_std*math.tanh(z) if raw_std>rt and std>wt else 0.
            err=max(abs(z-l['z']),abs(corr-l['correction']),abs(l['raw']+corr-l['weighted']))
            assert err<2e-10;max_score_error=max(max_score_error,err)
        order=[l['memory_id'] for l in sorted(ls,key=lambda l:-l['weighted'])]
        assert w['full_rank_memory_ids']==order and w['top5_memory_ids']==order[:5];group_checks+=1
    reqs=index(lines(folder/'requests.jsonl'),'record_id');reps={}
    for r in reqs.values():reps.setdefault(r['request_sha256'],r)
    outcomes=index(lines(folder/'outcomes.jsonl'),'request_sha256');logical=index(lines(folder/'responses.jsonl'),'record_id')
    fresh=lines(folder/'fresh_responses.jsonl');attempts=lines(folder/'api_attempts.jsonl')
    assert len(reqs)==len(logical)==742 and set(reqs)==set(logical) and set(reps)==set(outcomes)
    assert len(reps)==c['unique_requests'] and len({a['response_id'] for a in fresh})==len(fresh)
    prior_responses=lines(source/'weighted_evaluation/reader/fresh_responses.jsonl')
    assert not {a['response_id'] for a in prior_responses}&{a['response_id'] for a in fresh}
    fresh_by_hash=index(fresh,'request_sha256')
    assert set(fresh_by_hash)=={h for h,a in outcomes.items() if a['response_id'] is not None}
    for h,a in outcomes.items():
        r=reps[h];events=[e for e in attempts if e['request_sha256']==h]
        assert len(events)==a['attempts'] and all(e['execution_id']==a['execution_id'] for e in events)
        assert [e['attempt'] for e in events]==list(range(1,a['attempts']+1))
        for k,v in {'record_id':r['record_id'],'request_sha256':h,'payload_sha256':r['payload_sha256'],
                    'run_id':p['run_id'],'experiment_signature':p['experiment_signature'],'seed':42,'historical_response_reuse':False}.items():assert a[k]==v
        if a['status']=='provider_input_rejected':
            assert a['raw_response'] is None and a['response_id'] is None and a['predicted_choice'] is None and len(events)==3
            assert all(e['status']=='api_error' and e['http_status']==400 and e['provider_error_code']=='data_inspection_failed' for e in events)
            continue
        assert fresh_by_hash[h]==a and a['returned_model']==a['requested_model']==MODEL and a['response_id']
        assert a['enable_thinking'] is False and a['reasoning_content_present'] is False and a['reasoning_tokens'] in (None,0)
        if r['query_images']:assert a['usage']['prompt_tokens_details']['image_tokens']>0
        valid=a['finish_reason']=='stop' and not a.get('refusal') and choice(a['raw_response']) is not None
        assert (a['status']=='ok')==valid and a['status'] in ('ok','invalid_response')
        assert a['predicted_choice']==(choice(a['raw_response']) if valid else None)
        assert events[-1]['status']==a['status'] and events[-1]['response_id']==a['response_id']
    texts={mid:m['dialogue']+'\nMemory description: '+caps[m['image_sha256']]['caption'] for mid,m in history.items()}
    totals={n:{'correct':0,'gold_hits_top5':0,'failed_records':0,'recalls':[],'questions_with_any_gold':0} for n in NAMES}
    for rid,r in reqs.items():
        assert set(r)=={'record_id','question_id','task','condition','top_memory_ids','system','user_text','query_images','payload_sha256','request_sha256'}
        qid=r['question_id'];q=qs[qid];name=r['condition'];g=gold[qid]
        top=next(x['top5_memory_ids'] for x in ret[qid]['conditions'] if x['name']==name)
        assert r['top_memory_ids']==top and len(top)==len(set(top))==5
        text=('Retrieved dialogue and image descriptions (ranked by relevance):\n\n'+'\n\n'.join(f'Memory {i}: {texts[mid]}' for i,mid in enumerate(top,1))+
              '\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(f'{k}. {q["options"][k]}' for k in 'ABCD')+'\n\nAnswer with one letter only.')
        assert r['user_text']==text and r['query_images']==q['query_images']
        assert r['system']==read(source/'inputs/protocol.json')['reader_system_prompt']
        payload={k:r[k] for k in ('system','user_text','query_images')}
        assert digest(wire(payload,16))==r['request_sha256'] and digest(payload)==r['payload_sha256']
        a=outcomes[r['request_sha256']];final=logical[rid]
        for key in ('status','raw_response','predicted_choice','response_id','request_sha256','payload_sha256','execution_id','run_id','experiment_signature','seed'):assert final[key]==a[key]
        correct=a['status']=='ok' and a['predicted_choice']==g['gold_choice']
        assert final['correct']==correct and final['top_memory_ids']==top
        assert final['fresh_inference_executed_for_this_record']==(a['response_id'] is not None and a['record_id']==rid)
        assert final['same_run_shared_inference_record_id']==(None if a['record_id']==rid else a['record_id'])
        hits=len(set(top)&set(g['gold_ids']));t=totals[name]
        t['correct']+=correct;t['gold_hits_top5']+=hits;t['failed_records']+=a['status']!='ok'
        t['recalls'].append(hits/len(set(g['gold_ids'])));t['questions_with_any_gold']+=hits>0
    summary=read(output/'summary.json');total_gold=sum(len(set(g['gold_ids'])) for g in gold.values());assert total_gold==1041
    for n,t in totals.items():
        m=summary['overall'][n];recalls=t.pop('recalls');assert len(recalls)==371
        for k,v in t.items():assert m[k]==v
        assert m['accuracy']==t['correct']/371 and m['recall_at_5_micro']==t['gold_hits_top5']/1041
        assert abs(m['recall_at_5_macro']-sum(recalls)/371)<1e-14
    comp=read(output/'four_condition_comparison.json');previous=read(source/'weighted_evaluation/summary.json')
    assert comp['same_caption_sha256']==sha(source/'captions.jsonl') and comp['reader_batches_differ'] is True
    for n,m in comp['conditions'].items():
        expected=(summary if n in NAMES else previous)['overall'][n]
        for k,v in expected.items():assert m[k]==v
    report={'passed':True,'questions':371,'logical_records':742,'caption_seed':42,'reader_seed':42,
        'caption_requests_reconstructed':437,'reader_wire_requests_reconstructed':742,'historical_answers_reused':False,
        'full_domain_zscore_groups_checked':group_checks,'max_score_correction_error':max_score_error,
        'unweighted_master_rankings_identical':371,'fresh_model_responses':len(fresh),'unique_requests':len(outcomes),
        'same_run_shared_records':742-len(outcomes),'unique_outcome_counts':dict(Counter(a['status'] for a in outcomes.values())),
        'api_attempts':len(attempts),'overall':totals,'all_questions_retained':True,
        'script_sha256':sha(__file__),'source_signature':p['experiment_signature'],
        'artifact_sha256':{name:sha(output/name) for name in ('summary.json','four_condition_comparison.json','reader/protocol.json',
            'reader/requests.jsonl','reader/gold.jsonl','reader/fresh_responses.jsonl','reader/outcomes.jsonl','reader/api_attempts.jsonl','reader/responses.jsonl',
            'retrieval_results.jsonl','retrieval_protocol.json','source_validation.json','paired_results.json')},
        'whitening_scope':'Inverse ridge independently checked during GPU retrieval; this audit recomputes standardization, score correction and ranking.'}
    (output/'external_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True);return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    audit(parser.parse_args().output)
