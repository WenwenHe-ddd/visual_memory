"""Independent GPT-5 provenance, API-byte, retrieval and scoring audit (offline)."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/memeye_gpt5_base_seed42_full_20261024'
RAW=('baseline_question_only','master_answer_query')
CONDITIONS=(RAW[0],RAW[0]+'_zscore',RAW[1],RAW[1]+'_zscore')
REF=ROOT/'results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05'
REF_RAW=ROOT/'results/memeye_gpt54_alias_seed42_baselines_20261024'
REF_GPT55=ROOT/'results/memeye_gpt55_seed42_full_20261024'
REF_WEIGHTED=ROOT/'results/memeye_gpt54_alias_seed42_zscore_lambda0p01_alpha1_20261024'
FIELDS={'record_id','id','task','condition','reader_text','query_images','query_image_paths','top_memory_ids','payload_sha256'}
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def indexed(rs,key):
    d={r[key]:r for r in rs};assert len(d)==len(rs);return d
def choice(s):
    m=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',s.strip(),re.I)
    return m.group(1).upper() if m else None
def verify_hashes(hashes):
    for name,value in hashes.items():assert sha(name)==value,name
def wire(payload,limit,seed):
    assert set(payload)<= {'system','user_text','query_images'}
    content=[]
    for im in payload['query_images']:
        assert set(im)=={'path','sha256'}
        data=Path(im['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==im['sha256']
        mime='image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg' if data.startswith(b'\xff\xd8\xff') else None
        assert mime
        content.append({'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(data).decode(),'detail':'high'}})
    content.append({'type':'text','text':payload['user_text']})
    messages=[{'role':'system','content':payload['system']}] if 'system' in payload else []
    messages.append({'role':'user','content':content})
    body={'model':'gpt-5','messages':messages,'n':1,'stream':False,'store':False,'max_completion_tokens':limit,'reasoning_effort':'minimal'}
    if seed is not None:body['seed']=seed
    return body

def audit(source=BASE,inputs_only=False):
    source=Path(source).resolve();out=source/'reader'
    root=read(source/'protocol.json');rc=root['contract'];p=read(out/'protocol.json');c=p['contract']
    assert digest(rc)==root['experiment_signature'] and digest(c)==p['experiment_signature']
    assert rc['requested_model']==c['model']==c['requested_model']=='gpt-5'
    assert re.fullmatch(r'gpt-5(?:-\d{4}-\d{2}-\d{2})?',rc['model'])
    assert rc['model']==c['caption_model']==c['expected_returned_model']
    assert c['seed']==rc['reader_seed']==42 and rc['caption_seed'] is c['caption_seed'] is None
    assert c['conditions']==list(CONDITIONS) and (c['questions'],c['logical_records'],c['top_k'])==(371,1484,5)
    assert (c['lambda'],c['alpha'])==(.01,1.) and c['base_url']=='https://api.openai.com/v1'
    assert (c['temperature'],c['reasoning_effort'],c['max_completion_tokens'],c['image_detail'])==(None,'minimal',2048,'high')
    assert rc['caption_max_output_tokens']==rc['reader_max_output_tokens']==2048
    assert rc['temperature'] is None and rc['reasoning_effort']=='minimal'
    assert c['temperature_wire_policy']==rc['temperature_wire_policy']=='omitted; provider default'
    verify_hashes(rc['source_sha256']);verify_hashes(c['source_sha256'])
    inputs=read(source/'inputs/protocol.json');verify_hashes(inputs['source_sha256'])
    for name,value in inputs['artifact_sha256'].items():assert sha(source/'inputs'/name)==value
    for name in ('source_memories.jsonl','source_questions.jsonl','evaluation.jsonl','api_requests_caption.jsonl'):
        assert sha(source/'inputs'/name)==sha(REF/'inputs'/name)
    cp=read(source/'caption_generation/protocol.json')
    assert cp['run_id']==root['run_id'] and cp['experiment_signature']==root['experiment_signature']
    assert sha(source/'caption_generation/requests.jsonl')==cp['requests_sha256']
    verify_hashes(cp['source_sha256'])
    cap_requests=lines(source/'caption_generation/requests.jsonl');cap_answers=indexed(lines(source/'caption_generation/fresh_responses.jsonl'),'request_sha256')
    cap=indexed(lines(source/'captions.jsonl'),'id');jobs=indexed(lines(source/'inputs/api_requests_caption.jsonl'),'id')
    assert len(cap_requests)==len(cap_answers)==len(cap)==len(jobs)==437
    historical_ids={r['response_id'] for r in lines(REF/'caption_generation/fresh_responses.jsonl')}
    historical_ids.update(r['response_id'] for r in lines(REF_GPT55/'caption_generation/fresh_responses.jsonl'))
    historical_ids.add(read(source/'availability_probe.json')['response']['response_id'])
    for r in cap_requests:
        job=jobs[r['record_id']];payload={'user_text':job['user_text'],'query_images':job['query_images']}
        assert set(r)=={'record_id','user_text','query_images','payload_sha256','request_sha256'}
        assert r['user_text']==inputs['caption_prompt']==job['user_text'] and r['query_images']==job['query_images']
        assert len(r['query_images'])==1 and r['query_images'][0]['sha256']==r['record_id']
        assert digest(payload)==r['payload_sha256'] and digest(wire(payload,2048,None))==r['request_sha256']
        a=cap_answers[r['request_sha256']]
        assert a['requested_model']=='gpt-5' and a['returned_model']==rc['model']
        assert a['record_id']==r['record_id'] and a['payload_sha256']==r['payload_sha256']
        assert a['run_id']==root['run_id'] and a['experiment_signature']==root['experiment_signature']
        assert a['status']=='ok' and a['finish_reason']=='stop' and a['raw_response'].strip() and not a['refusal']
        assert a['reasoning_tokens']==((a.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
        assert a['reasoning_tokens'] is None or (type(a['reasoning_tokens']) is int and a['reasoning_tokens'] >= 0)
        assert not a['historical_response_reuse'] and a['response_id'] not in historical_ids
        caption=cap[r['record_id']]
        assert caption=={'id':r['record_id'],'caption':a['raw_response'].strip(),'model':rc['model'],
            'source_response_record_id':r['record_id'],'request_sha256':r['request_sha256']}
    assert len({r['response_id'] for r in cap_answers.values()})==437
    originals=indexed(lines(source/'inputs/source_memories.jsonl'),'id');qs=indexed(lines(source/'inputs/source_questions.jsonl'),'id')
    gold=indexed(lines(out/'inputs/gold.jsonl'),'id');source_gold=indexed(lines(source/'inputs/evaluation.jsonl'),'id')
    assert len(originals)==438 and set(gold)==set(qs)==set(source_gold) and len(gold)==371
    for ref in (REF_RAW,REF_WEIGHTED,REF_GPT55/'reader'):
        previous=indexed(lines(ref/'inputs/gold.jsonl'),'id');assert set(previous)==set(gold)
        for qid,g in gold.items():assert all(g[k]==previous[qid][k] for k in ('id','gold_choice','gold_ids'))
    for qid,g in gold.items():assert g=={**{k:source_gold[qid][k] for k in ('id','gold_choice','gold_ids')},'task':qs[qid]['task']}
    memories=indexed(lines(source/'offline/memories.jsonl'),'id');assert set(memories)==set(originals)
    graphs={d:read(source/'domains'/d/'memory/graph.json') for d in {m['task'] for m in originals.values()}}
    for mid,m in memories.items():
        origin=originals[mid];expected=cap[origin['image_sha256']]['caption']
        assert m['caption']==expected and m['dialogue']==origin['dialogue']
        assert m['retrieval_text']==origin['dialogue']+'\nMemory description: '+expected
        node=next(n for n in graphs[m['task']]['image_nodes'] if n['id']==m['image_id'])
        assert node['caption']==expected and node['dialogue']==m['dialogue']
    raw=indexed(lines(source/'retrieval_results.jsonl'),'question_id')
    weighted=indexed(lines(source/'zscore/retrieval_results.jsonl'),'question_id')
    assert set(raw)==set(weighted)==set(qs)
    ret=read(source/'retrieval_protocol.json');wret=read(source/'zscore/retrieval_protocol.json')
    verify_hashes(ret['source_sha256']);verify_hashes(ret['output_sha256']);verify_hashes(wret['source_sha256']);verify_hashes(wret['output_sha256'])
    assert ret['caption_model']==rc['model'] and not ret['gold_loaded_for_ranking']
    assert wret['unweighted_full_rankings_equal']==742 and not wret['vectors_persisted'] and not wret['memory_modified']
    ranks={};groups_checked=leaves_checked=0
    for qid,q in qs.items():
        r=raw[qid];wr=weighted[qid]
        query=q['question']+("\nquestion's image:\nimage_caption: "+' '.join(cap[im['sha256']]['caption'] for im in q['query_images']) if q['query_images'] else '')
        assert r['baseline_retrieval_query']==query and r['candidate_statements']==q['candidate_statements']
        graph=graphs[q['task']];gm={g['id']:g for g in graph['masters']};masters=indexed(r['masters'],'master_id')
        assert set(gm)==set(masters)
        expected_ids=[mid for mid,m in originals.items() if m['task']==q['task']]
        assert [l['memory_id'] for l in r['leaves']]==expected_ids
        raw_leaves=indexed(r['leaves'],'memory_id')
        for mid,g in masters.items():
            selected=max('ABCD',key=lambda k:g['clip_candidate_scores'][k])
            assert g['selected_choice']==selected and g['selected_statement']==q['candidate_statements'][selected]
            assert g['augmented_query']==query+'\nCandidate answer: '+g['selected_statement']
            assert set(g['member_ids'])==set(gm[mid]['member_ids'])
        for condition in r['conditions']:
            field='baseline_score' if condition['name']==RAW[0] else 'expanded_score'
            order=sorted(r['leaves'],key=lambda l:-l[field])
            assert condition['full_rank_memory_ids']==[l['memory_id'] for l in order]
            assert condition['top5_memory_ids']==[l['memory_id'] for l in order[:5]]
            ranks[(qid,condition['name'])]=condition['top5_memory_ids']
        for condition in wr['conditions']:
            name=condition['name'];ls=condition['leaves'];gs=condition['groups'];master=name.startswith('master_')
            assert [l['memory_id'] for l in ls]==expected_ids
            assert {g['group_id'] for g in gs}==(set(masters) if master else {'all_memory'})
            for g in gs:
                selected=[l for l in ls if l['group_id']==g['group_id']];assert len(selected)==g['size']
                if master:
                    mg=masters[g['group_id']]
                    assert g['query']==mg['augmented_query'] and g['selected_choice']==mg['selected_choice']
                    assert {originals[l['memory_id']]['image_id'] for l in selected}==set(gm[g['group_id']]['member_ids'])
                else:assert g['query']==query and g['selected_choice'] is None
                rv=np.array([l['raw'] for l in selected]);wv=np.array([l['white'] for l in selected]);z=np.array([l['z'] for l in selected])
                field='expanded_score' if master else 'baseline_score'
                np.testing.assert_allclose(rv,[raw_leaves[l['memory_id']][field] for l in selected],atol=2e-6,rtol=0)
                expected_z=wv if len(selected)==1 else np.zeros(len(selected)) if g['z_status']=='zero_variance' else (wv-wv.mean())/wv.std()
                np.testing.assert_allclose(z,expected_z,atol=1e-9,rtol=0)
                correction=np.zeros(len(selected)) if len(selected)==1 or g['raw_status']=='zero_variance' or g['z_status']=='zero_variance' else rv.std()*np.tanh(z)
                np.testing.assert_allclose([l['correction'] for l in selected],correction,atol=1e-14,rtol=0)
                np.testing.assert_allclose([l['weighted'] for l in selected],rv+correction,atol=1e-14,rtol=0)
                groups_checked+=1;leaves_checked+=len(selected)
            ranked=sorted(ls,key=lambda l:-l['weighted'])
            assert condition['full_rank_memory_ids']==[l['memory_id'] for l in ranked]
            assert condition['top5_memory_ids']==[l['memory_id'] for l in ranked[:5]]
            ranks[(qid,name)]=condition['top5_memory_ids']
    assert groups_checked==wret['independent_inverse_ridge_groups'] and leaves_checked==wret['group_leaf_scores_checked']
    requests_list=lines(out/'inputs/requests.jsonl');requests=indexed(requests_list,'record_id')
    wire_list=lines(out/'api_request_manifest.jsonl');manifest=indexed(wire_list,'record_id')
    assert len(requests)==len(manifest)==1484 and set(requests)==set(manifest)
    assert digest(wire_list)==c['request_manifest_sha256']
    representatives={};images=0
    for r in requests_list:
        assert set(r)==FIELDS and r['record_id']==r['id']+'::'+r['condition']
        q=qs[r['id']];ids=ranks[(r['id'],r['condition'])]
        assert r['top_memory_ids']==ids and len(set(ids))==len(ids)==5
        assert all(originals[mid]['task']==q['task'] for mid in ids)
        text='Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
        text+='\n\n'.join(f'Memory {i}: {memories[mid]["retrieval_text"]}' for i,mid in enumerate(ids,1))
        text+='\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(k+'. '+q['options'][k] for k in 'ABCD')+'\n\nAnswer with one letter only.'
        assert r['reader_text']==text and r['query_images']==q['query_images'] and r['task']==q['task']
        assert r['query_image_paths']==[im['path'] for im in q['query_images']]
        payload={'system':inputs['reader_system_prompt'],'user_text':text,'query_images':q['query_images']}
        assert c['reader_system_prompt']==inputs['reader_system_prompt']
        assert r['payload_sha256']==digest(payload)
        w=manifest[r['record_id']]
        assert w['payload_sha256']==r['payload_sha256'] and w['request_sha256']==digest(wire(payload,2048,42))
        assert w['requested_model']=='gpt-5' and w['query_image_count']==len(q['query_images'])
        representatives.setdefault(r['payload_sha256'],r['record_id']);images+=bool(q['query_images'])
    assert len(representatives)==c['unique_payloads'] and images==32
    assert Counter(r['condition'] for r in requests_list)==dict.fromkeys(CONDITIONS,371)
    checks={'passed':True,'requested_model':'gpt-5','caption_and_reader_returned_model':rc['model'],'caption_seed':None,'reader_seed':42,
        'fresh_caption_provenance_verified':437,'memory_leaf_caption_matches':438,'same_caption_memory_across_conditions':True,
        'questions':371,'reader_records':1484,'unique_reader_requests':len(representatives),'native_image_records':images,
        'unweighted_rankings_verified':742,'weighted_groups_checked':groups_checked,'weighted_leaf_scores_checked':leaves_checked,
        'reader_request_bodies_independently_reconstructed':1484,'caption_request_bodies_independently_reconstructed':437,
        'original_question_option_gold_ids_equal_gpt54':True,'gold_excluded_from_api_payloads':True,'source_hashes_verified':True}
    if inputs_only:
        (out/'inputs/independent_input_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps({'stage':'input_audit','passed':True,'unique_requests':len(representatives)}),flush=True)
        return checks
    summary=read(source/'summary.json');validation=read(out/'validation_output.json')
    responses=indexed(lines(out/'responses.jsonl'),'record_id');fresh_list=lines(out/'fresh_responses.jsonl');fresh=indexed(fresh_list,'payload_sha256')
    attempts=lines(out/'api_attempts.jsonl')
    assert set(responses)==set(requests) and set(fresh)==set(representatives)
    assert Counter(a['payload_sha256'] for a in attempts if a['status']=='success')==dict.fromkeys(fresh,1)
    for label in ('responses','fresh_responses'):assert sha(out/(label+'.jsonl'))==validation[label+'_sha256']
    prior={**indexed(lines(REF_RAW/'responses.jsonl'),'record_id'),**indexed(lines(REF_WEIGHTED/'responses.jsonl'),'record_id')}
    prior55=indexed(lines(REF_GPT55/'reader/responses.jsonl'),'record_id')
    historical_ids.update(r['response_id'] for r in prior55.values())
    historical_ids.update(r['response_id'] for r in prior.values());historical_ids.update(r['response_id'] for r in cap_answers.values())
    assert not historical_ids.intersection(r['response_id'] for r in fresh_list)
    assert len({r['response_id'] for r in fresh_list})==len(fresh_list)
    for r in fresh_list:
        rid=r['record_id'];assert rid==representatives[r['payload_sha256']]
        assert r['run_id']==p['run_id'] and r['experiment_signature']==p['experiment_signature']
        assert r['request_sha256']==manifest[rid]['request_sha256'] and r['requested_model']=='gpt-5' and r['returned_model']==rc['model']
        assert not r['historical_response_reuse']
        assert r['predicted_choice']==choice(r['raw_response']) and r['status']==('ok' if choice(r['raw_response']) else 'format_error')
        assert r['hit_token_limit']==(r['finish_reason']=='length')
        assert r['reasoning_tokens']==((r.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
        assert any(a['status']=='success' and a['response_id']==r['response_id'] and a['execution_id']==r['execution_id'] and a['request_sha256']==r['request_sha256'] for a in attempts)
    shared=formats=empty=truncated=0
    for rid,r in responses.items():
        q=requests[rid];origin=fresh[q['payload_sha256']]
        for key in ('raw_response','predicted_choice','response_id','request_id','request_sha256','run_id','experiment_signature','execution_id','returned_model','requested_model','usage','status','historical_response_reuse'):
            assert r[key]==origin[key]
        assert r['top_memory_ids']==q['top_memory_ids'] and r['id']==q['id'] and r['condition']==q['condition'] and r['task']==q['task']
        is_fresh=rid==origin['record_id']
        assert r['fresh_inference_executed_for_this_record']==is_fresh
        assert r['same_run_shared_inference_record_id']==(None if is_fresh else origin['record_id'])
        shared+=not is_fresh;formats+=r['status']!='ok';empty+=not r['raw_response'];truncated+=r['hit_token_limit']
    pairs=[]
    for qid,g in gold.items():
        row={'question_id':qid,'domain':g['task'],'gold_choice':g['gold_choice']}
        for name in CONDITIONS:
            rid=qid+'::'+name;pred=choice(responses[rid]['raw_response']);old=choice(prior[rid]['raw_response']);ids=requests[rid]['top_memory_ids']
            pred55=choice(prior55[rid]['raw_response'])
            row.update({name+'_answer':pred,name+'_correct':pred==g['gold_choice'],name+'_top5':ids,
                name+'_gold_hits':len(set(ids)&set(g['gold_ids'])),name+'_gpt54_answer':old,name+'_gpt54_correct':old==g['gold_choice'],
                name+'_gpt55_answer':pred55,name+'_gpt55_correct':pred55==g['gold_choice']})
        for name in RAW:
            same=requests[qid+'::'+name]['payload_sha256']==requests[qid+'::'+name+'_zscore']['payload_sha256']
            row[name+'_identical_payload']=same
            if same:assert row[name+'_answer']==row[name+'_zscore_answer']
        pairs.append(row)
    assert pairs==read(source/'paired_results.json')
    for name in CONDITIONS:
        correct=sum(r[name+'_correct'] for r in pairs)
        assert summary['overall'][name]=={'correct':correct,'questions':371,'accuracy':correct/371,
            'gold_hits_top5':sum(r[name+'_gold_hits'] for r in pairs),'gold_references':1041,
            'questions_with_any_gold':sum(r[name+'_gold_hits']>0 for r in pairs)}
        assert summary['method_annotations'][name]['5. 回答方式']['correct']==correct
        assert summary['method_annotations'][name]['1. 候选记忆结构']['caption_model']==rc['model']
        assert summary['gpt54_comparisons'][name]==comparison(pairs,name+'_gpt54',name)
        assert summary['gpt55_comparisons'][name]==comparison(pairs,name+'_gpt55',name)
    assert [summary['gpt55_comparisons'][n]['reference_correct'] for n in CONDITIONS]==[250,237,247,251]
    assert [summary['gpt54_comparisons'][n]['reference_correct'] for n in CONDITIONS]==[236,241,238,235]
    for name in RAW:
        expected=comparison(pairs,name,name+'_zscore')
        expected['identical_payload_questions']=sum(r[name+'_identical_payload'] for r in pairs)
        assert summary['within_gpt5_base_weighting_comparisons'][name]==expected
    for domain,values in summary['by_domain'].items():
        rows=[r for r in pairs if r['domain']==domain]
        assert values=={name:{'correct':sum(r[name+'_correct'] for r in rows),'questions':len(rows)} for name in CONDITIONS}
    for stage,rs in (('caption_generation',list(cap_answers.values())),('reader',fresh_list)):
        assert summary['generation_stages'][stage]=={'fresh_requests':len(rs),'returned_models':dict(Counter(r['returned_model'] for r in rs)),
            'usage':{k:sum((r.get('usage') or {}).get(k,0) or 0 for r in rs) for k in ('prompt_tokens','completion_tokens','total_tokens')}}
    assert validation['fresh_requests']==len(fresh) and validation['shared_records']==shared and validation['logical_records']==1484
    assert (validation['format_errors'],validation['empty_responses'],validation['token_limit_records'])==(formats,empty,truncated)
    assert validation['returned_models']==[rc['model']]
    cap_attempts=lines(source/'caption_generation/api_attempts.jsonl')
    assert Counter(a['request_sha256'] for a in cap_attempts if a['status']=='ok')==dict.fromkeys(cap_answers,1)
    cap_proof=read(source/'caption_model_provenance.json')
    assert cap_proof['passed'] and cap_proof['fresh_unique_caption_requests']==437
    assert cap_proof['requested_model_counts']=={'gpt-5':437} and cap_proof['returned_model_counts']=={rc['model']:437}
    assert cap_proof['previous_gpt54_or_gpt55_response_ids_reused']==cap_proof['refusals']==cap_proof['truncated']==0
    assert cap_proof['api_attempts']==len(cap_attempts) and cap_proof['reasoning_tokens']==sum(r.get('reasoning_tokens') or 0 for r in cap_answers.values())
    assert read(source/'retrieval_summary.json')=={n:{k:summary['overall'][n][k] for k in ('questions','gold_hits_top5','gold_references','questions_with_any_gold')} for n in CONDITIONS}
    checks.update(fresh_reader_requests=len(fresh),same_run_shared_records=shared,format_errors=formats,empty_responses=empty,
        token_limit_records=truncated,reader_api_attempts=len(attempts),caption_api_attempts=len(cap_attempts),
        transient_failed_reader_attempts=sum(a['status']!='success' for a in attempts),
        transient_failed_caption_attempts=sum(a['status']!='ok' for a in cap_attempts),
        caption_reasoning_tokens=sum(r['reasoning_tokens'] or 0 for r in cap_answers.values()),
        refusals=sum(bool(r['refusal']) for r in fresh_list),reasoning_tokens=sum(r['reasoning_tokens'] or 0 for r in fresh_list),
        historical_and_probe_response_ids_excluded=True,all_summary_counts_verified=True,overall=summary['overall'],
        artifact_sha256={name:sha(source/name) for name in ('protocol.json','captions.jsonl','offline/memories.jsonl',
            'caption_generation/fresh_responses.jsonl','reader/protocol.json','reader/responses.jsonl','reader/fresh_responses.jsonl',
            'summary.json','paired_results.json','paired_results.csv','caption_model_provenance.json','retrieval_summary.json')})
    (source/'independent_audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'stage':'independent_audit','passed':True,'correct':{n:v['correct'] for n,v in summary['overall'].items()}},ensure_ascii=False),flush=True)
    return checks


def comparison(pairs,before,after):
    wins=sum(not r[before+'_correct'] and r[after+'_correct'] for r in pairs)
    losses=sum(r[before+'_correct'] and not r[after+'_correct'] for r in pairs);n=wins+losses
    return {'reference_correct':sum(r[before+'_correct'] for r in pairs),'current_correct':sum(r[after+'_correct'] for r in pairs),
        'wrong_to_right':wins,'right_to_wrong':losses,'net_correct':wins-losses,
        'answer_changes':sum(r[before+'_answer']!=r[after+'_answer'] for r in pairs),
        'mcnemar_exact_p':min(1.,2*sum(math.comb(n,i) for i in range(min(wins,losses)+1))/2**n) if n else 1.}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=BASE);p.add_argument('--inputs-only',action='store_true')
    a=p.parse_args();audit(a.output,a.inputs_only)
