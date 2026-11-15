"""Independent five-candidate provenance, retrieval, API-byte and outcome audit."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
import numpy as np
from scripts.analysis.audit_memeye_gpt5_base import read,lines,sha,digest,indexed,choice,verify_hashes,wire,comparison,FIELDS
BASE=ROOT/'results/memeye_gpt5_qwen_five_validation_20261024'
FIRST=ROOT/'results/memeye_gpt5_qwen_selected_validation_20261024'
RAW=('baseline_question_only','master_answer_query')
PAIRS=((.01,1.),(.1,.6),(.03,1.),(.01,.6),(.003,.3))
def condition(m,l,a):return m+'__l'+format(l,'g').replace('.','p')+'_a'+format(a,'g').replace('.','p')
SPECS={m:{'method':m,'lambda':None,'alpha':0.} for m in RAW}
SPECS.update({condition(m,l,a):{'method':m,'lambda':l,'alpha':a} for l,a in PAIRS for m in RAW})
NAMES=tuple(SPECS)


def audit(output=BASE,inputs_only=False):
    output=Path(output).resolve();out=output/'reader';p=read(out/'protocol.json');c=p['contract']
    assert digest(c)==p['experiment_signature'];verify_hashes(c['source_sha256'])
    search=Path(c['qwen_search_path']);qsummary=read(search/'summary.json');qproof=read(search/'independent_audit.json')
    assert qproof['passed'];verify_hashes({str(search/n):v for n,v in qproof['artifact_sha256'].items()})
    b0,m0=(qsummary['overall'][n]['correct'] for n in RAW)
    eligible=[]
    for r in qsummary['search_results']:
        passed=r['baseline_correct']>b0 and r['master_correct']>m0 and r['master_correct']>r['baseline_correct']
        assert passed==r['passes_all_three']
        if passed:eligible.append(r)
    eligible.sort(key=lambda r:(-r['master_correct'],-r['baseline_correct'],r['alpha'],r['lambda']))
    assert len(qsummary['search_results'])==25 and eligible==c['parameter_pairs']
    assert tuple((r['lambda'],r['alpha']) for r in eligible)==PAIRS
    first=read(FIRST/'independent_audit.json');assert first['passed'] and first['validation_passed'] is False
    verify_hashes({str(FIRST/n):v for n,v in first['artifact_sha256'].items()})
    assert c['first_validation_path']==str(FIRST) and c['expanded_after_first_gpt5_failure']
    assert c['same_run_controls'] and c['candidate_list_from_qwen_only'] and c['conditions']==list(NAMES) and c['condition_specs']==SPECS
    assert (c['questions'],c['logical_records'],c['top_k'])==(371,4452,5)
    assert not any(c[k] for k in ('caption_regenerated','memory_rebuilt','historical_api_answer_reuse','gold_in_api_payload'))
    source=Path(c['memory_path']);prior=read(source/'reader/protocol.json');pc=prior['contract']
    assert digest(pc)==prior['experiment_signature'];verify_hashes(pc['source_sha256'])
    for k in ('model','requested_model','expected_returned_model','caption_model','seed','caption_seed','temperature','temperature_wire_policy',
              'reasoning_effort','max_completion_tokens','image_detail','base_url','api_config','reader_system_prompt','timeout_seconds','max_attempts'):
        assert c[k]==pc[k],k
    assert c['model']=='gpt-5' and c['caption_model']==c['expected_returned_model']=='gpt-5-2025-08-07'
    assert (c['seed'],c['reasoning_effort'],c['temperature'],c['max_completion_tokens'])==(42,'minimal',None,2048)
    old_audit=read(source/'independent_audit.json');assert old_audit['passed']
    verify_hashes({str(source/n):v for n,v in old_audit['artifact_sha256'].items()})
    qs=indexed(lines(source/'inputs/source_questions.jsonl'),'id');gold=indexed(lines(out/'inputs/gold.jsonl'),'id')
    assert gold==indexed(lines(source/'reader/inputs/gold.jsonl'),'id') and len(gold)==371 and set(qs)==set(gold)
    memories=indexed(lines(source/'offline/memories.jsonl'),'id');assert len(memories)==438
    ret=read(output/'retrieval_protocol.json');verify_hashes(ret['source_sha256']);verify_hashes(ret['output_sha256'])
    assert ret['condition_specs']==SPECS and ret['parameter_pairs']==eligible and ret['memory_path']==str(source)
    assert ret['unweighted_full_rankings_equal']==742 and ret['max_raw_score_difference']<2e-6 and ret['max_inverse_ridge_difference']<2e-12
    assert sum(ret['master_counts'].values())==72 and ret['all_raw_scores_and_groups_exactly_equal'] and ret['lambda0p01_alpha1_scores_exactly_equal']
    assert not any(ret[k] for k in ('vectors_persisted','memory_modified','old_images_reencoded','gold_loaded_for_ranking'))
    current=indexed(lines(output/'retrieval_results.jsonl'),'question_id');old=indexed(lines(source/'zscore/retrieval_results.jsonl'),'question_id')
    old_raw=indexed(lines(source/'retrieval_results.jsonl'),'question_id');assert set(current)==set(old)==set(old_raw)==set(gold)
    ranks={};groups=leaf_count=0
    for qid,gold_row in gold.items():
        cc=indexed(current[qid]['conditions'],'name');oc=indexed(old[qid]['conditions'],'name');rc=indexed(old_raw[qid]['conditions'],'name')
        assert set(cc)==set(NAMES) and current[qid]['task']==gold_row['task']
        for n in RAW:
            for k in ('top5_memory_ids','top5_scores','full_rank_memory_ids'):assert cc[n][k]==rc[n][k]
            ranks[qid,n]=cc[n]['top5_memory_ids']
        for n in (n for n in NAMES if n not in RAW):
            spec=SPECS[n];base_name=spec['method']+'_zscore'
            a,b=cc[n],oc[base_name];ls=a['leaves'];assert len(ls)==len(b['leaves']) and len(a['groups'])==len(b['groups'])
            for x,y in zip(ls,b['leaves']):
                for k in ('memory_id','leaf_index','group_id','raw'):assert x[k]==y[k]
                assert memories[x['memory_id']]['task']==gold_row['task']
            for x,y in zip(a['groups'],b['groups']):
                for k in ('group_id','size','query','selected_choice','selected_statement','raw_std','raw_status'):assert x[k]==y[k]
                subset=[l for l in ls if l['group_id']==x['group_id']];assert len(subset)==x['size']
                raw=np.array([l['raw'] for l in subset]);white=np.array([l['white'] for l in subset]);z=np.array([l['z'] for l in subset])
                expected=white if len(subset)==1 else np.zeros(len(subset)) if x['z_status']=='zero_variance' else (white-white.mean())/white.std()
                np.testing.assert_allclose(z,expected,atol=1e-9,rtol=0)
                correction=np.zeros(len(subset)) if len(subset)==1 or x['raw_status']=='zero_variance' or x['z_status']=='zero_variance' else spec['alpha']*raw.std()*np.tanh(z)
                np.testing.assert_allclose([l['correction'] for l in subset],correction,atol=1e-14,rtol=0)
                np.testing.assert_allclose([l['weighted'] for l in subset],raw+correction,atol=1e-14,rtol=0)
                groups+=1;leaf_count+=len(subset)
            if (spec['lambda'],spec['alpha'])==(.01,1.):
                assert ls==b['leaves'] and a['groups']==b['groups']
            rank=sorted(ls,key=lambda l:-l['weighted'])
            assert a['full_rank_memory_ids']==[l['memory_id'] for l in rank] and a['top5_memory_ids']==[l['memory_id'] for l in rank[:5]]
            np.testing.assert_array_equal(a['top5_scores'],[l['weighted'] for l in rank[:5]])
            ranks[qid,n]=a['top5_memory_ids']
    assert groups==ret['independent_inverse_ridge_groups'] and leaf_count==ret['group_leaf_scores_checked']
    requests_list=lines(out/'inputs/requests.jsonl');requests=indexed(requests_list,'record_id')
    manifest_list=lines(out/'api_request_manifest.jsonl');manifest=indexed(manifest_list,'record_id')
    old_requests=indexed(lines(source/'reader/inputs/requests.jsonl'),'record_id')
    assert len(requests)==4452 and set(requests)==set(manifest) and digest(manifest_list)==c['request_manifest_sha256']
    assert Counter(r['condition'] for r in requests_list)==dict.fromkeys(NAMES,371)
    representatives={};image_count=0;raw_matches=0
    for r in requests_list:
        assert set(r)==FIELDS and r['record_id']==r['id']+'::'+r['condition']
        q=qs[r['id']];ids=ranks[r['id'],r['condition']];assert r['top_memory_ids']==ids and len(set(ids))==5
        text='Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
        text+='\n\n'.join(f'Memory {j}: {memories[mid]["retrieval_text"]}' for j,mid in enumerate(ids,1))
        text+='\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(k+'. '+q['options'][k] for k in 'ABCD')+'\n\nAnswer with one letter only.'
        assert r['reader_text']==text and r['query_images']==q['query_images'] and r['task']==q['task']
        assert r['query_image_paths']==[im['path'] for im in q['query_images']]
        payload={'system':c['reader_system_prompt'],'user_text':text,'query_images':q['query_images']}
        w=manifest[r['record_id']];assert r['payload_sha256']==w['payload_sha256']==digest(payload)
        assert w['request_sha256']==digest(wire(payload,2048,42)) and w['requested_model']=='gpt-5' and w['query_image_count']==len(q['query_images'])
        if r['condition'] in RAW:assert r==old_requests[r['record_id']];raw_matches+=1
        representatives.setdefault(r['payload_sha256'],r['record_id']);image_count+=bool(q['query_images'])
    assert len(representatives)==c['unique_payloads'] and image_count==96 and raw_matches==742
    proof={'passed':True,'questions':371,'logical_records':4452,'unique_requests':len(representatives),'parameter_pairs':len(PAIRS),
           'all_five_qwen_eligible_pairs_verified':True,'previous_failed_first_trial_preserved':True,'fresh_controls':True,'same_gpt5_caption_memory_and_reader_settings':True,
           'request_bodies_independently_reconstructed':4452,'historical_raw_request_matches':raw_matches,'weighted_groups':groups,
           'gold_excluded_from_api_payload':True,'audit_script_sha256':sha(__file__)}
    if inputs_only:
        (out/'inputs/independent_input_audit.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True);return proof
    responses=indexed(lines(out/'responses.jsonl'),'record_id');fresh_list=lines(out/'fresh_responses.jsonl');fresh=indexed(fresh_list,'payload_sha256')
    attempts=lines(out/'api_attempts.jsonl');validation=read(out/'validation_output.json');summary=read(output/'summary.json')
    assert set(responses)==set(requests) and set(fresh)==set(representatives)
    assert Counter(a['payload_sha256'] for a in attempts if a['status']=='success')==dict.fromkeys(fresh,1)
    for n in ('responses','fresh_responses'):assert sha(out/(n+'.jsonl'))==validation[n+'_sha256']
    old_ids={r['response_id'] for r in lines(source/'reader/responses.jsonl')}|{r['response_id'] for r in lines(source/'caption_generation/fresh_responses.jsonl')}
    old_ids.update(r['response_id'] for r in lines(FIRST/'reader/responses.jsonl'))
    assert len({r['response_id'] for r in fresh_list})==len(fresh_list) and not old_ids.intersection(r['response_id'] for r in fresh_list)
    for r in fresh_list:
        rid=r['record_id'];assert rid==representatives[r['payload_sha256']]
        assert r['run_id']==p['run_id'] and r['experiment_signature']==p['experiment_signature'] and r['request_sha256']==manifest[rid]['request_sha256']
        assert r['requested_model']=='gpt-5' and r['returned_model']==c['expected_returned_model'] and not r['historical_response_reuse']
        assert r['predicted_choice']==choice(r['raw_response']) and r['status']==('ok' if choice(r['raw_response']) else 'format_error')
        assert r['hit_token_limit']==(r['finish_reason']=='length')
        assert any(a['status']=='success' and a['response_id']==r['response_id'] and a['request_sha256']==r['request_sha256'] and a['execution_id']==r['execution_id'] for a in attempts)
    shared=formats=empty=truncated=0
    for rid,r in responses.items():
        q=requests[rid];origin=fresh[q['payload_sha256']]
        for k in ('raw_response','predicted_choice','response_id','request_id','request_sha256','run_id','experiment_signature','execution_id','requested_model','returned_model','usage','reasoning_tokens','status','historical_response_reuse'):assert r[k]==origin[k]
        assert all(r[k]==q[k] for k in ('id','task','condition','top_memory_ids'))
        is_fresh=rid==origin['record_id'];assert r['fresh_inference_executed_for_this_record']==is_fresh
        assert r['same_run_shared_inference_record_id']==(None if is_fresh else origin['record_id'])
        shared+=not is_fresh;formats+=r['status']!='ok';empty+=not r['raw_response'];truncated+=r['hit_token_limit']
    pairs=[]
    for qid,g in gold.items():
        row={'question_id':qid,'domain':g['task'],'gold_choice':g['gold_choice']}
        for n in NAMES:
            rid=qid+'::'+n;pred=choice(responses[rid]['raw_response']);ids=requests[rid]['top_memory_ids']
            row.update({n+'_answer':pred,n+'_correct':pred==g['gold_choice'],n+'_top5':ids,n+'_gold_hits':len(set(ids)&set(g['gold_ids']))})
        by_payload={}
        for n in NAMES:
            h=requests[qid+'::'+n]['payload_sha256']
            if h in by_payload:assert row[n+'_answer']==by_payload[h]
            by_payload[h]=row[n+'_answer']
        pairs.append(row)
    assert pairs==read(output/'paired_results.json')
    for n in NAMES:
        correct=sum(r[n+'_correct'] for r in pairs)
        assert summary['overall'][n]=={'correct':correct,'questions':371,'accuracy':correct/371,'gold_hits_top5':sum(r[n+'_gold_hits'] for r in pairs),'gold_references':1041,'questions_with_any_gold':sum(r[n+'_gold_hits']>0 for r in pairs)}
        assert summary['method_annotations'][n]['5. 回答方式']['correct']==correct
    table=[]
    bc,mc=(summary['overall'][n]['correct'] for n in RAW)
    for rank,(ridge,alpha) in enumerate(PAIRS,1):
        bz,mz=(condition(n,ridge,alpha) for n in RAW)
        for a,b in ((RAW[0],bz),(RAW[1],mz),(bz,mz)):
            result=comparison(pairs,a,b);same=[r for r in pairs if requests[r['question_id']+'::'+a]['payload_sha256']==requests[r['question_id']+'::'+b]['payload_sha256']]
            assert all(r[a+'_answer']==r[b+'_answer'] for r in same)
            result.update(identical_request_questions=len(same),changed_request_questions=371-len(same),identical_request_answer_disagreements=0)
            assert result==summary['comparisons'][a+'__'+b]
        b,m=(summary['overall'][n]['correct'] for n in (bz,mz))
        table.append({'qwen_rank':rank,'lambda':ridge,'alpha':alpha,'baseline_correct':b,'master_correct':m,'baseline_gain':b-bc,'master_gain':m-mc,'master_minus_baseline':m-b,
                      'baseline_zscore_beats_baseline':b>bc,'master_zscore_beats_master':m>mc,'master_zscore_beats_baseline_zscore':m>b,'passes_all_three':b>bc and m>mc and m>b})
    assert table==summary['candidate_results'] and summary['passing_pairs']==[r for r in table if r['passes_all_three']]
    assert (validation['fresh_requests'],validation['shared_records'],validation['logical_records'])==(len(fresh),shared,4452)
    assert (validation['format_errors'],validation['empty_responses'],validation['token_limit_records'])==(formats,empty,truncated)
    assert summary['parameter_pairs']==eligible and summary['condition_specs']==SPECS
    assert summary['previous_first_validation_retained'] and summary['qwen_search_path']==c['qwen_search_path']
    repeat=read(output/'previous_first_repeatability.json');assert repeat['all_requests_identical'] and not repeat['historical_answers_reused']
    previous_answers=indexed(lines(FIRST/'reader/responses.jsonl'),'record_id');previous_requests=indexed(lines(FIRST/'reader/inputs/requests.jsonl'),'record_id')
    for old_name in (RAW[0],RAW[0]+'_zscore',RAW[1],RAW[1]+'_zscore'):
        name=condition(old_name.removesuffix('_zscore'),.01,1.) if old_name.endswith('_zscore') else old_name;rr=[]
        for qid,g in gold.items():
            rid=qid+'::'+name;oid=qid+'::'+old_name
            assert requests[rid]['payload_sha256']==previous_requests[oid]['payload_sha256']
            a,b=choice(previous_answers[oid]['raw_response']),choice(responses[rid]['raw_response'])
            rr.append({'question_id':qid,'old_answer':a,'new_answer':b,'old_correct':a==g['gold_choice'],'new_correct':b==g['gold_choice']})
        assert repeat['conditions'][old_name]=={**comparison(rr,'old','new'),'identical_requests':371,'changed_questions':[r['question_id'] for r in rr if r['old_answer']!=r['new_answer']]}
    proof.update(overall=summary['overall'],candidate_results=table,passing_pairs=len(summary['passing_pairs']),fresh_requests=len(fresh),shared_records=shared,
                 format_errors=formats,empty_responses=empty,token_limit_records=truncated,failed_attempts=sum(a['status']!='success' for a in attempts),
                 artifact_sha256={n:sha(output/n) for n in ('summary.json','paired_results.json','paired_results.csv','candidate_results.csv','previous_first_repeatability.json','reader/protocol.json','reader/responses.jsonl','reader/fresh_responses.jsonl','retrieval_protocol.json','retrieval_results.jsonl')})
    (output/'independent_audit.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True);return proof

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE);p.add_argument('--inputs-only',action='store_true');a=p.parse_args();audit(a.output,a.inputs_only)
