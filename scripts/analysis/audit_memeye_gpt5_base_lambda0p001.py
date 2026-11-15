"""Independent GPT-5 lambda=.001 control, API-byte, provenance and scoring audit."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# Independently implemented helpers; no generation or retrieval runner imports.
from scripts.analysis.audit_memeye_gpt5_base import read,lines,sha,digest,indexed,choice,verify_hashes,wire,comparison,FIELDS
BASE=ROOT/'results/memeye_gpt5_base_seed42_zscore_lambda0p001_alpha1_20261024'
NAMES=('baseline_question_only_zscore','master_answer_query_zscore')


def audit(output=BASE,inputs_only=False):
    output=Path(output).resolve();out=output/'reader'
    p=read(out/'protocol.json');c=p['contract'];assert digest(c)==p['experiment_signature']
    source=Path(c['memory_path']);prior=read(source/'reader/protocol.json');pc=prior['contract']
    assert digest(pc)==prior['experiment_signature'] and c['lambda_reference_path']==str(source)
    assert c['conditions']==list(NAMES) and (c['lambda'],pc['lambda'],c['alpha'],pc['alpha'])==(.001,.01,1.,1.)
    assert (c['logical_records'],c['questions'],c['top_k'])==(742,371,5)
    assert not c['caption_regenerated'] and not c['memory_rebuilt'] and not c['historical_api_answer_reuse'] and not c['gold_in_api_payload']
    for k in ('model','requested_model','expected_returned_model','caption_model','seed','caption_seed','temperature','temperature_wire_policy','reasoning_effort',
              'max_completion_tokens','image_detail','base_url','api_config','reader_system_prompt','timeout_seconds','max_attempts','memory_path','top_k','workers'):
        assert c[k]==pc[k],k
    assert c['model']=='gpt-5' and c['expected_returned_model']==c['caption_model']=='gpt-5-2025-08-07'
    assert (c['seed'],c['reasoning_effort'],c['temperature'],c['max_completion_tokens'])==(42,'minimal',None,2048)
    verify_hashes(c['source_sha256']);verify_hashes(pc['source_sha256'])
    source_audit=read(source/'independent_audit.json');assert source_audit['passed']
    verify_hashes({str(source/n):v for n,v in source_audit['artifact_sha256'].items()})
    gold=indexed(lines(out/'inputs/gold.jsonl'),'id');qs=indexed(lines(source/'inputs/source_questions.jsonl'),'id')
    assert gold==indexed(lines(source/'reader/inputs/gold.jsonl'),'id') and set(qs)==set(gold) and len(gold)==371
    memories=indexed(lines(source/'offline/memories.jsonl'),'id');assert len(memories)==438
    ret=read(output/'retrieval_protocol.json');old_ret=read(source/'zscore/retrieval_protocol.json')
    for r in (ret,old_ret):verify_hashes(r['source_sha256']);verify_hashes(r['output_sha256'])
    assert ret['memory_path']==str(source) and (ret['lambda'],ret['alpha'])==(.001,1.)
    assert ret['unweighted_full_rankings_equal']==742 and ret['max_raw_score_difference']==0
    assert ret['max_inverse_ridge_difference']<2e-12 and ret['lambda0p01_raw_scores_and_groups_exactly_equal']
    assert not ret['vectors_persisted'] and not ret['memory_modified'] and not ret['old_images_reencoded'] and not ret['gold_loaded_for_ranking']
    assert ret['encoding']==old_ret['encoding'] and ret['master_counts']==old_ret['master_counts'] and sum(ret['master_counts'].values())==72
    current=indexed(lines(output/'retrieval_results.jsonl'),'question_id');old=indexed(lines(source/'zscore/retrieval_results.jsonl'),'question_id')
    assert set(current)==set(old)==set(gold)
    ranks={};groups=leaf_count=0;changed=dict.fromkeys(NAMES,0)
    for qid in gold:
        assert current[qid]['task']==old[qid]['task']==gold[qid]['task']
        cc=indexed(current[qid]['conditions'],'name');oc=indexed(old[qid]['conditions'],'name');assert set(cc)==set(oc)==set(NAMES)
        for n in NAMES:
            a,b=cc[n],oc[n];ls=a['leaves'];assert len(ls)==len(b['leaves']) and len(a['groups'])==len(b['groups'])
            for x,y in zip(ls,b['leaves']):
                for k in ('memory_id','leaf_index','group_id','raw'):assert x[k]==y[k],(qid,n,k)
                assert memories[x['memory_id']]['task']==gold[qid]['task']
            for x,y in zip(a['groups'],b['groups']):
                for k in ('group_id','size','query','selected_choice','selected_statement','raw_std','raw_status'):assert x[k]==y[k],(qid,n,k)
                selected=[l for l in ls if l['group_id']==x['group_id']];assert len(selected)==x['size']
                raw=np.array([l['raw'] for l in selected]);white=np.array([l['white'] for l in selected]);z=np.array([l['z'] for l in selected])
                expected_z=white if len(selected)==1 else np.zeros(len(selected)) if x['z_status']=='zero_variance' else (white-white.mean())/white.std()
                np.testing.assert_allclose(z,expected_z,atol=1e-9,rtol=0)
                correction=np.zeros(len(selected)) if len(selected)==1 or x['raw_status']=='zero_variance' or x['z_status']=='zero_variance' else raw.std()*np.tanh(z)
                np.testing.assert_allclose([l['correction'] for l in selected],correction,atol=1e-14,rtol=0)
                np.testing.assert_allclose([l['weighted'] for l in selected],raw+correction,atol=1e-14,rtol=0)
                groups+=1;leaf_count+=len(selected)
            ranked=sorted(ls,key=lambda l:-l['weighted'])
            assert a['full_rank_memory_ids']==[l['memory_id'] for l in ranked]
            assert a['top5_memory_ids']==[l['memory_id'] for l in ranked[:5]]
            np.testing.assert_allclose(a['top5_scores'],[l['weighted'] for l in ranked[:5]],atol=0,rtol=0)
            ranks[qid,n]=a['top5_memory_ids'];changed[n]+=a['top5_memory_ids']!=b['top5_memory_ids']
    assert groups==ret['independent_inverse_ridge_groups'] and leaf_count==ret['group_leaf_scores_checked']
    requests_list=lines(out/'inputs/requests.jsonl');requests=indexed(requests_list,'record_id')
    manifest_list=lines(out/'api_request_manifest.jsonl');manifest=indexed(manifest_list,'record_id')
    old_requests=indexed(lines(source/'reader/inputs/requests.jsonl'),'record_id');old_manifest=indexed(lines(source/'reader/api_request_manifest.jsonl'),'record_id')
    assert len(requests)==742 and set(requests)==set(manifest) and digest(manifest_list)==c['request_manifest_sha256']
    assert Counter(r['condition'] for r in requests_list)==dict.fromkeys(NAMES,371)
    representatives={};same={n:[] for n in NAMES};image_count=0
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
        equal=w['request_sha256']==old_manifest[r['record_id']]['request_sha256']
        assert equal==(r['payload_sha256']==old_requests[r['record_id']]['payload_sha256'])
        if equal:same[r['condition']].append(r['id'])
        representatives.setdefault(r['payload_sha256'],r['record_id']);image_count+=bool(q['query_images'])
    assert len(representatives)==c['unique_payloads'] and image_count==16
    assert all(371-len(same[n])==changed[n] for n in NAMES)
    proof={'passed':True,'questions':371,'logical_records':742,'unique_requests':len(representatives),'lambda':.001,'reference_lambda':.01,'alpha':1.,
        'same_caption_memory_and_reader_settings':True,'caption_model':c['caption_model'],'requested_model':c['model'],'expected_returned_model':c['expected_returned_model'],
        'raw_scores_exactly_equal':leaf_count,'groups_and_queries_exactly_equal':groups,'unweighted_rankings_verified':742,
        'request_bodies_independently_reconstructed':742,'ordered_top5_changed_questions':changed,'identical_request_questions':{n:len(same[n]) for n in NAMES},
        'gold_excluded_from_api_payload':True,'source_artifacts_unchanged':True}
    if inputs_only:
        (out/'inputs/independent_input_audit.json').write_text(json.dumps(proof,indent=2)+'\n')
        print(json.dumps({'stage':'input_audit',**proof}),flush=True);return proof
    responses=indexed(lines(out/'responses.jsonl'),'record_id');fresh_list=lines(out/'fresh_responses.jsonl');fresh=indexed(fresh_list,'payload_sha256')
    prior_answers=indexed(lines(source/'reader/responses.jsonl'),'record_id')
    attempts=lines(out/'api_attempts.jsonl');validation=read(out/'validation_output.json');summary=read(output/'summary.json')
    assert set(responses)==set(requests) and set(fresh)==set(representatives)
    assert Counter(a['payload_sha256'] for a in attempts if a['status']=='success')==dict.fromkeys(fresh,1)
    old_ids={r['response_id'] for r in prior_answers.values()}|{r['response_id'] for r in lines(source/'caption_generation/fresh_responses.jsonl')}
    assert len({r['response_id'] for r in fresh_list})==len(fresh_list) and not old_ids.intersection(r['response_id'] for r in fresh_list)
    for r in fresh_list:
        rid=r['record_id'];assert rid==representatives[r['payload_sha256']]
        assert r['run_id']==p['run_id'] and r['experiment_signature']==p['experiment_signature'] and r['request_sha256']==manifest[rid]['request_sha256']
        assert r['requested_model']=='gpt-5' and r['returned_model']==c['expected_returned_model'] and not r['historical_response_reuse']
        assert r['predicted_choice']==choice(r['raw_response']) and r['status']==('ok' if choice(r['raw_response']) else 'format_error')
        assert r['hit_token_limit']==(r['finish_reason']=='length')
        assert r['reasoning_tokens']==((r.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
        assert any(a['status']=='success' and a['response_id']==r['response_id'] and a['request_sha256']==r['request_sha256'] and a['execution_id']==r['execution_id'] for a in attempts)
    shared=formats=empty=truncated=0
    for rid,r in responses.items():
        q=requests[rid];origin=fresh[q['payload_sha256']]
        for k in ('raw_response','predicted_choice','response_id','request_id','request_sha256','run_id','experiment_signature','execution_id','requested_model','returned_model','usage','reasoning_tokens','status','historical_response_reuse'):
            assert r[k]==origin[k]
        assert all(r[k]==q[k] for k in ('id','task','condition','top_memory_ids'))
        is_fresh=rid==origin['record_id'];assert r['fresh_inference_executed_for_this_record']==is_fresh
        assert r['same_run_shared_inference_record_id']==(None if is_fresh else origin['record_id'])
        shared+=not is_fresh;formats+=r['status']!='ok';empty+=not r['raw_response'];truncated+=r['hit_token_limit']
    pairs=[]
    for qid,g in gold.items():
        row={'question_id':qid,'domain':g['task'],'gold_choice':g['gold_choice']}
        for n in NAMES:
            rid=qid+'::'+n;answer=choice(responses[rid]['raw_response']);old_answer=choice(prior_answers[rid]['raw_response'])
            raw=choice(prior_answers[qid+'::'+n.removesuffix('_zscore')]['raw_response']);ids=requests[rid]['top_memory_ids']
            row.update({n+'_answer':answer,n+'_correct':answer==g['gold_choice'],n+'_reference_answer':old_answer,n+'_reference_correct':old_answer==g['gold_choice'],
                n+'_unweighted_answer':raw,n+'_unweighted_correct':raw==g['gold_choice'],n+'_top5':ids,n+'_gold_hits':len(set(ids)&set(g['gold_ids'])),
                n+'_same_payload':qid in same[n],n+'_same_request':qid in same[n]})
        pairs.append(row)
    assert pairs==read(output/'paired_results.json')
    for n in NAMES:
        expected={'correct':sum(r[n+'_correct'] for r in pairs),'questions':371,'accuracy':sum(r[n+'_correct'] for r in pairs)/371,
            'gold_hits_top5':sum(r[n+'_gold_hits'] for r in pairs),'gold_references':1041,'questions_with_any_gold':sum(r[n+'_gold_hits']>0 for r in pairs)}
        assert summary['overall'][n]==expected
        metric=comparison(pairs,n+'_reference',n)
        unchanged=[r for r in pairs if r[n+'_same_request']];changed_pairs=[r for r in pairs if not r[n+'_same_request']]
        metric.update(identical_request_questions=len(unchanged),identical_request_answer_disagreements=[r['question_id'] for r in unchanged if r[n+'_answer']!=r[n+'_reference_answer']],
            identical_request_net_correct=sum(r[n+'_correct']-r[n+'_reference_correct'] for r in unchanged),changed_request_questions=len(changed_pairs),
            changed_request_net_correct=sum(r[n+'_correct']-r[n+'_reference_correct'] for r in changed_pairs))
        assert summary['lambda_comparisons'][n]==metric and metric['reference_correct']==221
        assert summary['unweighted_historical_correct'][n]==sum(r[n+'_unweighted_correct'] for r in pairs)
        assert summary['method_annotations'][n]['5. 回答方式']['correct']==expected['correct']
        assert summary['method_annotations'][n]['3. 检索方式']['lambda']==.001
    assert list(summary['unweighted_historical_correct'].values())==[219,223]
    for d,v in summary['by_domain'].items():
        rs=[r for r in pairs if r['domain']==d];assert v=={n:{'correct':sum(r[n+'_correct'] for r in rs),'questions':len(rs)} for n in NAMES}
    assert summary['usage']=={k:sum((r.get('usage') or {}).get(k,0) or 0 for r in fresh_list) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    assert validation['passed'] and validation['logical_records']==742 and validation['fresh_requests']==len(fresh) and validation['shared_records']==shared
    assert (validation['format_errors'],validation['empty_responses'],validation['token_limit_records'])==(formats,empty,truncated)
    assert validation['reasoning_tokens_reported']==sum(r.get('reasoning_tokens') or 0 for r in fresh_list)
    for name in ('responses','fresh_responses'):assert sha(out/(name+'.jsonl'))==validation[name+'_sha256']
    proof.update(fresh_reader_requests=len(fresh),same_run_shared_records=shared,format_errors=formats,empty_responses=empty,token_limit_records=truncated,
        api_attempts=len(attempts),failed_attempts=sum(a['status']!='success' for a in attempts),refusals=sum(bool(r['refusal']) for r in fresh_list),
        historical_answers_reused=0,returned_models=dict(Counter(r['returned_model'] for r in fresh_list)),reasoning_tokens=validation['reasoning_tokens_reported'],
        overall=summary['overall'],lambda_comparisons=summary['lambda_comparisons'],artifact_sha256={n:sha(output/n) for n in ('retrieval_protocol.json','retrieval_results.jsonl',
            'reader/protocol.json','reader/responses.jsonl','reader/fresh_responses.jsonl','reader/validation_output.json','summary.json','paired_results.json','paired_results.csv','report.html')})
    (output/'independent_audit.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'stage':'independent_audit','passed':True,'overall':proof['overall']}),flush=True)
    return proof

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=BASE);p.add_argument('--inputs-only',action='store_true')
    a=p.parse_args();audit(a.output,a.inputs_only)
