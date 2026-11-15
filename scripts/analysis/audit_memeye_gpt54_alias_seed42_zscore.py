"""Independent audit of literal gpt-5.4 alias seed42 z-score against alias unweighted.

Does not import the experiment runner, API transport, or request builder.
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

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/memeye_gpt54_alias_seed42_zscore_lambda0p01_alpha1_20261024'
REFERENCE=ROOT/'results/memeye_gpt54_alias_seed42_baselines_20261024'
READER='gpt-5.4'
NAMES=('baseline_question_only_zscore','master_answer_query_zscore')
MODEL='gpt-5.4-2026-03-05'
FIELDS={'record_id','id','task','condition','reader_text','query_images','query_image_paths','top_memory_ids','payload_sha256'}

def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v,compact=True):
 kwargs={'sort_keys':True,'ensure_ascii':False,'allow_nan':False}
 if compact:kwargs['separators']=(',',':')
 return hashlib.sha256(json.dumps(v,**kwargs).encode()).hexdigest()
def indexed(rows,key):
 d={r[key]:r for r in rows};assert len(d)==len(rows);return d
def choice(text):
 m=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',text.strip(),re.I)
 return m.group(1).upper() if m else None

import numpy as np

def aggregate(pairs):
 n=len(pairs);b=sum(p[NAMES[0]+'_correct'] for p in pairs);m=sum(p[NAMES[1]+'_correct'] for p in pairs)
 wins=sum(not p[NAMES[0]+'_correct'] and p[NAMES[1]+'_correct'] for p in pairs)
 losses=sum(p[NAMES[0]+'_correct'] and not p[NAMES[1]+'_correct'] for p in pairs)
 d=wins+losses;p=min(1.,2*sum(math.comb(d,i) for i in range(min(wins,losses)+1))/2**d) if d else 1.
 return {'questions':n,'baseline_correct':b,'expanded_correct':m,'baseline_accuracy':b/n,'expanded_accuracy':m/n,
         'delta_percentage_points':100*(m-b)/n,'wrong_to_correct':wins,'correct_to_wrong':losses,'mcnemar_exact_p':p}


def audit(output=BASE,inputs_only=False):
 output=Path(output);protocol=read(output/'protocol.json');c=protocol['contract'];source=Path(c['memory_path'])
 assert digest(c)==protocol['experiment_signature']
 assert c['caption_model']==MODEL and c['model']==READER and c['requested_model']==READER and c['base_url']=='https://api.openai.com/v1'
 assert c['conditions']==list(NAMES) and (c['questions'],c['logical_records'],c['top_k'])==(371,742,5)
 assert (c['lambda'],c['alpha'])==(.01,1.)
 assert c['reasoning_effort']=='none' and c['temperature']==0 and c['seed']==42 and c['max_completion_tokens']==16
 for filename,expected in c['source_sha256'].items():assert sha(filename)==expected,filename
 origin=read(source/'protocol.json');assert digest(origin['contract'])==origin['experiment_signature']
 assert origin['contract']['scope']=='all_original_qwen_roles' and origin['contract']['model']==MODEL
 assert origin['contract']['reasoning_effort']==c['reasoning_effort'] and origin['contract']['temperature']==c['temperature']
 assert origin['contract']['reader_max_output_tokens']==c['max_completion_tokens'] and origin['contract']['seed'] is None
 assert read(source/'inputs/protocol.json')['reader_system_prompt']==c['reader_system_prompt']
 source_audit=read(source/'independent_audit.json');assert source_audit['audit_status']=='passed'
 reference=read(REFERENCE/'protocol.json');rc=reference['contract']
 assert digest(rc)==reference['experiment_signature'] and c['unweighted_reference']==str(REFERENCE)
 assert rc['frozen_memory_path']==str(source)
 for key in ('model','seed','temperature','reasoning_effort','max_completion_tokens','image_detail','reader_system_prompt','base_url'):
  assert c[key]==rc[key],key
 reference_audit=read(REFERENCE/'independent_audit.json');assert reference_audit['passed']
 for filename,expected in reference_audit['artifact_sha256'].items():assert sha(REFERENCE/filename)==expected
 reference_requests=indexed(lines(REFERENCE/'inputs/requests.jsonl'),'record_id')
 reference_wire=indexed(lines(REFERENCE/'api_request_manifest.jsonl'),'record_id')

 for filename,expected in source_audit['artifact_sha256'].items():assert sha(filename)==expected,filename
 originals=indexed(lines(source/'inputs/source_questions.jsonl'),'id')
 source_gold=indexed(lines(source/'inputs/evaluation.jsonl'),'id')
 gold=indexed(lines(output/'inputs/gold.jsonl'),'id')
 assert set(gold)==set(originals)==set(source_gold) and len(gold)==371
 for qid,g in gold.items():
  assert g=={**{k:source_gold[qid][k] for k in ('id','gold_choice','gold_ids')},'task':originals[qid]['task']}
 # Reconstruct every history and query caption from the original GPT API output.
 cap=indexed(lines(source/'captions.jsonl'),'id')
 cap_answers=indexed(lines(source/'caption_generation/fresh_responses.jsonl'),'request_sha256')
 assert len(cap)==len(cap_answers)==437
 for caption in cap.values():
  a=cap_answers[caption['request_sha256']]
  assert a['returned_model']==MODEL and a['run_id']==origin['run_id']
  assert a['raw_response'].strip()==caption['caption'] and a['record_id']==caption['source_response_record_id']
 source_memories=indexed(lines(source/'inputs/source_memories.jsonl'),'id')
 saved_memories=indexed(lines(source/'offline/memories.jsonl'),'id');memory={}
 graphs={d:read(source/'domains'/d/'memory/graph.json') for d in {q['task'] for q in originals.values()}}
 assert len(source_memories)==438
 for mid,m in source_memories.items():
  text=m['dialogue']+'\nMemory description: '+cap[m['image_sha256']]['caption']
  assert saved_memories[mid]['retrieval_text']==text and saved_memories[mid]['caption']==cap[m['image_sha256']]['caption']
  node=next(n for n in graphs[m['task']]['image_nodes'] if n['id']==m['image_id'])
  assert node['caption']==saved_memories[mid]['caption'] and node['dialogue']==m['dialogue']
  memory[mid]=text
 previous=indexed(lines(source/'retrieval_results.jsonl'),'question_id')
 retrieval=indexed(lines(output/'retrieval_results.jsonl'),'question_id')
 assert set(retrieval)==set(originals)==set(previous)
 retproof=read(output/'retrieval_protocol.json')
 assert retproof['unweighted_full_rankings_equal']==742 and not retproof['vectors_persisted'] and not retproof['memory_modified']
 source_ranks={};group_count=leaf_count=0
 for qid,q in originals.items():
  prior=previous[qid];r=retrieval[qid]
  expected_query=q['question']+("\nquestion's image:\nimage_caption: "+' '.join(cap[im['sha256']]['caption'] for im in q['query_images']) if q['query_images'] else '')
  assert prior['baseline_retrieval_query']==expected_query
  assert prior['candidate_statements']==q['candidate_statements']
  master_by={g['master_id']:g for g in prior['masters']}
  prior_leaves=indexed(prior['leaves'],'memory_id')
  graph=graphs[q['task']];graph_master={g['id']:g for g in graph['masters']}
  assert set(master_by)==set(graph_master)
  expected_ids=[mid for mid,m in source_memories.items() if m['task']==q['task']]
  for condition in r['conditions']:
   name=condition['name'];ls=condition['leaves'];gs=condition['groups']
   assert [l['memory_id'] for l in ls]==expected_ids
   assert [l['leaf_index'] for l in ls]==list(range(len(ls)))
   assert {g['group_id'] for g in gs}==({'all_memory'} if name==NAMES[0] else set(master_by))
   for group in gs:
    selected=[l for l in ls if l['group_id']==group['group_id']]
    assert group['size']==len(selected)
    if name==NAMES[0]:
     assert group['query']==expected_query and group['selected_choice'] is None
    else:
     mg=master_by[group['group_id']]
     assert group['query']==mg['augmented_query']==expected_query+'\nCandidate answer: '+q['candidate_statements'][mg['selected_choice']]
     assert group['selected_choice']==mg['selected_choice'] and group['selected_statement']==mg['selected_statement']
     assert {source_memories[l['memory_id']]['image_id'] for l in selected}==set(graph_master[group['group_id']]['member_ids'])
    raw=np.array([l['raw'] for l in selected]);white=np.array([l['white'] for l in selected]);z=np.array([l['z'] for l in selected])
    field='baseline_score' if name==NAMES[0] else 'expanded_score'
    np.testing.assert_allclose(raw,[prior_leaves[l['memory_id']][field] for l in selected],atol=2e-6,rtol=0)
    if len(selected)==1:expected_z=white.copy()
    elif group['z_status']=='zero_variance':expected_z=np.zeros(len(selected))
    else:expected_z=(white-white.mean())/white.std()
    np.testing.assert_allclose(z,expected_z,atol=1e-9,rtol=0)
    corr=np.zeros(len(raw)) if len(raw)==1 or group['raw_status']=='zero_variance' or group['z_status']=='zero_variance' else raw.std()*np.tanh(z)
    np.testing.assert_allclose([l['correction'] for l in selected],corr,atol=1e-14,rtol=0)
    np.testing.assert_allclose([l['weighted'] for l in selected],raw+corr,atol=1e-14,rtol=0)
    group_count+=1;leaf_count+=len(selected)
   ranked=sorted(ls,key=lambda l:-l['weighted'])
   assert [l['memory_id'] for l in ranked]==condition['full_rank_memory_ids']
   assert [l['memory_id'] for l in ranked[:5]]==condition['top5_memory_ids']
   source_ranks[(qid,name)]=condition['top5_memory_ids']
 assert group_count==retproof['independent_inverse_ridge_groups'] and leaf_count==retproof['group_leaf_scores_checked']
 requests_list=lines(output/'inputs/requests.jsonl');requests=indexed(requests_list,'record_id')
 wire_list=lines(output/'api_request_manifest.jsonl');wire=indexed(wire_list,'record_id')
 assert digest(wire_list)==c['request_manifest_sha256'] and set(requests)==set(wire) and len(requests)==742
 representative={};image_records=0;identical=Counter()
 for row in requests_list:
  assert set(row)==FIELDS and row['record_id']==row['id']+'::'+row['condition']
  q=originals[row['id']];ids=source_ranks[(row['id'],row['condition'])]
  assert row['task']==q['task'] and len(ids)==len(set(ids))==5 and row['top_memory_ids']==ids
  text='Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
  text+='\n\n'.join(f'Memory {i}: {memory[mid]}' for i,mid in enumerate(ids,1))
  text+='\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(k+'. '+q['options'][k] for k in 'ABCD')+'\n\nAnswer with one letter only.'
  assert row['reader_text']==text and row['query_images']==q['query_images']
  assert row['query_image_paths']==[im['path'] for im in q['query_images']]
  payload={'system':c['reader_system_prompt'],'user_text':text,'query_images':q['query_images']}
  assert digest(payload)==row['payload_sha256']
  content=[]
  for im in q['query_images']:
   assert sha(im['path'])==im['sha256']
   data=Path(im['path']).read_bytes()
   mime='image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg' if data.startswith(b'\xff\xd8\xff') else None
   assert mime is not None
   content.append({'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(data).decode(),'detail':'high'}})
  content.append({'type':'text','text':text})
  body={'model':READER,'seed':42,'messages':[{'role':'system','content':c['reader_system_prompt']},{'role':'user','content':content}],
        'temperature':0,'max_completion_tokens':16,'reasoning_effort':'none','n':1,'stream':False,'store':False}
  w=wire[row['record_id']]
  assert w['payload_sha256']==row['payload_sha256'] and w['request_sha256']==digest(body) and w['query_image_count']==len(q['query_images'])
  raw=row['condition'].removesuffix('_zscore');oldrid=row['id']+'::'+raw
  old=reference_requests[oldrid]
  old_body={**body,'messages':[body['messages'][0],{'role':'user','content':content[:-1]+[{'type':'text','text':old['reader_text']}]}]}
  assert digest(old_body)==reference_wire[oldrid]['request_sha256']
  old_top=next(z for z in previous[row['id']]['conditions'] if z['name']==raw)['top5_memory_ids']
  assert old['top_memory_ids']==old_top
  same=row['payload_sha256']==old['payload_sha256']
  assert same==(w['request_sha256']==reference_wire[oldrid]['request_sha256'])
  identical[raw]+=same
  assert w['requested_model']==READER
  representative.setdefault(row['payload_sha256'],row['record_id']);image_records+=bool(q['query_images'])
 assert len(representative)==c['unique_payloads'] and image_records==16
 assert Counter(r['condition'] for r in requests_list)==dict.fromkeys(NAMES,371)
 old_inputs=indexed(lines(REFERENCE/'inputs/requests.jsonl'),'record_id')
 old_answers=indexed(lines(REFERENCE/'responses.jsonl'),'record_id')
 old_wire=indexed(lines(REFERENCE/'api_request_manifest.jsonl'),'record_id')
 raw_names=('baseline_question_only','master_answer_query')
 old_correct={name:sum(choice(old_answers[qid+'::'+name]['raw_response'])==g['gold_choice'] for qid,g in gold.items()) for name in raw_names}
 assert old_correct==c['prior_unweighted_correct']==dict(zip(raw_names,(236,238)))
 assert c['reader_control_audit']=={'generation_parameters_equal':True,'request_pairs_checked':742,'identical_full_requests':dict(identical),'changes_confined_to_retrieved_context':True}
 checks={'passed':True,'questions':371,'records':742,'unique_payloads':len(representative),'native_image_records':image_records,
    'caption_model':MODEL,'reader_model':READER,'seed':42,'caption_api_provenance_verified':437,'source_unweighted_correct':old_correct,
    'frozen_source_hashes_verified':True,'gold_fields_absent_from_api_inputs':True,'api_request_bodies_independently_reconstructed':742,
    'weighting_groups_checked':group_count,'group_leaf_scores_checked':leaf_count,'unweighted_full_rankings_equal':742,'reader_control_audit':c['reader_control_audit']}
 if inputs_only:
  (output/'inputs/independent_input_audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps({'stage':'independent_input_audit','checks':checks}),flush=True);return checks
 validation=read(output/'validation_output.json');summary=read(output/'summary.json')
 response_list=lines(output/'responses.jsonl');responses=indexed(response_list,'record_id')
 fresh_list=lines(output/'fresh_responses.jsonl');fresh=indexed(fresh_list,'payload_sha256')
 attempts=lines(output/'api_attempts.jsonl')
 assert set(responses)==set(requests) and len(responses)==742 and set(fresh)==set(representative)
 assert len({r['response_id'] for r in fresh_list})==len(fresh)
 historical_ids={r['response_id'] for r in old_answers.values()}
 for historical in (source/'reader/fresh_responses.jsonl', ROOT/'results/memeye_gpt54_caption_zscore_lambda0p01_alpha1_20261024/fresh_responses.jsonl'):
  historical_ids.update(r['response_id'] for r in lines(historical))
 assert not (historical_ids & {r['response_id'] for r in fresh_list})
 assert len({r['returned_model'] for r in fresh_list})==1
 assert summary['returned_models']==validation['returned_models']==sorted({r['returned_model'] for r in fresh_list})

 successes=Counter(a['payload_sha256'] for a in attempts if a['status']=='success')
 assert successes==dict.fromkeys(fresh,1)
 for name in ('responses','fresh_responses'):assert sha(output/(name+'.jsonl'))==validation[name+'_sha256']
 same_run_shared=0;formats=empty=truncated=0;reasoning=Counter();image_fresh=0
 for r in fresh_list:
  rid=r['record_id'];q=requests[rid]
  assert rid==representative[r['payload_sha256']]
  assert r['run_id']==protocol['run_id'] and r['experiment_signature']==protocol['experiment_signature']
  assert r['request_sha256']==wire[rid]['request_sha256'] and re.fullmatch(r'gpt-5\.4(?:-\d{4}-\d{2}-\d{2})?',r['returned_model']) and r['requested_model']==READER and not r['historical_response_reuse']
  assert r['predicted_choice']==choice(r['raw_response']) and r['status']==('ok' if choice(r['raw_response']) else 'format_error')
  assert r['query_image_count']==len(q['query_images']) and r['hit_token_limit']==(r['finish_reason']=='length')
  assert any(a['status']=='success' and a['record_id']==rid and a['response_id']==r['response_id'] and a['execution_id']==r['execution_id'] and a['request_sha256']==r['request_sha256'] for a in attempts)
  value=((r.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
  assert r.get('reasoning_tokens')==value
  reasoning['unreported' if value is None else 'reported_zero' if value==0 else 'reported_nonzero']+=1
  reasoning['reported_token_total']+=value or 0;image_fresh+=bool(q['query_images'])
 for rid,r in responses.items():
  q=requests[rid];origin=fresh[q['payload_sha256']]
  assert r['payload_sha256']==q['payload_sha256'] and r['top_memory_ids']==q['top_memory_ids']
  assert r['id']==q['id'] and r['condition']==q['condition'] and r['task']==q['task']
  for k in ('raw_response','predicted_choice','response_id','request_id','request_sha256','run_id','experiment_signature','returned_model','requested_model','execution_id','usage','status','refusal','hit_token_limit','historical_response_reuse','query_image_count'):
   assert r[k]==origin[k],(rid,k)
  is_fresh=rid==origin['record_id']
  assert r['fresh_inference_executed_for_this_record']==is_fresh
  assert r['same_run_shared_inference_record_id']==(None if is_fresh else origin['record_id'])
  same_run_shared+=not is_fresh;formats+=r['status']!='ok';empty+=not r['raw_response'];truncated+=r['hit_token_limit']
 pairs=[]
 for qid,g in gold.items():
  r={'question_id':qid,'domain':g['task'],'gold_choice':g['gold_choice']}
  for name,raw in zip(NAMES,raw_names):
   rid=qid+'::'+name;oldrid=qid+'::'+raw
   pred=choice(responses[rid]['raw_response']);oldpred=choice(old_answers[oldrid]['raw_response'])
   ids=requests[rid]['top_memory_ids']
   r.update({name+'_answer':pred,name+'_correct':pred==g['gold_choice'],name+'_top5':ids,name+'_gold_hits':len(set(ids)&set(g['gold_ids'])),
             raw+'_historical_answer':oldpred,raw+'_historical_correct':oldpred==g['gold_choice'],
             raw+'_identical_historical_payload':requests[rid]['payload_sha256']==old_inputs[oldrid]['payload_sha256']})
  pairs.append(r)
 assert pairs==read(output/'paired_results.json') and aggregate(pairs)==summary['overall']
 for domain,values in summary['by_domain'].items():assert aggregate([p for p in pairs if p['domain']==domain])==values
 for name,raw in zip(NAMES,raw_names):
  v=summary['historical_comparisons'][name]
  assert v['historical_unweighted_correct']==old_correct[raw]
  assert v['current_weighted_correct']==sum(r[name+'_correct'] for r in pairs)
  wins=sum(not r[raw+'_historical_correct'] and r[name+'_correct'] for r in pairs)
  losses=sum(r[raw+'_historical_correct'] and not r[name+'_correct'] for r in pairs)
  assert (v['wrong_to_right'],v['right_to_wrong'],v['net_correct'])==(wins,losses,wins-losses)
  assert v['identical_payload_questions']==sum(r[raw+'_identical_historical_payload'] for r in pairs)
  assert v['identical_payload_answer_disagreements']==[r['question_id'] for r in pairs if r[raw+'_identical_historical_payload'] and r[raw+'_historical_answer']!=r[name+'_answer']]
  for label,same in (('identical_request',True),('changed_request',False)):
   rows=[r for r in pairs if r[raw+'_identical_historical_payload']==same]
   sw=sum(not r[raw+'_historical_correct'] and r[name+'_correct'] for r in rows)
   sl=sum(r[raw+'_historical_correct'] and not r[name+'_correct'] for r in rows)
   assert v['request_strata'][label]=={'questions':len(rows),'wrong_to_right':sw,'right_to_wrong':sl,'net_correct':sw-sl,'answer_changes':sum(r[raw+'_historical_answer']!=r[name+'_answer'] for r in rows)}
  discordant=wins+losses
  expected_p=min(1.,2*sum(math.comb(discordant,i) for i in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
  assert v['mcnemar_exact_p']==expected_p
  stats=summary['retrieval_stats'][name]
  assert stats['gold_hits_top5']==sum(r[name+'_gold_hits'] for r in pairs)
  assert stats['questions_with_any_gold']==sum(r[name+'_gold_hits']>0 for r in pairs)
  assert summary['method_annotations'][name]['5. 回答方式']['correct']==v['current_weighted_correct']
 assert summary['caption_model']==MODEL and summary['reader_model']==summary['requested_model']==READER and summary['seed']==42 and summary['memory_path']==str(source)
 usage={k:sum((r.get('usage') or {}).get(k,0) or 0 for r in fresh_list) for k in ('prompt_tokens','completion_tokens','total_tokens')}
 assert usage==summary['usage']
 assert validation['fresh_requests']==len(fresh) and validation['shared_records']==same_run_shared
 assert (validation['format_errors'],validation['empty_responses'],validation['token_limit_records'])==(formats,empty,truncated)
 assert validation['reasoning_tokens_reported']==reasoning['reported_token_total'] and validation['reasoning_usage_missing_requests']==reasoning['unreported']
 checks.update(fresh_requests=len(fresh),same_run_shared_records=same_run_shared,native_image_fresh_requests=image_fresh,
    format_errors=formats,empty_responses=empty,token_limit_records=truncated,reasoning_usage=dict(reasoning),
    api_attempts=len(attempts),transient_failed_attempts=sum(a['status']!='success' for a in attempts),
    refusals=sum(bool(r['refusal']) for r in fresh_list),returned_model_counts=dict(Counter(r['returned_model'] for r in fresh_list)),
    historical_response_ids_excluded=True,all_response_provenance_and_summary_counts_verified=True,all_requests_fresh_within_current_run=True,
    overall=summary['overall'],artifact_sha256={n:sha(output/n) for n in ('protocol.json','summary.json','responses.jsonl','fresh_responses.jsonl','api_attempts.jsonl','paired_results.csv')})
 (output/'independent_audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'stage':'independent_audit','passed':True,'fresh_requests':len(fresh),'overall':summary['overall']},ensure_ascii=False),flush=True)
 return checks


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=BASE);p.add_argument('--inputs-only',action='store_true')
 a=p.parse_args();audit(a.output,a.inputs_only)
