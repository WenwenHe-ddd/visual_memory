"""Independent, offline audit of GPT-5.4 on the two frozen z-score conditions.

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
BASE=ROOT/'results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024'
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

def aggregate(pairs):
 n=len(pairs);conditions={name:{'correct':sum(p[name+'_correct'] for p in pairs),'accuracy':sum(p[name+'_correct'] for p in pairs)/n} for name in NAMES}
 wins=sum(not p[NAMES[0]+'_correct'] and p[NAMES[1]+'_correct'] for p in pairs)
 losses=sum(p[NAMES[0]+'_correct'] and not p[NAMES[1]+'_correct'] for p in pairs)
 discordant=wins+losses
 p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
 return {'questions':n,'conditions':conditions,'master_vs_baseline':{'wrong_to_right':wins,'right_to_wrong':losses,'net_correct':wins-losses,
     'delta_percentage_points':100*(wins-losses)/n,'mcnemar_exact_two_sided_p':p}}


def audit(output=BASE,inputs_only=False):
 output=Path(output);protocol=read(output/'protocol.json');c=protocol['contract']
 assert digest(c)==protocol['experiment_signature']
 assert c['model']==MODEL and c['base_url']=='https://api.openai.com/v1'
 assert c['conditions']==list(NAMES) and c['questions']==371 and c['logical_records']==742
 assert (c['lambda'],c['alpha'],c['top_k'])==(.01,1.,5)
 assert c['reasoning_effort']=='none' and c['temperature']==0 and c['seed'] is None and c['max_completion_tokens']==16
 for filename,expected in c['source_sha256'].items():assert sha(filename)==expected,filename
 requests_list=lines(output/'inputs/requests.jsonl');requests=indexed(requests_list,'record_id')
 wire_list=lines(output/'api_request_manifest.jsonl');wire=indexed(wire_list,'record_id')
 assert digest(wire_list)==c['request_manifest_sha256'] and set(requests)==set(wire) and len(requests)==742
 originals=indexed(lines(ROOT/'data/independent_visual_inputs/qa_manifest.jsonl'),'id')
 memory={r['id']:r['reader_text'] for r in lines(ROOT/'data/independent_visual_inputs/retrieval_memories.jsonl')}
 gold=indexed(lines(output/'inputs/gold.jsonl'),'id');assert set(gold)==set(originals) and len(gold)==371
 assert all(g=={k:originals[qid][k] for k in ('id','task','gold_choice','gold_ids')} for qid,g in gold.items())
 source={};qwen_counts={};source_ranks={}
 for name in NAMES:
  origin=c['retrieval_sources'][name];directory=Path(origin['directory']);condition=origin['condition']
  rp=read(directory/'reader/protocol.json');unsigned=dict(rp);signature=unsigned.pop('experiment_signature')
  assert digest(unsigned,False)==signature
  assert rp['reader_settings']['reader_system_prompt']==c['reader_system_prompt']
  manifest=indexed(lines(directory/'reader/qa_manifest.jsonl'),'record_id');answers=indexed(lines(directory/'reader/responses.jsonl'),'record_id')
  retrieval=indexed(lines(directory/'retrieval_results.jsonl'),'question_id');assert set(retrieval)==set(originals)
  root=read(directory/'protocol.json')
  if name==NAMES[0]:assert condition=='lambda_0p01' and root['alpha']==1 and .01 in root['new_lambdas']
  else:assert condition=='master_answer_query_zscore' and root['lambda']==.01 and root['alpha']==1
  selected={row['id']:row for row in manifest.values() if row['condition']==condition};assert set(selected)==set(originals)
  count=0
  for qid,row in selected.items():
   ret=retrieval[qid];condition_row=next(t for t in ret['conditions'] if t['name']==condition)
   data=ret.get('weighting',{}).get(condition,ret)
   ranked=[l['memory_id'] for l in sorted(data['leaf_scores'],key=lambda l:-l['weighted_score'])[:5]]
   assert ranked==condition_row['top5_memory_ids']==row['top_memory_ids']
   if name==NAMES[0]:assert condition_row['lambda']==.01
   a=answers[row['record_id']]
   assert a['experiment_signature']==signature and a['payload_sha256']==row['payload_sha256']
   assert a['predicted_choice']==choice(a['raw_response']) and a['correct']==(a['predicted_choice']==gold[qid]['gold_choice'])
   count+=a['correct'];source_ranks[(qid,name)]=ranked
  source[name]=selected;qwen_counts[name]=count
 assert qwen_counts=={NAMES[0]:188,NAMES[1]:201}
 representative={};image_records=0
 for row in requests_list:
  assert set(row)==FIELDS and row['condition'] in NAMES and row['record_id']==row['id']+'::'+row['condition']
  q=originals[row['id']];ids=source_ranks[(row['id'],row['condition'])]
  assert row['task']==q['task'] and len(ids)==len(set(ids))==5 and ids==row['top_memory_ids']
  assert all(mid.split('/',1)[0]==q['task'] for mid in ids)
  text='Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
  text+='\n\n'.join(f'Memory {i}: {memory[mid]}' for i,mid in enumerate(ids,1))
  text+='\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'
  text+='\n'.join(k+'. '+q['options'][k] for k in 'ABCD')+'\n\nAnswer with one letter only.'
  images=[{'path':p,'sha256':sha(p)} for p in q['query_image_paths']]
  assert row['reader_text']==text and row['query_images']==images and row['query_image_paths']==q['query_image_paths']
  payload={'system':c['reader_system_prompt'],'user_text':text,'query_images':images}
  assert digest(payload,False)==row['payload_sha256']==source[row['condition']][row['id']]['payload_sha256']
  content=[]
  for im in images:
   data=Path(im['path']).read_bytes()
   mime='image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg' if data.startswith(b'\xff\xd8\xff') else None
   assert mime is not None
   content.append({'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(data).decode(),'detail':'high'}})
  content.append({'type':'text','text':text})
  body={'model':MODEL,'messages':[{'role':'system','content':c['reader_system_prompt']},{'role':'user','content':content}],
        'temperature':0,'max_completion_tokens':16,'reasoning_effort':'none','n':1,'stream':False,'store':False}
  w=wire[row['record_id']]
  assert w['payload_sha256']==row['payload_sha256'] and w['request_sha256']==digest(body) and w['query_image_count']==len(images)
  representative.setdefault(row['payload_sha256'],row['record_id']);image_records+=bool(images)
 assert len(representative)==c['unique_payloads'] and image_records==16
 assert Counter(r['condition'] for r in requests_list)==dict.fromkeys(NAMES,371)
 checks={'passed':True,'questions':371,'records':742,'unique_payloads':len(representative),'native_image_records':image_records,
    'frozen_source_hashes_verified':True,'all_payloads_reconstructed_from_original_dialogue_full_caption':True,
    'selected_lambda_alpha_and_frozen_top5_verified':True,'api_request_bodies_independently_reconstructed':742,
    'gold_fields_absent_from_api_inputs':True,'qwen_reference_correct':qwen_counts}
 if inputs_only:
  (output/'inputs/independent_input_audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps({'stage':'independent_input_audit','checks':checks}),flush=True);return checks
 validation=read(output/'validation_output.json');summary=read(output/'summary.json')
 response_list=lines(output/'responses.jsonl');responses=indexed(response_list,'record_id')
 fresh_list=lines(output/'fresh_responses.jsonl');fresh=indexed(fresh_list,'payload_sha256')
 attempts=lines(output/'api_attempts.jsonl')
 assert set(responses)==set(requests) and len(responses)==742 and set(fresh)==set(representative)
 assert len({r['response_id'] for r in fresh_list})==len(fresh)
 successes=Counter(a['payload_sha256'] for a in attempts if a['status']=='success')
 assert successes==dict.fromkeys(fresh,1)
 for name in ('responses','fresh_responses'):assert sha(output/(name+'.jsonl'))==validation[name+'_sha256']
 same_run_shared=0;formats=empty=truncated=0;reasoning=Counter();image_fresh=0
 for r in fresh_list:
  rid=r['record_id'];q=requests[rid]
  assert rid==representative[r['payload_sha256']]
  assert r['run_id']==protocol['run_id'] and r['experiment_signature']==protocol['experiment_signature']
  assert r['request_sha256']==wire[rid]['request_sha256'] and r['returned_model']==MODEL and not r['historical_response_reuse']
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
  for k in ('raw_response','predicted_choice','response_id','request_id','request_sha256','run_id','experiment_signature','returned_model','execution_id','usage','status','refusal','hit_token_limit','historical_response_reuse','query_image_count'):
   assert r[k]==origin[k],(rid,k)
  is_fresh=rid==origin['record_id']
  assert r['fresh_inference_executed_for_this_record']==is_fresh
  assert r['same_run_shared_inference_record_id']==(None if is_fresh else origin['record_id'])
  same_run_shared+=not is_fresh;formats+=r['status']!='ok';empty+=not r['raw_response'];truncated+=r['hit_token_limit']
 pairs=[]
 for qid,g in gold.items():
  p={'question_id':qid,'domain':g['task'],'gold_choice':g['gold_choice']}
  for name in NAMES:
   rid=qid+'::'+name;ans=choice(responses[rid]['raw_response']);ids=requests[rid]['top_memory_ids']
   p.update({name+'_answer':ans,name+'_correct':ans==g['gold_choice'],name+'_top5':ids,name+'_gold_hits':len(set(ids)&set(g['gold_ids']))})
  pairs.append(p)
 assert pairs==read(output/'paired_results.json') and aggregate(pairs)==summary['overall']
 for domain,values in summary['by_domain'].items():assert aggregate([p for p in pairs if p['domain']==domain])==values
 for name in NAMES:
  stats=summary['retrieval_stats'][name]
  assert stats['gold_hits_top5']==sum(p[name+'_gold_hits'] for p in pairs)
  assert stats['questions_with_any_gold']==sum(p[name+'_gold_hits']>0 for p in pairs)
  assert summary['method_annotations'][name]['5. 回答方式']['correct']==summary['overall']['conditions'][name]['correct']
 assert summary['qwen_reference_correct']==qwen_counts
 usage={k:sum((r.get('usage') or {}).get(k,0) or 0 for r in fresh_list) for k in ('prompt_tokens','completion_tokens','total_tokens')}
 assert usage==summary['usage_fresh_requests_only']
 assert validation['fresh_requests']==len(fresh) and validation['shared_records']==same_run_shared
 assert (validation['format_errors'],validation['empty_responses'],validation['token_limit_records'])==(formats,empty,truncated)
 assert validation['reasoning_tokens_reported']==reasoning['reported_token_total'] and validation['reasoning_usage_missing_requests']==reasoning['unreported']
 # Recompute the historical, unweighted comparison and its same-input instability.
 old_dir=Path(c['prior_unweighted_gpt54_reference']);old=indexed(lines(old_dir/'responses.jsonl'),'record_id')
 old_inputs=indexed(lines(old_dir/'inputs/requests.jsonl'),'record_id')
 for name,raw_name in zip(NAMES,('baseline_question_only','master_answer_query')):
  counts=Counter();different=[]
  for qid,g in gold.items():
   rid=qid+'::'+name;old_id=qid+'::'+raw_name
   before=choice(old[old_id]['raw_response'])==g['gold_choice'];after=choice(responses[rid]['raw_response'])==g['gold_choice']
   counts['prior_unweighted_correct']+=before;counts['wrong_to_right']+=not before and after;counts['right_to_wrong']+=before and not after
   if requests[rid]['payload_sha256']==old_inputs[old_id]['payload_sha256']:
    counts['identical_payload_questions']+=1
    if choice(old[old_id]['raw_response'])!=choice(responses[rid]['raw_response']):different.append(qid)
  expected=summary['prior_unweighted_gpt54_comparisons'][name]
  for k,v in counts.items():assert expected[k]==v
  assert expected['net_correct']==counts['wrong_to_right']-counts['right_to_wrong']
  assert expected['identical_payload_answer_disagreements']==different
 checks.update(fresh_requests=len(fresh),same_run_shared_records=same_run_shared,native_image_fresh_requests=image_fresh,
    format_errors=formats,empty_responses=empty,token_limit_records=truncated,reasoning_usage=dict(reasoning),
    api_attempts=len(attempts),transient_failed_attempts=sum(a['status']!='success' for a in attempts),
    refusals=sum(bool(r['refusal']) for r in fresh_list),returned_model_counts=dict(Counter(r['returned_model'] for r in fresh_list)),
    all_response_provenance_and_summary_counts_verified=True,all_requests_fresh_within_current_run=True,
    overall=summary['overall'],artifact_sha256={n:sha(output/n) for n in ('protocol.json','summary.json','responses.jsonl','fresh_responses.jsonl','api_attempts.jsonl','paired_results.csv')})
 (output/'independent_audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'stage':'independent_audit','passed':True,'fresh_requests':len(fresh),'overall':summary['overall']},ensure_ascii=False),flush=True)
 return checks


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=BASE);p.add_argument('--inputs-only',action='store_true')
 a=p.parse_args();audit(a.output,a.inputs_only)
