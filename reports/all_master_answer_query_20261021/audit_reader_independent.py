from pathlib import Path
import json,hashlib,datetime,re,math,sys,os
ROOT=Path('/storage/hewenwen/VisualMemory');OUT=ROOT/'reports/all_master_answer_query_20261021';RD=OUT/'reader'
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
assert (RD/'summary.json').exists(),'Only audit completed results.'
protocol=read(RD/'protocol.json');rootprotocol=read(OUT/'protocol.json');snapshot=read(RD/'independent_audit_start_snapshot.json')
assert sha(OUT/'protocol.json')==snapshot['root_protocol_sha256']
assert sha(RD/'protocol.json')==snapshot['reader_protocol_sha256']
sig=protocol['experiment_signature'];unsigned={k:v for k,v in protocol.items()if k!='experiment_signature'};assert digest(unsigned)==sig
for p,h in rootprotocol['source_sha256'].items():assert sha(p)==h,p
for p,h in {**protocol['source_sha256'],**protocol['model_source_sha256']}.items():assert sha(p)==h,p
for w in protocol['weight_files']:
 st=Path(w['path']).stat();assert st.st_size==w['bytes'] and st.st_mtime_ns==w['mtime_ns']
images={x['path']:x['sha256']for x in protocol['query_images']};assert len(images)==6
for p,h in images.items():assert sha(p)==h,p
manifest=lines(RD/'qa_manifest.jsonl');responses=lines(RD/'responses.jsonl')
M={r['record_id']:r for r in manifest};A={r['record_id']:r for r in responses}
assert len(M)==len(manifest)==len(A)==len(responses)==742 and set(M)==set(A)
originals={r['id']:r for r in lines(ROOT/'data/independent_visual_inputs/qa_manifest.jsonl')}
retrieval={r['question_id']:r for r in lines(OUT/'retrieval_results.jsonl')}
assert set(originals)==set(retrieval) and len(originals)==371
texts={}
for domain in (ROOT/'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain').iterdir():
 p=domain/'inputs.json'
 if not p.exists():continue
 for leaf in read(p)['leaves']:texts[leaf['memory_id']]=leaf['retrieval_text']
expected_record_order=[]
for i,qid in enumerate(originals):
 pair=[qid+'::baseline_question_only',qid+'::master_answer_query'];expected_record_order.extend(pair if i%2==0 else pair[::-1])
assert [m['record_id']for m in manifest]==expected_record_order
environments={}
for a in responses:
 eid=a['execution_id']
 if eid not in environments:
  env=read(RD/('generation_environment_'+eid+'.json'));assert env['execution_id']==eid and env['run_id']==protocol['experiment_run_id']
  assert env['dtype']=='torch.bfloat16';environments[eid]=env
 assert a['input_tokens']+16<=environments[eid]['context_limit']
settings=protocol['reader_settings'];assert settings['seed']==42 and settings['do_sample'] is False and settings['max_new_tokens']==16
assert settings['batch_size_text']==settings['batch_size_query_images']==1
assert not protocol['retrieval_expansion_sent_to_reader'] and not protocol['leaf_distance_used'] and not protocol['historical_response_reuse']
# Tokenizer/template loading only: no model weights and no image or embedding encoding.
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)];os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoProcessor
processor=AutoProcessor.from_pretrained(str(ROOT/'models/Qwen3-VL-8B-Instruct'),local_files_only=True)
eos=processor.tokenizer.eos_token_id
payloads={};positions={r['record_id']:i for i,r in enumerate(responses)};image_records=0
for m in manifest:
 q=originals[m['id']];r=retrieval[m['id']];condition=next(c for c in r['conditions']if c['name']==m['condition'])
 assert m['record_id']==m['id']+'::'+m['condition']
 for field in ('question','options','gold_choice','gold_ids','query_image_paths','task'):assert m[field]==q[field]
 assert m['top_memory_ids']==condition['top5_memory_ids'] and m['top_query_scores']==condition['top5_scores']
 expected=('Retrieved dialogue and image descriptions (ranked by relevance):\n\n'+'\n\n'.join(f'Memory {i}: {texts[mid]}'for i,mid in enumerate(m['top_memory_ids'],1))+'\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(f'{k}. {q["options"][k]}'for k in 'ABCD')+'\n\nAnswer with one letter only.')
 assert m['reader_text']==expected
 imgs=[{'path':p,'sha256':images[p]}for p in q['query_image_paths']];assert m['query_images']==imgs
 payload={'system':settings['reader_system_prompt'],'user_text':expected,'query_images':imgs}
 assert digest(payload)==m['payload_sha256'];payloads[m['record_id']]=m['payload_sha256'];image_records+=bool(imgs)
 a=A[m['record_id']];assert a['question_id']==m['id'] and a['condition']==m['condition']
 assert a['payload_sha256']==m['payload_sha256'] and a['experiment_signature']==sig and a['run_id']==protocol['experiment_run_id']
 assert a['gold_choice']==q['gold_choice'] and not a['historical_response_reuse']
 assert a['batch_size']==1 and a['query_image_count']==len(q['query_image_paths'])
 rendered=processor.apply_chat_template([{'role':'system','content':settings['reader_system_prompt']},{'role':'user','content':[{'type':'image'}for _ in imgs]+[{'type':'text','text':expected}]}],tokenize=False,add_generation_prompt=True)
 assert hashlib.sha256(rendered.encode()).hexdigest()==a['rendered_prompt_sha256']
 raw=processor.tokenizer.decode(a['generated_token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False).strip();assert raw==a['raw_response']
 match=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',raw,re.I);choice=match.group(1).upper()if match else None
 assert choice==a['predicted_choice'] and a['correct']==(choice==q['gold_choice'])
 assert a['status']==('ok'if match else 'format_error')
 tokens=a['generated_token_ids'];emitted=next((i+1 for i,t in enumerate(tokens)if t==eos),len(tokens))
 assert a['generated_tokens']==emitted and a['hit_token_limit']==(emitted>=16 and eos not in tokens)
 if a['fresh_inference_executed_for_this_record']:
  assert a['same_run_shared_inference_record_id'] is None
 else:
  source=A[a['same_run_shared_inference_record_id']];assert source['fresh_inference_executed_for_this_record']
  assert positions[source['record_id']]<positions[a['record_id']]
  for k in ('payload_sha256','rendered_prompt_sha256','raw_response','predicted_choice','generated_token_ids','execution_id','run_id'):assert a[k]==source[k]
  assert a['elapsed_seconds']==0
assert image_records==16
fresh=[a for a in responses if a['fresh_inference_executed_for_this_record']]
assert len(fresh)==len({a['payload_sha256']for a in fresh})==len(set(payloads.values()))==719
assert len(responses)-len(fresh)==23
pairs=[]
for qid,q in originals.items():
 a,b=[A[qid+'::'+c]for c in ['baseline_question_only','master_answer_query']]
 ma,mb=[M[x['record_id']]for x in (a,b)]
 ordered=ma['top_memory_ids']==mb['top_memory_ids'];sameset=set(ma['top_memory_ids'])==set(mb['top_memory_ids'])
 samepayload=ma['payload_sha256']==mb['payload_sha256'];assert samepayload==ordered
 if samepayload:assert a['predicted_choice']==b['predicted_choice']
 pairs.append({'question_id':qid,'domain':q['task'],'query_has_image':bool(q['query_image_paths']),'baseline_answer':a['predicted_choice'],'expanded_answer':b['predicted_choice'],'gold_choice':q['gold_choice'],'baseline_correct':a['predicted_choice']==q['gold_choice'],'expanded_correct':b['predicted_choice']==q['gold_choice'],'same_reader_payload':samepayload,'same_top5_order':ordered,'same_top5_set':sameset,'group':'same_ordered_top5'if ordered else 'same_set_reordered'if sameset else 'changed_top5_set'})
def aggregate(rows):
 n=len(rows);a=sum(r['baseline_correct']for r in rows);b=sum(r['expanded_correct']for r in rows)
 w=sum(not r['baseline_correct']and r['expanded_correct']for r in rows);l=sum(r['baseline_correct']and not r['expanded_correct']for r in rows);d=w+l
 return {'questions':n,'baseline_correct':a,'expanded_correct':b,'baseline_accuracy':a/n if n else None,'expanded_accuracy':b/n if n else None,'delta_percentage_points':100*(b-a)/n if n else None,'wrong_to_correct':w,'correct_to_wrong':l,'answer_changed':sum(r['baseline_answer']!=r['expanded_answer']for r in rows),'same_reader_payload':sum(r['same_reader_payload']for r in rows),'same_top5_order':sum(r['same_top5_order']for r in rows),'same_top5_set':sum(r['same_top5_set']for r in rows),'mcnemar_exact_two_sided_p':min(1.,2*sum(math.comb(d,k)for k in range(min(w,l)+1))/2**d)if d else 1.}
summary=read(RD/'summary.json');overall=aggregate(pairs);bydomain={d:aggregate([r for r in pairs if r['domain']==d])for d in sorted({r['domain']for r in pairs})}
assert summary['overall']==overall and summary['by_domain']==bydomain
for name,v in [('text_only',False),('with_query_image',True)]:assert summary['by_query_modality'][name]==aggregate([r for r in pairs if r['query_has_image']==v])
assert summary['logical_conditions']==742 and summary['fresh_model_inferences']==719 and summary['same_run_shared_records']==23
assert summary['format_errors']==sum(r['status']!='ok'for r in responses)
assert summary['token_limit_hits']==sum(r['hit_token_limit']for r in responses)
for name in ['baseline_question_only','master_answer_query']:
 rr=[a for a in responses if a['condition']==name]
 assert summary['conditions'][name]=={'questions':371,'correct':sum(a['correct']for a in rr),'format_errors':sum(a['status']!='ok'for a in rr),'token_limit_hits':sum(a['hit_token_limit']for a in rr)}
groups={g:aggregate([r for r in pairs if r['group']==g])for g in ['same_ordered_top5','same_set_reordered','changed_top5_set']}
effect={'definitions':{'same_ordered_top5':'Both membership and order unchanged; identical reader payload shared within this run.','same_set_reordered':'Exactly the same five memories, changed order.','changed_top5_set':'At least one of five memories changed; order can also change.'},'groups':groups,'by_domain':{d:{g:aggregate([r for r in pairs if r['domain']==d and r['group']==g])for g in groups}for d in bydomain},'changed_correctness':[r for r in pairs if r['baseline_correct']!=r['expanded_correct']],'question_group_assignments':[{k:r[k]for k in ('question_id','group')}for r in pairs],'limitations':'Post hoc descriptive breakdown by retrieval change; not causal attribution or a prespecified model-selection criterion.','responses_sha256':sha(RD/'responses.jsonl')}
dump(OUT/'effect_breakdown.json',effect)
report={'status':'passed','auditor':'all_master_answer_protocol','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'questions':371,'records':742,'unique_record_ids':742,'fresh_inferences':719,'same_run_shared_records':23,'historical_response_reuse':False,'original_questions_options_labels_verified':True,'original_dialogue_fullcaption_evidence_verified':True,'query_image_questions':8,'image_condition_records':16,'distinct_original_image_files_hashed':6,'all_reader_payload_and_rendered_prompt_hashes_recomputed':True,'generated_token_decoding_and_choice_parsing_recomputed':True,'root_source_hashes_verified':len(rootprotocol['source_sha256']),'reader_source_and_model_config_hashes_verified':len(protocol['source_sha256'])+len(protocol['model_source_sha256']),'model_weight_files_metadata_unchanged':True,'model_weights_sha_recomputed_by_generation_prepare_not_duplicated_in_this_audit':True,'root_and_reader_protocols_unchanged_from_audit_snapshot':True,'experiment_signature_recomputed':True,'overall':overall,'by_domain':bydomain,'format_errors':summary['format_errors'],'token_limit_hits':summary['token_limit_hits'],'responses_sha256':sha(RD/'responses.jsonl'),'reader_protocol_sha256':sha(RD/'protocol.json'),'root_protocol_sha256':sha(OUT/'protocol.json'),'effect_breakdown_path':str(OUT/'effect_breakdown.json')}
dump(RD/'validation_independent.json',report)
print(json.dumps({'status':'passed','overall':overall,'groups':groups,'format_errors':report['format_errors'],'token_limit_hits':report['token_limit_hits']}))
