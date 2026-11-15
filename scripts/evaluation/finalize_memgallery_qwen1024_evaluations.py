"""Offline recovery of two evaluated Gallery jobs; no generation or API calls."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation import required_grid_suite as suite
from scripts.evaluation import memgallery_multi_method_judge as judge
from scripts.evaluation.mem_gallery_master_reader import score_answer

METHODS=('gen_agents','memoryos')
REPORT=c.ROOT/'reports/memgallery_qwen_seed1024_evaluation_recovery_20261115'

def index(values):
 result={r['id']:r for r in values}
 if len(values)!=1711 or len(result)!=1711:raise ValueError('Expected exactly 1711 unique questions')
 return result

def verify_hashes(mapping):
 def check(item):
  p,h=item
  if c.sha(p)!=h:raise ValueError('Original artifact changed: '+p)
 with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,mapping.items()))
 return len(mapping)

def recover(method):
 job=c.read(c.RUN/'jobs'/f'memgallery__qwen__s1024__{method}.json')
 root=Path(job['run_root'])/'text_methods'/method;out=Path(job['method_root']);jr=out/'judge'
 with c.lock(out/'execution.lock',False):
  audit=c.read(root/'independent_audit.json');manifest=c.read(root/'completion_manifest.json');summary=c.read(root/'summary.json')
  assert audit['status']=='passed' and audit['seed']==1024 and audit['questions']==1711
  assert manifest['status']=='complete' and manifest['seed']==1024 and manifest['questions']==1711
  for field,p in [('responses_sha256',root/'reader/responses.jsonl'),('protocol_sha256',root/'protocol.json')]:assert c.sha(p)==manifest[field]
  # evaluation reran audit() after export(), changing audited_at and leaving
  # the native export's old audit hash stale. Preserve both and independently
  # rerun the full native audit into this recovery directory.
  fresh_path=REPORT/(method+'_independent_audit.json')
  # Preserve the stale export references explicitly. The saved passed audit
  # pins native construction, retrieval, actual generation events and raw batches.
  # Revalidate those pinned bytes in parallel rather than re-importing models.
  stale_path=REPORT/(method+'_stale_audit_reference.json')
  c.dump(stale_path,dict(cause='native export followed by a second audit which updates audited_at',
      export_expected_audit_sha256=manifest['audit_sha256'],current_audit_sha256=c.sha(root/'independent_audit.json'),
      original_files_preserved=True,old_export_audit_reference_valid=False))
  print({'method':method,'stage':'native_hash_audit'},flush=True)
  count=verify_hashes(audit['artifact_sha256'])
  events=c.rows(root/'runtime_generation_calls.jsonl')
  assert len(events)==audit['actual_seed_generation_events']
  for event in events:
   assert event['experiment_signature']==audit['experiment_signature']
   assert event['actual_torch_seed']==1024 and event['observed_model_generate']
   assert event['actual_categorical_seed']==(1024 if event['temperature']>0 else None)
   assert audit['artifact_sha256'][event['raw_batch_path']]==event['raw_batch_sha256']
  c.dump(fresh_path,dict(status='passed',seed=1024,questions=1711,native_audit_sha256=c.sha(root/'independent_audit.json'),
      immutable_native_artifacts_verified=count,actual_generation_events_verified=len(events),
      audit_policy='Revalidate every native-audit artifact hash plus actual seed events, original reader seal, per-question metrics and exact Judge wires/outputs',
      api_calls=0,verified_at=c.now()))
  verify_hashes({p:h for p,h in summary['source_sha256'].items() if Path(p)!=root/'independent_audit.json'})
  assert summary['source_sha256'][str(root/'independent_audit.json')]==manifest['audit_sha256']
  protocol=c.read(root/'protocol.json');verify_hashes(protocol['input_sha256'])
  assert protocol['seed']==1024
  responses=index(c.rows(root/'reader/responses.jsonl'))
  questions=index(c.rows(c.ROOT/'results/mem_gallery_master_answer_20261021/inputs/questions.jsonl'))
  gold=index(c.rows(c.ROOT/'results/mem_gallery_master_answer_20261021/inputs/gold.jsonl'))
  scores=index(c.rows(jr/'per_question.jsonl'));requests=index(c.rows(jr/'requests.jsonl'))
  assert set(responses)==set(questions)==set(gold)==set(scores)==set(requests)
  jp=c.read(jr/'protocol.json');verify_hashes(jp['source_sha256'])
  for k,v in judge.judge_settings().items():assert jp[k]==v,(k,'judge policy changed')
  assert jp['answer_generation_seed']==1024 and jp['method']==method
  js=c.read(jr/'summary.json');assert js['status']=='complete' and js['questions']==1711
  verify_hashes(js['artifacts_sha256'])
  metrics=[];values=[];grades=[];unique=set()
  for qid,q in questions.items():
   r=responses[qid];s=scores[qid];req=requests[qid]
   assert r['experiment_signature']==protocol['experiment_signature']
   item=dict(id=qid,task=q['task'],point=q['point'],question=q['question'],ground_truth=gold[qid]['answer'],model_output=r['raw_response'])
   wire=judge.build_request(item,judge.PROMPT.read_text());key=judge.transport.request_sha256(wire)
   assert req==dict(item,request=wire,request_sha256=key)
   for k,v in item.items():assert s[k]==v
   assert s['request']==wire and s['request_sha256']==key
   saved=c.read(jr/'responses'/(key+'.json'));judge.validate_score(saved,key)
   for k in ('score','reasoning','response_id','raw_response'):assert s[k]==saved[k]
   grades.append(saved['score']);unique.add(key)
   metrics.append(score_answer(r['raw_response'],gold[qid]['answer']))
   values.append(dict(id=qid,model_output=r['raw_response'],task=q['task'],point=q['point'],question=q['question']))
  numbers={k:sum(r[k] for r in metrics)/1711 for k in ('em','f1','bleu_1')};numbers['em_correct']=sum(int(r['em']) for r in metrics)
  for k in ('em','f1','bleu_1'):assert math.isclose(numbers[k],summary[k],abs_tol=1e-12)
  assert numbers['em_correct']==summary['correct']
  assert math.isclose(sum(grades)/1711,js['llm_judge'],abs_tol=1e-12)
  evidence=[root/n for n in ('protocol.json','reader/responses.jsonl','reader/manifest.jsonl','independent_audit.json','completion_manifest.json','summary.json')]
  evidence += [jr/n for n in ('protocol.json','requests.jsonl','per_question.jsonl','summary.json')]+[Path(__file__),fresh_path,REPORT/(method+'_stale_audit_reference.json')]
  # caption_seed is already supplied by complete(); do not pass it twice.
  proof=c.complete(job,None,values,jr,evidence,dict(**numbers,caption_source_run_seed=42,original_native_snapshot=True,
    finalization_recovery='Offline validation of existing native answers and exact saved Judge requests/responses; no API calls'))
  suite.completed(job)
  result=dict(method=method,status='complete',questions=1711,**numbers,llm_judge=sum(grades)/1711,
    unique_judge_requests=len(unique),native_artifact_hashes_verified=count,score_distribution=dict(Counter(map(str,grades))),
    preserved_token_limit_answers=audit['final_reader_token_limit_hits'],completion=str(out/'completion.json'),completion_sha256=c.sha(out/'completion.json'))
  print(result,flush=True);return job,result

if __name__=='__main__':
 REPORT.mkdir(parents=True,exist_ok=True)
 with c.lock(c.RUN/'scheduler/suite.lock',False):
  state=c.read(c.RUN/'progress.json')
  if any(v['status']=='running' for v in state['jobs'].values()):raise ValueError('Recovery requires the suite to remain paused')
  backup=REPORT/'progress_before_recovery.json'
  if not backup.exists():c.dump(backup,state)
  recovered=[recover(m) for m in METHODS]
  for job,result in recovered:
   old=dict(state['jobs'][job['id']]);state['jobs'][job['id']].update(status='complete',recovered_at=c.now(),recovery_report=str(REPORT/'completion.json'),previous_failure=old)
   suite.event('method_finalized_offline',id=job['id'],api_calls=0,completion_sha256=result['completion_sha256'])
  # Preserve the paused suite status and all other jobs.
  suite.publish(c.read(c.RUN/'plan.json'),state)
  c.dump(REPORT/'completion.json',dict(status='complete',methods=[r for _,r in recovered],questions_per_method=1711,api_calls=0,
    completed_at=c.now(),suite_resumed=False,source_sha256={str(Path(__file__).resolve()):c.sha(__file__)},progress_sha256=c.sha(c.RUN/'progress.json')))
