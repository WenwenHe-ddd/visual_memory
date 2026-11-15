"""Continue the unchanged seed42 run after an explicitly terminal provider rejection.

No prompts, model settings, context, labels or frozen implementations are changed.
The rejected record remains in the denominator and is stored as a missing answer,
never as a fabricated model response. Other requests retain the signed runner.
"""
from scripts.analysis.memeye_deepseek_flash_seed42_zscore import *
from scripts.analysis import memeye_deepseek_flash_seed42_zscore as original


def load_rejections(output,p,records):
    protocol=read(Path(output)/'reader/continuation_protocol.json')
    assert protocol['source_experiment_signature']==p['experiment_signature']
    checked_hashes(protocol['source_sha256'])
    rows=lines(Path(output)/'reader/terminal_rejections.jsonl')
    rejected=indexed(rows,'request_sha256');reqs=indexed(records,'record_id')
    attempts=lines(Path(output)/'reader/api_attempts.jsonl')
    for h,r in rejected.items():
        req=reqs[r['record_id']]
        assert r['status']=='provider_input_rejected' and r['seed']==42
        assert req['request_sha256']==h and r['provider_error']['code']=='data_inspection_failed'
        errors=[a for a in attempts if a['request_sha256']==h and a['status']=='api_error']
        assert len(errors)==3 and all(a['http_status']==400 for a in errors)
        assert r['retry_policy']=='Three identical seed42 requests rejected; no input rewriting.'
    return rejected


def run(output=OUTPUT, limit=0):
    output = Path(output); p, records = verify_reader(output); c = p['contract']; folder = output/'reader'
    config = transport.load_config(c['api_env_file']); assert config.public_dict() == c['api_config']
    representatives = {}
    for row in records:
        representatives.setdefault(row['request_sha256'], row)
    completed = lines(folder/'fresh_responses.jsonl') if (folder/'fresh_responses.jsonl').exists() else []
    fresh = indexed(completed, 'request_sha256')
    for h, ans in fresh.items():
        validate_response(ans, representatives[h], p, 'reader')
    rejected = load_rejections(output,p,records)
    assert not set(fresh)&set(rejected)
    pending = [r for h,r in representatives.items() if h not in fresh and h not in rejected]
    if limit:
        pending = pending[:limit]
    execution_id = str(uuid.uuid4())

    def execute(row):
        payload = {k: row[k] for k in ('system','user_text','query_images')}
        body = transport.build_request(payload, model=MODEL, max_output_tokens=16, image_detail='high')
        assert digest(body) == row['request_sha256']
        attempts = []
        for attempt in range(1, c['max_attempts']+1):
            entry = {'record_id': row['record_id'], 'request_sha256': row['request_sha256'],
                     'execution_id': execution_id, 'attempt': attempt}
            try:
                response = transport.chat_request(config, body, timeout=c['timeout_seconds'])
            except transport.APIError as error:
                attempts.append({**entry, 'status': 'api_error', 'kind': error.kind,
                                 'http_status': error.http_status, 'message': error.message})
                if not error.retryable or attempt == c['max_attempts']:
                    return None, attempts
                time.sleep(min(2**attempt, 8)); continue
            valid = (choice(response['raw_response']) is not None and response['finish_reason']=='stop'
                     and not response.get('refusal') and response['returned_model']==MODEL)
            answer = {**response, 'record_id': row['record_id'], 'question_id': row['question_id'], 'condition': row['condition'],
                      'payload_sha256': row['payload_sha256'], 'request_sha256': row['request_sha256'],
                      'run_id': p['run_id'], 'experiment_signature': p['experiment_signature'], 'execution_id': execution_id,
                      'attempts': attempt, 'retry_count': attempt-1, 'status': 'ok' if valid else 'invalid_response',
                      'predicted_choice': choice(response['raw_response']), 'historical_response_reuse': False}
            attempts.append({**entry, 'status': answer['status'], 'response_id': answer['response_id']})
            return answer, attempts

    with ThreadPoolExecutor(max_workers=c['workers']) as pool:
        for start in range(0, len(pending), c['workers']):
            failed = []
            futures = {pool.submit(execute, row): row for row in pending[start:start+c['workers']]}
            for future in as_completed(futures):
                row = futures[future]; answer, attempts = future.result()
                for a in attempts: append(folder/'api_attempts.jsonl', a)
                if answer is not None:
                    append(folder/'fresh_responses.jsonl', answer)
                    if answer['status']=='ok':
                        validate_response(answer, row, p, 'reader'); fresh[row['request_sha256']] = answer
                if answer is None or answer['status']!='ok': failed.append(row['record_id'])
            dump(output/'execution_status.json', {'status': 'stopped_on_error' if failed else 'running',
                 'fresh_completed': len(fresh), 'unique_requests': len(representatives), 'failed_records': failed})
            print(json.dumps({'stage': 'reader', 'fresh_completed': len(fresh), 'unique_requests': len(representatives), 'failed': failed}), flush=True)
            if failed: raise RuntimeError('API batch failed; stopped before any further batches.')
    if set(fresh)|set(rejected) == set(representatives):
        return analyze(output)
    return {'status': 'partial', 'fresh_completed': len(fresh)}


def analyze(output=OUTPUT):
    output = Path(output); p, records = verify_reader(output); c=p['contract'];folder=output/'reader'
    gold = indexed(lines(folder/'gold.jsonl'), 'id'); assert len(gold)==371
    fresh_rows=lines(folder/'fresh_responses.jsonl'); fresh=indexed(fresh_rows,'request_sha256')
    reps={}
    for row in records: reps.setdefault(row['request_sha256'],row)
    rejected=load_rejections(output,p,records)
    assert not set(fresh)&set(rejected)
    assert set(fresh)|set(rejected)==set(reps) and len(fresh)+len(rejected)==c['unique_requests']
    for h,ans in fresh.items():
        validate_response(ans,reps[h],p,'reader'); assert choice(ans['raw_response'])==ans['predicted_choice']
    previous=lines(ROOT/'results/memeye_deepseek_flash_zscore_lambda0p03_alpha1_20261025/reader/fresh_responses.jsonl')
    questions=indexed(lines(Path(c['memory_path'])/'inputs/source_questions.jsonl'),'id')
    assert not ({r['response_id'] for r in previous}&{r['response_id'] for r in fresh_rows})
    assert len({r['response_id'] for r in fresh_rows})==len(fresh_rows)
    attempt_rows=lines(folder/'api_attempts.jsonl')
    assert Counter(a['request_sha256'] for a in attempt_rows if a['status']=='ok')==dict.fromkeys(fresh,1)
    answers={}; logical=[]; per_question=[]
    for row in records:
        q=gold[row['question_id']]
        if row['request_sha256'] in rejected:
            rejected_entry=rejected[row['request_sha256']]
            r={**rejected_entry,'record_id':row['record_id'],'question_id':row['question_id'],'condition':row['condition'],
                'top_memory_ids':row['top_memory_ids'],'correct':False,'predicted_choice':None,'raw_response':None,
                'response_id':None,'payload_sha256':row['payload_sha256'],'run_id':p['run_id'],
                'experiment_signature':p['experiment_signature'],'fresh_inference_executed_for_this_record':False,
                'same_run_shared_inference_record_id':None,'historical_response_reuse':False,
                'attempted_request_without_model_answer':True}
            logical.append(r);answers[row['question_id'],row['condition']]=r
            continue
        ans=fresh[row['request_sha256']]; predicted=choice(ans['raw_response'])
        r={**ans,'record_id':row['record_id'],'question_id':row['question_id'],'condition':row['condition'],
           'top_memory_ids':row['top_memory_ids'],'correct':predicted==q['gold_choice'],
           'fresh_inference_executed_for_this_record':ans['record_id']==row['record_id'],
           'same_run_shared_inference_record_id':None if ans['record_id']==row['record_id'] else ans['record_id']}
        logical.append(r);answers[row['question_id'],row['condition']]=r
    write_lines(folder/'responses.jsonl',logical)
    for qid,q in gold.items():
        row={'question_id':qid,'domain':questions[qid]['task'],'gold_choice':q['gold_choice'],'gold_count':len(q['gold_ids'])}
        for name in CONDITIONS:
            ans=answers[qid,name];hits=len(set(ans['top_memory_ids'])&set(q['gold_ids']))
            row.update({name+'_answer':ans['predicted_choice'],name+'_correct':ans['correct'],name+'_gold_hits':hits,
                        name+'_top5':ans['top_memory_ids']})
        per_question.append(row)
    total=sum(r['gold_count'] for r in per_question);assert total==1041
    overall={}
    for name in CONDITIONS:
        correct=sum(r[name+'_correct'] for r in per_question);hits=sum(r[name+'_gold_hits'] for r in per_question)
        overall[name]={'correct':correct,'questions':371,'accuracy':correct/371,'gold_hits_top5':hits,'gold_references':total,
                       'recall_at_5_micro':hits/total,'recall_at_5_macro':sum(r[name+'_gold_hits']/r['gold_count'] for r in per_question)/371,
                       'questions_with_any_gold':sum(r[name+'_gold_hits']>0 for r in per_question)}
    wins=sum(not r[BASELINE+'_correct'] and r[WEIGHTED+'_correct'] for r in per_question)
    losses=sum(r[BASELINE+'_correct'] and not r[WEIGHTED+'_correct'] for r in per_question)
    method_annotations=c['method_annotations']
    for name in CONDITIONS:method_annotations[name]['5. 回答方式'].update(overall[name])
    summary={'status':'complete','model':MODEL,'caption_model':MODEL,'provider':c['provider'],'lambda':RIDGE,'alpha':ALPHA,
             'caption_source':c['memory_path'],'captions_regenerated':True,'caption_seed':42,'caption_images':437,'memory_leaves':438,'masters':read(output/'source_validation.json')['masters'],
             'overall':overall,'comparison':{'wrong_to_right':wins,'right_to_wrong':losses,'net_correct':wins-losses},
             'seed':42,'fresh_requests':len(fresh),'logical_records':742,
             'same_run_shared_records':742-len(reps),'failed_attempts':sum(r['status']!='ok' for r in attempt_rows),
             'format_errors':0,'refusals':0,'length_stops':0,'historical_answers_reused':False,'provider_rejected_unique_requests':len(rejected),
             'provider_rejected_records':[r['record_id'] for r in logical if r.get('status')=='provider_input_rejected'],
             'provider_rejections_count_as_incorrect':True,
             'method_annotations':method_annotations,'generation_parameters':{k:c[k] for k in ('temperature','seed','enable_thinking','reader_max_output_tokens')},
             'usage':{k:sum((r.get('usage') or {}).get(k,0) for r in fresh_rows) for k in ('prompt_tokens','completion_tokens','total_tokens')},
             'limitations':['API model is an undated alias; the provider does not establish a fixed weight revision.',
                            'Only the requested baseline and master+zscore conditions were freshly answered; historical unweighted master is not a matched new control.']}
    cap_responses=lines(Path(c['memory_path'])/'caption_generation/fresh_responses.jsonl')
    summary['caption_fresh_requests']=len(cap_responses)
    summary['total_fresh_requests']=len(cap_responses)+len(fresh)
    summary['caption_generation_parameters']={'model':MODEL,'seed':42,'temperature':0,'enable_thinking':False,'max_tokens':1024}
    summary['master_counts']=read(output/'source_validation.json')['master_counts']
    summary['by_domain']={}
    for domain in sorted({r['domain'] for r in per_question}):
        rows=[r for r in per_question if r['domain']==domain]; domain_total=sum(r['gold_count'] for r in rows)
        summary['by_domain'][domain]={name:{'correct':sum(r[name+'_correct'] for r in rows),
            'questions':len(rows),'accuracy':sum(r[name+'_correct'] for r in rows)/len(rows),
            'gold_hits_top5':sum(r[name+'_gold_hits'] for r in rows),'gold_references':domain_total,
            'recall_at_5_micro':sum(r[name+'_gold_hits'] for r in rows)/domain_total} for name in CONDITIONS}
    summary['continuation_protocol']=read(output/'reader/continuation_protocol.json')
    dump(output/'summary.json',summary);dump(output/'paired_results.json',per_question)
    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        table=[{'condition':name,**r} for name,r in overall.items()]
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    with (output/'paired_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_question[0]));writer.writeheader();writer.writerows(per_question)
    audit={'passed':True,'caption_and_graph_source_verified':True,'questions':371,'gold_references':1041,
           'records':742,'fresh_requests':len(fresh),'same_run_shared_records':742-len(reps),
           'full_request_hashes_reconstructed':742,'historical_answer_reuse':False,
           'successful_response_identity_and_disabled_thinking_verified':True,'metric_recomputed_from_raw_answers_and_retrieved_ids':True,
           'artifact_sha256':{n:sha(output/n) for n in ('reader/protocol.json','reader/requests.jsonl','reader/gold.jsonl',
               'reader/fresh_responses.jsonl','reader/responses.jsonl','reader/api_attempts.jsonl','summary.json','paired_results.json',
               'retrieval_protocol.json','retrieval_results.jsonl','source_validation.json','independent_input_audit.json')}}
    dump(output/'independent_audit.json',audit)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>DeepSeek Flash MemEye</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 16px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
          '<h1>DeepSeek V4.1 Flash：baseline 与 master + zscore</h1>',
          '<p>DashScope International，deepseek-v4.1-flash。重新生成全部 437 张去重图片的 caption，重新构建 438 条记忆及 master。全部 371 题重新回答；未加权 baseline 对比 master_answer_query + zscore（λ=0.03，α=1）。</p>',
          '<table><tr><th>方法</th><th>QA</th><th>Recall@5 micro</th><th>Recall@5 macro</th></tr>']
    for name,result in overall.items():
        body.append(f'<tr><td>{name}</td><td>{result["correct"]}/371 ({result["accuracy"]:.4%})</td><td>{result["gold_hits_top5"]}/1041 ({result["recall_at_5_micro"]:.4%})</td><td>{result["recall_at_5_macro"]:.4%}</td></tr>')
    body+=['</table>',f'<p>本轮 {len(fresh)} 次新推理，{742-len(reps)} 条同请求共享。caption 与回答请求均显式发送 seed=42；1 条请求被平台输入内容检查拒绝，计为错误并保留在 371 题分母中。本轮没有复用历史 caption 或回答。</p>',
           '<p>temperature=0，enable_thinking=false，回答预算 16 tokens；caption 和回答均固定 seed=42。接口模型为无日期别名，实际权重版本未由服务返回确认。</p>',
           '<p><a href="summary.json">统计和五项标注</a> · <a href="paired_results.csv">逐题 CSV</a> · <a href="external_audit.json">独立核验</a> · <a href="source_validation.json">Caption 来源核验</a></p>',
           '<pre>'+html.escape(json.dumps(method_annotations,ensure_ascii=False,indent=2))+'</pre></html>']
    (output/'report.html').write_text('\n'.join(body))
    dump(output/'execution_status.json',{'status':'complete_with_provider_rejection','fresh_completed':len(fresh),'provider_rejected':len(rejected),'unique_requests':len(reps),'logical_records':742})
    print(json.dumps({'status':'complete','overall':overall,'fresh_requests':len(fresh)}),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--stage',choices=['run','analyze'],default='run')
    parser.add_argument('--limit',type=int,default=0)
    a=parser.parse_args()
    run(a.output,a.limit) if a.stage=='run' else analyze(a.output)
