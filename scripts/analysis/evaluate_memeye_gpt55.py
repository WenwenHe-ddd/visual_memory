"""GPT-5.5 readers on fresh GPT-5.5 captions: both retrievers with/without z-score."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import html
import json
import math
from pathlib import Path
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
from scripts.evaluation import memeye_gpt5_reader as api
from scripts.evaluation import memeye_gpt55_captions as captions
from scripts.evaluation import openai_gpt55_transport as transport
read,lines,sha,digest,dump=api.read,api.lines,api.sha,api.digest,api.dump
indexed,append,write_lines=api.indexed,api.append,api.write_lines
parse_choice,logical_responses,verify=api.parse_choice,api.logical_responses,api.verify
BASE=captions.DEFAULT_OUTPUT
DEFAULT_OUTPUT=BASE/'reader'
RAW=('baseline_question_only','master_answer_query')
CONDITIONS=(RAW[0],RAW[0]+'_zscore',RAW[1],RAW[1]+'_zscore')
REFERENCE_RAW=ROOT/'results/memeye_gpt54_alias_seed42_baselines_20261024'
REFERENCE_WEIGHTED=ROOT/'results/memeye_gpt54_alias_seed42_zscore_lambda0p01_alpha1_20261024'


def request_body(protocol,row):
    c=protocol['contract']
    assert c['model']=='gpt-5.5' and c['seed']==42
    payload={'system':c['reader_system_prompt'],'user_text':row['reader_text'],'query_images':row['query_images']}
    assert digest(payload)==row['payload_sha256']
    return transport.build_request(payload,seed=42,max_output_tokens=16,image_detail='high')


def annotations(source,model):
    result={}
    for name in CONDITIONS:
        master=name.startswith('master_');weighted=name.endswith('_zscore')
        result[name]={
            '1. 候选记忆结构':{'memory_path':str(source),'offline_method':'caption_visual_master' if master else 'natural',
                'caption_model':model,'caption_source':'本轮全部重新生成','memory_leaves':438,
                'text':'原始对话＋本轮 GPT-5.5 完整 caption','weighting_group':'每个新 master 下的全部叶子' if master else '同 domain 全部记忆'},
            '2. 查询内容':'原问题＋本轮 GPT-5.5 题图 caption'+('＋该 master 视觉匹配选中的完整候选答案陈述' if master else '；不添加选项陈述'),
            '3. 检索方式':{'retrieval':'multi_path' if master else 'full_scan','online_weighting':'z_score' if weighted else 'none',
                'top_k':5,'lambda':.01 if weighted else None,'alpha':1. if weighted else None,
                'score':'raw + alpha * std_pop(raw in group) * tanh(white_z)' if weighted else 'raw cosine similarity',
                'merge':'同 domain 全部叶子按最终分数取 Top-5，同分保留顺序'},
            '4. 编码方式':{'BGE':'BAAI/bge-small-en-v1.5, 384 dimensions, L2 normalized',
                'query_prefix':'Represent this sentence for searching relevant passages: ',
                'CLIP':'已有视觉编码器及外部基底不变；新 caption 重建 master 与视觉聚合量',
                'online_vectors':'使用后丢弃，不写回记忆库','length_policy':'原始 BGE 512-token 上限，逐条记录编码截断，记忆与回答保留完整文本'},
            '5. 回答方式':{'requested_model':'gpt-5.5','expected_returned_model':model,'seed':42,'temperature':0,
                'reasoning_effort':'none','max_completion_tokens':16,'image_detail':'high',
                'input':'Top-5 原始对话＋完整 caption、原问题、选项、原始题图；检索候选陈述不进入回答上下文',
                'same_run_identical_requests':'四种条件中完整请求相同则共享本轮新回答','historical_answer_reuse':False}}
    return result


def prepare(source=BASE,workers=3):
    from scripts.analysis.retrieve_memeye_gpt55 import verify_source
    source=Path(source).resolve();output=source/'reader'
    if (output/'protocol.json').exists():return verify(output)
    root=captions.verify(source);proof=verify_source(source)
    weighted=read(source/'zscore/retrieval_protocol.json')
    for name,value in {**weighted['source_sha256'],**weighted['output_sha256']}.items():assert sha(name)==value,name
    assert (weighted['lambda'],weighted['alpha'])==(.01,1.) and weighted['unweighted_full_rankings_equal']==742
    questions=indexed(lines(source/'inputs/source_questions.jsonl'),'id')
    evaluation=indexed(lines(source/'inputs/evaluation.jsonl'),'id')
    memories={m['id']:m['retrieval_text'] for m in lines(source/'offline/memories.jsonl')}
    raw=lines(source/'retrieval_results.jsonl');weighted_rows=indexed(lines(source/'zscore/retrieval_results.jsonl'),'question_id')
    system=read(source/'inputs/protocol.json')['reader_system_prompt']
    records=[];gold=[]
    for i,row in enumerate(raw):
        q=questions[row['question_id']];g=evaluation[q['id']]
        gold.append({**{k:g[k] for k in ('id','gold_choice','gold_ids')},'task':q['task']})
        conditions={c['name']:c for c in row['conditions']+weighted_rows[q['id']]['conditions']}
        assert set(conditions)==set(CONDITIONS)
        batch=[]
        for name in CONDITIONS:
            ids=conditions[name]['top5_memory_ids'];assert len(set(ids))==len(ids)==5
            text='Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
            text+='\n\n'.join(f'Memory {j}: {memories[mid]}' for j,mid in enumerate(ids,1))
            text+='\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(k+'. '+q['options'][k] for k in 'ABCD')+'\n\nAnswer with one letter only.'
            payload={'system':system,'user_text':text,'query_images':q['query_images']}
            batch.append({'record_id':q['id']+'::'+name,'id':q['id'],'task':q['task'],'condition':name,
                'reader_text':text,'query_images':q['query_images'],'query_image_paths':[im['path'] for im in q['query_images']],
                'top_memory_ids':ids,'payload_sha256':digest(payload)})
        offset=i%4;records.extend(batch[offset:]+batch[:offset])
    output.mkdir(exist_ok=True);(output/'inputs').mkdir(exist_ok=True)
    write_lines(output/'inputs/requests.jsonl',records);write_lines(output/'inputs/gold.jsonl',gold)
    config=transport.load_config(root['contract']['api_env_file'])
    sources={**proof['source_sha256'],**weighted['source_sha256'],**weighted['output_sha256']}
    paths=[Path(__file__),ROOT/'scripts/analysis/audit_memeye_gpt55.py',ROOT/'scripts/tests/test_memeye_gpt55.py',
        source/'zscore/retrieval_protocol.json',output/'inputs/requests.jsonl',output/'inputs/gold.jsonl',
        ROOT/'scripts/evaluation/openai_gpt55_transport.py']
    for reference in (REFERENCE_RAW,REFERENCE_WEIGHTED):
        verify(reference)
        paths.extend(reference/name for name in ('protocol.json','summary.json','responses.jsonl','independent_audit.json'))
    paths.extend(Path(im['path']) for q in questions.values() for im in q['query_images'])
    sources.update({str(p):sha(p) for p in paths})
    contract={'model':'gpt-5.5','expected_returned_model':root['contract']['model'],'caption_model':root['contract']['model'],
        'requested_model':'gpt-5.5','seed':42,'caption_seed':None,'temperature':0,'reasoning_effort':'none','max_completion_tokens':16,
        'image_detail':'high','base_url':config.base_url,'api_config':config.public_dict(),'api_env_file':root['contract']['api_env_file'],
        'questions':371,'logical_records':1484,'unique_payloads':len({r['payload_sha256'] for r in records}),
        'conditions':list(CONDITIONS),'memory_path':str(source),'source_run_id':root['run_id'],'lambda':.01,'alpha':1.,'top_k':5,
        'reader_system_prompt':system,'workers':workers,'timeout_seconds':60,'max_attempts':3,
        'source_sha256':sources,'method_annotations':annotations(source,root['contract']['model']),
        'historical_api_answer_reuse':False,'gold_in_api_payload':False,
        'within_run_sharing':'One fresh answer per complete request across all four conditions.',
        'interpretation':'Fresh GPT-5.5 captions and answers. Compare weighting within this same rebuilt memory. GPT-5.4 comparison changes both caption generation and answering.'}
    wire=[{'record_id':r['record_id'],'payload_sha256':r['payload_sha256'],
           'request_sha256':digest(request_body({'contract':contract},r)),
           'query_image_count':len(r['query_images']),'requested_model':'gpt-5.5'} for r in records]
    write_lines(output/'api_request_manifest.jsonl',wire);contract['request_manifest_sha256']=digest(wire)
    p={'run_id':str(uuid.uuid4()),'experiment_signature':digest(contract),'contract':contract,
       'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(output/'protocol.json',p);verify(output)
    print(json.dumps({'stage':'reader_prepared','records':len(records),'unique_requests':contract['unique_payloads']}),flush=True)
    return p


def compare(pairs,before,after):
    wins=sum(not r[before+'_correct'] and r[after+'_correct'] for r in pairs)
    losses=sum(r[before+'_correct'] and not r[after+'_correct'] for r in pairs)
    n=wins+losses
    return {'reference_correct':sum(r[before+'_correct'] for r in pairs),'current_correct':sum(r[after+'_correct'] for r in pairs),
        'wrong_to_right':wins,'right_to_wrong':losses,'net_correct':wins-losses,
        'answer_changes':sum(r[before+'_answer']!=r[after+'_answer'] for r in pairs),
        'mcnemar_exact_p':min(1.,2*sum(math.comb(n,i) for i in range(min(wins,losses)+1))/2**n) if n else 1.}


def analyze(source=BASE):
    source=Path(source).resolve();output=source/'reader';p=verify(output);c=p['contract']
    validation=read(output/'validation_output.json');assert validation['passed']
    for n in ('responses','fresh_responses'):assert sha(output/(n+'.jsonl'))==validation[n+'_sha256']
    requests=indexed(lines(output/'inputs/requests.jsonl'),'record_id');responses=indexed(lines(output/'responses.jsonl'),'record_id')
    gold=indexed(lines(output/'inputs/gold.jsonl'),'id');assert set(requests)==set(responses) and len(responses)==1484
    prior={**indexed(lines(REFERENCE_RAW/'responses.jsonl'),'record_id'),**indexed(lines(REFERENCE_WEIGHTED/'responses.jsonl'),'record_id')}
    pairs=[]
    for qid,g in gold.items():
        row={'question_id':qid,'domain':g['task'],'gold_choice':g['gold_choice']}
        for name in CONDITIONS:
            rid=qid+'::'+name;pred=parse_choice(responses[rid]['raw_response']);old=parse_choice(prior[rid]['raw_response'])
            ids=requests[rid]['top_memory_ids']
            row.update({name+'_answer':pred,name+'_correct':pred==g['gold_choice'],name+'_top5':ids,
                name+'_gold_hits':len(set(ids)&set(g['gold_ids'])),name+'_gpt54_answer':old,name+'_gpt54_correct':old==g['gold_choice']})
        for raw in RAW:
            same=requests[qid+'::'+raw]['payload_sha256']==requests[qid+'::'+raw+'_zscore']['payload_sha256']
            row[raw+'_identical_payload']=same
            if same:assert row[raw+'_answer']==row[raw+'_zscore_answer']
        pairs.append(row)
    overall={name:{'correct':sum(r[name+'_correct'] for r in pairs),'questions':371,
        'accuracy':sum(r[name+'_correct'] for r in pairs)/371,
        'gold_hits_top5':sum(r[name+'_gold_hits'] for r in pairs),'gold_references':1041,
        'questions_with_any_gold':sum(r[name+'_gold_hits']>0 for r in pairs)} for name in CONDITIONS}
    within={raw:compare(pairs,raw,raw+'_zscore') for raw in RAW}
    for raw in RAW:within[raw]['identical_payload_questions']=sum(r[raw+'_identical_payload'] for r in pairs)
    cross={name:compare(pairs,name+'_gpt54',name) for name in CONDITIONS}
    by_domain={d:{name:{'correct':sum(r[name+'_correct'] for r in pairs if r['domain']==d),
        'questions':sum(r['domain']==d for r in pairs)} for name in CONDITIONS} for d in sorted({r['domain'] for r in pairs})}
    methods=c['method_annotations']
    for name in CONDITIONS:methods[name]['5. 回答方式'].update(correct=overall[name]['correct'],returned_models=validation['returned_models'])
    caps=lines(source/'caption_generation/fresh_responses.jsonl');fresh=lines(output/'fresh_responses.jsonl')
    stages={}
    for label,rs in (('caption_generation',caps),('reader',fresh)):
        stages[label]={'fresh_requests':len(rs),'returned_models':dict(Counter(r['returned_model'] for r in rs)),
            'usage':{k:sum((r.get('usage') or {}).get(k,0) or 0 for r in rs) for k in ('prompt_tokens','completion_tokens','total_tokens')}}
    summary={'status':'complete','requested_model':'gpt-5.5','returned_model':c['expected_returned_model'],'caption_model':c['caption_model'],
        'caption_seed':None,'reader_seed':42,'lambda':.01,'alpha':1.,'memory_path':str(source),'overall':overall,
        'within_gpt55_weighting_comparisons':within,'gpt54_comparisons':cross,'by_domain':by_domain,
        'generation_stages':stages,'method_annotations':methods,'validation':validation,'interpretation':c['interpretation']}
    dump(source/'summary.json',summary);dump(source/'paired_results.json',pairs)
    with (source/'paired_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>MemEye GPT-5.5 全流程评测</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1150px;margin:32px auto;padding:0 20px}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap}</style>',
        '<h1>GPT-5.5：新 caption、新记忆、新回答</h1>',
        '<p>全部 371 题；437 张独立图片的 caption 全部重新生成，重建 438 条记忆及 master。请求 gpt-5.5，实际返回 '+html.escape(c['expected_returned_model'])+'。caption 未传 seed；回答 seed=42。温度 0、reasoning=none；caption 256 tokens、回答 16 tokens。固定选项排列。</p>',
        '<table><tr><th>检索方案</th><th>GPT-5.4 同设置参考</th><th>GPT-5.5</th><th>准确率</th><th>相对 GPT-5.4</th></tr>']
    for name in CONDITIONS:
        body.append(f'<tr><td>{name}</td><td>{cross[name]["reference_correct"]}/371</td><td>{overall[name]["correct"]}/371</td><td>{overall[name]["accuracy"]:.4%}</td><td>{cross[name]["net_correct"]:+d}</td></tr>')
    body+=['</table><h2>同一 GPT-5.5 记忆上的 z-score 消融</h2>',
        '<p>λ=0.01、α=1；全量按同 domain 分组，多路径按新 master 分组。四种条件中完整输入相同则共享本轮新回答，减少同输入跨调用波动。GPT-5.4 对照同时改变 caption 生成与回答，不能解释为只替换回答模型的效果。</p>',
        '<pre>'+html.escape(json.dumps(within,ensure_ascii=False,indent=2))+'</pre>',
        '<h2>分 domain 统计</h2><pre>'+html.escape(json.dumps(by_domain,ensure_ascii=False,indent=2))+'</pre>',
        '<h2>方法与核验</h2><pre>'+html.escape(json.dumps(validation,ensure_ascii=False,indent=2))+'</pre>']
    for name in CONDITIONS:body.append('<details><summary>'+name+'</summary><pre>'+html.escape(json.dumps(methods[name],ensure_ascii=False,indent=2))+'</pre></details>')
    body+=['<p><a href="summary.json">完整统计</a> · <a href="paired_results.csv">逐题 CSV</a> · <a href="independent_audit.json">独立核验</a></p></html>']
    (source/'report.html').write_text('\n'.join(body));dump(source/'execution_status.json',{'status':'complete','overall':overall})
    print(json.dumps({'stage':'analyzed','overall':overall,'within_gpt55':within}),flush=True)
    return summary


def run(output=DEFAULT_OUTPUT, smoke=False):
    from scripts.evaluation.openai_gpt55_transport import APIError, chat_request, load_config
    output = Path(output).resolve()
    protocol = verify(output)
    c = protocol['contract']
    config = load_config(c['api_env_file'], model=c['model'], base_url=c['base_url'])
    if config.public_dict() != c['api_config']:
        raise ValueError('API configuration differs from prepared contract')
    records = lines(output / 'inputs/requests.jsonl')
    by_record = indexed(records, 'record_id')
    wire = indexed(lines(output / 'api_request_manifest.jsonl'), 'record_id')
    completed = lines(output / 'fresh_responses.jsonl') if (output / 'fresh_responses.jsonl').exists() else []
    fresh = indexed(completed, 'payload_sha256')
    for r in completed:
        original = by_record[r['record_id']]
        if (r['run_id'] != protocol['run_id'] or r['experiment_signature'] != protocol['experiment_signature'] or
                r['payload_sha256'] != original['payload_sha256'] or
                r['request_sha256'] != wire[r['record_id']]['request_sha256']):
            raise ValueError('API response resume provenance mismatch')
        if (r['predicted_choice'] != parse_choice(r['raw_response']) or not transport.valid_returned_model(r['returned_model']) or r['returned_model'] != c['expected_returned_model'] or r.get('requested_model') != c['model']
                or r['historical_response_reuse'] is not False):
            raise ValueError('Saved prediction parser mismatch')
    unique = {}
    for r in records:
        unique.setdefault(r['payload_sha256'], r)
    if len(unique) != c['unique_payloads']:
        raise ValueError('Unique payload coverage changed')
    if smoke:
        selected = []
        for has_image in (False, True):
            selected.append(next(r for r in unique.values() if bool(r['query_images']) == has_image))
    else:
        selected = list(unique.values())
    pending = [r for r in selected if r['payload_sha256'] not in fresh]
    execution_id = str(uuid.uuid4())

    def execute(row):
        body = request_body(protocol, row)
        wire_hash = digest(body)
        if wire_hash != wire[row['record_id']]['request_sha256']:
            raise ValueError('Actual API request differs from frozen request')
        attempts = []
        for attempt in range(1, c['max_attempts'] + 1):
            meta = {'record_id': row['record_id'], 'payload_sha256': row['payload_sha256'],
                    'request_sha256': wire_hash, 'execution_id': execution_id,
                    'run_id': protocol['run_id'], 'requested_model': c['model'], 'attempt': attempt,
                    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            try:
                answer = chat_request(config, body, timeout=c['timeout_seconds'])
            except APIError as e:
                attempts.append({**meta, 'status': 'api_error', 'kind': e.kind,
                                 'http_status': e.http_status, 'code': e.code,
                                 'message': e.message, 'retryable': e.retryable})
                if not e.retryable or attempt == c['max_attempts']:
                    return None, attempts
                time.sleep(min(2 ** attempt, 30))
                continue
            returned = answer['returned_model']
            if not transport.valid_returned_model(returned) or returned != c['expected_returned_model'] or answer.get('requested_model') != c['model']:
                attempts.append({**meta, 'status': 'model_mismatch', 'returned_model': returned})
                return None, attempts
            choice = parse_choice(answer['raw_response'])
            attempts.append({**meta, 'status': 'success', 'response_id': answer['response_id'],
                             'request_id': answer['request_id'], 'returned_model': returned,
                             'elapsed_seconds': answer['elapsed_seconds']})
            result = {**answer, 'record_id': row['record_id'], 'id': row['id'], 'task': row['task'],
                      'condition': row['condition'], 'payload_sha256': row['payload_sha256'],
                      'request_sha256': wire_hash, 'run_id': protocol['run_id'],
                      'experiment_signature': protocol['experiment_signature'], 'execution_id': execution_id,
                      'attempts': attempt, 'query_image_count': len(row['query_images']),
                      'predicted_choice': choice, 'status': 'ok' if choice else 'format_error',
                      'hit_token_limit': answer['finish_reason'] == 'length',
                      'historical_response_reuse': False}
            return result, attempts

    failures = []
    # Bounded batches prevent a permanent provider failure from launching all 719 calls.
    workers = c['workers']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(pending), workers):
            batch = pending[start:start + workers]
            futures = {pool.submit(execute, r): r for r in batch}
            for future in as_completed(futures):
                row = futures[future]
                result, attempts = future.result()
                for a in attempts:
                    append(output / 'api_attempts.jsonl', a)
                if result is None:
                    failures.append({'record_id': row['record_id'], 'last_attempt': attempts[-1]})
                    continue
                fresh[row['payload_sha256']] = result
                append(output / 'fresh_responses.jsonl', result)
            logical = logical_responses(records, fresh)
            write_lines(output / 'responses.jsonl', logical)
            progress = {'stage': 'api_reader', 'model': c['model'], 'completed': len(logical),
                        'total': c['logical_records'], 'fresh_requests': len(fresh),
                        'expected_fresh_requests': c['unique_payloads'], 'smoke': smoke,
                        'format_errors': sum(r['status'] != 'ok' for r in logical),
                        'token_limit_records': sum(r['hit_token_limit'] for r in logical),
                        'failed_requests_this_execution': len(failures)}
            dump(output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            if failures:
                dump(output / 'failed_requests.json', failures)
                raise RuntimeError('API requests failed; inspect failed_requests.json. Successful responses retained for resume.')
    if smoke:
        selected_results = [fresh[r['payload_sha256']] for r in selected]
        report = {'passed': all(r['status'] == 'ok' and not r['hit_token_limit']
                                and r.get('reasoning_tokens') in (None, 0) for r in selected_results),
                  'records': [{k: r[k] for k in ('record_id', 'query_image_count', 'status', 'returned_model',
                                                'finish_reason', 'request_sha256', 'response_id', 'reasoning_tokens')} for r in selected_results],
                  'same_protocol_as_full_run': True}
        dump(output / 'smoke_validation.json', report)
        if not report['passed']:
            raise RuntimeError('Smoke outputs failed format/token-limit validation; preserved for diagnosis')
        return report
    if set(fresh) != set(unique):
        raise ValueError('Incomplete API coverage')
    verify(output)
    logical = logical_responses(records, fresh)
    write_lines(output / 'responses.jsonl', logical)
    returned_models = sorted({r['returned_model'] for r in fresh.values()})
    if len(returned_models) != 1:
        raise ValueError('Provider returned multiple model revisions in this run')
    validation = {'passed': True, 'questions': 371, 'logical_records': len(logical),
                  'fresh_requests': len(fresh), 'shared_records': len(logical) - len(fresh),
                  'requested_model': c['model'], 'seed': c['seed'], 'returned_models': returned_models,
                  'system_fingerprints': dict(Counter(r['system_fingerprint'] for r in fresh.values())),
                  'format_errors': sum(r['status'] != 'ok' for r in logical),
                  'empty_responses': sum(not r['raw_response'] for r in logical),
                  'token_limit_records': sum(r['hit_token_limit'] for r in logical),
                  'reasoning_tokens_reported': sum(r.get('reasoning_tokens') or 0 for r in fresh.values()),
                  'reasoning_usage_missing_requests': sum(r.get('reasoning_tokens') is None for r in fresh.values()),
                  'responses_sha256': sha(output / 'responses.jsonl'),
                  'fresh_responses_sha256': sha(output / 'fresh_responses.jsonl'),
                  'gold_loaded_during_run': False, 'historical_api_answer_reuse': False}
    dump(output / 'validation_output.json', validation)
    return validation



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=BASE)
    p.add_argument('--stage',choices=['prepare','smoke','run','analyze','all'],default='all')
    p.add_argument('--api-workers',type=int,default=3)
    a=p.parse_args();assert a.api_workers>0
    if a.stage in ('prepare','all'):prepare(a.output,a.api_workers)
    from scripts.analysis.audit_memeye_gpt55 import audit
    if a.stage in ('prepare','smoke','run','all'):audit(a.output,inputs_only=True)
    if a.stage in ('smoke','all'):run(a.output/'reader',smoke=True)
    if a.stage in ('run','all'):run(a.output/'reader')
    if a.stage in ('analyze','all'):
        analyze(a.output);audit(a.output)


if __name__=='__main__':main()
