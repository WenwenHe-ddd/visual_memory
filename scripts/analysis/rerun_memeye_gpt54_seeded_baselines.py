"""Rerun both unweighted GPT-caption baselines, changing only the reader seed."""
from pathlib import Path
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import html
import json
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memeye_gpt5_reader as api
from scripts.evaluation import openai_gpt54_seeded_transport as transport

SOURCE = ROOT / 'results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05'
OUTPUT = ROOT / 'results/memeye_gpt54_seed42_baselines_20261024'
DEFAULT_OUTPUT = OUTPUT
CONDITIONS = ('baseline_question_only', 'master_answer_query')
read, lines, sha, digest, dump = api.read, api.lines, api.sha, api.digest, api.dump
indexed, append, write_lines = api.indexed, api.append, api.write_lines
parse_choice, logical_responses = api.parse_choice, api.logical_responses


def request_body(protocol, row):
    body = transport.build_request({'system':protocol['contract']['reader_system_prompt'],
        'user_text':row['reader_text'], 'query_images':row['query_images']},
        seed=protocol['contract']['seed'])
    assert digest({k:v for k,v in body.items() if k != 'seed'}) == row['historical_request_sha256']
    return body


def method_annotations(seed):
    result = {}
    for name in CONDITIONS:
        master = name == CONDITIONS[1]
        result[name] = {
            '1. 候选记忆结构': {'memory_path':str(SOURCE), 'memory_leaves':438,
                'offline_method':'caption_visual_master' if master else 'natural',
                'caption_model':api.MODEL, 'text':'原始对话＋冻结的 GPT-5.4 完整 caption'},
            '2. 查询内容':('问题＋已有 GPT-5.4 题图 caption＋各 master 视觉匹配选中的完整候选答案陈述'
                           if master else '问题＋已有 GPT-5.4 题图 caption；不拼入候选答案陈述'),
            '3. 检索方式':{'method':'master_answer_query' if master else 'full_scan',
                'top_k':5, 'score':'原始 BGE 余弦相似度', 'zscore':False,
                'scope':'同 domain 全部叶子排序后取 Top-5',
                'execution':'复用此前 236/241 两套基线的完整排名、分数及有序 Top-5'},
            '4. 编码方式':{'BGE':'BAAI/bge-small-en-v1.5, 384 dimensions, L2 normalized',
                'leaf_text':'对话＋完整 caption',
                'CLIP':'master 视觉聚合量匹配完整候选陈述' if master else '不参与该条件排序',
                'execution':'冻结原有编码和检索产物；本次不重新编码'},
            '5. 回答方式':{'model':api.MODEL, 'seed':seed, 'temperature':0,
                'reasoning_effort':'none', 'max_completion_tokens':16, 'image_detail':'high',
                'input':'有序 Top-5 对话＋完整 caption、问题、固定排列选项、原始题图；不传检索假设',
                'execution':'本轮重新请求回答；同轮相同完整请求共享一次新回答；不复用历史或探测答案'}}
    return result


def prepare(output=OUTPUT, seed=42):
    output = Path(output).resolve()
    if (output/'protocol.json').exists():
        p = verify(output)
        assert p['contract']['seed'] == seed
        return p
    assert not (output/'fresh_responses.jsonl').exists()
    from scripts.analysis.evaluate_memeye_gpt54_caption_zscore import verify_sources
    proof = verify_sources(SOURCE)
    probe = read(OUTPUT/'seed_support_probe.json')
    assert probe['status'] == 'seed_accepted' and probe['seed'] == seed
    old = read(SOURCE/'protocol.json')['contract']
    config = transport.load_config(old['api_env_file'], model=api.MODEL)
    assert config.public_dict() == old['api_config']
    requests = lines(SOURCE/'reader/requests.jsonl')
    manifest = indexed(lines(SOURCE/'reader_manifest.jsonl'), 'record_id')
    assert len(requests) == len(manifest) == 742
    assert len({r['system'] for r in requests}) == 1
    rows = []
    for r in requests:
        rid = r['record_id']; qid, condition = rid.rsplit('::',1)
        assert condition in CONDITIONS
        payload = {k:r[k] for k in ('system','user_text','query_images')}
        assert digest(payload) == r['payload_sha256']
        rows.append(dict(record_id=rid,id=qid,task=manifest[rid]['task'],condition=condition,
            reader_text=r['user_text'],query_images=r['query_images'],
            top_memory_ids=manifest[rid]['top_memory_ids'],payload_sha256=r['payload_sha256'],
            historical_request_sha256=r['request_sha256']))
    assert all(len(r['top_memory_ids']) == 5 for r in rows)
    gold = lines(SOURCE/'inputs/evaluation.jsonl')
    assert len(gold) == 371 and {r['id'] for r in gold} == {r['id'] for r in rows}
    output.mkdir(parents=True,exist_ok=True); (output/'inputs').mkdir(exist_ok=True)
    write_lines(output/'inputs/requests.jsonl',rows)
    write_lines(output/'inputs/gold.jsonl',gold)
    dump(output/'source_validation.json',proof)
    c = {'model':api.MODEL,'caption_model':api.MODEL,'seed':seed,'temperature':0,
         'reasoning_effort':'none','max_completion_tokens':16,'image_detail':'high',
         'questions':371,'logical_records':742,'unique_payloads':len({r['payload_sha256'] for r in rows}),
         'base_url':config.base_url,'api_env_file':old['api_env_file'],'api_config':config.public_dict(),
         'reader_system_prompt':requests[0]['system'],'workers':3,'max_attempts':3,'timeout_seconds':60,
         'changed_component':'Only add the explicit provider seed to otherwise byte-identical API requests',
         'frozen_memory_path':str(SOURCE),'caption_rebuilt':False,'retrieval_rebuilt':False,
         'historical_api_answer_reuse':False,'probe_answer_reuse':False,'gold_in_api_payload':False,
         'method_annotations':method_annotations(seed),
         'seed_caveat':'API accepts seed; identical seeded requests already produced different answers in the probe.',
         'metric_policy':'Fixed option order MCQ accuracy; all 371 questions retained including format errors.'}
    assert c['unique_payloads'] == 720
    wire = [{'record_id':r['record_id'],'payload_sha256':r['payload_sha256'],
             'request_sha256':digest(request_body({'contract':c},r)),
             'query_image_count':len(r['query_images'])} for r in rows]
    c['request_manifest_sha256'] = digest(wire)
    paths = [Path(__file__),Path(transport.__file__),Path(api.__file__),
             output/'inputs/requests.jsonl',output/'inputs/gold.jsonl',output/'source_validation.json',
             OUTPUT/'seed_support_probe.json']
    c['source_sha256'] = {**proof['source_sha256'],**{str(p):sha(p) for p in paths}}
    p = {'run_id':str(uuid.uuid4()),'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'contract':c,'experiment_signature':digest(c)}
    write_lines(output/'api_request_manifest.jsonl',wire); dump(output/'protocol.json',p)
    dump(output/'input_audit.json',{'passed':True,'requests_equal_to_historical_except_seed':742,
        'logical_records':742,'unique_requests':720,'all_top5_frozen':True,'caption_source_correct':[236,241],
        'seed':seed,'native_image_records':sum(bool(r['query_images']) for r in rows),
        'gold_fields_absent_from_wire':True})
    print(json.dumps({'stage':'prepared','seed':seed,'records':742,'unique_requests':720}),flush=True)
    return p


def verify(output):
    return api.verify(output)


def analyze(output=OUTPUT):
    output = Path(output); protocol=verify(output)
    validation = read(output/'validation_output.json')
    requests = indexed(lines(output/'inputs/requests.jsonl'),'record_id')
    responses = indexed(lines(output/'responses.jsonl'),'record_id')
    gold = indexed(lines(output/'inputs/gold.jsonl'),'id')
    fresh = indexed(lines(output/'fresh_responses.jsonl'),'payload_sha256')
    wire = indexed(lines(output/'api_request_manifest.jsonl'),'record_id')
    assert set(requests) == set(responses) and len(responses) == 742 and len(fresh) == 720
    assert sha(output/'responses.jsonl') == validation['responses_sha256']
    assert sha(output/'fresh_responses.jsonl') == validation['fresh_responses_sha256']
    old_pairs = indexed(lines(SOURCE/'paired_results.jsonl'),'question_id')
    old_requests = indexed(lines(SOURCE/'reader/requests.jsonl'),'record_id')
    for rid,r in responses.items():
        q = requests[rid]; f = fresh[q['payload_sha256']]
        assert r['predicted_choice'] == parse_choice(r['raw_response'])
        assert r['request_sha256'] == wire[rid]['request_sha256'] == digest(request_body(protocol,q))
        assert r['payload_sha256'] == old_requests[rid]['payload_sha256']
        assert r['top_memory_ids'] == q['top_memory_ids']
        assert r['historical_response_reuse'] is False and r['returned_model'] == api.MODEL
        assert r['run_id'] == protocol['run_id'] and r['experiment_signature'] == protocol['experiment_signature']
        assert r['fresh_inference_executed_for_this_record'] == (rid == f['record_id'])
        assert r['same_run_shared_inference_record_id'] == (None if rid == f['record_id'] else f['record_id'])
        for key in ['raw_response','response_id','request_sha256','usage','execution_id']:
            assert r[key] == f[key]
    pairs = []
    for qid,g in gold.items():
        b,m = [responses[qid+'::'+name] for name in CONDITIONS]
        old = old_pairs[qid]
        pairs.append({'question_id':qid,'domain':b['task'],'gold_choice':g['gold_choice'],
            'baseline_answer':b['predicted_choice'],'expanded_answer':m['predicted_choice'],
            'baseline_correct':b['predicted_choice']==g['gold_choice'],
            'expanded_correct':m['predicted_choice']==g['gold_choice'],
            'historical_baseline_answer':old['baseline_answer'],'historical_master_answer':old['expanded_answer'],
            'historical_baseline_correct':old['baseline_correct'],'historical_master_correct':old['expanded_correct']})
    comparisons = {}
    for name,new,old in [('baseline_question_only','baseline_correct','historical_baseline_correct'),
                         ('master_answer_query','expanded_correct','historical_master_correct')]:
        comparisons[name] = {'historical_correct':sum(r[old] for r in pairs),'seeded_correct':sum(r[new] for r in pairs),
            'wrong_to_right':sum(not r[old] and r[new] for r in pairs),
            'right_to_wrong':sum(r[old] and not r[new] for r in pairs)}
    annotations=method_annotations(protocol['contract']['seed'])
    for name in CONDITIONS:
        annotations[name]['5. 回答方式']['correct'] = comparisons[name]['seeded_correct']
    usage = {k:sum((r.get('usage') or {}).get(k,0) for r in fresh.values())
             for k in ['prompt_tokens','completion_tokens','total_tokens']}
    probe=read(OUTPUT/'seed_support_probe.json')
    summary={'status':'complete','model':api.MODEL,'caption_model':api.MODEL,'seed':protocol['contract']['seed'],
        'overall':api.aggregate(pairs),'by_domain':{d:api.aggregate([r for r in pairs if r['domain']==d])
                                                  for d in sorted({r['domain'] for r in pairs})},
        'historical_comparisons':comparisons,'method_annotations':annotations,'validation':validation,
        'seed_probe_pairs':probe['repeat_pairs'],'seed_caveat':protocol['contract']['seed_caveat'],
        'usage_scored_fresh_calls_only':usage,'probe_calls_excluded_from_scores':len(probe['attempts']),
        'retrieval_metrics':read(SOURCE/'summary.json')['recall_at_5'],
        'interpretation':'Single seeded rerun on frozen GPT-caption retrieval. Score changes cannot establish a causal seed effect.'}
    dump(output/'summary.json',summary);dump(output/'paired_results.json',pairs)
    with (output/'paired_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    parts=['<!doctype html><meta charset="utf-8"><title>MemEye seed=42 基线复测</title>',
           '<style>body{font:16px/1.7 system-ui;max-width:1050px;margin:35px auto}td,th{padding:12px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>',
           '<h1>GPT-5.4 seed=42：两套未加权基线</h1>',
           '<p>GPT-5.4 caption、memory、完整检索排名和有序 Top-5 全部冻结，仅回答请求新增 seed=42。问题固定为 371 道；模型快照、温度、推理和输出长度不变。</p>',
           '<table><tr><th>方案</th><th>历史未传 seed</th><th>本次 seed=42</th><th>错转对 / 对转错</th></tr>']
    for name,s in comparisons.items():
        parts.append(f'<tr><td>{name}</td><td>{s["historical_correct"]}/371</td><td>{s["seeded_correct"]}/371 ({s["seeded_correct"]/371:.2%})</td><td>{s["wrong_to_right"]} / {s["right_to_wrong"]}</td></tr>')
    parts += ['</table><p>720 次本轮新请求支撑 742 条逻辑记录；22 条相同请求共享本轮回答。4 次先行探测不计入成绩，不复用历史答案。</p>',
              '<p>seed 接受验证中，Health/Q38 相同请求重复调用返回 A 和 C。因此固定 seed 不能保证确定性；本次分数变化也不能直接归因于 seed。</p>',
              '<p><a href="paired_results.csv">逐题对照</a> · <a href="summary.json">完整统计</a> · <a href="protocol.json">协议</a></p>',
              '<h2>五项方法标注</h2><pre>'+html.escape(json.dumps(annotations,ensure_ascii=False,indent=2))+'</pre>']
    (output/'report.html').write_text('\n'.join(parts)+'\n')
    dump(output/'execution_status.json',{'status':'complete','model':api.MODEL,'seed':protocol['contract']['seed'],
        'questions':371,'logical_records':742,'fresh_requests':720,'overall':summary['overall']})
    print(json.dumps({'stage':'complete','overall':summary['overall'],'validation':validation}),flush=True)
    return summary


# The run function below is a frozen copy of the existing audited reader loop,
# with only its transport import changed to the explicit-seed adapter.

def run(output=DEFAULT_OUTPUT, smoke=False):
    from scripts.evaluation.openai_gpt54_seeded_transport import APIError, chat_request, load_config
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
        if (r['predicted_choice'] != parse_choice(r['raw_response']) or r['returned_model'] != c['model']
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
                    'run_id': protocol['run_id'], 'attempt': attempt,
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
            if returned != c['model']:
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
                  'returned_models': returned_models,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare','run','analyze','all'], default='all')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage in ('prepare','all'):
        prepare(args.output,args.seed)
    if args.stage in ('run','all'):
        assert verify(args.output)['contract']['seed'] == args.seed
        run(args.output,smoke=args.smoke)
    if args.stage in ('analyze','all') and not args.smoke:
        analyze(args.output)


if __name__ == '__main__':
    main()
