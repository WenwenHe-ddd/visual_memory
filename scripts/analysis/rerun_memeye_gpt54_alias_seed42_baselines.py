"""Request literal gpt-5.4 with seed 42 on the frozen 371-question baselines."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import html
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis import rerun_memeye_gpt54_seeded_baselines as previous
from scripts.evaluation import memeye_gpt5_reader as api
from scripts.evaluation import openai_gpt54_alias_transport as transport

SOURCE = previous.SOURCE
REFERENCE = previous.OUTPUT
REFERENCE1042 = ROOT/'results/memeye_gpt54_seed1042_baselines_20261024'
OUTPUT = ROOT/'results/memeye_gpt54_alias_seed42_baselines_20261024'
DEFAULT_OUTPUT = OUTPUT
SEED = 42
CONDITIONS = previous.CONDITIONS
read, lines, sha, digest, dump = api.read, api.lines, api.sha, api.digest, api.dump
indexed, append, write_lines = api.indexed, api.append, api.write_lines
parse_choice, logical_responses, verify = api.parse_choice, api.logical_responses, api.verify


def request_body(protocol, row):
    c = protocol['contract']
    assert c['model'] == transport.MODEL and c['seed'] == SEED
    body = transport.build_request({'system': c['reader_system_prompt'],
        'user_text': row['reader_text'], 'query_images': row['query_images']}, seed=SEED)
    historical = {k: v for k, v in body.items() if k != 'seed'}
    historical['model'] = api.MODEL
    assert digest(historical) == row['historical_request_sha256']
    return body


def method_annotations():
    result = previous.method_annotations(SEED)
    for item in result.values():
        item['5. 回答方式'].update(model=transport.MODEL, requested_model=transport.MODEL,
            returned_model_policy='Record the actual provider revision separately; reject mixed revisions.',
            execution='本轮直接请求 gpt-5.4 别名、seed=42；同轮相同完整请求共享新回答；不复用历史答案')
    return result


def prepare(output=OUTPUT, seed=SEED):
    output = Path(output).resolve()
    assert seed == SEED
    if (output/'protocol.json').exists():
        p = verify(output)
        assert p['contract']['model'] == transport.MODEL and p['contract']['seed'] == SEED
        return p
    assert not (output/'fresh_responses.jsonl').exists()
    old = verify(REFERENCE)
    assert old['contract']['seed'] == SEED and old['contract']['model'] == api.MODEL
    audit = read(REFERENCE/'independent_audit.json')
    assert audit['passed'] and audit['correct'] == {'baseline_question_only': 233, 'master_answer_query': 231}
    for name, expected in audit['artifact_sha256'].items():
        assert sha(REFERENCE/name) == expected, name
    records, gold = lines(REFERENCE/'inputs/requests.jsonl'), lines(REFERENCE/'inputs/gold.jsonl')
    assert len(records) == 742 and len(gold) == 371
    config = transport.load_config(old['contract']['api_env_file'], base_url=old['contract']['base_url'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'inputs').mkdir(exist_ok=True)
    write_lines(output/'inputs/requests.jsonl', records)
    write_lines(output/'inputs/gold.jsonl', gold)
    dump(output/'source_validation.json', read(REFERENCE/'source_validation.json'))
    c = copy.deepcopy(old['contract'])
    c.update(model=transport.MODEL, requested_model=transport.MODEL, api_config=config.public_dict(),
        changed_component='Only the reader request model changes from gpt-5.4-2026-03-05 to gpt-5.4; seed stays 42.',
        method_annotations=method_annotations(), snapshot_seed42_reference=str(REFERENCE),
        seed_caveat='Alias requests do not guarantee deterministic answers; record actual returned revisions.',
        model_revision_policy='Accept GPT-5.4 alias or dated GPT-5.4 revision; require one returned revision per run.',
        probe_calls_this_experiment=0)
    wire = []
    old_wire = indexed(lines(REFERENCE/'api_request_manifest.jsonl'), 'record_id')
    for row in records:
        before = previous.request_body(old, row)
        after = request_body({'contract': c}, row)
        assert before['model'] == api.MODEL and after['model'] == transport.MODEL
        assert {k: v for k, v in before.items() if k != 'model'} == {k: v for k, v in after.items() if k != 'model'}
        assert digest(before) == old_wire[row['record_id']]['request_sha256']
        wire.append({'record_id': row['record_id'], 'payload_sha256': row['payload_sha256'],
            'request_sha256': digest(after), 'requested_model': transport.MODEL,
            'query_image_count': len(row['query_images'])})
    c['request_manifest_sha256'] = digest(wire)
    paths = [Path(__file__), Path(transport.__file__),
        ROOT/'scripts/analysis/audit_memeye_gpt54_alias_seed42_baselines.py',
        ROOT/'scripts/analysis/audit_memeye_gpt54_seed1042_baselines.py',
        ROOT/'scripts/tests/test_memeye_gpt54_alias_seed42_baselines.py',
        output/'inputs/requests.jsonl', output/'inputs/gold.jsonl', output/'source_validation.json']
    for directory in [REFERENCE, REFERENCE1042]:
        paths.extend(directory/name for name in ['protocol.json', 'summary.json', 'paired_results.json',
            'responses.jsonl', 'fresh_responses.jsonl', 'independent_audit.json', 'api_request_manifest.jsonl'])
    c['source_sha256'].update({str(p): sha(p) for p in paths})
    p = {'run_id': str(uuid.uuid4()), 'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'contract': c, 'experiment_signature': digest(c)}
    write_lines(output/'api_request_manifest.jsonl', wire)
    dump(output/'protocol.json', p)
    dump(output/'input_audit.json', {'passed': True, 'seed': SEED, 'requested_model': transport.MODEL,
        'questions': 371, 'records': 742, 'unique_requests': 720,
        'wire_only_difference_from_snapshot_seed42': 'model', 'complete_requests_checked': 742,
        'caption_memory_and_retrieval_frozen': True, 'historical_or_probe_answer_reuse': False,
        'gold_fields_absent_from_wire': True})
    print(json.dumps({'stage': 'prepared', 'model': transport.MODEL, 'seed': SEED, 'unique_requests': 720}), flush=True)
    return p


def analyze(output=OUTPUT):
    output = Path(output)
    p = verify(output)
    validation = read(output/'validation_output.json')
    records = indexed(lines(output/'inputs/requests.jsonl'), 'record_id')
    responses = indexed(lines(output/'responses.jsonl'), 'record_id')
    fresh = indexed(lines(output/'fresh_responses.jsonl'), 'payload_sha256')
    wire = indexed(lines(output/'api_request_manifest.jsonl'), 'record_id')
    gold = indexed(lines(output/'inputs/gold.jsonl'), 'id')
    assert set(records) == set(responses) and len(responses) == 742 and len(fresh) == 720
    for name in ['responses', 'fresh_responses']:
        assert sha(output/(name+'.jsonl')) == validation[name+'_sha256']
    references = {'snapshot_seed42': indexed(read(REFERENCE/'paired_results.json'), 'question_id'),
        'snapshot_seed1042': indexed(read(REFERENCE1042/'paired_results.json'), 'question_id'),
        'snapshot_unseeded': indexed(lines(SOURCE/'paired_results.jsonl'), 'question_id')}
    for rid, answer in responses.items():
        row, original = records[rid], fresh[records[rid]['payload_sha256']]
        assert answer['request_sha256'] == wire[rid]['request_sha256'] == digest(request_body(p, row))
        assert answer['run_id'] == p['run_id'] and answer['experiment_signature'] == p['experiment_signature']
        assert answer['requested_model'] == transport.MODEL and transport.valid_returned_model(answer['returned_model'])
        assert answer['historical_response_reuse'] is False
        assert answer['predicted_choice'] == parse_choice(answer['raw_response'])
        assert answer['top_memory_ids'] == row['top_memory_ids']
        assert answer['fresh_inference_executed_for_this_record'] == (rid == original['record_id'])
        assert answer['same_run_shared_inference_record_id'] == (None if rid == original['record_id'] else original['record_id'])
        for key in ['response_id', 'raw_response', 'request_sha256', 'usage', 'execution_id', 'returned_model']:
            assert answer[key] == original[key]
    pairs, long_rows = [], []
    for qid, g in gold.items():
        pair = {'question_id': qid, 'domain': qid.rsplit('/', 1)[0], 'gold_choice': g['gold_choice']}
        for condition, prefix in zip(CONDITIONS, ['baseline', 'expanded']):
            answer = responses[qid+'::'+condition]
            pred = answer['predicted_choice']
            correct = pred == g['gold_choice']
            pair.update({prefix+'_answer': pred, prefix+'_correct': correct})
            row = {'question_id': qid, 'condition': condition, 'gold_choice': g['gold_choice'],
                'alias_seed42_answer': pred, 'alias_seed42_correct': correct,
                'requested_model': answer['requested_model'], 'returned_model': answer['returned_model']}
            for label, ref in references.items():
                pair.update({label+'_'+prefix+'_answer': ref[qid][prefix+'_answer'],
                             label+'_'+prefix+'_correct': ref[qid][prefix+'_correct']})
                row[label+'_answer'] = ref[qid][prefix+'_answer']
                row[label+'_correct'] = ref[qid][prefix+'_correct']
            row['answer_changed_vs_snapshot_seed42'] = pred != row['snapshot_seed42_answer']
            long_rows.append(row)
        pairs.append(pair)
    comparisons = {}
    for label in references:
        comparisons[label] = {}
        for condition, prefix in zip(CONDITIONS, ['baseline', 'expanded']):
            stats = api.aggregate([{'baseline_correct': r[label+'_'+prefix+'_correct'],
                                   'expanded_correct': r[prefix+'_correct']} for r in pairs])
            comparisons[label][condition] = {'reference_correct': stats['baseline_correct'],
                'alias_seed42_correct': stats['expanded_correct'], 'wrong_to_right': stats['wrong_to_correct'],
                'right_to_wrong': stats['correct_to_wrong'], 'net_correct': stats['expanded_correct']-stats['baseline_correct'],
                'mcnemar_exact_p': stats['mcnemar_exact_p'],
                'answer_changed': sum(r[prefix+'_answer'] != r[label+'_'+prefix+'_answer'] for r in pairs)}
    annotations = method_annotations()
    for condition in CONDITIONS:
        annotations[condition]['5. 回答方式'].update(
            correct=comparisons['snapshot_seed42'][condition]['alias_seed42_correct'],
            returned_models=validation['returned_models'])
    summary = {'status': 'complete', 'model': transport.MODEL, 'requested_model': transport.MODEL,
        'returned_models': validation['returned_models'], 'caption_model': api.MODEL, 'seed': SEED,
        'overall': api.aggregate(pairs), 'by_domain': {d: api.aggregate([r for r in pairs if r['domain'] == d])
            for d in sorted({r['domain'] for r in pairs})}, 'comparisons': comparisons,
        'method_annotations': annotations, 'validation': validation,
        'retrieval_metrics': read(REFERENCE/'summary.json')['retrieval_metrics'],
        'usage_scored_fresh_calls_only': {k: sum((r.get('usage') or {}).get(k, 0) for r in fresh.values())
            for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']},
        'interpretation': 'One alias rerun versus one snapshot run at seed 42; differences do not establish an alias effect or reproducibility improvement.'}
    dump(output/'summary.json', summary)
    dump(output/'paired_results.json', pairs)
    for name, rows in [('paired_results.csv', pairs), ('alias_comparison.csv', long_rows)]:
        with (output/name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    parts = ['<!doctype html><meta charset="utf-8"><title>MemEye GPT-5.4 别名 seed=42</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:35px auto}td,th{padding:12px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>',
        '<h1>GPT-5.4 别名与日期快照：seed=42</h1>',
        '<p>371 题，两套未加权基线。沿用历史 GPT-5.4 caption、memory、完整排名、有序 Top-5、原图、提示词及全部生成参数，只将请求 model 改为 gpt-5.4。seed=42、temperature=0、reasoning_effort=none、max_completion_tokens=16。</p>',
        '<p>实际返回版本：'+html.escape(', '.join(validation['returned_models']))+'</p>',
        '<table><tr><th>方案</th><th>快照未传 seed</th><th>快照 seed=42</th><th>别名 seed=42</th><th>错→对 / 对→错</th><th>答案改变</th></tr>']
    for condition in CONDITIONS:
        c = comparisons['snapshot_seed42'][condition]
        old = comparisons['snapshot_unseeded'][condition]
        parts.append(f'<tr><td>{condition}</td><td>{old["reference_correct"]}/371</td><td>{c["reference_correct"]}/371</td><td>{c["alias_seed42_correct"]}/371 ({c["alias_seed42_correct"]/371:.2%})</td><td>{c["wrong_to_right"]} / {c["right_to_wrong"]}</td><td>{c["answer_changed"]}</td></tr>')
    parts += ['</table><p>720 次本轮新调用，742 条逻辑记录，22 条同轮共享；先行文本／图片检查计入同轮。没有复用历史或独立探测回答。每种设置仅一轮，不能将分差归因于别名，也不能据此判断更可复现。</p>',
        '<p><a href="alias_comparison.csv">逐题对比</a> · <a href="summary.json">完整统计</a> · <a href="protocol.json">协议</a> · <a href="independent_audit.json">独立核验</a></p>',
        '<h2>五项方法标注</h2><pre>'+html.escape(json.dumps(annotations, ensure_ascii=False, indent=2))+'</pre>']
    (output/'report.html').write_text('\n'.join(parts)+'\n')
    dump(output/'execution_status.json', {'status': 'complete', 'requested_model': transport.MODEL,
        'returned_models': validation['returned_models'], 'seed': SEED, 'overall': summary['overall']})
    print(json.dumps({'stage': 'complete', 'overall': summary['overall'], 'validation': validation}), flush=True)
    return summary


# Versioned copy of the audited seed-42 runner with alias-aware model checks.
def run(output=DEFAULT_OUTPUT, smoke=False):
    from scripts.evaluation.openai_gpt54_alias_transport import APIError, chat_request, load_config
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
        if (r['predicted_choice'] != parse_choice(r['raw_response']) or not transport.valid_returned_model(r['returned_model']) or r.get('requested_model') != c['model']
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
            if not transport.valid_returned_model(returned) or answer.get('requested_model') != c['model']:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'run', 'analyze', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--model', choices=['gpt-5.4'], default='gpt-5.4')
    parser.add_argument('--seed', type=int, choices=[42], default=42)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage in ['prepare', 'all']:
        prepare(args.output, args.seed)
    if args.stage in ['run', 'all']:
        run(args.output, smoke=args.smoke)
    if args.stage in ['analyze', 'all'] and not args.smoke:
        analyze(args.output)


if __name__ == '__main__':
    main()
