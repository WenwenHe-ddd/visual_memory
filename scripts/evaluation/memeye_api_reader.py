"""Frozen-input MemEye reader ablation through an OpenAI-compatible API.

Only the answer model changes. Gold labels stay in a separate evaluation file;
the runtime transmits allowlisted prompts and original query image bytes only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / 'reports/all_master_answer_query_20261021/reader'
DEFAULT_OUTPUT = ROOT / 'results/memeye_reader_ablation_20261021/gpt-4o-mini'
DEFAULT_ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
CONDITIONS = ('baseline_question_only', 'master_answer_query')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()


def write_lines(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    tmp.replace(path)


def indexed(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate ' + key)
    return result


def parse_choice(raw):
    match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else None


def request_body(protocol, row):
    from scripts.evaluation.openai_chat_transport import build_request
    c = protocol['contract']
    return build_request(c['reader_system_prompt'], row['reader_text'], row['query_images'],
                         model=c['model'], max_tokens=c['max_tokens'], seed=c['seed'],
                         temperature=c['temperature'], image_detail=c['image_detail'])


def prepare(output=DEFAULT_OUTPUT, source=DEFAULT_SOURCE, env_file=DEFAULT_ENV,
            model='gpt-4o-mini', workers=3):
    from scripts.evaluation.memeye_reader_inputs import prepare_inputs
    from scripts.evaluation.openai_chat_transport import load_config
    output = Path(output).resolve()
    config = load_config(env_file, model=model)
    if workers < 1:
        raise ValueError('workers must be positive')
    metadata = prepare_inputs(output / 'inputs', source_reader=Path(source).resolve())
    source_hashes = {**metadata['source_sha256'],
                     **{str(output / 'inputs' / name): value for name, value in metadata['artifact_sha256'].items()}}
    for path in (Path(__file__), Path(__file__).with_name('openai_chat_transport.py'),
                 output / 'inputs/input_metadata.json'):
        source_hashes[str(path)] = sha(path)
    contract = {
        'dataset': 'MemEye', 'questions': 371, 'domains': 8,
        'logical_records': 742, 'unique_payloads': 719, 'image_questions': 8,
        'model': config.model, 'base_url': config.base_url,
        'api_config': config.public_dict(), 'api_env_file': str(Path(env_file).resolve()),
        'api': 'chat/completions', 'temperature': 0, 'seed': 42, 'max_tokens': 16,
        'image_detail': 'high', 'workers': workers, 'timeout_seconds': 60,
        'max_attempts': 3, 'retry_policy': 'Only transient transport/rate-limit/server errors; same request. No answer-format retries.',
        'reader_system_prompt': metadata['reader_system_prompt'],
        'qwen_reference_settings': metadata['reader_settings'],
        'answer_parser': r'(?:Answer\s*:\s*)?([ABCD])[.!]?',
        'source_sha256': source_hashes,
        'frozen_retrieval': 'Both full same-domain image-leaf rankings and ordered Top5; 438 image leaves total.',
        'changed_component': 'Final answer model and its native API/image processor only.',
        'image_policy': 'Original query images in original order; no history image pixels; no local resize.',
        'evidence_policy': 'Frozen original dialogue and full caption, question and fixed option order. No retrieval hypotheses.',
        'gold_in_api_payload': False, 'historical_api_answer_reuse': False,
        'deduplication': 'Identical model request/payload within this experiment shares one fresh API result.',
        'determinism': 'Temperature zero and seed42 requested; provider determinism is not guaranteed. Returned model and system fingerprint are recorded.',
        'interpretation': 'Fixed option-order paired MCQ accuracy, not four-rotation paper EM.',
    }
    provisional = {'contract': contract}
    records = lines(output / 'inputs/requests.jsonl')
    requests = []
    for r in records:
        body = request_body(provisional, r)
        requests.append({'record_id': r['record_id'], 'payload_sha256': r['payload_sha256'],
                         'request_sha256': digest(body), 'query_image_count': len(r['query_images'])})
    contract['request_manifest_sha256'] = digest(requests)
    signature = digest(contract)
    path = output / 'protocol.json'
    if path.exists():
        p = read(path)
        if p['experiment_signature'] != signature or p['contract'] != contract:
            raise ValueError('Existing API experiment differs; use a new --output directory')
        if lines(output / 'api_request_manifest.jsonl') != requests:
            raise ValueError('Prepared API request hashes changed')
        verify(output)
        return p
    if (output / 'fresh_responses.jsonl').exists():
        raise ValueError('Responses exist without their protocol')
    p = {'run_id': str(uuid.uuid4()), 'experiment_signature': signature,
         'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'contract': contract}
    write_lines(output / 'api_request_manifest.jsonl', requests)
    dump(path, p)
    print(json.dumps({'stage': 'prepared_api_reader', 'model': config.model,
                      'base_url': config.base_url, 'questions': 371, 'unique_requests': 719}), flush=True)
    return p


def verify(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    if digest(p['contract']) != p['experiment_signature']:
        raise ValueError('Protocol signature mismatch')
    for path, expected in p['contract']['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    request_manifest = lines(output / 'api_request_manifest.jsonl')
    if digest(request_manifest) != p['contract']['request_manifest_sha256']:
        raise ValueError('API request manifest changed')
    return p


def logical_responses(records, fresh):
    result = []
    for r in records:
        original = fresh.get(r['payload_sha256'])
        if original is None:
            continue
        is_fresh = r['record_id'] == original['record_id']
        item = {**original, 'record_id': r['record_id'], 'id': r['id'], 'task': r['task'],
                'condition': r['condition'], 'top_memory_ids': r['top_memory_ids'],
                'fresh_inference_executed_for_this_record': is_fresh,
                'same_run_shared_inference_record_id': None if is_fresh else original['record_id']}
        result.append(item)
    return result


def run(output=DEFAULT_OUTPUT, smoke=False):
    from scripts.evaluation.openai_chat_transport import APIError, chat_request, load_config
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
        if r['predicted_choice'] != parse_choice(r['raw_response']):
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
            if not (returned == c['model'] or re.fullmatch(re.escape(c['model']) + r'-\d{4}-\d{2}-\d{2}', returned)):
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
        report = {'passed': all(r['status'] == 'ok' and not r['hit_token_limit'] for r in selected_results),
                  'records': [{k: r[k] for k in ('record_id', 'query_image_count', 'status', 'returned_model',
                                                'finish_reason', 'request_sha256', 'response_id')} for r in selected_results],
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
                  'responses_sha256': sha(output / 'responses.jsonl'),
                  'fresh_responses_sha256': sha(output / 'fresh_responses.jsonl'),
                  'gold_loaded_during_run': False, 'historical_api_answer_reuse': False}
    dump(output / 'validation_output.json', validation)
    return validation


def aggregate(pairs):
    n = len(pairs)
    b = sum(r['baseline_correct'] for r in pairs)
    m = sum(r['expanded_correct'] for r in pairs)
    wins = sum(not r['baseline_correct'] and r['expanded_correct'] for r in pairs)
    losses = sum(r['baseline_correct'] and not r['expanded_correct'] for r in pairs)
    discordant = wins + losses
    p = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2 ** discordant) if discordant else 1.
    return {'questions': n, 'baseline_correct': b, 'expanded_correct': m,
            'baseline_accuracy': b / n if n else None, 'expanded_accuracy': m / n if n else None,
            'delta_percentage_points': 100 * (m - b) / n if n else None,
            'wrong_to_correct': wins, 'correct_to_wrong': losses, 'mcnemar_exact_p': p}


def analyze(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    protocol = verify(output)
    validation = read(output / 'validation_output.json')
    for name in ('responses', 'fresh_responses'):
        if sha(output / (name + '.jsonl')) != validation[name + '_sha256']:
            raise ValueError('Response output hash mismatch')
    requests = indexed(lines(output / 'inputs/requests.jsonl'), 'record_id')
    responses = indexed(lines(output / 'responses.jsonl'), 'record_id')
    fresh = indexed(lines(output / 'fresh_responses.jsonl'), 'payload_sha256')
    gold = indexed(lines(output / 'inputs/gold.jsonl'), 'id')
    wire = indexed(lines(output / 'api_request_manifest.jsonl'), 'record_id')
    if len(responses) != 742 or set(responses) != set(requests) or len(gold) != 371 or len(fresh) != 719:
        raise ValueError('Incomplete paired benchmark coverage')
    for rid, r in responses.items():
        q = requests[rid]
        original = fresh[r['payload_sha256']]
        if (r['run_id'] != protocol['run_id'] or r['experiment_signature'] != protocol['experiment_signature'] or
                r['payload_sha256'] != q['payload_sha256'] or r['request_sha256'] != wire[rid]['request_sha256'] or
                r['top_memory_ids'] != q['top_memory_ids'] or r['predicted_choice'] != parse_choice(r['raw_response'])):
            raise ValueError('Answer provenance mismatch')
        if not r['fresh_inference_executed_for_this_record']:
            if r['same_run_shared_inference_record_id'] != original['record_id']:
                raise ValueError('Shared inference source mismatch')
        for key in ('raw_response', 'response_id', 'request_sha256', 'execution_id', 'usage', 'returned_model'):
            if r[key] != original[key]:
                raise ValueError('Shared/fresh result differs: ' + key)
    pairs = []
    for p in read(output / 'inputs/pairs.json'):
        qid = p['question_id']
        b, m = [responses[qid + '::' + c] for c in CONDITIONS]
        g = gold[qid]['gold_choice']
        pairs.append({**p, 'gold_choice': g,
                      'baseline_answer': b['predicted_choice'], 'expanded_answer': m['predicted_choice'],
                      'baseline_raw': b['raw_response'], 'expanded_raw': m['raw_response'],
                      'baseline_correct': b['predicted_choice'] == g, 'expanded_correct': m['predicted_choice'] == g})
    reference = read(output / 'inputs/qwen_reference.json')
    qwen_records = indexed(reference['responses'], 'record_id')
    changes = {}
    for condition in CONDITIONS:
        improved = worsened = 0
        for rid, qwen in qwen_records.items():
            if qwen['condition'] != condition:
                continue
            api = responses[rid]
            if api['payload_sha256'] != qwen['payload_sha256']:
                raise ValueError('API input differs from Qwen reference')
            g = gold[requests[rid]['id']]['gold_choice']
            old, new = parse_choice(qwen['raw_response']) == g, api['predicted_choice'] == g
            improved += not old and new
            worsened += old and not new
        changes[condition] = {'qwen_wrong_api_correct': improved, 'qwen_correct_api_wrong': worsened}
    usage = {k: sum(int((r.get('usage') or {}).get(k, 0) or 0) for r in fresh.values())
             for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    summary = {'status': 'complete', 'model_requested': protocol['contract']['model'],
               'model_returned': validation['returned_models'], 'overall': aggregate(pairs),
               'by_domain': {v: aggregate([p for p in pairs if p['domain'] == v]) for v in sorted({p['domain'] for p in pairs})},
               'by_query_modality': {name: aggregate([p for p in pairs if p['query_has_image'] == value])
                                     for name, value in (('text_only', False), ('with_query_image', True))},
               'qwen_reference': reference['summary'], 'reader_model_changes': changes,
               'usage_fresh_requests_only': usage, 'validation': validation,
               'metric_policy': protocol['contract']['interpretation']}
    dump(output / 'summary.json', summary)
    dump(output / 'paired_results.json', pairs)
    with (output / 'paired_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    old = reference['summary']['overall']
    new = summary['overall']
    text = [
        '# MemEye 回答模型消融', '',
        '固定 371 道题、8 个领域、选项排列、原问题图片、两路 Top5、原始对话与完整 caption，只替换最终回答模型。', '',
        '| 回答模型 | 全量检索基线 | caption_visual_master | 方法增益（百分点） |',
        '|---|---:|---:|---:|',
        f'| Qwen3-VL-8B-Instruct（既有冻结运行） | {old["baseline_correct"]}/371 ({old["baseline_accuracy"]:.2%}) | {old["expanded_correct"]}/371 ({old["expanded_accuracy"]:.2%}) | {old["delta_percentage_points"]:+.2f} |',
        f'| {summary["model_returned"][0]}（本次 API） | {new["baseline_correct"]}/371 ({new["baseline_accuracy"]:.2%}) | {new["expanded_correct"]}/371 ({new["expanded_accuracy"]:.2%}) | {new["delta_percentage_points"]:+.2f} |', '',
        f'API 内新方法相对基线：错转对 {new["wrong_to_correct"]} 题，对转错 {new["correct_to_wrong"]} 题；McNemar 双侧精确 p={new["mcnemar_exact_p"]:.4g}。', '',
        '这是固定一次选项顺序的选择题准确率，不是论文四次选项轮换 EM。全量检索池沿用原实验的 438 条图片叶子，不扩展为全部纯文本历史轮次。', '',
        'API 请求使用原 system/user 文本与原始问题图，temperature=0、seed=42、max_tokens=16。API 原生图像预处理及 tokenizer 与 Qwen 不同；seed 不保证服务端确定性。没有重生成 caption 或检索假设，没有重做检索。', '',
        f'完整覆盖 742 条记录，其中 {validation["fresh_requests"]} 次新 API 回答、{validation["shared_records"]} 条本轮相同输入共享；格式错误 {validation["format_errors"]}，空回答 {validation["empty_responses"]}，输出达到上限 {validation["token_limit_records"]}。所有错误保留在 371 题分母中，不做回答格式修复重试。', '',
        '旧 Qwen 输入迁移造成的原路径/源码哈希差异在 inputs/input_audit.json 明确保留；冻结 payload、Top5、证据字段和问题原图逐条核验相同。旧协议和答案未改写。', '',
        '[逐题对照](paired_results.csv) · [完整指标](summary.json) · [实验协议](protocol.json)', '',
        '接口依据：[OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。', '',
    ]
    (output / 'comparison.md').write_text('\n'.join(text))
    print(json.dumps({'stage': 'api_reader_complete', 'model': summary['model_returned'],
                      'overall': new, 'validation': validation}, ensure_ascii=False), flush=True)
    return summary


def pipeline(args):
    output = Path(args.output)
    if args.stage in ('prepare', 'all'):
        prepare(output, args.api_source_reader, args.api_env_file, args.api_model, args.api_workers)
    if args.stage in ('run', 'all'):
        run(output, smoke=args.api_smoke)
    if args.stage in ('analyze', 'all') and not args.api_smoke:
        analyze(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run', 'analyze', 'all'), default='all')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--api-source-reader', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--api-env-file', type=Path, default=DEFAULT_ENV)
    parser.add_argument('--api-model', default='gpt-4o-mini')
    parser.add_argument('--api-workers', type=int, default=3)
    parser.add_argument('--api-smoke', action='store_true')
    pipeline(parser.parse_args())


if __name__ == '__main__':
    main()
