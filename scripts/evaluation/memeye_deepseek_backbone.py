"""Replace every Qwen role with the pinned DashScope DeepSeek vision backbone.

Each model gets fresh history/query captions, rebuilt memories and retrieval,
and fresh answers. The earlier frozen-retrieval reader experiments are separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.memeye_gpt5_reader import (
    aggregate, append, digest, dump, indexed, lines, parse_choice, read, sha, write_lines,
)

DEFAULT_ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
DEFAULT_ROOT = ROOT / 'results/memeye_full_backbone_20261021'
CONDITIONS = ('baseline_question_only', 'master_answer_query')


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen artifact changed: ' + str(path))
    else:
        dump(path, value)


def freeze_lines(path, rows):
    path = Path(path)
    if path.exists():
        if lines(path) != rows:
            raise ValueError('Frozen records changed: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_lines(path, rows)


def prepare(output, model, env_file=DEFAULT_ENV, workers=3):
    from scripts.data.memeye_backbone_inputs import prepare_inputs
    from scripts.evaluation.deepseek_backbone_transport import canonical_model, load_config
    if workers < 1:
        raise ValueError('workers must be positive')
    output = Path(output).resolve()
    model = canonical_model(model)
    config = load_config(env_file, model=model)
    prepare_inputs(output / 'inputs', model)
    sources = [output / 'inputs/protocol.json', Path(__file__),
               ROOT / 'scripts/data/memeye_backbone_inputs.py',
               ROOT / 'scripts/evaluation/deepseek_backbone_transport.py',
               ROOT / 'scripts/evaluation/openai_chat_transport.py',
               ROOT / 'scripts/evaluation/memeye_gpt5_reader.py',
               ROOT / 'scripts/memory/memeye_backbone_retrieval.py',
               ROOT / 'scripts/memory/memeye_backbone_long_text.py',
               ROOT / 'scripts/evaluation/memeye_backbone_completion.py',
               ROOT / 'scripts/visual_memory_pipeline.py',
               ROOT / 'scripts/graph/caption_ward_visual.py',
               ROOT / 'scripts/graph/visual_image_encoder.py',
               ROOT / 'scripts/memory/caption_visual_master.py',
               ROOT / 'scripts/analysis/prepare_all_domain_adaptive_inputs.py',
               ROOT / 'scripts/analysis/probe_caption_visual_online.py']
    contract = {
        'scope': 'all_original_qwen_roles', 'model': model,
        'api_config': config.public_dict(), 'api_env_file': str(Path(env_file).resolve()),
        'caption_max_output_tokens': 256, 'reader_max_output_tokens': 16,
        'image_detail': 'high', 'temperature': 0,
        'provider': 'DashScope International', 'enable_thinking': False,
        'web_search': False, 'reasoning_effort': None, 'seed': None,
        'workers': workers, 'max_attempts': 3, 'timeout_seconds': 60,
        'replaced_roles': ['history_image_caption', 'query_image_caption', 'answer_reader'],
        'rebuilt_dependencies': ['caption_embeddings', 'streaming_caption_clusters',
                                 'master_visual_sums', 'leaf_text_embeddings',
                                 'query_embeddings', 'both_retrieval_rankings'],
        'fixed_components': ['CLIP', 'BGE', 'external_2d_bases', 'Ward_partition_rule',
                             'assistant_authored_option_statements', 'reader_prompt',
                             'question_option_order', 'dialogue', 'arrival_order', 'top_k_5'],
        'caption_cache': 'fresh within this model/run, keyed by image SHA256; shared across history/query roles',
        'prior_qwen_caption_reuse': False, 'prior_qwen_retrieval_reuse': False,
        'prior_answer_reuse': False, 'gold_sent_to_api': False,
        'memory_policy': 'Master full visual aggregate plus two coordinates per leaf; no per-image full visual vectors or image copies saved in memory.',
        'source_sha256': {str(p): sha(p) for p in sources},
    }
    path = output / 'protocol.json'
    if path.exists():
        protocol = read(path)
        if protocol['contract'] != contract or protocol['experiment_signature'] != digest(contract):
            raise ValueError('Existing full-backbone experiment differs; use a fresh output directory')
    else:
        protocol = {'run_id': str(uuid.uuid4()), 'experiment_signature': digest(contract),
                    'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'contract': contract}
        dump(path, protocol)
    return verify(output)


def verify(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    if digest(p['contract']) != p['experiment_signature']:
        raise ValueError('Experiment signature mismatch')
    for path, expected in p['contract']['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen input or implementation changed: ' + path)
    # The input module owns its source/artifact verification and never reads
    # API secrets. Re-preparation only checks equality of already frozen files.
    from scripts.data.memeye_backbone_inputs import prepare_inputs
    prepare_inputs(output / 'inputs', p['contract']['model'])
    return p


def sanitized_payload(record):
    """The only fields allowed to reach the API transport."""
    result = {key: record[key] for key in ('user_text', 'query_images')}
    if record.get('system') is not None:
        result['system'] = record['system']
    return result


def verify_stage(output, stage, require_complete=False):
    """Validate saved stage results without transmitting anything."""
    output = Path(output)
    p = verify(output)
    c = p['contract']
    directory = output / stage
    meta = read(directory / 'protocol.json')
    if (meta['run_id'] != p['run_id'] or meta['experiment_signature'] != p['experiment_signature']
            or meta['model'] != c['model']):
        raise ValueError('Stage belongs to another experiment')
    for path, expected in meta['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Stage input changed: ' + path)
    if sha(directory / 'requests.jsonl') != meta['requests_sha256']:
        raise ValueError('Prepared requests changed')
    records = lines(directory / 'requests.jsonl')
    original = indexed(records, 'record_id')
    unique = {r['request_sha256'] for r in records}
    if len(records) != meta['logical_records'] or len(unique) != meta['unique_requests']:
        raise ValueError('Prepared stage coverage differs')
    for row in records:
        if digest(sanitized_payload(row)) != row['payload_sha256']:
            raise ValueError('Saved API payload hash differs')
    completed = lines(directory / 'fresh_responses.jsonl') if (directory / 'fresh_responses.jsonl').exists() else []
    fresh = indexed(completed, 'request_sha256')
    for answer in completed:
        row = original[answer['record_id']]
        if (answer['run_id'] != p['run_id'] or answer['experiment_signature'] != p['experiment_signature']
                or answer['request_sha256'] != row['request_sha256']
                or answer['payload_sha256'] != row['payload_sha256']
                or answer['returned_model'] != c['model']
                or answer['historical_response_reuse'] is not False):
            raise ValueError('Saved response provenance differs')
        valid = (answer['status'] == 'ok' and answer['finish_reason'] == 'stop'
                 and bool(answer['raw_response'].strip()) and not answer.get('refusal')
                 and isinstance(answer.get('response_id'), str) and bool(answer['response_id'].strip()))
        if stage == 'reader':
            valid = valid and parse_choice(answer['raw_response']) is not None
        if c['model'].startswith('deepseek-'):
            valid = (valid and answer.get('reasoning_tokens') in (None, 0)
                     and answer.get('reasoning_content_present') is False
                     and answer.get('enable_thinking') is False)
        if not valid:
            raise ValueError('Saved response failed validation')
    if require_complete and set(fresh) != unique:
        raise ValueError('Stage responses are incomplete')
    return records, fresh


def stage_requests(output, stage, records, source_paths):
    from scripts.evaluation.deepseek_backbone_transport import build_request
    output = Path(output)
    p = verify(output)
    c = p['contract']
    directory = output / stage
    directory.mkdir(parents=True, exist_ok=True)
    limit = c['caption_max_output_tokens'] if stage == 'caption_generation' else c['reader_max_output_tokens']
    prepared = []
    for row in records:
        payload = sanitized_payload(row)
        body = build_request(payload, model=c['model'], max_output_tokens=limit,
                             image_detail=c['image_detail'])
        prepared.append({**row, 'payload_sha256': digest(payload),
                         'request_sha256': digest(body)})
    indexed(prepared, 'record_id')
    freeze_lines(directory / 'requests.jsonl', prepared)
    metadata = {'stage': stage, 'run_id': p['run_id'],
                'experiment_signature': p['experiment_signature'], 'model': c['model'],
                'max_output_tokens': limit, 'logical_records': len(prepared),
                'unique_requests': len({r['request_sha256'] for r in prepared}),
                'requests_sha256': sha(directory / 'requests.jsonl'),
                'source_sha256': {str(Path(path).resolve()): sha(path) for path in source_paths}}
    freeze(directory / 'protocol.json', metadata)
    return metadata


def execute_stage(output, stage, limit=0):
    """Resume only this run's successes; bounded batches stop on provider errors."""
    from scripts.evaluation.deepseek_backbone_transport import APIError, build_request, chat_request, load_config
    output = Path(output)
    p = verify(output)
    c = p['contract']
    directory = output / stage
    meta = read(directory / 'protocol.json')
    if meta['run_id'] != p['run_id'] or meta['experiment_signature'] != p['experiment_signature']:
        raise ValueError('Stage belongs to another experiment')
    for path, expected in meta['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Stage input changed: ' + path)
    if sha(directory / 'requests.jsonl') != meta['requests_sha256']:
        raise ValueError('Prepared requests changed')
    records = lines(directory / 'requests.jsonl')
    original = indexed(records, 'record_id')
    unique = {}
    for row in records:
        unique.setdefault(row['request_sha256'], row)
    completed = lines(directory / 'fresh_responses.jsonl') if (directory / 'fresh_responses.jsonl').exists() else []
    fresh = indexed(completed, 'request_sha256')
    for answer in completed:
        row = original[answer['record_id']]
        if (answer['run_id'] != p['run_id'] or answer['experiment_signature'] != p['experiment_signature']
                or answer['request_sha256'] != row['request_sha256']
                or answer['payload_sha256'] != row['payload_sha256']
                or answer['returned_model'] != c['model']):
            raise ValueError('Saved response provenance differs')
        if answer['status'] != 'ok':
            raise ValueError('An earlier response failed validation; inspect this run before continuing')
    # Resume must apply the full validator, including disabled-thinking and
    # same-run provenance, before a new request can be submitted.
    verify_stage(output, stage)
    pending = [row for key, row in unique.items() if key not in fresh]
    if limit:
        pending = pending[:limit]
    config = load_config(c['api_env_file'], model=c['model'])
    if config.public_dict() != c['api_config']:
        raise ValueError('API configuration changed')
    execution_id = str(uuid.uuid4())

    def execute(row):
        body = build_request(sanitized_payload(row), model=c['model'],
                             max_output_tokens=meta['max_output_tokens'], image_detail=c['image_detail'])
        if digest(body) != row['request_sha256']:
            raise ValueError('Wire payload differs from frozen request')
        attempts = []
        for attempt in range(1, c['max_attempts'] + 1):
            a = {'record_id': row['record_id'], 'request_sha256': row['request_sha256'],
                 'execution_id': execution_id, 'attempt': attempt, 'stage': stage}
            try:
                response = chat_request(config, body, timeout=c['timeout_seconds'])
            except APIError as error:
                attempts.append({**a, 'status': 'api_error', 'kind': error.kind,
                                 'http_status': error.http_status, 'message': error.message})
                if not error.retryable or attempt == c['max_attempts']:
                    return None, attempts
                time.sleep(min(2 ** attempt, 30))
                continue
            valid = (response['returned_model'] == c['model'] and response['finish_reason'] == 'stop'
                     and bool(response['raw_response'].strip()) and not response.get('refusal')
                     and isinstance(response.get('response_id'), str) and bool(response['response_id'].strip()))
            if stage == 'reader':
                valid = valid and parse_choice(response['raw_response']) is not None
            if c['model'].startswith('deepseek-'):
                valid = (valid and response.get('reasoning_tokens') in (None, 0)
                         and response.get('reasoning_content_present') is False
                         and response.get('enable_thinking') is False)
            result = {**response, 'record_id': row['record_id'],
                      'payload_sha256': row['payload_sha256'], 'request_sha256': row['request_sha256'],
                      'run_id': p['run_id'], 'experiment_signature': p['experiment_signature'],
                      'execution_id': execution_id, 'attempts': attempt, 'retry_count': attempt - 1,
                      'status': 'ok' if valid else 'invalid_response', 'historical_response_reuse': False}
            attempts.append({**a, 'status': result['status'], 'response_id': result.get('response_id')})
            return result, attempts

    workers = c['workers']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(pending), workers):
            failed = []
            futures = {pool.submit(execute, row): row for row in pending[start:start + workers]}
            for future in as_completed(futures):
                row = futures[future]
                result, attempts = future.result()
                for attempt in attempts:
                    append(directory / 'api_attempts.jsonl', attempt)
                if result is not None:
                    append(directory / 'fresh_responses.jsonl', result)
                    fresh[row['request_sha256']] = result
                if result is None or result['status'] != 'ok':
                    failed.append(row['record_id'])
            progress = {'stage': stage, 'model': c['model'], 'completed_unique': len(fresh),
                        'expected_unique': len(unique), 'failed_this_batch': failed}
            dump(directory / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            if failed:
                raise RuntimeError('Generation failed validation; completed responses retained. See ' + str(directory))
    return len(fresh) == len(unique)


def generate_captions(output, limit=0):
    output = Path(output)
    source = output / 'inputs/api_requests_caption.jsonl'
    records = [{'record_id': r['id'], 'user_text': r['user_text'], 'query_images': r['query_images']}
               for r in lines(source)]
    stage_requests(output, 'caption_generation', records, [source])
    if not execute_stage(output, 'caption_generation', limit):
        return False
    requests = lines(output / 'caption_generation/requests.jsonl')
    fresh = indexed(lines(output / 'caption_generation/fresh_responses.jsonl'), 'request_sha256')
    captions = []
    for row in requests:
        result = fresh[row['request_sha256']]
        captions.append({'id': row['record_id'], 'caption': result['raw_response'].strip(),
                         'model': result['returned_model'], 'source_response_record_id': result['record_id'],
                         'request_sha256': row['request_sha256'],
                         'response_id': result['response_id']})
    freeze_lines(output / 'captions.jsonl', captions)
    return True


def verify_captions(output):
    output = Path(output)
    records, fresh = verify_stage(output, 'caption_generation', require_complete=True)
    captions = indexed(lines(output / 'captions.jsonl'), 'id')
    if set(captions) != {r['record_id'] for r in records}:
        raise ValueError('Caption coverage differs')
    for row in records:
        caption = captions[row['record_id']]
        result = fresh[row['request_sha256']]
        if (caption['caption'] != result['raw_response'].strip()
                or caption['model'] != result['returned_model']
                or caption['source_response_record_id'] != result['record_id']
                or caption['request_sha256'] != row['request_sha256']
                or caption['response_id'] != result['response_id']):
            raise ValueError('Caption artifact is not this run fresh API result')
    return captions


def verify_retrieval(output):
    output = Path(output)
    p = verify(output)
    protocol = read(output / 'retrieval_protocol.json')
    if (protocol['source_run_id'] != p['run_id']
            or protocol['source_experiment_signature'] != p['experiment_signature']):
        raise ValueError('Retrieval belongs to another experiment')
    for path, expected in {**protocol['source_sha256'], **protocol['output_sha256']}.items():
        if sha(path) != expected:
            raise ValueError('Retrieval provenance changed: ' + path)
    return protocol


def prepare_reader(output):
    output = Path(output)
    p = verify(output)
    verify_retrieval(output)
    questions = indexed(lines(output / 'inputs/source_questions.jsonl'), 'id')
    captions = verify_captions(output)
    memories_by_id = indexed(lines(output / 'inputs/source_memories.jsonl'), 'id')
    expected_texts = {mid: m['dialogue'] + '\nMemory description: ' + captions[m['image_sha256']]['caption']
                      for mid, m in memories_by_id.items()}
    retrieval = lines(output / 'retrieval_results.jsonl')
    if set(questions) != {r['question_id'] for r in retrieval} or len(retrieval) != len(questions):
        raise ValueError('Retrieval/question coverage mismatch')
    source_protocol = ROOT / 'data/independent_visual_inputs/protocol.json'
    system = read(source_protocol)['reader_system_prompt']
    records, manifest = [], []
    for index, result in enumerate(retrieval):
        q = questions[result['question_id']]
        if (result['question'] != q['question'] or result['options'] != q['options']
                or result['query_image_paths'] != [im['path'] for im in q['query_images']]):
            raise ValueError('Retrieval question fields differ from the prepared benchmark')
        conditions = indexed(result['conditions'], 'name')
        if set(conditions) != set(CONDITIONS):
            raise ValueError('Condition coverage differs')
        pair = []
        for condition in CONDITIONS:
            row = conditions[condition]
            ids = row['top5_memory_ids']
            memories = row['reader_memories']
            if len(ids) != len(set(ids)) or len(ids) != 5 or [r['memory_id'] for r in memories] != ids:
                raise ValueError('Invalid Top5 reader memory list')
            if any(memories_by_id[m['memory_id']]['task'] != q['task']
                   or m['reader_text'] != expected_texts[m['memory_id']] for m in memories):
                raise ValueError('Reader evidence differs from this backbone fresh captions and source dialogue')
            context = '\n\n'.join(f'Memory {i}: {m["reader_text"]}' for i, m in enumerate(memories, 1))
            if context != row['reader_context']:
                raise ValueError('Reader context differs from selected memories')
            text = ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + context
                    + '\n\nQuestion:\n' + q['question'] + '\n\nOptions:\n'
                    + '\n'.join(f'{k}. {q["options"][k]}' for k in 'ABCD')
                    + '\n\nAnswer with one letter only.')
            record_id = q['id'] + '::' + condition
            pair.append({'record_id': record_id, 'system': system, 'user_text': text,
                         'query_images': q['query_images']})
            manifest.append({'record_id': record_id, 'question_id': q['id'], 'task': q['task'],
                             'condition': condition, 'top_memory_ids': ids})
        records.extend(pair if index % 2 == 0 else reversed(pair))
    freeze_lines(output / 'reader_manifest.jsonl', manifest)
    stage_requests(output, 'reader', records, [output / 'retrieval_results.jsonl',
                                             output / 'retrieval_protocol.json', output / 'captions.jsonl',
                                             output / 'reader_manifest.jsonl',
                                             output / 'inputs/source_questions.jsonl', source_protocol])
    return p


def retrieval_metrics(rows, evaluation):
    total_gold = sum(len(set(evaluation[r['question_id']]['gold_ids'])) for r in rows)
    output = {'questions': len(rows), 'gold_references': total_gold}
    for name in CONDITIONS:
        hits, recalls, any_hits, all_hits = 0, [], 0, 0
        for row in rows:
            gold = set(evaluation[row['question_id']]['gold_ids'])
            condition = next(c for c in row['conditions'] if c['name'] == name)
            top = condition['top5_memory_ids']
            if not gold or len(top) != len(set(top)) or len(top) != 5:
                raise ValueError('Recall requires nonempty gold and five distinct selected memories')
            n = len(gold & set(top))
            hits += n
            recalls.append(n / len(gold))
            any_hits += n > 0
            all_hits += n == len(gold)
        output[name] = {'gold_hits_top5': hits, 'gold_recall': hits / total_gold,
                        'macro_recall_at_5': sum(recalls) / len(recalls),
                        'questions_with_gold_hit': any_hits, 'questions_with_all_gold': all_hits}
    return output


def analyze(output, require_reader=True):
    output = Path(output)
    p = verify(output)
    verify_retrieval(output)
    verify_captions(output)
    rows = lines(output / 'retrieval_results.jsonl')
    evaluation = indexed(lines(output / 'inputs/evaluation.jsonl'), 'id')
    questions = indexed(lines(output / 'inputs/source_questions.jsonl'), 'id')
    if set(evaluation) != {r['question_id'] for r in rows} or len(rows) != len(evaluation):
        raise ValueError('Evaluation coverage differs')
    domains = sorted({q['task'] for q in questions.values()})
    recall = {'overall': retrieval_metrics(rows, evaluation), 'by_domain': {
        task: retrieval_metrics([r for r in rows if questions[r['question_id']]['task'] == task], evaluation)
        for task in domains}}
    dump(output / 'retrieval_summary.json', recall)
    if not (output / 'reader/fresh_responses.jsonl').exists():
        if require_reader:
            raise ValueError('Reader results are not available')
        return recall
    saved_requests, fresh = verify_stage(output, 'reader', require_complete=require_reader)
    verify_stage(output, 'caption_generation', require_complete=True)
    requests = indexed(saved_requests, 'record_id')
    if set(fresh) != {r['request_sha256'] for r in requests.values()}:
        if require_reader:
            raise ValueError('Reader responses are incomplete')
        return recall
    manifest = indexed(lines(output / 'reader_manifest.jsonl'), 'record_id')
    pairs = []
    for qid, question in questions.items():
        ids = [qid + '::' + name for name in CONDITIONS]
        responses = [fresh[requests[rid]['request_sha256']] for rid in ids]
        if any(r['status'] != 'ok' or r['returned_model'] != p['contract']['model'] for r in responses):
            raise ValueError('Invalid reader response')
        choices = [parse_choice(r['raw_response']) for r in responses]
        gold = evaluation[qid]['gold_choice']
        pairs.append({'question_id': qid, 'domain': question['task'],
                      'baseline_answer': choices[0], 'expanded_answer': choices[1],
                      'baseline_correct': choices[0] == gold, 'expanded_correct': choices[1] == gold,
                      'same_reader_payload': requests[ids[0]]['payload_sha256'] == requests[ids[1]]['payload_sha256'],
                      'same_top5_order': manifest[ids[0]]['top_memory_ids'] == manifest[ids[1]]['top_memory_ids']})
    write_lines(output / 'paired_results.jsonl', pairs)
    stages = {}
    for stage in ('caption_generation', 'reader'):
        responses = lines(output / stage / 'fresh_responses.jsonl')
        stages[stage] = {'fresh_requests': len(responses),
                         'returned_models': dict(Counter(r['returned_model'] for r in responses)),
                         'errors': sum(r['status'] != 'ok' for r in responses),
                         'refusals': sum(bool(r.get('refusal')) for r in responses),
                         'length_stops': sum(r['finish_reason'] == 'length' for r in responses),
                         'retry_count': sum(r['retry_count'] for r in responses),
                         'usage': {key: sum((r.get('usage') or {}).get(key, 0) for r in responses)
                                   for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}}
    summary = {'scope': 'all_original_qwen_roles', 'model': p['contract']['model'],
               'run_id': p['run_id'], 'overall': aggregate(pairs),
               'by_domain': {task: aggregate([p for p in pairs if p['domain'] == task]) for task in domains},
               'recall_at_5': recall, 'generation_stages': stages,
               'logical_reader_records': len(requests),
               'same_run_shared_reader_records': len(requests) - len(fresh),
               'historical_qwen_caption_or_answer_reuse': False}
    dump(output / 'summary.json', summary)
    table = ['# MemEye full backbone replacement', '',
             f'Model: `{summary["model"]}`. All history/query captions, caption-dependent memory/retrieval, and answers were regenerated.', '',
             '| Condition | Recall@5 (micro) | Recall@5 (macro) | Answer accuracy |',
             '|---|---:|---:|---:|']
    for name, field in zip(CONDITIONS, ('baseline', 'expanded')):
        r = recall['overall'][name]
        table.append(f'| {name} | {r["gold_recall"]:.2%} ({r["gold_hits_top5"]}/{recall["overall"]["gold_references"]}) | '
                     f'{r["macro_recall_at_5"]:.2%} | {summary["overall"][field + "_accuracy"]:.2%} '
                     f'({summary["overall"][field + "_correct"]}/{len(pairs)}) |')
    table += ['', 'CLIP, BGE, fixed external 2D bases, clustering/retrieval algorithms, question/option order and assistant-authored candidate statements are held constant.',
              'These are fixed-option-order results. Earlier API experiments replaced only the reader and are not full-backbone results.', '']
    (output / 'comparison.md').write_text('\n'.join(table))
    return summary


def pipeline(args):
    from scripts.evaluation.deepseek_backbone_transport import canonical_model
    if args.api_workers < 1 or args.api_limit < 0:
        raise ValueError('workers must be positive and request limit must be nonnegative')
    model = canonical_model(args.api_model)
    output = Path(args.output or DEFAULT_ROOT / model).resolve()
    stage = args.stage
    if stage in ('prepare', 'all'):
        prepare(output, model, args.api_env_file, args.api_workers)
    elif verify(output)['contract']['model'] != model:
        raise ValueError('Requested backbone differs from prepared experiment')
    if stage in ('caption', 'all'):
        if not generate_captions(output, args.api_limit):
            return
    if stage in ('retrieve', 'all'):
        verify_captions(output)
        if (output / 'retrieval_protocol.json').exists():
            verify_retrieval(output)
        else:
            from scripts.memory.memeye_backbone_retrieval import rebuild_and_retrieve
            overflow = None
            try:
                rebuild_and_retrieve(output, device=args.device)
            except ValueError as error:
                if ' exceeds BGE token limit ' not in str(error):
                    raise
                overflow = str(error)
            if overflow:
                freeze(output / 'long_text_transition.json', {
                    'reason': overflow,
                    'action': 'Restore the original BGE 512-token encoding limit with an explicit truncation audit; keep full reader evidence.',
                    'caption_generation_changed': False, 'memory_construction_changed': False,
                    'reader_evidence_changed': False,
                })
                from scripts.memory.memeye_backbone_long_text import complete_retrieval
                complete_retrieval(output, device=args.device)
        analyze(output, require_reader=False)
    if stage in ('reader', 'all'):
        prepare_reader(output)
        if not execute_stage(output, 'reader', args.api_limit):
            return
    if stage in ('analyze', 'all'):
        print(json.dumps(analyze(output), ensure_ascii=False, indent=2), flush=True)


def add_arguments(parser):
    parser.add_argument('--api-model', default='deepseek-v4.1-flash-1017')
    parser.add_argument('--api-env-file', type=Path, default=DEFAULT_ENV)
    parser.add_argument('--api-workers', type=int, default=3)
    parser.add_argument('--api-limit', type=int, default=0,
                        help='Smoke test: maximum new requests in this stage; zero means all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', choices=['prepare', 'caption', 'retrieve', 'reader', 'analyze', 'all'], default='prepare')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    pipeline(parser.parse_args())
