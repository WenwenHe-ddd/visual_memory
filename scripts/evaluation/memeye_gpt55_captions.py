"""Generate fresh GPT-5.5 captions with a frozen provenance chain.

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
DEFAULT_OUTPUT = ROOT / 'results/memeye_gpt55_seed42_full_20261024'
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


def prepare(output=DEFAULT_OUTPUT, model='gpt-5.5', env_file=DEFAULT_ENV, workers=3):
    from scripts.data.memeye_backbone_inputs import prepare_inputs
    from scripts.evaluation.openai_gpt55_transport import load_config, valid_returned_model
    assert model == 'gpt-5.5' and workers > 0
    output = Path(output).resolve()
    probe = read(output / 'availability_probe.json')
    assert probe['passed'] and probe['request']['model'] == model and probe['request']['seed'] == 42
    actual_model = probe['response']['returned_model']
    assert valid_returned_model(actual_model)
    config = load_config(env_file, model=model)
    inputs = prepare_inputs(output / 'inputs', actual_model)
    # Preserve all caption-free controls from the preceding GPT-5.4 experiment.
    reference = ROOT/'results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05'
    for name in ('source_memories.jsonl', 'source_questions.jsonl', 'evaluation.jsonl', 'api_requests_caption.jsonl'):
        assert sha(output/'inputs'/name) == sha(reference/'inputs'/name), name
    sources = [output/'inputs/protocol.json', output/'availability_probe.json', Path(__file__),
        ROOT/'scripts/data/memeye_backbone_inputs.py',
        ROOT/'scripts/evaluation/openai_gpt55_transport.py',
        ROOT/'scripts/evaluation/openai_backbone_transport.py',
        ROOT/'scripts/evaluation/openai_chat_transport.py',
        ROOT/'scripts/evaluation/openai_gpt54_seeded_transport.py',
        ROOT/'scripts/evaluation/memeye_gpt5_reader.py',
        ROOT/'scripts/memory/memeye_backbone_retrieval.py',
        ROOT/'scripts/memory/memeye_backbone_long_text.py',
        ROOT/'scripts/graph/caption_ward_visual.py',
        ROOT/'scripts/graph/visual_image_encoder.py',
        ROOT/'scripts/memory/caption_visual_master.py',
        ROOT/'scripts/analysis/prepare_all_domain_adaptive_inputs.py',
        ROOT/'scripts/analysis/probe_caption_visual_online.py']
    contract = {
        'scope': 'fresh_gpt55_caption_memory_retrieval_and_answers',
        'model': actual_model, 'requested_model': model,
        'api_config': config.public_dict(), 'api_env_file': str(Path(env_file).resolve()),
        'caption_max_output_tokens': 256, 'reader_max_output_tokens': 16,
        'image_detail': 'high', 'temperature': 0, 'reasoning_effort': 'none',
        'seed': None, 'caption_seed': None, 'reader_seed': 42,
        'workers': workers, 'max_attempts': 3, 'timeout_seconds': 60,
        'lambda': .01, 'alpha': 1., 'top_k': 5,
        'conditions': list(CONDITIONS)+[n+'_zscore' for n in CONDITIONS],
        'replaced_roles': ['history_image_caption', 'query_image_caption', 'answer_reader'],
        'fixed_components': ['dialogue', 'question_option_order', 'original_images', 'caption_prompt',
            'reader_prompt', 'CLIP', 'BGE', 'external_2d_bases', 'Ward_partition_rule',
            'assistant_authored_option_statements', 'arrival_order', 'top_k_5'],
        'caption_cache': 'Fresh within this run, keyed by image SHA256; shared between history/query.',
        'prior_caption_reuse': False, 'prior_answer_reuse': False, 'gold_sent_to_api': False,
        'caption_free_inputs_equal_gpt54': True,
        'dataset': 'Public MemEyeBench/MemEye; 371 questions, 438 leaves, 437 unique images',
        'public_input_provenance': inputs['history_source_policy'],
        'source_sha256': {str(p):sha(p) for p in sources},
    }
    path = output/'protocol.json'
    if path.exists():
        existing = read(path)
        assert existing['contract'] == contract and existing['experiment_signature'] == digest(contract)
    else:
        dump(path, {'run_id':str(uuid.uuid4()), 'experiment_signature':digest(contract),
            'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), 'contract':contract})
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
                or answer['requested_model'] != c['requested_model']
                or answer['returned_model'] != c['model']
                or answer['historical_response_reuse'] is not False):
            raise ValueError('Saved response provenance differs')
        valid = (answer['status'] == 'ok' and answer['finish_reason'] == 'stop'
                 and bool(answer['raw_response'].strip()) and not answer.get('refusal'))
        if stage == 'reader':
            valid = valid and parse_choice(answer['raw_response']) is not None
        if c['model'].startswith('gpt-5'):
            valid = valid and answer.get('reasoning_tokens') in (None, 0)
        if not valid:
            raise ValueError('Saved response failed validation')
    if require_complete and set(fresh) != unique:
        raise ValueError('Stage responses are incomplete')
    return records, fresh


def stage_requests(output, stage, records, source_paths):
    from scripts.evaluation.openai_gpt55_transport import build_request
    output = Path(output)
    p = verify(output)
    c = p['contract']
    directory = output / stage
    directory.mkdir(parents=True, exist_ok=True)
    limit = c['caption_max_output_tokens'] if stage == 'caption_generation' else c['reader_max_output_tokens']
    prepared = []
    for row in records:
        payload = sanitized_payload(row)
        body = build_request(payload, seed=None if stage == 'caption_generation' else c['reader_seed'], max_output_tokens=limit,
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
    from scripts.evaluation.openai_gpt55_transport import APIError, build_request, chat_request, load_config
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
                or answer['requested_model'] != c['requested_model']
                or answer['returned_model'] != c['model']):
            raise ValueError('Saved response provenance differs')
        if answer['status'] != 'ok':
            raise ValueError('An earlier response failed validation; inspect this run before continuing')
    pending = [row for key, row in unique.items() if key not in fresh]
    if limit:
        pending = pending[:limit]
    config = load_config(c['api_env_file'], model=c['requested_model'])
    if config.public_dict() != c['api_config']:
        raise ValueError('API configuration changed')
    execution_id = str(uuid.uuid4())

    def execute(row):
        body = build_request(sanitized_payload(row), seed=None if stage == 'caption_generation' else c['reader_seed'],
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
            valid = (response['returned_model'] == c['model'] and response['requested_model'] == c['requested_model'] and response['finish_reason'] == 'stop'
                     and bool(response['raw_response'].strip()) and not response.get('refusal'))
            if stage == 'reader':
                valid = valid and parse_choice(response['raw_response']) is not None
            if c['model'].startswith('gpt-5'):
                valid = valid and response.get('reasoning_tokens') in (None, 0)
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
                         'request_sha256': row['request_sha256']})
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
                or caption['request_sha256'] != row['request_sha256']):
            raise ValueError('Caption artifact is not this run fresh API result')
    return captions



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--stage', choices=['prepare','caption','all'], default='all')
    parser.add_argument('--api-env-file', type=Path, default=DEFAULT_ENV)
    parser.add_argument('--api-workers', type=int, default=3)
    parser.add_argument('--api-limit', type=int, default=0)
    args=parser.parse_args()
    if args.stage in ('prepare','all'):
        prepare(args.output, env_file=args.api_env_file, workers=args.api_workers)
    if args.stage in ('caption','all'):
        complete=generate_captions(args.output,args.api_limit)
        print(json.dumps({'stage':'caption','complete':complete,'model':verify(args.output)['contract']['model']}),flush=True)
        if complete:
            verify_captions(args.output)


if __name__=='__main__':
    main()
