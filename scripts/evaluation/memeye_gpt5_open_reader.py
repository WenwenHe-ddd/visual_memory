"""Audited GPT-5 readers for frozen, gold-free MemEye Open evidence manifests."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import mem_gallery_master_reader as common
from scripts.evaluation import clip_image_dialogue_api as api
from scripts.evaluation import gpt5_text_memory_transport as text_api
from scripts.evaluation import openai_gpt5_base_transport as gpt

MODEL = 'gpt-5'
REVISION = 'gpt-5-2025-08-07'
CANONICAL = ROOT / 'results/memeye_qwen_semrag_open_20261105'
RUN = ROOT / 'results/memeye_gpt5_open_20261105'
sha, digest, rows, dump = common.sha, common.digest, common.read_lines, common.dump


def read(path):
    return json.loads(Path(path).read_text())


def append(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def write_frozen(path, records):
    value = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records)
    if path.exists() and path.read_text() != value:
        raise ValueError('Frozen artifact differs: ' + str(path))
    path.write_text(value)


def messages(row):
    if 'reader_messages' in row:
        result = row['reader_messages']
    else:
        images = [{'type': 'image', 'path': p, 'sha256': sha(p)} for p in row['query_image_paths']]
        result = [{'role': 'system', 'content': row['system']},
                  {'role': 'user', 'content': images + [{'type': 'text', 'text': row['user_text']}]}]
    if not isinstance(result, list) or not result:
        raise ValueError('Reader requires explicit messages')
    for message in result:
        if set(message) != {'role', 'content'} or message['role'] not in {'system', 'user', 'assistant'}:
            raise ValueError('Unexpected reader role or metadata')
        content = message['content']
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or message['role'] != 'user':
            raise ValueError('Typed content is allowed only in user messages')
        for part in content:
            if part.get('type') == 'text':
                if set(part) != {'type', 'text'} or not isinstance(part['text'], str):
                    raise ValueError('Invalid text content')
            elif part.get('type') == 'image':
                if set(part) != {'type', 'path', 'sha256'} or sha(part['path']) != part['sha256']:
                    raise ValueError('Original image bytes or metadata differ')
            else:
                raise ValueError('Unrecognized content type')
    return result


def build_request(row):
    value = messages(row)
    if all(isinstance(m['content'], str) for m in value):
        return text_api.build_request(value, 2048, 0)
    if len(value) != 2 or value[0]['role'] != 'system' or value[1]['role'] != 'user':
        raise ValueError('Multimodal reader must preserve source system/user roles')
    import base64
    content = []
    for part in value[1]['content']:
        if part['type'] == 'text':
            content.append(dict(part))
        else:
            data = Path(part['path']).read_bytes()
            if data.startswith(b'\x89PNG\r\n\x1a\n'):
                mime = 'image/png'
            elif data.startswith(b'\xff\xd8\xff'):
                mime = 'image/jpeg'
            else:
                raise ValueError('Only unchanged PNG/JPEG bytes are supported')
            content.append({'type': 'image_url', 'image_url': {'detail': 'high',
                'url': 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')}})
    return {**api.settings(MODEL), 'messages': [dict(value[0]), {'role': 'user', 'content': content}]}


def code_files():
    return [Path(__file__), Path(common.__file__), Path(api.__file__), Path(text_api.__file__),
            Path(gpt.__file__), ROOT / 'scripts/evaluation/openai_backbone_transport.py',
            ROOT / 'scripts/evaluation/openai_chat_transport.py']


def prepare(args):
    source = args.root / 'reader_manifest.jsonl'
    manifest = rows(source)
    canonical = common.by_id(rows(CANONICAL / 'inputs/questions.jsonl'))
    if len(manifest) != 371 or set(common.by_id(manifest)) != set(canonical):
        raise ValueError('Expected all 371 unique canonical Open questions')
    images = {}
    for row in manifest:
        if any(row[k] != canonical[row['id']][k] for k in ('question', 'task', 'point')):
            raise ValueError('Open question metadata changed')
        if set(row) & {'options', 'gold_choice', 'ground_truth', 'answer', 'clue', 'gold_ids'}:
            raise ValueError('Evaluation-only field in reader input')
        for message in messages(row):
            if isinstance(message['content'], list):
                for part in message['content']:
                    if part['type'] == 'image':
                        images[part['path']] = part['sha256']
    config = gpt.load_config(args.env_file)
    args.cache.mkdir(parents=True, exist_ok=True)
    shared_contract = {'answer_model': MODEL, 'expected_returned_model': REVISION,
        'settings': api.settings(MODEL), 'api': config.public_dict(),
        'source_sha256': {str(p.resolve()): sha(p) for p in code_files()},
        'cache_policy': 'Exact wire SHA256, per-request process lock, durable raw attempts and output; no answer-quality retries.'}
    with (args.cache / 'protocol.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cache_protocol = common.ensure_protocol(args.cache / 'protocol.json', shared_contract)
    files = [source, args.root / 'protocol.json', CANONICAL / 'inputs/questions.jsonl',
             args.cache / 'protocol.json', *code_files()]
    contract = {'dataset': 'MemEye Open', 'method_id': args.method_id, 'answer_model': MODEL,
        'expected_returned_model': REVISION, 'questions': 371,
        'reference_mcq_correct': args.reference_correct,
        'settings': api.settings(MODEL), 'image_policy': 'Original PNG/JPEG, high, no resizing, native message order.',
        'source_sha256': {str(p.resolve()): sha(p) for p in files}, 'image_sha256': images,
        'gold_sha256': sha(CANONICAL / 'inputs/gold.jsonl'), 'gold_parsed_during_generation': False,
        'cache_root': str(args.cache.resolve()), 'cache_signature': cache_protocol['experiment_signature'],
        'raw_answer_policy': 'Retain empty, refused and token-limited returned answers without quality retries.',
        'retry_policy': 'At most four attempts per wire, transient API errors only; no missing-answer imputation.'}
    out = args.root / 'reader'
    out.mkdir(exist_ok=True)
    protocol = common.ensure_protocol(out / 'protocol.json', contract)
    return protocol, manifest, config


def validate_output(output, key):
    if (output.get('returned_model') != REVISION or output.get('requested_model') != MODEL
            or output.get('request_sha256') != key or not isinstance(output.get('raw_response'), str)
            or not output.get('response_id') or output.get('seed') != 42
            or output.get('reasoning_effort') != 'minimal'):
        raise ValueError('GPT output identity or wire metadata differs')
    if output.get('hit_token_limit') != (output.get('finish_reason') == 'length'):
        raise ValueError('Output token-limit flag differs')


def cached_output(cache, request):
    key = digest(request)
    directory = cache / 'calls' / key
    output = read(directory / 'response.json')
    wire = json.loads(gzip.decompress((directory / 'wire.json.gz').read_bytes()))
    if wire != request or digest(wire) != key:
        raise ValueError('Saved exact wire changed')
    validate_output(output, key)
    attempts = rows(directory / 'attempts.jsonl')
    if [a['attempt'] for a in attempts] != list(range(1, len(attempts) + 1)):
        raise ValueError('Attempt sequence changed')
    successful = [a for a in attempts if a['status'] == 'ok']
    if len(successful) != 1 or successful[0]['response'] != output:
        raise ValueError('Original GPT output lacks one successful raw attempt')
    return output


def call(config, cache, request):
    key = digest(request)
    directory = cache / 'calls' / key
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'call.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (directory / 'response.json').exists():
            return cached_output(cache, request)
        wire_path = directory / 'wire.json.gz'
        wire = gzip.compress(json.dumps(request, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode(), mtime=0)
        if wire_path.exists() and wire_path.read_bytes() != wire:
            raise ValueError('Request cache hash collision')
        wire_path.write_bytes(wire)
        attempt_path = directory / 'attempts.jsonl'
        previous = rows(attempt_path) if attempt_path.exists() else []
        if previous and previous[-1]['status'] == 'ok':
            dump(directory / 'response.json', previous[-1]['response'])
            return cached_output(cache, request)
        if previous and (len(previous) >= 4 or not previous[-1].get('retryable')):
            raise RuntimeError('Saved GPT failure exhausted the bounded retry policy: ' + key)
        for number in range(len(previous) + 1, 5):
            event = {'request_sha256': key, 'attempt': number, 'at_unix': time.time()}
            try:
                response = api.chat(config, request, timeout=120)
                usage = response.get('usage') or {}
                response.update(requested_model=MODEL, request_sha256=key, seed=42,
                    reasoning_effort='minimal', input_tokens=usage.get('prompt_tokens', 0),
                    generated_tokens=usage.get('completion_tokens', 0),
                    hit_token_limit=response['finish_reason'] == 'length')
                # Record all received outputs before rejecting a model revision drift.
                event.update(status='ok' if response['returned_model'] == REVISION else 'model_mismatch',
                             response=response, retryable=False)
                append(attempt_path, event)
                validate_output(response, key)
                dump(directory / 'response.json', response)
                return cached_output(cache, request)
            except api.APIError as error:
                event.update(status='error', retryable=error.retryable, error_kind=error.kind,
                    http_status=error.http_status, code=error.code)
                append(attempt_path, event)
                if not error.retryable or number == 4:
                    raise
                time.sleep(min(2 ** number, 16))


def validate(root, protocol, manifest):
    contract = protocol['contract']
    if digest(contract) != protocol['experiment_signature']:
        raise ValueError('Reader protocol signature differs')
    for field in ('source_sha256', 'image_sha256'):
        for path, expected in contract[field].items():
            if sha(path) != expected:
                raise ValueError('Frozen reader source changed: ' + path)
    source = common.by_id(manifest)
    path = root / 'reader/responses.jsonl'
    saved = rows(path) if path.exists() else []
    common.by_id(saved)
    for row in saved:
        if row['id'] not in source or row['experiment_signature'] != protocol['experiment_signature']:
            raise ValueError('Response belongs to another experiment')
        request = build_request(source[row['id']])
        output = cached_output(Path(contract['cache_root']), request)
        if any(row.get(k) != v for k, v in output.items()):
            raise ValueError('Answer differs from its original raw API output')
    return saved


def run(args):
    protocol, manifest, config = prepare(args)
    existing = validate(args.root, protocol, manifest)
    done = {r['id'] for r in existing}
    pending = [r for r in manifest if r['id'] not in done]
    if args.limit:
        pending = pending[:args.limit]
    io_lock = threading.Lock()
    stop = threading.Event()
    def one(row):
        if stop.is_set():
            return
        try:
            output = call(config, args.cache, build_request(row))
        except Exception:
            stop.set()
            raise
        record = {**output, **{k: row[k] for k in ('id', 'task', 'point')},
            'method_id': args.method_id, 'answer_model': MODEL,
            'top_memory_ids': row.get('top_memory_ids', []),
            'experiment_signature': protocol['experiment_signature'], 'run_id': protocol['run_id']}
        with io_lock:
            append(args.root / 'reader/responses.jsonl', record)
            done.add(row['id'])
            if len(done) % 10 == 0 or len(done) == 371:
                print(json.dumps({'method': args.method_id, 'completed': len(done), 'total': 371}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one, row) for row in pending]):
            future.result()
    saved = validate(args.root, protocol, manifest)
    status = {'status': 'complete' if len(saved) == 371 else 'partial', 'questions': 371,
        'questions_answered': len(saved), 'responses_sha256': sha(args.root / 'reader/responses.jsonl'),
        'protocol_sha256': sha(args.root / 'reader/protocol.json'),
        'token_limit_hits': sum(r['hit_token_limit'] for r in saved),
        'empty_answers': sum(not r['raw_response'] for r in saved)}
    dump(args.root / 'reader/completion.json', status)
    print(json.dumps(status), flush=True)


def collect(args):
    protocol = read(args.root / 'reader/protocol.json')
    manifest = rows(args.root / 'reader_manifest.jsonl')
    saved = common.by_id(validate(args.root, protocol, manifest))
    seal = read(args.root / 'reader/completion.json')
    if seal['status'] != 'complete' or len(saved) != 371 or seal['responses_sha256'] != sha(args.root / 'reader/responses.jsonl'):
        raise ValueError('Require all 371 sealed original answers')
    gold_path = CANONICAL / 'inputs/gold.jsonl'
    if sha(gold_path) != protocol['contract']['gold_sha256']:
        raise ValueError('Canonical gold changed')
    gold = common.by_id(rows(gold_path))
    records = []
    for q in manifest:
        response = saved[q['id']]
        records.append({**{k: q[k] for k in ('id', 'task', 'point', 'question')},
            'ground_truth': gold[q['id']]['answer'], 'model_output': response['raw_response'],
            'answer_model': MODEL, 'answer_model_revision': response['returned_model'],
            'source_record_id': q['id'], 'source_hit_token_limit': response['hit_token_limit'],
            'source_responses_sha256': seal['responses_sha256'],
            'source_api_request_sha256': response['request_sha256']})
    write_frozen(args.root / 'judge_inputs.jsonl', records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('run', 'collect'), required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--method-id', required=True)
    p.add_argument('--reference-correct', type=int, required=True)
    p.add_argument('--cache', type=Path, default=RUN / 'answer_api_cache')
    p.add_argument('--env-file', type=Path, default=Path('/storage/hewenwen/NTU Decentralized Agent/.env'))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    if args.workers < 1 or args.limit < 0:
        p.error('workers must be positive; limit nonnegative')
    args.root, args.cache = args.root.resolve(), args.cache.resolve()
    with (args.root / 'reader.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run if args.phase == 'run' else collect)(args)


if __name__ == '__main__':
    main()
