"""Audited fresh DeepSeek Flash hypotheses for answer-free MemEye Open retrieval."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from scripts.evaluation import mem_gallery_master_reader as common
from scripts.evaluation import deepseek_flash_seed42_audited_transport as transport
from scripts.evaluation import memeye_deepseek_open_caption_inputs as inputs

ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
LOCK = threading.Lock()


def append(path, record):
    with LOCK:
        with Path(path).open('a', encoding='utf-8') as out:
            out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            out.flush(); os.fsync(out.fileno())


def run(root, workers, limit):
    from transformers import AutoTokenizer
    inputs.prepare(root)
    source = root / 'inputs/sanitized_candidate_inputs.jsonl'
    rows = inputs.rows(source)
    assert len(rows) == 371 and all(set(r) == {'id', 'question', 'query_image_caption'} for r in rows)
    assert 1 <= workers <= 4
    tokenizer = AutoTokenizer.from_pretrained(str(common.CLIP), local_files_only=True)
    config = transport.load_config(ENV)
    source_paths = [source, Path(__file__), Path(inputs.__file__), Path(common.__file__),
        Path(transport.__file__), ROOT / 'scripts/evaluation/openai_chat_transport.py']
    contract = {'stage': 'gold_free_DeepSeek_Open_candidates', 'questions': 371,
        'requested_model': inputs.MODEL, 'expected_returned_model': inputs.MODEL, 'seed': 42,
        'enable_thinking': False, 'temperature': 0,
        'max_tokens': 2048, 'max_format_attempts': 3, 'max_transport_attempts': 5,
        'workers': workers, 'candidate_count': 4, 'clip_token_limit': 77, 'truncation': False,
        'api_config': config.public_dict(), 'system_prompt': common.CANDIDATE_SYSTEM,
        'input_fields': ['id', 'question', 'query_image_caption'], 'gold_or_history_or_options_used': False,
        'source_sha256': {str(p): inputs.sha(p) for p in source_paths},
        'clip_tokenizer_sha256': {p.name: inputs.sha(p) for p in common.CLIP.iterdir()
            if p.name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')}}
    protocol = {'contract': contract, 'experiment_signature': inputs.digest(contract)}
    inputs.freeze(root / 'candidate_generation_protocol.json', protocol)
    signature = protocol['experiment_signature']
    cache = root / 'candidate_calls'; cache.mkdir(exist_ok=True)
    output = root / 'candidates.jsonl'
    done = inputs.native.unique(inputs.rows(output)) if output.exists() else {}
    by_id = inputs.native.unique(rows)
    for qid, item in done.items():
        assert item['source_input_sha256'] == inputs.digest(by_id[qid]) and item['experiment_signature'] == signature
        response = inputs.read(cache / (item['request_sha256'] + '.json'))
        assert inputs.digest(response['request']) == item['request_sha256']
        parsed, lengths = common.parse_candidates(response['response']['raw_response'], tokenizer)
        assert parsed == item['candidate_statements'] and lengths == item['clip_token_lengths']
        assert response['response']['returned_model'] == inputs.MODEL
    stop = threading.Event()

    def one(row):
        for fmt in range(1, 4):
            if stop.is_set(): raise RuntimeError('Candidate run stopped following another request failure')
            payload = {'system': common.CANDIDATE_SYSTEM, 'user_text': common.candidate_user_text(row, fmt), 'query_images': []}
            body = transport.build_request(payload, model=inputs.MODEL, max_output_tokens=2048, image_detail='high')
            key = inputs.digest(body); path = cache / (key + '.json')
            if path.exists():
                cached = inputs.read(path); assert cached['request'] == body
                response = cached['response']
            else:
                response = None
                with LOCK:
                    prior = [r for r in inputs.rows(root / 'candidate_api_attempts.jsonl') if r['request_sha256'] == key] if (root / 'candidate_api_attempts.jsonl').exists() else []
                for attempt in range(len(prior) + 1, 6):
                    if stop.is_set(): raise RuntimeError('Candidate transport stopped')
                    try:
                        response = transport.chat_request(config, body, timeout=120)
                        append(root / 'candidate_api_attempts.jsonl', {'id': row['id'], 'format_attempt': fmt,
                            'transport_attempt': attempt, 'request_sha256': key, 'status': 'ok', 'response': response})
                        inputs.freeze(path, {'request': body, 'request_sha256': key, 'response': response,
                            'experiment_signature': signature})
                        if response['returned_model'] != inputs.MODEL:
                            stop.set(); raise RuntimeError('Returned DeepSeek Flash revision differs from frozen source: ' + response['returned_model'])
                        break
                    except transport.APIError as exc:
                        append(root / 'candidate_api_attempts.jsonl', {'id': row['id'], 'format_attempt': fmt,
                            'transport_attempt': attempt, 'request_sha256': key, 'status': 'api_error',
                            'error_type': type(exc).__name__, 'message': str(exc), 'retryable': bool(exc.retryable), 'provider_error_code': getattr(exc, 'provider_error_code', None)})
                        if not exc.retryable or 'quota' in str(exc).lower() or attempt == 5:
                            stop.set(); raise
                        time.sleep(min(2 ** attempt, 20))
                if response is None: stop.set(); raise RuntimeError('Bounded transport attempts exhausted: ' + row['id'])
            error = None
            try:
                if response['returned_model'] != inputs.MODEL:
                    stop.set(); raise RuntimeError('Cached returned model differs')
                if response['finish_reason'] != 'stop' or response.get('refusal'):
                    raise ValueError('Candidate request did not finish normally or refused')
                parsed, lengths, form = common.parse_candidates(response['raw_response'], tokenizer, return_format=True)
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
            record = {'id': row['id'], 'format_attempt': fmt, 'request_sha256': key,
                'source_input_sha256': inputs.digest(row), 'validation_error': error, 'response': response}
            append(root / 'candidate_attempts.jsonl', record)
            if error is None:
                item = {'id': row['id'], 'candidate_statements': parsed, 'clip_token_lengths': lengths,
                    'source_input_sha256': inputs.digest(row), 'attempt': fmt, 'response_format': form,
                    'request_sha256': key, 'experiment_signature': signature, 'gold_or_history_used': False}
                append(output, item)
                return item
        stop.set(); raise RuntimeError('Three candidate format attempts exhausted: ' + row['id'])

    pending = [r for r in rows if r['id'] not in done]
    if limit: pending = pending[:limit]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, row): row for row in pending}
        for future in as_completed(futures):
            try:
                item = future.result(); done[item['id']] = item
                progress = {'stage': 'DeepSeek_candidates', 'complete': len(done), 'total': 371}
                inputs.dump(root / 'candidate_progress.json', progress)
                if len(done) % 10 == 0 or len(done) == 371: print(json.dumps(progress), flush=True)
            except BaseException:
                stop.set()
                for f in futures: f.cancel()
                raise
    inputs.native.verify_hashes(contract['source_sha256'])
    if len(done) == 371:
        temp = output.with_suffix('.jsonl.tmp')
        temp.write_text(''.join(json.dumps(done[r['id']], ensure_ascii=False) + '\n' for r in rows))
        temp.replace(output)
        result = {'passed': True, 'questions': 371, 'candidates': 1484,
            'max_clip_tokens': max(max(r['clip_token_lengths']) for r in done.values()),
            'candidates_sha256': inputs.sha(output), 'gold_or_history_used': False,
            'model': inputs.MODEL, 'unique_model_calls': len(list(cache.glob('*.json')))}
        inputs.freeze(root / 'candidate_validation.json', result)
        print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, default=inputs.DEFAULT)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--limit', type=int, default=0)
    args = p.parse_args(); root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root / '.candidate_lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root, args.workers, args.limit)


if __name__ == '__main__': main()
