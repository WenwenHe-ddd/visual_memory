"""Seed42 text transport for pinned memory cores, bound to DashScope Intl.

Unlike the image/MCQ-only transport, this accepts upstream multi-message roles
and native memory-operation temperatures. No credentials enter saved records.
"""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
from pathlib import Path
import socket
import threading
import time
import urllib.error
import urllib.request

from scripts.evaluation.deepseek_flash_seed42_audited_transport import (
    DEEPSEEK_MODEL, DASHSCOPE_BASE_URL, load_config, request_sha256,
    _validate_config, _NoRedirect, _http_error, APIError,
)


def build_request(messages, max_tokens, temperature):
    if not isinstance(messages, list) or not messages:
        raise ValueError('nonempty text messages required')
    for m in messages:
        if set(m) != {'role', 'content'} or m['role'] not in {'system', 'user', 'assistant'} or not isinstance(m['content'], str):
            raise ValueError('only explicit text message roles accepted')
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('positive integer token budget required')
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError('invalid temperature')
    return dict(model=DEEPSEEK_MODEL, messages=messages, max_tokens=max_tokens,
                temperature=float(temperature), seed=42, enable_thinking=False,
                n=1, stream=False)


def chat_request(config, request, timeout=120):
    _validate_config(config)
    expected = build_request(request['messages'], request['max_tokens'], request['temperature'])
    if request != expected:
        raise ValueError('wire request differs from fixed model/seed policy')
    encoded = json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
    http = urllib.request.Request(DASHSCOPE_BASE_URL + '/chat/completions', data=encoded,
        method='POST', headers={'Authorization': 'Bearer ' + config.api_key,
        'Content-Type': 'application/json', 'Accept': 'application/json'})
    start = time.monotonic()
    try:
        with urllib.request.build_opener(_NoRedirect()).open(http, timeout=timeout) as response:
            body = response.read()
            request_id = response.headers.get('x-request-id')
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(65536)
        finally:
            exc.close()
        error = _http_error(exc.code, body)
        error.provider_error_code = None
        try:
            if json.loads(body).get('error', {}).get('code') == 'data_inspection_failed':
                error.provider_error_code = 'data_inspection_failed'
        except (ValueError, AttributeError, UnicodeError):
            pass
        raise error from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, http.client.HTTPException):
        raise APIError('network', retryable=True, message='API network request failed or timed out') from None
    try:
        data = json.loads(body)
        assert len(data['choices']) == 1
        choice = data['choices'][0]
        message = choice['message']
        raw = message.get('content')
        if raw is None and message.get('refusal'):
            raw = ''
        assert isinstance(raw, str) and isinstance(data['id'], str) and data['id']
        assert isinstance(choice['finish_reason'], str)
        assert isinstance(data['model'], str)
        usage = data.get('usage') or {}
        reasoning_tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
    except (AssertionError, ValueError, KeyError, TypeError, AttributeError):
        raise APIError('protocol', retryable=False, message='Invalid Chat Completions response') from None
    if data['model'] != DEEPSEEK_MODEL:
        raise APIError('model_mismatch', retryable=False, message='Unexpected returned model')
    if message.get('reasoning_content') or reasoning_tokens not in (None, 0):
        raise APIError('unexpected_reasoning', retryable=False, message='Reasoning returned despite thinking=false')
    return dict(raw_response=raw, requested_model=DEEPSEEK_MODEL, returned_model=data['model'],
        response_id=data['id'], finish_reason=choice['finish_reason'],
        system_fingerprint=data.get('system_fingerprint'), usage=usage,
        input_tokens=usage.get('prompt_tokens', 0), generated_tokens=usage.get('completion_tokens', 0),
        hit_token_limit=choice['finish_reason'] == 'length', seed=42, enable_thinking=False,
        request_sha256=request_sha256(request), request_id=request_id,
        elapsed_seconds=time.monotonic()-start, refusal=message.get('refusal'))


class APIModel:
    """Broker-compatible model with parallel API requests and append-only audit."""
    def __init__(self, root, env_file, workers=2):
        self.root = Path(root)
        self.config = load_config(env_file)
        self.workers = workers
        self.lock = threading.Lock()
        self.fatal_errors = []
        import torch
        self.torch = torch
        self.environment = dict(provider='DashScope International', model=DEEPSEEK_MODEL,
            seed=42, enable_thinking=False, api_workers=workers)

    def log(self, name, record):
        with self.lock:
            with (self.root / name).open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def one(self, messages, cap, temperature):
        request = build_request(messages, cap, temperature)
        key = request_sha256(request)
        for attempt in range(1, 4):
            try:
                result = chat_request(self.config, request)
                self.log('api_attempts.jsonl', dict(request_sha256=key, attempt=attempt,
                    status='ok', response_id=result['response_id']))
                self.log('api_responses.jsonl', dict(request=request, output=result))
                return result
            except APIError as exc:
                failure = dict(request_sha256=key, attempt=attempt, status='error',
                    error_type=getattr(exc, 'kind', type(exc).__name__),
                    retryable=exc.retryable, provider_error_code=getattr(exc, 'provider_error_code', None))
                self.log('api_attempts.jsonl', failure)
                if attempt == 3 or not exc.retryable:
                    with self.lock:
                        self.fatal_errors.append(failure)
                    raise
                time.sleep(attempt * 2)

    def generate(self, messages, max_new_tokens, sampling_temperature=0.0):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            jobs = [pool.submit(self.one, row, max_new_tokens, sampling_temperature) for row in messages]
            return [job.result() for job in jobs]
