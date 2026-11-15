"""Seed42 GPT-5 text transport for pinned memory cores, official endpoint only.

Unlike the image/MCQ-only transport, this accepts upstream multi-message roles
with recorded native temperatures; GPT-5 omits temperature on the wire. No credentials enter saved records.
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

from scripts.evaluation.openai_gpt5_base_transport import MODEL, load_config, valid_returned_model
from scripts.evaluation.openai_backbone_transport import request_sha256, OFFICIAL_BASE_URL
from scripts.evaluation.openai_chat_transport import _NoRedirect, _http_error, APIError


def classify_http_error(status, body):
    """Recognize exhausted credits even when the provider adds a new code."""
    try:
        error=json.loads(body).get('error',{})
        if error.get('code')=='credit_balance_exhausted' or error.get('type')=='insufficient_quota':
            result=APIError('quota',http_status=status,retryable=False,code='insufficient_quota',
                message='API credits exhausted; funding must be restored before retrying')
            result.provider_error_code='credit_balance_exhausted' if error.get('code')=='credit_balance_exhausted' else 'insufficient_quota'
            return result
    except (ValueError,AttributeError,TypeError):
        pass
    return _http_error(status,body)


def _validate_config(config):
    if config.model != MODEL or config.base_url != OFFICIAL_BASE_URL or config.credential_variable != 'OPENAI_API_KEY':
        raise ValueError('GPT-5 must use official OpenAI credentials and endpoint')



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
    return dict(model=MODEL, messages=messages, max_completion_tokens=max(2048,max_tokens),
                seed=42, reasoning_effort='minimal', n=1, stream=False, store=False)


def chat_request(config, request, timeout=120):
    _validate_config(config)
    expected = build_request(request['messages'], request['max_completion_tokens'], 0)
    if request != expected:
        raise ValueError('wire request differs from fixed model/seed policy')
    encoded = json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
    http = urllib.request.Request(OFFICIAL_BASE_URL + '/chat/completions', data=encoded,
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
        error = classify_http_error(exc.code, body)
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
    if not valid_returned_model(data['model']):
        raise APIError('model_mismatch', retryable=False, message='Unexpected returned model')
    return dict(raw_response=raw, requested_model=MODEL, returned_model=data['model'],
        response_id=data['id'], finish_reason=choice['finish_reason'],
        system_fingerprint=data.get('system_fingerprint'), usage=usage,
        input_tokens=usage.get('prompt_tokens', 0), generated_tokens=usage.get('completion_tokens', 0),
        hit_token_limit=choice['finish_reason'] == 'length', seed=42, reasoning_effort='minimal', reasoning_tokens=reasoning_tokens,
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
        self.expected_model=json.loads((self.root/'protocol.json').read_text())['caption_model']
        self.environment = dict(provider='OpenAI official', model=MODEL,
            seed=42, reasoning_effort='minimal', temperature_wire_policy='omitted',
            completion_budget_policy='max(2048,native_requested_cap)', api_workers=workers)

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
                if result['returned_model'] != self.expected_model:
                    raise APIError('model_mismatch', retryable=False, message='Caption and text model revisions differ')
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
