"""Isolated wire-seed1024 transport for the existing DeepSeek Flash condition.

HTTP response/error handling is derived unchanged from the audited seed42
transport, with the response seed metadata changed to1024. Request validators
below independently require actual transmitted seed1024, exact model and the
original credential-bound DashScope endpoint. No judge transport is modified.
"""
from __future__ import annotations
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import http.client
import json
import math
from pathlib import Path
import socket
import threading
import time
from typing import Any, Mapping
import urllib.error
import urllib.request
from scripts.evaluation import deepseek_flash_seed42_audited_transport as original

MODEL=DEEPSEEK_MODEL=original.DEEPSEEK_MODEL
SEED=1024
DASHSCOPE_BASE_URL=original.DASHSCOPE_BASE_URL
MODEL_IDENTITY_KIND='provider_alias'
APIError=original.APIError
APIConfig=original.APIConfig
load_config=original.load_config
canonical_model=original.canonical_model
request_sha256=original.request_sha256
_validate_config=original._validate_config
_NoRedirect=original._NoRedirect
_http_error=original._http_error
_WIRE_KEYS=frozenset({'model','messages','temperature','max_tokens','n','stream','enable_thinking','seed'})


def _seed(seed):
    if type(seed) is not int or seed != SEED:
        raise ValueError('This replication requires actual wire seed1024')


def _messages(messages):
    if not isinstance(messages,list) or not messages:
        raise ValueError('Nonempty native messages required')
    value=copy.deepcopy(messages)
    for message in value:
        if set(message)!={'role','content'} or message['role'] not in {'system','assistant','user'}:
            raise ValueError('Unexpected native message shape')
        content=message['content']
        if isinstance(content,str):
            continue
        if not isinstance(content,list) or message['role']!='user':
            raise ValueError('Typed image content only supported in user role')
        for part in content:
            if part.get('type')=='text':
                if set(part)!={'type','text'} or not isinstance(part['text'],str):
                    raise ValueError('Invalid typed text')
            elif part.get('type')=='image':
                if set(part)!={'type','path','sha256'}:
                    raise ValueError('Original image path and hash required')
                data=Path(part['path']).read_bytes()
                if hashlib.sha256(data).hexdigest()!=part['sha256']:
                    raise ValueError('Original image changed')
                mime='image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg' if data.startswith(b'\xff\xd8\xff') else None
                if mime is None:
                    raise ValueError('Only original PNG/JPEG images supported')
                part.clear();part.update(type='image_url',image_url={'detail':'high','url':'data:'+mime+';base64,'+base64.b64encode(data).decode('ascii')})
            elif part.get('type')=='image_url':
                if set(part)!={'type','image_url'} or set(part['image_url'])!={'url','detail'}:
                    raise ValueError('Unexpected image URL shape')
                uri=part['image_url']['url']
                if not isinstance(uri,str) or not uri.startswith(('data:image/png;base64,','data:image/jpeg;base64,')):
                    raise ValueError('Only original inline PNG/JPEG accepted')
                if part['image_url']['detail'] not in {'auto','high','low'}:
                    raise ValueError('Unexpected image detail')
            else:
                raise ValueError('Unexpected native content type')
    return value


def build_message_request(messages,max_tokens=2048,temperature=0,*,seed=SEED,model=MODEL):
    _seed(seed);canonical_model(model)
    if type(max_tokens)is not int or max_tokens<1:
        raise ValueError('Positive output cap required')
    if not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or not 0<=temperature<=2:
        raise ValueError('Invalid native temperature')
    return dict(model=model,messages=_messages(messages),max_tokens=max_tokens,temperature=float(temperature),seed=seed,enable_thinking=False,n=1,stream=False)


def build_request(payload,max_tokens=2048,temperature=0,*,model=MODEL,max_output_tokens=None,image_detail='high',seed=SEED):
    _seed(seed)
    if isinstance(payload,Mapping):
        request=original.build_request(payload,model=model,max_output_tokens=max_tokens if max_output_tokens is None else max_output_tokens,image_detail=image_detail)
        request['seed']=seed
        request['temperature']=float(temperature)
        return request
    return build_message_request(payload,max_tokens if max_output_tokens is None else max_output_tokens,temperature,seed=seed,model=model)


def _validate_request(config,request):
    _validate_config(config)
    if set(request)!=_WIRE_KEYS or request.get('model')!=config.model:
        raise ValueError('Wire fields/model differ')
    expected=build_message_request(request['messages'],request['max_tokens'],request['temperature'],seed=request['seed'],model=request['model'])
    if expected!=request:
        raise ValueError('Wire request differs from seed1024/native-message policy')

def _raw_chat_request(config: APIConfig, request: Mapping[str, Any], timeout: float=60) -> dict[str, Any]:
    """One call without redirects/retries; permanent failure on API model-identifier mismatch."""
    _validate_request(config, request)
    if not config.api_key or any((ord(char) < 33 or ord(char) > 126 for char in config.api_key)):
        raise ValueError('API credential is missing or contains invalid characters')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('API timeout must be positive and finite')
    wire_hash = request_sha256(request)
    encoded = json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8')
    http_request = urllib.request.Request(DASHSCOPE_BASE_URL + '/chat/completions', data=encoded, method='POST', headers={'Authorization': 'Bearer ' + config.api_key, 'Content-Type': 'application/json', 'Accept': 'application/json'})
    start = time.monotonic()
    try:
        with urllib.request.build_opener(_NoRedirect()).open(http_request, timeout=timeout) as response:
            body = response.read()
            request_id = response.headers.get('x-request-id')
    except urllib.error.HTTPError as error:
        try:
            body = error.read(65536)
        except Exception:
            body = b''
        finally:
            error.close()
        failure = _http_error(error.code, body)
        failure.provider_error_code = None
        try:
            provider_error = json.loads(body).get('error', {})
            if isinstance(provider_error, dict) and provider_error.get('code') == 'data_inspection_failed':
                failure.provider_error_code = 'data_inspection_failed'
        except (ValueError, AttributeError, UnicodeError):
            pass
        raise failure from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, http.client.HTTPException):
        raise APIError('network', retryable=True, message='API network request failed or timed out') from None
    elapsed = time.monotonic() - start
    try:
        payload = json.loads(body)
        choices = payload['choices']
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        message = choice['message']
        text, refusal = (message.get('content'), message.get('refusal'))
        reasoning_content = message.get('reasoning_content')
        if reasoning_content is not None and (not isinstance(reasoning_content, str)):
            raise ValueError
        if text is None and isinstance(refusal, str) and refusal:
            text = ''
        if not isinstance(text, str) or not isinstance(payload['model'], str):
            raise ValueError
        if not isinstance(payload.get('id'), str) or not payload['id'].strip():
            raise ValueError
        if not isinstance(choice['finish_reason'], str):
            raise ValueError
        if refusal is not None and (not isinstance(refusal, str)):
            raise ValueError
        fingerprint = payload.get('system_fingerprint')
        if fingerprint is not None and (not isinstance(fingerprint, str)):
            raise ValueError
        usage = payload.get('usage')
        if usage is not None and (not isinstance(usage, dict)):
            raise ValueError
        details = (usage or {}).get('completion_tokens_details') or {}
        if not isinstance(details, dict):
            raise ValueError
        reasoning_tokens = details.get('reasoning_tokens')
        if reasoning_tokens is not None and (type(reasoning_tokens) is not int or reasoning_tokens < 0):
            raise ValueError
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        raise APIError('protocol', retryable=False, message='API returned an invalid Chat Completions response') from None
    if payload['model'] != config.model:
        raise APIError('model_mismatch', retryable=False, message='API returned an unexpected model identifier')
    if reasoning_tokens not in (None, 0) or reasoning_content:
        raise APIError('unexpected_reasoning', retryable=False, message='API returned reasoning despite enable_thinking=false')
    return {'model_identity_kind': MODEL_IDENTITY_KIND, 'resolved_model_revision': None, 'reasoning_content_present': False, 'enable_thinking': False, 'seed': 1024, 'raw_response': text, 'returned_model': payload['model'], 'requested_model': config.model, 'response_id': payload.get('id'), 'system_fingerprint': fingerprint, 'finish_reason': choice['finish_reason'], 'usage': usage, 'reasoning_tokens': reasoning_tokens, 'request_id': request_id, 'elapsed_seconds': elapsed, 'refusal': refusal, 'request_sha256': wire_hash, 'attempts': 1, 'retry_count': 0}

def chat_request(config,request,timeout=120):
    result=_raw_chat_request(config,request,timeout)
    usage=result.get('usage') or {}
    result.update(input_tokens=usage.get('prompt_tokens',0),generated_tokens=usage.get('completion_tokens',0),hit_token_limit=result['finish_reason']=='length')
    return result


class APIModel:
    """Broker-compatible model with parallel API requests and append-only audit."""

    def __init__(self, root, env_file, workers=8, seed=SEED, model=MODEL):
        _seed(seed)
        canonical_model(model)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = load_config(env_file)
        self.workers = workers
        self.lock = threading.Lock()
        self.fatal_errors = []
        import torch
        self.torch = torch
        self.environment = dict(provider='DashScope International', model=DEEPSEEK_MODEL, seed=1024, enable_thinking=False, api_workers=workers)

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
                self.log('api_attempts.jsonl', dict(request_sha256=key, attempt=attempt, status='ok', response_id=result['response_id']))
                self.log('api_responses.jsonl', dict(request=request, output=result))
                return result
            except APIError as exc:
                failure = dict(request_sha256=key, attempt=attempt, status='error', error_type=getattr(exc, 'kind', type(exc).__name__), retryable=exc.retryable, provider_error_code=getattr(exc, 'provider_error_code', None))
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
