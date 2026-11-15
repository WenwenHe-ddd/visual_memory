"""Strict loopback-only client for the pinned local DeepSeek-VL2-Small run."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping
import urllib.error
import urllib.request

from scripts.evaluation.openai_chat_transport import (
    APIError, _NoRedirect, _http_error, _validate_request as _validate_messages,
    build_request as _verified_image_request,
)

MODEL = 'deepseek-vl2-small'
REVISION = '6033e16432a1d771cf9fe4a6f894ff5e5e1459af'
BASE_URL = 'http://127.0.0.1:8312/v1'
CONTEXT_LIMIT = 4096
ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MANIFEST = ROOT / 'reports/model_runtime/deepseek-vl2-small/runtime_manifest.json'


@dataclass(frozen=True)
class LocalConfig:
    model: str = MODEL
    base_url: str = BASE_URL

    def public_dict(self):
        runtime = json.loads(RUNTIME_MANIFEST.read_text())
        for path, expected in runtime['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen VL2 runtime source changed: ' + path)
        return {'api_type': 'local_chat_completions', 'model': self.model,
                'base_url': self.base_url, 'checkpoint_revision': REVISION,
                'context_limit': CONTEXT_LIMIT, 'credentials': 'none',
                'runtime_manifest': str(RUNTIME_MANIFEST),
                'runtime_manifest_sha256': hashlib.sha256(RUNTIME_MANIFEST.read_bytes()).hexdigest()}


def canonical_model(model):
    if model != MODEL:
        raise ValueError('The local experiment requires deepseek-vl2-small')
    return model


def load_config(env_file=None, model=MODEL, base_url=None):
    if env_file is not None:
        raise ValueError('The local VL2 experiment does not read API credentials')
    if base_url is not None and base_url.rstrip('/') != BASE_URL:
        raise ValueError('VL2 requests must stay on the fixed loopback endpoint')
    return LocalConfig(canonical_model(model))


def request_sha256(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_request(payload: Mapping[str, Any], *, model=MODEL,
                  max_output_tokens=1024, image_detail='high'):
    if not isinstance(payload, Mapping) or set(payload) - {'record_id', 'user_text', 'system', 'query_images'}:
        raise ValueError('Local generation payload contains unexpected fields')
    if 'user_text' not in payload or not isinstance(payload['user_text'], str):
        raise ValueError('Local generation requires user_text')
    system = payload.get('system')
    if system is not None and not isinstance(system, str):
        raise ValueError('system must be a string when supplied')
    images = payload.get('query_images', [])
    if not isinstance(images, (list, tuple)) or len(images) > 1:
        raise ValueError('This MemEye protocol supports at most one original query image')
    body = _verified_image_request(system or '', payload['user_text'], images,
        model=canonical_model(model), max_tokens=max_output_tokens, seed=42,
        temperature=0, image_detail=image_detail)
    if system is None:
        body['messages'] = body['messages'][1:]
    del body['store']
    body['return_token_ids'] = True
    return body


def _validate(config, body):
    if config != load_config(model=config.model, base_url=config.base_url):
        raise ValueError('Local inference configuration changed')
    if body.get('return_token_ids') is not True:
        raise ValueError('Local inference must return generated token IDs')
    if type(body.get('max_tokens')) is not int or body['max_tokens'] <= 0:
        raise ValueError('max_tokens must be a positive integer')
    checked = dict(body)
    checked.pop('return_token_ids')
    checked['store'] = False
    messages = checked.get('messages')
    if isinstance(messages, list) and len(messages) == 1:
        checked['messages'] = [{'role': 'system', 'content': ''}, messages[0]]
    _validate_messages(checked, config.model)
    if body['temperature'] != 0 or body['seed'] != 42:
        raise ValueError('VL2 experiments require greedy generation and seed 42')


def _post(url, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False,
        allow_nan=False).encode(), method='POST', headers={'Content-Type': 'application/json'})
    # Ignore process proxy settings; benchmark inputs are only sent to loopback.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as error:
        data = error.read(65536)
        status = error.code
        error.close()
        raise _http_error(status, data) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise APIError('network', retryable=True, message='Local VL2 server unavailable or timed out') from None
    try:
        return json.loads(data)
    except (ValueError, UnicodeError):
        raise APIError('protocol', message='Local VL2 returned invalid JSON') from None


def preflight(config, request, timeout=600):
    _validate(config, request)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    body = {'model': config.model, 'messages': request['messages'],
            'add_generation_prompt': True}
    response = _post(config.base_url.removesuffix('/v1') + '/tokenize', body, timeout)
    if not isinstance(response, dict):
        raise APIError('protocol', message='Local VL2 tokenizer returned an invalid response')
    count, tokens = response.get('count'), response.get('tokens')
    if (type(count) is not int or count <= 0 or not isinstance(tokens, list)
            or any(type(t) is not int or t < 0 for t in tokens)
            or len(tokens) != count or response.get('max_model_len') != CONTEXT_LIMIT):
        raise APIError('protocol', message='Local VL2 tokenizer did not report the expected context limit')
    image_count = sum(part.get('type') == 'image_url'
                      for message in request['messages'] if isinstance(message['content'], list)
                      for part in message['content'])
    if image_count and tokens.count(100003) <= image_count:
        raise APIError('protocol', message='VL2 image placeholders were not expanded by the processor')
    if count + request['max_tokens'] > CONTEXT_LIMIT:
        raise APIError('context_budget', retryable=False, code='context_length_exceeded',
                       message=f'VL2 input {count} + output budget {request["max_tokens"]} exceeds {CONTEXT_LIMIT}; no truncation applied')
    return response


def chat_request(config, request, timeout=600):
    start = time.monotonic()
    measured = preflight(config, request, timeout)
    response = _post(config.base_url + '/chat/completions', request, timeout)
    try:
        assert response['model'] == config.model
        assert isinstance(response['id'], str) and response['id']
        assert len(response['choices']) == 1
        choice = response['choices'][0]
        message = choice['message']
        text = message['content']
        assert isinstance(text, str)
        usage = response['usage']
        assert type(usage['prompt_tokens']) is int
        assert type(usage['completion_tokens']) is int
        assert type(usage['total_tokens']) is int
        assert usage['prompt_tokens'] == measured['count']
        assert 1 <= usage['completion_tokens'] <= request['max_tokens']
        assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
        prompt_ids, output_ids = response['prompt_token_ids'], choice['token_ids']
        assert isinstance(prompt_ids, list) and all(type(t) is int and t >= 0 for t in prompt_ids)
        assert prompt_ids == measured['tokens']
        assert isinstance(output_ids, list) and len(output_ids) == usage['completion_tokens']
        assert all(type(t) is int and t >= 0 for t in output_ids)
        assert choice['finish_reason'] in {'stop', 'length'}
        assert not message.get('reasoning_content')
    except (AssertionError, KeyError, TypeError):
        raise APIError('protocol', message='VL2 response identity, tokens, or usage differs from the checked request') from None
    return {
        'raw_response': text, 'requested_model': config.model, 'returned_model': response['model'],
        'response_id': response['id'], 'request_id': response['id'],
        'system_fingerprint': response.get('system_fingerprint'),
        'finish_reason': choice['finish_reason'], 'stop_reason': choice.get('stop_reason'),
        'usage': usage, 'input_tokens': measured['count'], 'context_limit': CONTEXT_LIMIT,
        'prompt_token_ids': prompt_ids, 'generated_token_ids': output_ids,
        'expanded_image_tokens': prompt_ids.count(100003),
        'checkpoint_revision': REVISION, 'input_truncated': False,
        'refusal': message.get('refusal'), 'reasoning_content_present': False,
        'reasoning_tokens': None, 'enable_thinking': None,
        'elapsed_seconds': time.monotonic() - start,
        'request_sha256': request_sha256(request), 'attempts': 1, 'retry_count': 0,
    }
