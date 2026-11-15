"""Small, credential-safe Chat Completions transport for reader ablations.

The caller owns retries, concurrency, result persistence, and benchmark labels.
This module sends one request and never imports another project's data or code.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request


_ENV_KEYS = frozenset({
    'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL',
    'CODEX_API_KEY', 'CODEX_BASE_URL',
})
_REQUEST_KEYS = frozenset({
    'model', 'messages', 'temperature', 'seed', 'max_tokens',
    'n', 'stream', 'store',
})
_SAFE_ERROR_CODES = frozenset({
    'invalid_api_key', 'insufficient_quota', 'rate_limit_exceeded',
    'model_not_found', 'invalid_request_error', 'billing_hard_limit_reached',
    'context_length_exceeded', 'content_filter', 'unsupported_parameter',
    'invalid_value', 'account_deactivated', 'server_error',
})
_QUOTA_CODES = frozenset({'insufficient_quota', 'billing_hard_limit_reached'})


@dataclass(frozen=True)
class APIConfig:
    model: str
    base_url: str
    api_key: str = field(repr=False)
    credential_variable: str = 'OPENAI_API_KEY'

    def public_dict(self) -> dict[str, str]:
        return {'api_type': 'chat_completions', 'model': self.model,
                'base_url': self.base_url, 'credential_variable': self.credential_variable}


class APIError(RuntimeError):
    """Only fixed messages and allowlisted codes can be attached to API failures."""

    def __init__(self, kind: str, *, http_status: int | None = None,
                 retryable: bool = False, code: str | None = None,
                 message: str = 'API request failed') -> None:
        self.kind = kind
        self.http_status = http_status
        self.retryable = retryable
        self.code = code if code in _SAFE_ERROR_CODES else None
        self.message = message
        super().__init__(message)


def _base_url(value: str) -> str:
    value = value.strip().rstrip('/')
    for suffix in ('/chat/completions', '/responses'):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    parts = urllib.parse.urlsplit(value)
    if (parts.scheme not in {'http', 'https'} or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment):
        raise ValueError('API base URL must have a host and no embedded credentials or query')
    if parts.scheme == 'http' and parts.hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('Remote API endpoints must use HTTPS')
    return value


def load_config(env_file: str | Path | None, model: str | None = None,
                base_url: str | None = None) -> APIConfig:
    """Read only relevant settings, matching NTU dotenv parsing and env precedence.

    No environment mutation and no fallback to other credential files. Explicit
    GPT-5.4 selects CODEX credentials; its endpoint must also be explicit.
    """
    values: dict[str, str] = {}
    if env_file is not None:
        for raw in Path(env_file).read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export '):].strip()
            key, separator, value = line.partition('=')
            key, value = key.strip(), value.strip()
            if not separator or key not in _ENV_KEYS:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            values.setdefault(key, value)
    values.update({key: os.environ[key] for key in _ENV_KEYS if key in os.environ})
    selected_model = (model or values.get('OPENAI_MODEL', 'gpt-4.1-mini')).strip() or 'gpt-4.1-mini'
    prefix = 'CODEX' if selected_model == 'gpt-5.4' else 'OPENAI'
    credential_variable = prefix + '_API_KEY'
    key = values.get(credential_variable, '').strip()
    if not key:
        raise ValueError(f'Missing required environment variable: {credential_variable}')
    endpoint = base_url if base_url is not None else values.get(prefix + '_BASE_URL', '').strip()
    if not endpoint:
        if prefix == 'CODEX':
            raise ValueError('GPT-5.4 requires an explicit CODEX_BASE_URL or base_url')
        endpoint = 'https://api.openai.com/v1'
    return APIConfig(model=selected_model, base_url=_base_url(endpoint), api_key=key,
                     credential_variable=credential_variable)


def build_request(system: str, user_text: str,
                  query_images: Sequence[Mapping[str, Any]], *, model: str,
                  max_tokens: int = 16, seed: int = 42, temperature: float = 0,
                  image_detail: str = 'high') -> dict[str, Any]:
    """Use original verified query pixels; no resizing or label-bearing inputs."""
    if not isinstance(system, str) or not isinstance(user_text, str):
        raise ValueError('System and user prompts must be strings')
    if not isinstance(model, str) or not model.strip():
        raise ValueError('Model must be a nonempty string')
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('max_tokens must be a positive integer')
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError('seed must be an integer between 0 and 2^31 - 1')
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError('temperature must be between 0 and 2')
    if image_detail not in {'low', 'high', 'auto'}:
        raise ValueError('Unsupported image_detail')
    content: list[dict[str, Any]] = []
    for item in query_images:
        if set(item) != {'path', 'sha256'}:
            raise ValueError('Each query image must contain only path and sha256')
        expected = item['sha256']
        if not isinstance(expected, str) or re.fullmatch(r'[a-f0-9]{64}', expected) is None:
            raise ValueError('Query image requires a lowercase SHA-256 digest')
        data = Path(item['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Query image SHA-256 mismatch')
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            mime = 'image/png'
        elif data.startswith(b'\xff\xd8\xff'):
            mime = 'image/jpeg'
        else:
            raise ValueError('Only original PNG and JPEG query images are supported')
        url = f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')
        content.append({'type': 'image_url', 'image_url': {'url': url, 'detail': image_detail}})
    content.append({'type': 'text', 'text': user_text})
    return {
        'model': model,
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
        'temperature': temperature, 'seed': seed, 'max_tokens': max_tokens,
        'n': 1, 'stream': False, 'store': False,
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_error(status: int, body: bytes) -> APIError:
    code = None
    try:
        payload = json.loads(body)
        candidate = payload.get('error', {}).get('code')
        if isinstance(candidate, str) and candidate in _SAFE_ERROR_CODES:
            code = candidate
    except (ValueError, AttributeError, TypeError):
        pass
    if status in {401, 403}:
        kind, retryable, message = 'authentication', False, 'API authentication or access was rejected'
    elif code in _QUOTA_CODES:
        kind, retryable, message = 'quota', False, 'API quota or billing limit was reached'
    elif status == 429:
        kind, retryable, message = 'rate_limit', True, 'API rate limit was reached'
    elif 500 <= status <= 599:
        kind, retryable, message = 'server', True, 'API server returned a temporary error'
    elif 300 <= status <= 399:
        kind, retryable, message = 'redirect', False, 'API redirect was refused'
    elif status == 404:
        kind, retryable, message = 'not_found', False, 'API model or endpoint was not found'
    else:
        kind, retryable, message = 'http', False, 'API request was rejected'
    return APIError(kind, http_status=status, retryable=retryable, code=code, message=message)


def _validate_request(request: Mapping[str, Any], model: str) -> None:
    if set(request) != _REQUEST_KEYS or request.get('model') != model:
        raise ValueError('Request model or fields differ from the frozen API configuration')
    if request['n'] != 1 or request['stream'] is not False or request['store'] is not False:
        raise ValueError('Only one non-streamed, non-stored completion is supported')
    messages = request['messages']
    if (not isinstance(messages, list) or len(messages) != 2
            or any(not isinstance(m, dict) or set(m) != {'role', 'content'} for m in messages)
            or messages[0]['role'] != 'system' or not isinstance(messages[0]['content'], str)
            or messages[1]['role'] != 'user' or not isinstance(messages[1]['content'], list)):
        raise ValueError('Request must contain exactly a system message and typed user message')
    content = messages[1]['content']
    if not content or not isinstance(content[-1], dict) or content[-1].get('type') != 'text':
        raise ValueError('User content must end with text after any query images')
    for index, item in enumerate(content):
        if not isinstance(item, dict):
            raise ValueError('Invalid user content')
        if index == len(content) - 1:
            valid = set(item) == {'type', 'text'} and item['type'] == 'text' and isinstance(item['text'], str)
        else:
            image = item.get('image_url')
            valid = (set(item) == {'type', 'image_url'} and item['type'] == 'image_url'
                     and isinstance(image, dict) and set(image) == {'url', 'detail'}
                     and image['detail'] in {'low', 'high', 'auto'}
                     and isinstance(image['url'], str)
                     and image['url'].startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')))
        if not valid:
            raise ValueError('User content contains unexpected fields or external image URLs')


def chat_request(config: APIConfig, request: Mapping[str, Any], timeout: float = 60) -> dict[str, Any]:
    """Send one request. Errors omit provider bodies, URLs, prompts, and secrets."""
    _validate_request(request, config.model)
    if not config.api_key or any(ord(char) < 33 or ord(char) > 126 for char in config.api_key):
        raise ValueError('API credential is missing or contains invalid characters')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('API timeout must be positive and finite')
    endpoint = _base_url(config.base_url) + '/chat/completions'
    encoded = json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8')
    http_request = urllib.request.Request(endpoint, data=encoded, method='POST', headers={
        'Authorization': 'Bearer ' + config.api_key,
        'Content-Type': 'application/json', 'Accept': 'application/json',
    })
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
        raise _http_error(error.code, body) from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, http.client.HTTPException):
        raise APIError('network', retryable=True,
                       message='API network request failed or timed out') from None
    elapsed = time.monotonic() - start
    try:
        payload = json.loads(body)
        choices = payload['choices']
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        message = choice['message']
        text = message.get('content')
        refusal = message.get('refusal')
        if text is None and isinstance(refusal, str) and refusal:
            text = ''
        if not isinstance(text, str) or not isinstance(payload['model'], str):
            raise ValueError
        if not isinstance(choice['finish_reason'], str):
            raise ValueError
        if refusal is not None and not isinstance(refusal, str):
            raise ValueError
        usage = payload.get('usage')
        if usage is not None and not isinstance(usage, dict):
            raise ValueError
        return {
            'raw_response': text, 'returned_model': payload['model'],
            'response_id': payload.get('id'), 'system_fingerprint': payload.get('system_fingerprint'),
            'finish_reason': choice['finish_reason'], 'usage': usage,
            'request_id': request_id, 'elapsed_seconds': elapsed, 'refusal': refusal,
        }
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        raise APIError('protocol', retryable=False,
                       message='API returned an invalid Chat Completions response') from None
