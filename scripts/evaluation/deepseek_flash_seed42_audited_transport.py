"""Seed42 DashScope requests with a safe content-rejection code for evaluation.

Caption requests preserve Qwen's user-only message with image before text.
Reader requests optionally retain their original system message. This module
owns one HTTP attempt; the experiment runner owns retries and persistence.

The undated API identifier does not establish a fixed weight revision or prove
that its weights differ from the 1017 model. Provider identity fields are kept
exactly as returned; an absent fingerprint remains unknown.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping
import urllib.error
import urllib.request

from scripts.evaluation.openai_chat_transport import (
    APIConfig, APIError, _NoRedirect, _base_url, _http_error,
    _validate_request as _validate_message_shape,
    build_request as _build_verified_images,
)


DEEPSEEK_MODEL = 'deepseek-v4.1-flash'
MODEL_IDENTITY_KIND = 'provider_alias'
DASHSCOPE_BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
_PAYLOAD_KEYS = frozenset({'record_id', 'user_text', 'system', 'query_images'})
_WIRE_KEYS = frozenset({'model', 'messages', 'temperature', 'max_tokens', 'n',
                        'stream', 'enable_thinking', 'seed'})
_ENV_KEYS = frozenset({'DEEPSEEKV4_API_KEY', 'DEEPSEEKV4_BASE_URL'})


def canonical_model(model: str) -> str:
    if model != DEEPSEEK_MODEL:
        raise ValueError('DeepSeek backbone requires the exact deepseek-v4.1-flash API alias')
    return model


def _validate_config(config: APIConfig) -> None:
    canonical_model(config.model)
    if config.credential_variable != 'DEEPSEEKV4_API_KEY':
        raise ValueError('DeepSeek backbone requires DEEPSEEKV4_API_KEY credentials')
    if _base_url(config.base_url) != DASHSCOPE_BASE_URL:
        raise ValueError('DeepSeek credentials are bound to the configured DashScope endpoint')


def load_config(env_file: str | Path | None, model: str = DEEPSEEK_MODEL,
                base_url: str | None = None) -> APIConfig:
    """Read only this provider's credential pair; never route it to DeepSeek/OpenAI."""
    values = {}
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
    key = values.get('DEEPSEEKV4_API_KEY', '').strip()
    if not key:
        raise ValueError('Missing required environment variable: DEEPSEEKV4_API_KEY')
    configured_endpoint = values.get('DEEPSEEKV4_BASE_URL', '').strip()
    if not configured_endpoint:
        raise ValueError('Missing required environment variable: DEEPSEEKV4_BASE_URL')
    endpoint = _base_url(configured_endpoint)
    if base_url is not None and _base_url(base_url) != endpoint:
        raise ValueError('Explicit endpoint differs from the paired DeepSeek credential endpoint')
    config = APIConfig(model=canonical_model(model), base_url=endpoint, api_key=key,
                       credential_variable='DEEPSEEKV4_API_KEY')
    _validate_config(config)
    return config


def request_sha256(request: Mapping[str, Any]) -> str:
    serialized = json.dumps(request, sort_keys=True, ensure_ascii=False,
                            separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


def build_request(payload: Mapping[str, Any], *, model: str,
                  max_output_tokens: int = 1024, image_detail: str = 'auto') -> dict[str, Any]:
    """Build a request from a gold-free allowlist, preserving exact image bytes.

    Input: user_text, optional system, optional query_images, optional record_id.
    record_id and local image paths are audit metadata and are never transmitted.
    Images contain path/sha256 and may declare mime_type for additional checking.
    """
    if not isinstance(payload, Mapping) or set(payload) - _PAYLOAD_KEYS or 'user_text' not in payload:
        raise ValueError('Generation payload contains missing or unexpected fields')
    if 'record_id' in payload and not isinstance(payload['record_id'], str):
        raise ValueError('record_id must be a string when supplied')
    system = payload.get('system')
    if system is not None and not isinstance(system, str):
        raise ValueError('Optional system prompt must be a string')
    images = payload.get('query_images', [])
    if not isinstance(images, (list, tuple)):
        raise ValueError('Image inputs must be a sequence')
    normalized, declared_types = [], []
    for image in images:
        if (not isinstance(image, Mapping) or set(image) - {'path', 'sha256', 'mime_type'}
                or not {'path', 'sha256'}.issubset(image)):
            raise ValueError('Image input must contain path, sha256, and optional mime_type only')
        mime = image.get('mime_type')
        if mime is not None and mime not in {'image/jpeg', 'image/png'}:
            raise ValueError('Only JPEG and PNG input MIME types are supported')
        normalized.append({'path': image['path'], 'sha256': image['sha256']})
        declared_types.append(mime)
    selected_model = canonical_model(model)
    request = _build_verified_images(system if system is not None else '', payload['user_text'],
                                     normalized, model=selected_model, max_tokens=max_output_tokens,
                                     seed=42, temperature=0, image_detail=image_detail)
    user = request['messages'][1]
    for declared, part in zip(declared_types, user['content']):
        if declared is not None and not part['image_url']['url'].startswith('data:' + declared + ';base64,'):
            raise ValueError('Declared image MIME type differs from its original bytes')
    if system is None:
        request['messages'] = [user]
    # DashScope controls thinking explicitly. No tools/search configuration is
    # supplied, so requests contain only the frozen text/image evidence.
    # Seed is part of the transmitted body, not only experiment metadata.
    del request['store']
    request['enable_thinking'] = False
    return request


def _validate_request(config: APIConfig, request: Mapping[str, Any]) -> None:
    _validate_config(config)
    if set(request) != _WIRE_KEYS or request.get('model') != config.model:
        raise ValueError('Request fields or model differ from the frozen backbone configuration')
    if type(request['seed']) is not int or request['seed'] != 42:
        raise ValueError('This experiment requires an explicit wire seed of 42')
    limit = request['max_tokens']
    if type(limit) is not int or limit < 1:
        raise ValueError('Output token limit must be a positive integer')
    if request['temperature'] != 0 or type(request['n']) is not int or request['n'] != 1:
        raise ValueError('Backbone experiments require temperature zero and one completion')
    if request['enable_thinking'] is not False:
        raise ValueError('DeepSeek backbone requires enable_thinking=false')
    messages = request['messages']
    if not isinstance(messages, list) or len(messages) not in {1, 2}:
        raise ValueError('Generation requires user-only or system-plus-user messages')
    # Reuse the established strict typed-content validator. Added validation
    # fields/messages exist only in this temporary mapping, never on the wire.
    validated = {key: value for key, value in request.items()
                 if key != 'enable_thinking'}
    validated.update(max_tokens=limit, seed=42, store=False)
    if len(messages) == 1:
        validated['messages'] = [{'role': 'system', 'content': ''}, messages[0]]
    _validate_message_shape(validated, config.model)


def chat_request(config: APIConfig, request: Mapping[str, Any], timeout: float = 60) -> dict[str, Any]:
    """One call without redirects/retries; permanent failure on API model-identifier mismatch."""
    _validate_request(config, request)
    if not config.api_key or any(ord(char) < 33 or ord(char) > 126 for char in config.api_key):
        raise ValueError('API credential is missing or contains invalid characters')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('API timeout must be positive and finite')
    wire_hash = request_sha256(request)
    encoded = json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8')
    http_request = urllib.request.Request(DASHSCOPE_BASE_URL + '/chat/completions',
                                         data=encoded, method='POST', headers={
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
        failure = _http_error(error.code, body)
        # Only this public fixed code is retained. Never expose arbitrary error
        # messages, headers, API credentials, or provider echoes of user data.
        failure.provider_error_code = None
        try:
            provider_error = json.loads(body).get('error', {})
            if isinstance(provider_error, dict) and provider_error.get('code') == 'data_inspection_failed':
                failure.provider_error_code = 'data_inspection_failed'
        except (ValueError, AttributeError, UnicodeError):
            pass
        raise failure from None
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
        text, refusal = message.get('content'), message.get('refusal')
        reasoning_content = message.get('reasoning_content')
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            raise ValueError
        if text is None and isinstance(refusal, str) and refusal:
            text = ''
        if not isinstance(text, str) or not isinstance(payload['model'], str):
            raise ValueError
        if not isinstance(payload.get('id'), str) or not payload['id'].strip():
            raise ValueError
        if not isinstance(choice['finish_reason'], str):
            raise ValueError
        if refusal is not None and not isinstance(refusal, str):
            raise ValueError
        fingerprint = payload.get('system_fingerprint')
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise ValueError
        usage = payload.get('usage')
        if usage is not None and not isinstance(usage, dict):
            raise ValueError
        details = (usage or {}).get('completion_tokens_details') or {}
        if not isinstance(details, dict):
            raise ValueError
        reasoning_tokens = details.get('reasoning_tokens')
        if reasoning_tokens is not None and (type(reasoning_tokens) is not int or reasoning_tokens < 0):
            raise ValueError
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        raise APIError('protocol', retryable=False,
                       message='API returned an invalid Chat Completions response') from None
    if payload['model'] != config.model:
        raise APIError('model_mismatch', retryable=False,
                       message='API returned an unexpected model identifier')
    if reasoning_tokens not in (None, 0) or reasoning_content:
        raise APIError('unexpected_reasoning', retryable=False,
                       message='API returned reasoning despite enable_thinking=false')
    return {
        'model_identity_kind': MODEL_IDENTITY_KIND, 'resolved_model_revision': None,
        'reasoning_content_present': False, 'enable_thinking': False, 'seed': 42,
        'raw_response': text, 'returned_model': payload['model'], 'requested_model': config.model,
        'response_id': payload.get('id'), 'system_fingerprint': fingerprint,
        'finish_reason': choice['finish_reason'], 'usage': usage, 'reasoning_tokens': reasoning_tokens,
        'request_id': request_id, 'elapsed_seconds': elapsed, 'refusal': refusal,
        'request_sha256': wire_hash, 'attempts': 1, 'retry_count': 0,
    }
