"""Official GPT-5.4 snapshot transport for the fixed MemEye reader ablation.

No seed is sent: support was not established for this experiment. The token
budget uses max_completion_tokens and reasoning is explicitly disabled.
Existing GPT-4o-mini experiment code and its frozen protocol stay unchanged.
"""
from __future__ import annotations

import http.client
import json
import math
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request

from scripts.evaluation.openai_chat_transport import (
    APIConfig,
    APIError,
    _NoRedirect,
    _base_url,
    _http_error,
    _validate_request as _validate_legacy_shape,
    build_request as _build_image_messages,
    load_config as _load_openai_config,
)


MODEL = 'gpt-5.4-2026-03-05'
OFFICIAL_BASE_URL = 'https://api.openai.com/v1'
_REQUEST_KEYS = frozenset({
    'model', 'messages', 'temperature', 'max_completion_tokens',
    'reasoning_effort', 'n', 'stream', 'store',
})


def canonical_model(model: str | None) -> str:
    if model not in {None, 'gpt-5.4', MODEL}:
        raise ValueError('This transport accepts only the fixed GPT-5.4 snapshot')
    return MODEL


def _official_config(config: APIConfig) -> None:
    if config.model != MODEL:
        raise ValueError('GPT-5.4 requests must use the fixed snapshot')
    if config.credential_variable != 'OPENAI_API_KEY':
        raise ValueError('GPT-5.4 official requests require OPENAI_API_KEY credentials')
    if _base_url(config.base_url) != OFFICIAL_BASE_URL:
        raise ValueError('GPT-5.4 requests require the official OpenAI endpoint')


def load_config(env_file: str | Path | None, model: str | None = MODEL,
                base_url: str | None = None) -> APIConfig:
    """Always use OPENAI credentials; never route bare GPT-5.4 through CODEX."""
    # The old loader only selects CODEX for the bare alias. Passing the pinned
    # snapshot explicitly chooses OPENAI settings, including env precedence.
    config = _load_openai_config(env_file, model=canonical_model(model), base_url=base_url)
    _official_config(config)
    return config


def build_request(system: str, user_text: str,
                  query_images: Sequence[Mapping[str, Any]], *, model: str = MODEL,
                  max_completion_tokens: int = 16, reasoning_effort: str = 'none',
                  temperature: float = 0, seed: None = None,
                  image_detail: str = 'high') -> dict[str, Any]:
    """Preserve exact verified query bytes and omit unsupported generation knobs."""
    if seed is not None:
        raise ValueError('Provider seed is intentionally not sent in the GPT-5.4 experiment')
    if reasoning_effort != 'none':
        raise ValueError('The frozen GPT-5.4 ablation requires reasoning_effort=none')
    if temperature != 0:
        raise ValueError('The frozen GPT-5.4 ablation requires temperature=0')
    request = _build_image_messages(system, user_text, query_images,
                                   model=canonical_model(model), max_tokens=max_completion_tokens,
                                   temperature=temperature, image_detail=image_detail)
    del request['seed']
    del request['max_tokens']
    request['max_completion_tokens'] = max_completion_tokens
    request['reasoning_effort'] = reasoning_effort
    return request


def _validate_request(config: APIConfig, request: Mapping[str, Any]) -> None:
    _official_config(config)
    if set(request) != _REQUEST_KEYS or request.get('model') != MODEL:
        raise ValueError('GPT-5.4 request fields or model differ from the frozen API configuration')
    if request['reasoning_effort'] != 'none' or request['temperature'] != 0:
        raise ValueError('GPT-5.4 requests require reasoning_effort=none and temperature=0')
    limit = request['max_completion_tokens']
    if type(limit) is not int or limit < 1:
        raise ValueError('max_completion_tokens must be a positive integer')
    if type(request['n']) is not int or request['n'] != 1:
        raise ValueError('Exactly one completion is required')
    # Reuse message shape validation only. This temporary mapping is never sent.
    legacy_shape = {key: value for key, value in request.items()
                    if key not in {'max_completion_tokens', 'reasoning_effort'}}
    legacy_shape.update(max_tokens=limit, seed=42)
    _validate_legacy_shape(legacy_shape, MODEL)


def chat_request(config: APIConfig, request: Mapping[str, Any], timeout: float = 60) -> dict[str, Any]:
    """One official API request; caller owns bounded retries and persistence."""
    _validate_request(config, request)
    if not config.api_key or any(ord(char) < 33 or ord(char) > 126 for char in config.api_key):
        raise ValueError('API credential is missing or contains invalid characters')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('API timeout must be positive and finite')
    encoded = json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8')
    http_request = urllib.request.Request(OFFICIAL_BASE_URL + '/chat/completions',
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
        text, refusal = message.get('content'), message.get('refusal')
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
        details = (usage or {}).get('completion_tokens_details') or {}
        if not isinstance(details, dict):
            raise ValueError
        reasoning_tokens = details.get('reasoning_tokens')
        if reasoning_tokens is not None and (type(reasoning_tokens) is not int or reasoning_tokens < 0):
            raise ValueError
        return {
            'raw_response': text, 'returned_model': payload['model'],
            'response_id': payload.get('id'), 'system_fingerprint': payload.get('system_fingerprint'),
            'finish_reason': choice['finish_reason'], 'usage': usage,
            'reasoning_tokens': reasoning_tokens,
            'request_id': request_id, 'elapsed_seconds': elapsed, 'refusal': refusal,
        }
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        raise APIError('protocol', retryable=False,
                       message='API returned an invalid Chat Completions response') from None
