"""Versioned seed-enabled adapter; historical no-seed adapters remain frozen."""
import http.client
import json
import math
import socket
import time
import urllib.error
import urllib.request

from scripts.evaluation.openai_gpt5_chat_transport import (
    APIError, MODEL, OFFICIAL_BASE_URL, load_config,
    _validate_request as validate_unseeded,
)
from scripts.evaluation.openai_chat_transport import _NoRedirect, _http_error
from scripts.evaluation.openai_backbone_transport import build_request as build_unseeded


def validate_seed(seed):
    if type(seed) is not int or not -(2**63) <= seed < 2**63:
        raise ValueError('seed must be a signed 64-bit integer')


def build_request(payload, *, seed=42):
    validate_seed(seed)
    body = build_unseeded(payload, model=MODEL, max_output_tokens=16, image_detail='high')
    return {**body, 'seed': seed}


def chat_request(config, request, timeout=60):
    validate_seed(request.get('seed'))
    validate_unseeded(config, {k:v for k,v in request.items() if k != 'seed'})
    if not config.api_key or any(ord(c) < 33 or ord(c) > 126 for c in config.api_key):
        raise ValueError('Invalid API credential')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Timeout must be positive and finite')
    wire = urllib.request.Request(OFFICIAL_BASE_URL+'/chat/completions',
        data=json.dumps(request, ensure_ascii=False, allow_nan=False).encode(), method='POST',
        headers={'Authorization':'Bearer '+config.api_key, 'Content-Type':'application/json',
                 'Accept':'application/json'})
    start = time.monotonic()
    try:
        with urllib.request.build_opener(_NoRedirect()).open(wire, timeout=timeout) as response:
            data = response.read()
            request_id = response.headers.get('x-request-id')
    except urllib.error.HTTPError as error:
        try:
            data = error.read(65536)
        except Exception:
            data = b''
        finally:
            error.close()
        raise _http_error(error.code, data) from None
    except (OSError, socket.timeout, http.client.HTTPException):
        raise APIError('network', retryable=True, message='API network request failed') from None
    try:
        payload = json.loads(data)
        assert payload['model'] == MODEL
        assert len(payload['choices']) == 1
        choice = payload['choices'][0]
        message = choice['message']
        refusal = message.get('refusal')
        text = message.get('content')
        if text is None and refusal:
            text = ''
        assert isinstance(text, str) and isinstance(choice['finish_reason'], str)
        assert refusal is None or isinstance(refusal, str)
        usage = payload.get('usage')
        assert usage is None or isinstance(usage, dict)
        reasoning = ((usage or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
        assert reasoning is None or (type(reasoning) is int and reasoning >= 0)
        return dict(raw_response=text, returned_model=payload['model'], response_id=payload.get('id'),
                    system_fingerprint=payload.get('system_fingerprint'), finish_reason=choice['finish_reason'],
                    usage=usage, reasoning_tokens=reasoning, request_id=request_id,
                    elapsed_seconds=time.monotonic()-start, refusal=refusal)
    except (ValueError, KeyError, TypeError, AttributeError, AssertionError, UnicodeError):
        raise APIError('protocol', retryable=False,
                       message='Invalid response or returned model differs from frozen snapshot') from None
