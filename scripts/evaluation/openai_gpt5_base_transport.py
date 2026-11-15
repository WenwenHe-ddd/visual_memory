"""Official GPT-5 alias for fresh captions and readers; record actual revision."""
from dataclasses import replace
import http.client
import json
import math
import re
import socket
import time
import urllib.error
import urllib.request

from scripts.evaluation import openai_backbone_transport as snapshot
from scripts.evaluation.openai_gpt54_seeded_transport import validate_seed
from scripts.evaluation.openai_chat_transport import _NoRedirect, _http_error

APIError = snapshot.APIError
MODEL = 'gpt-5'
OFFICIAL_BASE_URL = snapshot.OFFICIAL_BASE_URL


def valid_returned_model(model):
    return isinstance(model, str) and re.fullmatch(r'gpt-5(?:-\d{4}-\d{2}-\d{2})?', model) is not None


def load_config(env_file, model=MODEL, base_url=None):
    if model != MODEL:
        raise ValueError('This experiment requests the literal gpt-5 alias')
    # Use the audited OPENAI credential route, not the legacy CODEX alias route.
    config = snapshot.load_config(env_file, model=snapshot.GPT5_MODEL, base_url=base_url)
    return replace(config, model=MODEL)


def build_request(payload, *, seed=None, max_output_tokens=2048, image_detail='high'):
    if seed is not None:
        validate_seed(seed)
    body = snapshot.build_request(payload, model=snapshot.GPT5_MODEL,
        max_output_tokens=max_output_tokens, image_detail=image_detail)
    body['model'] = MODEL
    body.pop('temperature')
    body['reasoning_effort'] = 'minimal'
    if seed is not None:
        body['seed'] = seed
    return body


def chat_request(config, request, timeout=60):
    if config.model != MODEL or request.get('model') != MODEL:
        raise ValueError('The alias must remain literal in configuration and wire request')
    if 'seed' in request:
        validate_seed(request['seed'])
    if 'temperature' in request or request.get('reasoning_effort') != 'minimal':
        raise ValueError('GPT-5 requires temperature omitted and reasoning_effort=minimal in this experiment')
    # Validate message shape via a temporary mapping; never transmit these compatibility fields.
    snapshot._validate_request(replace(config, model=snapshot.GPT5_MODEL),
        {**{k: v for k, v in request.items() if k != 'seed'}, 'model': snapshot.GPT5_MODEL,
         'temperature': 0, 'reasoning_effort': 'none'})
    if not config.api_key or any(ord(c) < 33 or ord(c) > 126 for c in config.api_key):
        raise ValueError('Invalid API credential')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Timeout must be positive and finite')
    wire = urllib.request.Request(OFFICIAL_BASE_URL + '/chat/completions',
        data=json.dumps(request, ensure_ascii=False, allow_nan=False).encode(), method='POST',
        headers={'Authorization': 'Bearer ' + config.api_key, 'Content-Type': 'application/json',
                 'Accept': 'application/json'})
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
        assert valid_returned_model(payload['model'])
        assert len(payload['choices']) == 1
        choice = payload['choices'][0]
        message = choice['message']
        refusal, text = message.get('refusal'), message.get('content')
        if text is None and refusal:
            text = ''
        assert isinstance(text, str) and isinstance(choice['finish_reason'], str)
        assert refusal is None or isinstance(refusal, str)
        usage = payload.get('usage')
        assert usage is None or isinstance(usage, dict)
        reasoning = ((usage or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
        assert reasoning is None or (type(reasoning) is int and reasoning >= 0)
        return dict(raw_response=text, requested_model=MODEL, returned_model=payload['model'],
            response_id=payload.get('id'), system_fingerprint=payload.get('system_fingerprint'),
            finish_reason=choice['finish_reason'], usage=usage, reasoning_tokens=reasoning,
            request_id=request_id, elapsed_seconds=time.monotonic()-start, refusal=refusal)
    except (ValueError, KeyError, TypeError, AttributeError, AssertionError, UnicodeError):
        raise APIError('protocol', retryable=False,
                       message='Invalid response or returned model is outside GPT-5 family') from None
