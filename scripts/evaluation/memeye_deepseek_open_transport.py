"""Fixed Open-answer settings over the existing audited DeepSeek transports."""
import http.client
import json
import time
import urllib.error
import urllib.request
from scripts.evaluation.openai_chat_transport import _NoRedirect
from scripts.evaluation import deepseek_flash_seed42_audited_transport as vision
from scripts.evaluation import deepseek_text_memory_transport as text

MODEL = vision.DEEPSEEK_MODEL
APIError = vision.APIError
load_config = vision.load_config


def settings(model=MODEL):
    if model != MODEL:
        raise ValueError('Unexpected DeepSeek answer alias')
    return dict(model=MODEL, max_tokens=2048, temperature=0, seed=42,
                enable_thinking=False, n=1, stream=False)


def chat(config, request, timeout=120):
    if {k: v for k, v in request.items() if k != 'messages'} != settings():
        raise ValueError('Open reader settings differ')
    if all(isinstance(m['content'], str) for m in request['messages']):
        output = text.chat_request(config, request, timeout=timeout)
    else:
        output = multimodal_chat(config, request, timeout=timeout)
    output.update(model_identity_kind='provider_alias', resolved_model_revision=None)
    return output


def multimodal_chat(config, body, timeout=120):
    if {k: v for k, v in body.items() if k != 'messages'} != settings(config.model):
        raise ValueError('Wire settings changed')
    vision._validate_config(config)
    req = urllib.request.Request(config.base_url + '/chat/completions',
        data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(), method='POST',
        headers={'Authorization': 'Bearer ' + config.api_key, 'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=timeout) as response:
            raw, request_id = response.read(), response.headers.get('x-request-id')
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(65536)
        finally:
            exc.close()
        error = vision._http_error(exc.code, raw)
        try:
            if json.loads(raw).get('error', {}).get('code') == 'data_inspection_failed':
                error.kind = 'data_inspection_failed'
        except (ValueError, AttributeError, TypeError):
            pass
        raise error from None
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        raise APIError('network', retryable=True, message='API network request failed') from None
    try:
        data = json.loads(raw)
        assert len(data['choices']) == 1
        choice = data['choices'][0]
        message = choice['message']
        text, refusal = message.get('content'), message.get('refusal')
        if text is None and refusal:
            text = ''
        assert isinstance(text, str) and isinstance(choice['finish_reason'], str)
        assert isinstance(data['id'], str) and data['id']
        returned = data['model']
        usage = data.get('usage') or {}
        reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
    except (ValueError, KeyError, TypeError, AttributeError, AssertionError):
        raise APIError('protocol', message='Invalid API response') from None
    if returned != config.model:
        raise APIError('model_mismatch', message='Returned model mismatch')
    if message.get('reasoning_content') or reasoning not in (None, 0):
        raise APIError('unexpected_reasoning', message='Thinking was not disabled')
    return dict(raw_response=text, returned_model=returned, response_id=data['id'], request_id=request_id,
        system_fingerprint=data.get('system_fingerprint'), finish_reason=choice['finish_reason'],
        usage=usage, refusal=refusal, reasoning_tokens=reasoning, elapsed_seconds=time.monotonic()-started)
