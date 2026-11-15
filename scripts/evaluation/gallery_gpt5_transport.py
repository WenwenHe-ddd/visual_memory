"""Official GPT-5 transport for the Gallery GA experiment, with audited retries.

Keep the existing signed experiments untouched. In particular, avoid the old
text transport's `http` local variable masking `http.client` on network errors.
"""
import http.client
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from scripts.evaluation import openai_gpt5_base_transport as api
from scripts.evaluation.gpt5_text_memory_transport import build_request, classify_http_error
from scripts.evaluation.openai_backbone_transport import request_sha256, OFFICIAL_BASE_URL
from scripts.evaluation.openai_chat_transport import _NoRedirect, APIError


def chat(config, request, timeout=120):
    if (config.base_url != OFFICIAL_BASE_URL or config.credential_variable != 'OPENAI_API_KEY'
            or config.model != 'gpt-5' or request.get('model') != 'gpt-5'
            or request.get('seed') != 42 or request.get('reasoning_effort') != 'minimal'
            or 'temperature' in request or request.get('store') is not False):
        raise ValueError('Unexpected GPT-5 routing or request policy')
    wire = urllib.request.Request(OFFICIAL_BASE_URL + '/chat/completions',
        data=json.dumps(request, ensure_ascii=False, allow_nan=False).encode(), method='POST',
        headers={'Authorization': 'Bearer ' + config.api_key, 'Content-Type': 'application/json'})
    start = time.monotonic()
    try:
        with urllib.request.build_opener(_NoRedirect()).open(wire, timeout=timeout) as response:
            body, request_id = response.read(), response.headers.get('x-request-id')
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(65536)
        finally:
            exc.close()
        raise classify_http_error(exc.code, body) from None
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        raise APIError('network', retryable=True, message='Official GPT-5 network request failed') from None
    try:
        data = json.loads(body)
        assert api.valid_returned_model(data['model']) and len(data['choices']) == 1
        choice = data['choices'][0]
        raw = choice['message'].get('content')
        refusal = choice['message'].get('refusal')
        if raw is None and refusal:
            raw = ''
        assert isinstance(raw, str) and isinstance(choice['finish_reason'], str)
        assert isinstance(data['id'], str) and data['id']
        usage = data.get('usage') or {}
    except (AssertionError, KeyError, TypeError, ValueError, AttributeError):
        raise APIError('protocol', retryable=False, message='Invalid GPT-5 response or model') from None
    return dict(raw_response=raw, requested_model='gpt-5', returned_model=data['model'],
        response_id=data['id'], request_id=request_id, finish_reason=choice['finish_reason'],
        system_fingerprint=data.get('system_fingerprint'), usage=usage, refusal=refusal,
        input_tokens=usage.get('prompt_tokens', 0), generated_tokens=usage.get('completion_tokens', 0),
        reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
        seed=42, reasoning_effort='minimal', hit_token_limit=choice['finish_reason'] == 'length',
        request_sha256=request_sha256(request), elapsed_seconds=time.monotonic() - start)


class Model:
    def __init__(self, root, env_file, expected_model):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = api.load_config(env_file)
        self.expected_model = expected_model
        self.lock = threading.Lock()
        self.fatal_errors = []
        import torch
        self.torch = torch
        self.environment = dict(provider='OpenAI official', requested_model='gpt-5',
            returned_model=expected_model, seed=42, reasoning_effort='minimal', temperature='omitted')

    def log(self, name, value):
        with self.lock:
            with (self.root / name).open('a') as f:
                f.write(json.dumps(value, ensure_ascii=False) + '\n')

    def invoke(self, request, stage, record_id=None):
        key = request_sha256(request)
        for attempt in range(1, 4):
            if self.fatal_errors:
                raise RuntimeError('Experiment stopped after fatal API error; see api_attempts.jsonl')
            try:
                output = chat(self.config, request)
                if output['returned_model'] != self.expected_model:
                    raise APIError('model_mismatch', retryable=False, message='GPT-5 revision changed')
                self.log('api_attempts.jsonl', dict(stage=stage, id=record_id,
                    request_sha256=key, attempt=attempt, status='ok', response_id=output['response_id']))
                # Images are reconstructed from hashed payload manifests, not copied as base64 into logs.
                self.log('api_responses.jsonl', dict(stage=stage, id=record_id, output=output,
                    **({'request': request} if stage == 'memory' else {})))
                return output
            except APIError as exc:
                failure = dict(stage=stage, id=record_id, request_sha256=key, attempt=attempt,
                    status='error', error_type=exc.kind, http_status=exc.http_status,
                    code=getattr(exc, 'provider_error_code', exc.code), retryable=exc.retryable)
                self.log('api_attempts.jsonl', failure)
                if not exc.retryable or attempt == 3:
                    with self.lock:
                        self.fatal_errors.append(failure)
                    raise
                time.sleep(2 * attempt)

    def one(self, messages, cap, temperature):
        return self.invoke(build_request(messages, cap, temperature), 'memory')
