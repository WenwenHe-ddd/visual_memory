"""Gallery API calls: original-byte MIME support, bounded DeepSeek images, explicit rejections."""
import json
import base64
import io
from pathlib import Path
import time

from scripts.memory import hippomm as base, hippomm_inputs as inputs
from scripts.evaluation import openai_gpt5_base_transport as gpt
from scripts.evaluation import deepseek_flash_seed42_audited_transport as deep
from scripts.evaluation.mem_gallery_master_reader import READER_SYSTEM
from scripts.evaluation import gallery_gpt5_transport as gpt_chat
from scripts.evaluation import gallery_deepseek_transport as deep_chat
from scripts.evaluation.gallery_deepseek_bounded_images import jpeg_to_limit, MAX_IMAGE_BYTES
from PIL import Image

MODELS = {'gpt5': 'gpt-5-2025-08-07', 'deepseek': deep.DEEPSEEK_MODEL}


class ProviderRejected(RuntimeError):
    """An API rejection is an evaluated failure, never a generated answer."""


def encoded_image(path, backbone):
    data = Path(path).read_bytes()
    with Image.open(io.BytesIO(data)) as source:
        if getattr(source, 'n_frames', 1) != 1 and source.format != 'MPO':
            raise ValueError('Animated input requires an explicit frame policy')
        mime = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp',
                'GIF': 'image/gif', 'MPO': 'image/jpeg'}.get(source.format)
        if mime is None:
            raise ValueError('Unsupported input image format')
        metadata = {'original_bytes': len(data), 'dimensions': list(source.size),
                    'resized': False, 'cropped': False, 'reencoded': False,
                    'source_format': source.format, 'source_frames': getattr(source, 'n_frames', 1),
                    'primary_image_index': 0}
    if backbone == 'deepseek' and len(data) > MAX_IMAGE_BYTES:
        data, details = jpeg_to_limit(data)
        metadata.update(details, reencoded=True)
        mime = 'image/jpeg'
    metadata.update(encoded_bytes=len(data), mime_type=mime)
    return data, metadata


def build_wire(api, backbone, prompt, paths, token_limit):
    kwargs = {'seed': 42} if backbone == 'gpt5' else {'model': deep.DEEPSEEK_MODEL}
    wire = api.build_request({'system': READER_SYSTEM, 'user_text': prompt, 'query_images': []},
                             max_output_tokens=token_limit, image_detail='high', **kwargs)
    parts, encodings = [], []
    for path in paths:
        data, metadata = encoded_image(path, backbone)
        parts.append({'type': 'image_url', 'image_url': {'detail': 'high',
            'url': 'data:' + metadata['mime_type'] + ';base64,' + base64.b64encode(data).decode()}})
        encodings.append(metadata)
    wire['messages'][-1]['content'] = parts + wire['messages'][-1]['content']
    return wire, encodings


class APIChat:
    """One instance per event/question, so parallel jobs never share cache files."""

    def __init__(self, directory, backbone, env_file, token_limit=2048):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.backbone, self.token_limit = backbone, token_limit
        self.api = gpt if backbone == 'gpt5' else deep
        self.config = self.api.load_config(env_file)
        self.last_call_id = None
        self.index = {}
        for path in sorted(self.directory.glob('call_*.json')):
            row = json.loads(path.read_text())
            key = json.dumps(row['request'], sort_keys=True, ensure_ascii=False)
            if key in self.index:
                raise ValueError('Duplicate API request in record cache')
            self.index[key] = path.stem

    def __call__(self, role, prompt, image_paths=()):
        request = {'role': role, 'prompt': prompt, 'image_paths': [str(p) for p in image_paths],
                   'backbone': self.backbone, 'max_output_tokens': self.token_limit}
        key = json.dumps(request, sort_keys=True, ensure_ascii=False)
        call_id = self.index.get(key, f'call_{len(self.index) + 1:04d}')
        path = self.directory / (call_id + '.json')
        if path.exists():
            row = json.loads(path.read_text())
            if row['request'] != request:
                raise ValueError('API cache request differs')
        else:
            wire, encodings = build_wire(self.api, self.backbone, prompt, image_paths, self.token_limit)
            response = None
            for attempt in range(1, 5):
                try:
                    response = (gpt_chat.chat(self.config, wire, timeout=120) if self.backbone == 'gpt5'
                                else deep_chat.chat(self.config, wire))
                    break
                except self.api.APIError as exc:
                    base.append(self.directory / 'api_attempts.jsonl', {'call_id': call_id,
                        'attempt': attempt, 'kind': exc.kind, 'retryable': exc.retryable,
                        'http_status': exc.http_status, 'code': exc.code,
                        'provider_error_code': getattr(exc, 'provider_error_code', None)})
                    if getattr(exc, 'provider_error_code', None) == 'data_inspection_failed':
                        response = {'status': 'provider_rejection', 'raw_response': '',
                                    'returned_model': None, 'requested_model': MODELS[self.backbone],
                                    'finish_reason': 'provider_rejection',
                                    'refusal': 'provider_data_inspection_failed'}
                        break
                    if not exc.retryable or attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 30))
            response = {k: v for k, v in response.items()
                        if k not in ('request_sha256', 'system_fingerprint')}
            row = {'call_id': call_id, 'request': request, 'response': response, 'image_encoding': encodings}
            base.dump(path, row)
            self.index[key] = call_id
        self.last_call_id = call_id
        response = row['response']
        if response.get('status') == 'provider_rejection':
            raise ProviderRejected('Provider rejected ' + role + '; outcome preserved: ' + str(path))
        if response['returned_model'] != MODELS[self.backbone]:
            raise ValueError('Returned model differs from the caption backbone')
        if response['finish_reason'] != 'stop' or response.get('refusal'):
            raise ValueError('API response did not finish normally; raw output preserved: ' + str(path))
        if not response['raw_response'].strip():
            raise ValueError('Empty API response; raw output preserved: ' + str(path))
        return response['raw_response']
