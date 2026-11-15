"""Per-record, resumable GPT/DeepSeek calls with short IDs and safe API errors."""
import json
from pathlib import Path
import time

from scripts.memory import hippomm as base, hippomm_inputs as inputs
from scripts.evaluation import openai_gpt5_base_transport as gpt
from scripts.evaluation import deepseek_flash_seed42_audited_transport as deep
from scripts.evaluation.mem_gallery_master_reader import READER_SYSTEM

MODELS = {'gpt5': 'gpt-5-2025-08-07', 'deepseek': deep.DEEPSEEK_MODEL}


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
            # Transport fingerprints only validate bytes locally; they never enter
            # prompts or new cache records. Original image bytes go on the wire.
            images = [{'path': str(p), 'sha256': inputs.sha(p)} for p in image_paths]
            payload = {'system': READER_SYSTEM, 'user_text': prompt, 'query_images': images}
            kwargs = {'seed': 42} if self.backbone == 'gpt5' else {'model': deep.DEEPSEEK_MODEL}
            wire = self.api.build_request(payload, max_output_tokens=self.token_limit,
                                          image_detail='high', **kwargs)
            response = None
            for attempt in range(1, 5):
                try:
                    response = self.api.chat_request(self.config, wire, timeout=60)
                    break
                except self.api.APIError as exc:
                    base.append(self.directory / 'api_attempts.jsonl', {'call_id': call_id,
                        'attempt': attempt, 'kind': exc.kind, 'retryable': exc.retryable,
                        'http_status': exc.http_status, 'code': exc.code})
                    if not exc.retryable or attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 30))
            response = {k: v for k, v in response.items()
                        if k not in ('request_sha256', 'system_fingerprint')}
            row = {'call_id': call_id, 'request': request, 'response': response}
            base.dump(path, row)
            self.index[key] = call_id
        self.last_call_id = call_id
        response = row['response']
        if response['returned_model'] != MODELS[self.backbone]:
            raise ValueError('Returned model differs from the caption backbone')
        if response['finish_reason'] != 'stop' or response.get('refusal'):
            raise ValueError('API response did not finish normally; raw output preserved: ' + str(path))
        if not response['raw_response'].strip():
            raise ValueError('Empty API response; raw output preserved: ' + str(path))
        return response['raw_response']
