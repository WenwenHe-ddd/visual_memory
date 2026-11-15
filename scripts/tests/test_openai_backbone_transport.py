"""Offline tests of exact caption/reader messages and official model routes."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from scripts.evaluation.openai_backbone_transport import (
    APIConfig, APIError, GPT5_MODEL, MINI_MODEL, OFFICIAL_BASE_URL,
    build_request, canonical_model, chat_request, load_config, request_sha256,
)


class BackboneTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / '.env'
        self.env.write_text('OPENAI_API_KEY=FAKE_OFFICIAL_SECRET\n'
                            'OPENAI_BASE_URL=https://unexpected.example/v1\n'
                            'CODEX_API_KEY=FAKE_CODEX_SECRET\n'
                            'CODEX_BASE_URL=https://third-party.example/v1\n')
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.data = b'\x89PNG\r\n\x1a\noriginal caption image bytes'
        self.path = self.root / 'incoming.png'
        self.path.write_bytes(self.data)
        self.image = {'path': str(self.path), 'sha256': hashlib.sha256(self.data).hexdigest(),
                      'mime_type': 'image/png'}
        self.payload = {'record_id': 'internal-id', 'user_text': 'Original caption prompt.',
                        'query_images': [self.image]}

    def test_both_aliases_pin_snapshots_and_force_official_openai_credentials(self):
        for alias, expected in (('gpt-4o-mini', MINI_MODEL), ('gpt-5.4', GPT5_MODEL)):
            config = load_config(self.env, alias)
            self.assertEqual(config.model, expected)
            self.assertEqual(config.base_url, OFFICIAL_BASE_URL)
            self.assertEqual(config.credential_variable, 'OPENAI_API_KEY')
            self.assertEqual(config.api_key, 'FAKE_OFFICIAL_SECRET')
            self.assertNotIn('FAKE_OFFICIAL_SECRET', repr(config) + json.dumps(config.public_dict()))
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'PROCESS_SECRET'}):
            self.assertEqual(load_config(self.env, MINI_MODEL).api_key, 'PROCESS_SECRET')
        with self.assertRaisesRegex(ValueError, 'official OpenAI endpoint'):
            load_config(self.env, MINI_MODEL, base_url='https://third-party.example/v1')
        with self.assertRaises(ValueError):
            canonical_model('gpt-5.4-latest')

    def test_caption_user_only_exact_image_and_original_prompt(self):
        for model in (MINI_MODEL, GPT5_MODEL):
            request = build_request(self.payload, model=model, max_output_tokens=256, image_detail='high')
            self.assertEqual(len(request['messages']), 1)
            self.assertEqual(request['messages'][0]['role'], 'user')
            content = request['messages'][0]['content']
            self.assertEqual([part['type'] for part in content], ['image_url', 'text'])
            self.assertEqual(content[-1]['text'], self.payload['user_text'])
            self.assertEqual(base64.b64decode(content[0]['image_url']['url'].split(',', 1)[1]), self.data)
            self.assertEqual(content[0]['image_url']['detail'], 'high')
            serialized = json.dumps(request)
            self.assertNotIn('internal-id', serialized)
            self.assertNotIn(str(self.path), serialized)
            self.assertNotIn('system', [m['role'] for m in request['messages']])
        self.assertEqual(build_request(self.payload, model=MINI_MODEL)['messages'][0]['content'][0]['image_url']['detail'], 'auto')

    def test_reader_retains_system_and_image_before_text(self):
        payload = {**self.payload, 'system': 'Original reader system.', 'user_text': 'Question plus evidence.'}
        request = build_request(payload, model=MINI_MODEL, max_output_tokens=16)
        self.assertEqual([m['role'] for m in request['messages']], ['system', 'user'])
        self.assertEqual(request['messages'][0]['content'], payload['system'])
        self.assertEqual(request['messages'][1]['content'][-1]['text'], payload['user_text'])
        self.assertEqual(request['max_tokens'], 16)
        text_only = build_request({'user_text': 'Question'}, model=MINI_MODEL)
        self.assertEqual(text_only['messages'], [{'role': 'user', 'content': [{'type': 'text', 'text': 'Question'}]}])

    def test_model_specific_generation_fields(self):
        mini = build_request(self.payload, model=MINI_MODEL)
        gpt5 = build_request(self.payload, model=GPT5_MODEL)
        self.assertEqual((mini['max_tokens'], mini['seed'], mini['temperature']), (256, 42, 0))
        self.assertNotIn('max_completion_tokens', mini)
        self.assertNotIn('reasoning_effort', mini)
        self.assertEqual((gpt5['max_completion_tokens'], gpt5['reasoning_effort'], gpt5['temperature']), (256, 'none', 0))
        self.assertNotIn('seed', gpt5)
        self.assertNotIn('max_tokens', gpt5)
        for request in (mini, gpt5):
            self.assertEqual(request['n'], 1)
            self.assertFalse(request['stream'])
            self.assertFalse(request['store'])

    def test_gold_mime_and_image_hash_tamper_are_rejected(self):
        with self.assertRaises(ValueError):
            build_request({**self.payload, 'gold_choice': 'A'}, model=MINI_MODEL)
        with self.assertRaises(ValueError):
            build_request({**self.payload, 'query_images': [{**self.image, 'gold_caption': 'secret'}]}, model=MINI_MODEL)
        with self.assertRaisesRegex(ValueError, 'MIME type differs'):
            build_request({**self.payload, 'query_images': [{**self.image, 'mime_type': 'image/jpeg'}]}, model=MINI_MODEL)
        self.path.write_bytes(self.data + b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            build_request(self.payload, model=MINI_MODEL)

    @staticmethod
    def opener_for(model, content='A', reasoning=0):
        payload = {'id': 'response-test', 'model': model,
                   'choices': [{'finish_reason': 'stop', 'message': {'content': content, 'refusal': None}}],
                   'usage': {'prompt_tokens': 50, 'completion_tokens': 1, 'total_tokens': 51,
                             'completion_tokens_details': {'reasoning_tokens': reasoning}}}
        opener, response = MagicMock(), MagicMock()
        response.read.return_value = json.dumps(payload).encode()
        response.headers.get.return_value = 'request-test'
        opener.open.return_value.__enter__.return_value = response
        return opener

    def test_single_call_preserves_wire_and_normalizes_response_audit(self):
        for model in (MINI_MODEL, GPT5_MODEL):
            with self.subTest(model=model):
                config = load_config(self.env, model)
                request = build_request(self.payload, model=model)
                opener = self.opener_for(model, 'A complete caption.')
                with patch('urllib.request.build_opener', return_value=opener) as factory:
                    result = chat_request(config, request, timeout=25)
                opener.open.assert_called_once()
                sent = opener.open.call_args.args[0]
                self.assertEqual(sent.full_url, OFFICIAL_BASE_URL + '/chat/completions')
                self.assertEqual(json.loads(sent.data), request)
                self.assertEqual(result['raw_response'], 'A complete caption.')
                self.assertEqual(result['request_sha256'], request_sha256(request))
                self.assertEqual(result['returned_model'], model)
                self.assertEqual((result['attempts'], result['retry_count']), (1, 0))
                self.assertEqual(result['reasoning_tokens'], 0)
                self.assertIsNone(factory.call_args.args[0].redirect_request(None, None, 302, '', {}, 'https://other.test'))
                self.assertNotIn('FAKE_OFFICIAL_SECRET', json.dumps(result))

    def test_snapshot_mismatch_is_permanent(self):
        config = load_config(self.env, GPT5_MODEL)
        request = build_request(self.payload, model=GPT5_MODEL)
        opener = self.opener_for('gpt-5.4-2026-04-01')
        with patch('urllib.request.build_opener', return_value=opener):
            with self.assertRaises(APIError) as raised:
                chat_request(config, request)
        self.assertEqual(raised.exception.kind, 'model_mismatch')
        self.assertFalse(raised.exception.retryable)
        opener.open.assert_called_once()

    def test_tampered_schema_or_endpoint_rejected_before_network(self):
        config = load_config(self.env, MINI_MODEL)
        request = build_request(self.payload, model=MINI_MODEL)
        variants = [{**request, 'gold': 'A'}, {**request, 'model': GPT5_MODEL},
                    {**request, 'seed': 1}, {**request, 'store': True},
                    {**request, 'messages': [{'role': 'assistant', 'content': 'A'}]}]
        with patch('urllib.request.build_opener') as opener:
            for variant in variants:
                with self.subTest(variant=list(variant)), self.assertRaises(ValueError):
                    chat_request(config, variant)
            with self.assertRaises(ValueError):
                chat_request(APIConfig(MINI_MODEL, 'https://third-party.example/v1', 'KEY'), request)
            with self.assertRaises(ValueError):
                chat_request(APIConfig(MINI_MODEL, OFFICIAL_BASE_URL, 'KEY', 'CODEX_API_KEY'), request)
            opener.assert_not_called()

    def test_http_failure_is_sanitized_without_internal_retries(self):
        config = load_config(self.env, MINI_MODEL)
        request = build_request(self.payload, model=MINI_MODEL)
        body = json.dumps({'error': {'code': 'insufficient_quota', 'message': 'FAKE_OFFICIAL_SECRET PROMPT'}}).encode()
        opener = MagicMock()
        opener.open.side_effect = urllib.error.HTTPError(OFFICIAL_BASE_URL, 429, 'FAKE_OFFICIAL_SECRET', {}, io.BytesIO(body))
        with patch('urllib.request.build_opener', return_value=opener):
            with self.assertRaises(APIError) as raised:
                chat_request(config, request)
        opener.open.assert_called_once()
        self.assertEqual(raised.exception.kind, 'quota')
        self.assertFalse(raised.exception.retryable)
        self.assertNotIn('FAKE_OFFICIAL_SECRET', str(raised.exception) + json.dumps(vars(raised.exception)))


if __name__ == '__main__':
    unittest.main()
