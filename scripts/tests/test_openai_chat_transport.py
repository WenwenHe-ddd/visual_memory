"""Offline boundary tests; no credentials or external network are used."""
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

from scripts.evaluation.openai_chat_transport import (
    APIConfig, APIError, _NoRedirect, build_request, chat_request, load_config,
)


class OpenAIChatTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = APIConfig('gpt-4o-mini', 'https://api.openai.com/v1', 'FAKE_SECRET')
        self.request = build_request('system prompt', 'question', [], model=self.config.model)

    def test_allowlisted_env_precedence_and_secret_safe_config(self):
        env = self.root / '.env'
        env.write_text('export OPENAI_API_KEY="FILE_SECRET"\nOPENAI_MODEL=gpt-4o-mini\n'
                       'OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions\n'
                       'OPENAI_MODEL=ignored_duplicate\nPRIVATE_DATA_PATH=/do/not/use\n')
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'ENV_SECRET'}, clear=True):
            config = load_config(env)
            self.assertEqual(config.api_key, 'ENV_SECRET')
            self.assertEqual(config.model, 'gpt-4o-mini')
            self.assertEqual(config.base_url, 'https://api.openai.com/v1')
            self.assertNotIn('PRIVATE_DATA_PATH', os.environ)
            self.assertNotIn('OPENAI_MODEL', os.environ)
        public = json.dumps(config.public_dict()) + repr(config)
        self.assertNotIn('ENV_SECRET', public)
        self.assertNotIn('FILE_SECRET', public)

    def test_explicit_codex_model_needs_its_endpoint(self):
        env = self.root / '.env'
        env.write_text('OPENAI_MODEL=gpt-4o-mini\nOPENAI_API_KEY=ONE\n'
                       'OPENAI_BASE_URL=https://api.openai.com/v1\nCODEX_API_KEY=TWO\n')
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'explicit CODEX_BASE_URL'):
                load_config(env, model='gpt-5.4')
            config = load_config(env, model='gpt-5.4', base_url='https://example.test/v1')
        self.assertEqual(config.credential_variable, 'CODEX_API_KEY')
        self.assertEqual(config.api_key, 'TWO')

    def test_original_png_jpeg_bytes_precede_text(self):
        images, originals = [], [b'\x89PNG\r\n\x1a\noriginal PNG', b'\xff\xd8\xfforiginal JPEG']
        for index, data in enumerate(originals):
            path = self.root / str(index)
            path.write_bytes(data)
            images.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()})
        request = build_request('S', 'U', images, model='gpt-4o-mini')
        content = request['messages'][1]['content']
        self.assertEqual([p['type'] for p in content], ['image_url', 'image_url', 'text'])
        for item, expected in zip(content, originals):
            self.assertEqual(base64.b64decode(item['image_url']['url'].split(',', 1)[1]), expected)
            self.assertEqual(item['image_url']['detail'], 'high')
        self.assertEqual(set(request), {'model', 'messages', 'temperature', 'seed', 'max_tokens', 'n', 'stream', 'store'})
        self.assertEqual(request['seed'], 42)
        self.assertEqual(request['max_tokens'], 16)
        self.assertFalse(request['store'])
        Path(images[0]['path']).write_bytes(originals[0] + b'tamper')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            build_request('S', 'U', images, model='gpt-4o-mini')

    def test_no_evaluation_fields_or_external_images(self):
        image = {'path': 'unused', 'sha256': 'a' * 64, 'gold': 'SECRET_ANSWER'}
        with self.assertRaises(ValueError):
            build_request('S', 'U', [image], model='gpt-4o-mini')
        self.request['gold'] = 'SECRET_ANSWER'
        with patch('urllib.request.build_opener') as opener:
            with self.assertRaises(ValueError):
                chat_request(self.config, self.request)
            opener.assert_not_called()

    def test_single_http_call_and_normalized_response(self):
        payload = {'id': 'chatcmpl-test', 'model': 'gpt-4o-mini-2024-07-18',
                   'system_fingerprint': 'fp-test', 'usage': {'prompt_tokens': 10, 'completion_tokens': 1},
                   'choices': [{'finish_reason': 'stop', 'message': {'content': 'B', 'refusal': None}}]}
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode()
        response.headers.get.return_value = 'req-test'
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = response
        with patch('urllib.request.build_opener', return_value=opener):
            result = chat_request(self.config, self.request, timeout=23)
        opener.open.assert_called_once()
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'https://api.openai.com/v1/chat/completions')
        self.assertEqual(json.loads(request.data), self.request)
        self.assertEqual(opener.open.call_args.kwargs['timeout'], 23)
        self.assertEqual(result['raw_response'], 'B')
        self.assertEqual(result['returned_model'], payload['model'])
        self.assertEqual(result['request_id'], 'req-test')
        self.assertNotIn('FAKE_SECRET', json.dumps(result))

    def test_errors_never_expose_provider_body_or_credentials(self):
        cases = [(401, 'invalid_api_key', 'authentication', False),
                 (429, 'insufficient_quota', 'quota', False),
                 (429, 'rate_limit_exceeded', 'rate_limit', True),
                 (503, 'server_error', 'server', True),
                 (404, 'model_not_found', 'not_found', False),
                 (302, None, 'redirect', False),
                 (400, 'FAKE_SECRET', 'http', False)]
        for status, code, kind, retryable in cases:
            with self.subTest(status=status, code=code):
                body = json.dumps({'error': {'code': code, 'message': 'FAKE_SECRET USER_PROMPT'}}).encode()
                failure = urllib.error.HTTPError('https://api.openai.com/v1', status,
                                                'FAKE_SECRET', {}, io.BytesIO(body))
                opener = MagicMock()
                opener.open.side_effect = failure
                with patch('urllib.request.build_opener', return_value=opener):
                    with self.assertRaises(APIError) as raised:
                        chat_request(self.config, self.request)
                error = raised.exception
                self.assertEqual((error.kind, error.retryable), (kind, retryable))
                self.assertEqual(error.http_status, status)
                public = str(error) + repr(error) + json.dumps(vars(error))
                self.assertNotIn('FAKE_SECRET', public)
                self.assertNotIn('USER_PROMPT', public)
                opener.open.assert_called_once()

    def test_network_and_invalid_response_are_safe(self):
        opener = MagicMock()
        opener.open.side_effect = urllib.error.URLError('FAKE_SECRET in network error')
        with patch('urllib.request.build_opener', return_value=opener):
            with self.assertRaises(APIError) as raised:
                chat_request(self.config, self.request)
        self.assertEqual(raised.exception.kind, 'network')
        self.assertTrue(raised.exception.retryable)
        self.assertNotIn('FAKE_SECRET', str(raised.exception))
        response = MagicMock()
        response.read.return_value = b'FAKE_SECRET malformed payload'
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = response
        with patch('urllib.request.build_opener', return_value=opener):
            with self.assertRaises(APIError) as raised:
                chat_request(self.config, self.request)
        self.assertEqual(raised.exception.kind, 'protocol')
        self.assertFalse(raised.exception.retryable)
        self.assertNotIn('FAKE_SECRET', str(raised.exception))

    def test_redirects_disabled_and_remote_http_rejected(self):
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.test'))
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'HTTPS'):
                load_config(None, base_url='http://example.test/v1')
            with self.assertRaisesRegex(ValueError, 'embedded credentials'):
                load_config(None, base_url='https://user:password@example.test/v1')


if __name__ == '__main__':
    unittest.main()
